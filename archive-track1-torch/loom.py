"""
LOOM — Looped Operations Over Memory. Complete reference implementation.

Mechanisms (all fully implemented, no stubs):
  1. Recursive core: n_blocks unique transformer blocks applied up to k_max
     times with SHARED weights (recursion adds depth, zero parameters).
  2. Adaptive halting (ACT, Graves 2016): per-token halting probability each
     loop; output is the halting-weighted mixture of loop states; ponder cost
     added to the loss. Weights provably sum to 1 per token (tests.py).
  3. Product-key memory (Lample et al. 2019): N = n_sub^2 value slots
     addressed by two sub-key tables. The two-stage top-k retrieval is EXACT
     for product-structured scores; proven against brute force in tests.py.
  4. Carry registers: a per-position latent state c_t persisting across
     recursion steps (GRU-style gated update). Per-position by design so
     causality is preserved — cross-position information flows only through
     causally masked attention. (The sequence-level registers in the original
     blueprint would leak future tokens into past positions; this is the
     corrected, causal formulation.)
  5. Hypernetwork ("Jacquard head"): reads the carry state and emits a
     per-token, per-loop rank-r additive delta on the shared FFN's first
     linear. Delta generator's B-side is zero-initialised, so at init the
     delta path is an exact no-op (proven in tests.py).

Everything runs in fp32 on CPU. No external dependencies beyond PyTorch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        def heads(t):
            return t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(heads(q), heads(k), heads(v),
                                           is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class DynamicFFN(nn.Module):
    """FFN whose first linear can receive a per-token rank-r additive delta:
         h = W1 x + B_t (A_t x) / r
    A_t: (B,T,r,d)  B_t: (B,T,m,r) are produced by the hypernetwork."""

    def __init__(self, d_model: int, mult: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, mult * d_model)
        self.w2 = nn.Linear(mult * d_model, d_model)

    def forward(self, x, A=None, Bm=None):
        h = self.w1(x)
        if A is not None:
            r = A.shape[2]
            u = torch.einsum('btrd,btd->btr', A, x)          # (B,T,r)
            h = h + torch.einsum('btmr,btr->btm', Bm, u) / r  # (B,T,m)
        return self.w2(F.gelu(h))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = DynamicFFN(d_model, mult)

    def forward(self, x, A=None, Bm=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x), A, Bm)
        return x


# --------------------------------------------------------------------------
# product-key memory
# --------------------------------------------------------------------------

class ProductKeyMemory(nn.Module):
    """N = n_sub^2 slots. Slot (i, j) has key concat(k1_i, k2_j) and score
    q1·k1_i + q2·k2_j. Retrieval: top-t on each sub-score, combine the t*t
    candidate sums, take global top-t. This is exact: if pair (i,j) is in the
    true global top-t, then i is in top-t of s1 (else >= t pairs (a,j) with
    s1_a > s1_i all beat it — contradiction), and symmetrically for j.
    tests.py verifies this against O(N) brute force."""

    def __init__(self, d_model: int, n_sub: int = 32, d_key: int = 32,
                 topk: int = 8, qnorm: bool = False, key_init: float = 0.02):
        super().__init__()
        self.n_sub, self.topk = n_sub, topk
        self.ln = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, 2 * d_key)
        self.qn1 = nn.LayerNorm(d_key) if qnorm else None
        self.qn2 = nn.LayerNorm(d_key) if qnorm else None
        self.sub_keys1 = nn.Parameter(torch.randn(n_sub, d_key) * key_init)
        self.sub_keys2 = nn.Parameter(torch.randn(n_sub, d_key) * key_init)
        self.values = nn.Embedding(n_sub * n_sub, d_model)
        nn.init.normal_(self.values.weight, std=0.02)

    def sub_scores(self, x):
        q1, q2 = self.query(self.ln(x)).chunk(2, dim=-1)
        if self.qn1 is not None:
            q1, q2 = self.qn1(q1), self.qn2(q2)
        return q1 @ self.sub_keys1.t(), q2 @ self.sub_keys2.t()  # (B,T,n_sub)

    def forward(self, x):
        t = self.topk
        s1, s2 = self.sub_scores(x)
        v1, i1 = s1.topk(t, dim=-1)                       # (B,T,t)
        v2, i2 = s2.topk(t, dim=-1)
        comb = v1.unsqueeze(-1) + v2.unsqueeze(-2)        # (B,T,t,t)
        sc, flat = comb.flatten(-2).topk(t, dim=-1)       # (B,T,t)
        a, b = flat // t, flat % t
        idx = i1.gather(-1, a) * self.n_sub + i2.gather(-1, b)
        w = F.softmax(sc, dim=-1)
        out = (w.unsqueeze(-1) * self.values(idx)).sum(dim=-2)
        return out, idx


# --------------------------------------------------------------------------
# carry registers + hypernetwork
# --------------------------------------------------------------------------

class CarryCell(nn.Module):
    """Per-position latent scratchpad, persistent across recursion steps."""

    def __init__(self, d_model: int):
        super().__init__()
        self.read = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(2 * d_model, d_model)
        self.cand = nn.Linear(2 * d_model, d_model)

    def inject(self, x, c):
        return x + self.read(c)

    def update(self, x, c):
        xc = torch.cat([x, c], dim=-1)
        z = torch.sigmoid(self.gate(xc))
        return (1.0 - z) * c + z * torch.tanh(self.cand(xc))


class Hypernet(nn.Module):
    """Emits per-token rank-r deltas for the shared core's FFN first linear
    (one delta per loop, applied to every block — 'one core, re-instructed').
    B-side zero-initialised: exact identity at init."""

    def __init__(self, d_model: int, mult: int, rank: int):
        super().__init__()
        self.rank, self.d, self.m = rank, d_model, mult * d_model
        self.to_A = nn.Linear(d_model, rank * d_model)
        self.to_B = nn.Linear(d_model, rank * self.m)
        nn.init.zeros_(self.to_B.weight)
        nn.init.zeros_(self.to_B.bias)

    def forward(self, c):
        B_, T, _ = c.shape
        A = self.to_A(c).view(B_, T, self.rank, self.d)
        Bm = self.to_B(c).view(B_, T, self.m, self.rank)
        return A, Bm


# --------------------------------------------------------------------------
# LOOM language model
# --------------------------------------------------------------------------

class LoomLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_head: int = 4,
                 n_blocks: int = 2, ffn_mult: int = 2, k_max: int = 4,
                 act_eps: float = 0.01, ponder_tau: float = 0.01,
                 block_size: int = 96, mem_n_sub: int = 32,
                 mem_d_key: int = 32, mem_topk: int = 8, hyper_rank: int = 2,
                 mem_qnorm: bool = False, mem_key_init: float = 0.02):
        super().__init__()
        self.k_max, self.act_eps, self.ponder_tau = k_max, act_eps, ponder_tau
        self.block_size = block_size

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_head, ffn_mult) for _ in range(n_blocks))
        self.memory = ProductKeyMemory(d_model, mem_n_sub, mem_d_key,
                                       mem_topk, mem_qnorm, mem_key_init)
        self.carry = CarryCell(d_model)
        self.hyper = Hypernet(d_model, ffn_mult, hyper_rank)
        self.halt_ln = nn.LayerNorm(d_model)
        self.halt = nn.Linear(d_model, 1)
        nn.init.constant_(self.halt.bias, -1.0)   # start pondering ~3 steps
        self.ln_f = nn.LayerNorm(d_model)
        # output head is TIED to self.tok.weight (no separate matrix)

        self.apply(self._init)
        # re-apply the deliberate zero inits clobbered by _init
        nn.init.zeros_(self.hyper.to_B.weight); nn.init.zeros_(self.hyper.to_B.bias)
        nn.init.constant_(self.halt.bias, -1.0)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---- one recursion step -------------------------------------------------
    def core_step(self, x, c, disable_memory: bool, disable_hyper: bool):
        x = self.carry.inject(x, c)
        A = Bm = None
        if not disable_hyper:
            A, Bm = self.hyper(c)
        for blk in self.blocks:
            x = blk(x, A, Bm)
        mem_idx = None
        if not disable_memory:
            m, mem_idx = self.memory(x)
            x = x + m
        c = self.carry.update(x, c)
        p = torch.sigmoid(self.halt(self.halt_ln(x))).squeeze(-1)  # (B,T)
        return x, c, p, mem_idx

    # ---- full forward with ACT ----------------------------------------------
    def forward(self, idx, targets=None, force_k=None,
                disable_memory: bool = False, disable_hyper: bool = False):
        B, T = idx.shape
        assert T <= self.block_size
        device = idx.device
        x = self.tok(idx) + self.pos(torch.arange(T, device=device))
        c = torch.zeros_like(x)
        k_max = force_k if force_k is not None else self.k_max

        acc = torch.zeros(B, T, device=device)      # cumulative halt prob
        still = torch.ones(B, T, device=device)     # 1.0 while running
        out = torch.zeros_like(x)
        w_sum = torch.zeros(B, T, device=device)
        n_steps = torch.zeros(B, T, device=device)
        remainder = torch.zeros(B, T, device=device)
        mem_indices = []
        loops_executed = 0

        for n in range(k_max):
            x, c, p, mem_idx = self.core_step(x, c, disable_memory,
                                              disable_hyper)
            if mem_idx is not None:
                mem_indices.append(mem_idx)
            loops_executed = n + 1
            last = (n == k_max - 1)

            if force_k is not None:
                # fixed-depth ablation: only the final state is the output
                if last:
                    out = x
                    n_steps = torch.full_like(n_steps, float(k_max))
                    w_sum = torch.ones_like(w_sum)
                continue

            if last:
                will_halt = torch.ones_like(p, dtype=torch.bool)
            else:
                will_halt = (acc + p) > (1.0 - self.act_eps)
            halted_now = still * will_halt.float()
            w = halted_now * (1.0 - acc) + (still - halted_now) * p
            out = out + w.unsqueeze(-1) * x
            w_sum = w_sum + w
            remainder = remainder + halted_now * (1.0 - acc)
            acc = acc + p * still
            n_steps = n_steps + still
            still = still * (1.0 - halted_now)

            if (not self.training) and still.sum() == 0:
                break  # batch-level early exit at inference

        y = self.ln_f(out)
        logits = F.linear(y, self.tok.weight)

        stats = {
            'steps': n_steps.detach(),
            'w_sum': w_sum.detach(),
            'loops_executed': loops_executed,
            'mem_indices': (torch.cat([m.reshape(-1) for m in mem_indices])
                            if mem_indices else None),
        }
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 targets.reshape(-1),
                                 reduction='none').view(B, T)
            stats['ce'] = ce.mean().detach()
            stats['ce_per_token'] = ce.detach()
            loss = ce.mean()
            if force_k is None:
                ponder = (n_steps + remainder).mean()
                stats['ponder'] = ponder.detach()
                loss = loss + self.ponder_tau * ponder
        return logits, loss, stats


# --------------------------------------------------------------------------
# parameter-matched vanilla baseline
# --------------------------------------------------------------------------

class VanillaLM(nn.Module):
    """Standard pre-LN transformer, tied head, same embeddings/positions.
    Layer count / FFN mult chosen in train.py so total params match LOOM."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_head: int = 4,
                 n_layer: int = 3, ffn_mult: int = 4, block_size: int = 96):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_head, ffn_mult) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.apply(LoomLM._init)

    def forward(self, idx, targets=None, **_):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = F.linear(self.ln_f(x), self.tok.weight)
        stats, loss = {}, None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 targets.reshape(-1),
                                 reduction='none').view(B, T)
            loss = ce.mean()
            stats['ce'] = ce.mean().detach()
            stats['ce_per_token'] = ce.detach()
        return logits, loss, stats


# --------------------------------------------------------------------------
# shared utilities
# --------------------------------------------------------------------------

def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def param_breakdown(model: nn.Module) -> dict:
    groups = {}
    for name, p in model.named_parameters():
        top = name.split('.')[0]
        groups[top] = groups.get(top, 0) + p.numel()
    groups['TOTAL'] = sum(groups.values())
    return groups


@torch.no_grad()
def generate(model, idx, n_new: int, temperature: float = 0.8,
             top_k: int = 40):
    model.eval()
    for _ in range(n_new):
        ctx = idx[:, -model.block_size:]
        logits, _, _ = model(ctx)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx
