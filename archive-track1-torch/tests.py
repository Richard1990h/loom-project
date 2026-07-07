"""
Correctness proofs for every LOOM mechanism. Each test either passes with a
printed measured value or fails loudly. No mocks, no fixtures, no skips.

  1. PKM two-stage top-k == O(N) brute force (exactness of retrieval)
  2. ACT invariants: mixture weights sum to exactly 1; 1 <= steps <= k_max
  3. Hypernet zero-init is an exact identity (delta path wired correctly)
  4. Causality: perturbing future tokens leaves past logits bit-identical
     (covers attention, memory, carry, hypernet, halting simultaneously)
  5. Gradient flow: every parameter tensor receives nonzero gradient
  6. Weight tying: k_max=1 and k_max=8 models have identical param counts;
     no separate output head exists
  7. Halting head controls depth: biasing it moves mean steps as predicted
  8. End-to-end trainability: overfit one batch to near-zero loss
"""

import time
import torch
import torch.nn.functional as F

from loom import LoomLM, ProductKeyMemory, n_params

torch.manual_seed(1337)
VOCAB = 65
RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    assert ok, name


# ---------------------------------------------------------------- 1. PKM
def test_pkm_exact():
    mem = ProductKeyMemory(d_model=64, n_sub=16, d_key=16, topk=8)
    x = torch.randn(3, 11, 64)
    out, idx = mem(x)

    # brute force over all N = n_sub^2 = 256 slots
    s1, s2 = mem.sub_scores(x)                       # (B,T,16) each
    full = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(-2)   # (B,T,256)
    bf_sc, bf_idx = full.topk(mem.topk, dim=-1)
    bf_w = F.softmax(bf_sc, dim=-1)
    bf_out = (bf_w.unsqueeze(-1) * mem.values(bf_idx)).sum(dim=-2)

    same_sets = torch.equal(idx.sort(dim=-1).values,
                            bf_idx.sort(dim=-1).values)
    close = torch.allclose(out, bf_out, atol=1e-6)
    check("PKM two-stage == brute force",
          same_sets and close,
          f"indices identical over {idx.numel()} retrievals; "
          f"max|out-bf| = {(out - bf_out).abs().max():.2e}")


# ---------------------------------------------------------------- 2. ACT
def test_act_invariants():
    m = LoomLM(VOCAB, k_max=4)
    m.eval()
    x = torch.randint(0, VOCAB, (4, 48))
    _, _, st = m(x, targets=x)
    w_err = (st['w_sum'] - 1.0).abs().max().item()
    smin, smax = st['steps'].min().item(), st['steps'].max().item()
    check("ACT weights sum to 1", w_err < 1e-5, f"max|sum-1| = {w_err:.2e}")
    check("ACT step bounds", 1 <= smin and smax <= 4,
          f"steps in [{smin:.0f}, {smax:.0f}], k_max=4")


# ---------------------------------------------------------------- 3. hypernet
def test_hypernet_zero_init_identity():
    torch.manual_seed(7)
    m = LoomLM(VOCAB, k_max=3)
    m.eval()
    x = torch.randint(0, VOCAB, (2, 40))
    with torch.no_grad():
        l_on, _, _ = m(x)
        l_off, _, _ = m(x, disable_hyper=True)
    d = (l_on - l_off).abs().max().item()
    check("hypernet zero-init identity", d == 0.0,
          f"max|logits_on - logits_off| = {d:.1e} (exact)")


# ---------------------------------------------------------------- 4. causality
def test_causality():
    torch.manual_seed(11)
    m = LoomLM(VOCAB, k_max=4, block_size=96)
    m.eval()
    B, T, split = 2, 64, 40
    x1 = torch.randint(0, VOCAB, (B, T))
    x2 = x1.clone()
    x2[:, split:] = torch.randint(0, VOCAB, (B, T - split))
    assert not torch.equal(x1, x2)
    with torch.no_grad():
        l1, _, _ = m(x1)
        l2, _, _ = m(x2)
    d = (l1[:, :split] - l2[:, :split]).abs().max().item()
    check("causality (memory+carry+hyper+halt)", d == 0.0,
          f"future perturbed at t>={split}; max|Δlogits(t<{split})| = {d:.1e}")


# ---------------------------------------------------------------- 5. gradients
def test_gradient_flow():
    # With the hypernet's B-side zero-initialised (the identity guarantee of
    # test 3), the delta d = B(Ax)/r has dd/dA proportional to B = 0, so
    # to_A's gradient is mathematically zero at step 0 — the same lazy
    # wiring as LoRA. It must be EXACTLY that set at step 0, and empty by
    # step 2 once B has moved.
    torch.manual_seed(3)
    m = LoomLM(VOCAB, k_max=4, block_size=48)
    x = torch.randint(0, VOCAB, (2, 48))          # T == block_size: all pos used
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def dead():
        return sorted(n for n, p in m.named_parameters()
                      if p.grad is None or p.grad.abs().sum() == 0)

    _, loss, _ = m(x, targets=x)
    loss.backward()
    d0 = dead()
    check("step-0 zero-grad set == LoRA-lazy theory",
          d0 == ['hyper.to_A.bias', 'hyper.to_A.weight'],
          f"step-0 dead set: {d0}")
    opt.step()
    for _ in range(1):
        opt.zero_grad(set_to_none=True)
        _, loss, _ = m(x, targets=x)
        loss.backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    _, loss, _ = m(x, targets=x)
    loss.backward()
    d2 = dead()
    check("gradient reaches every parameter tensor by step 2", d2 == [],
          f"{sum(1 for _ in m.parameters())} tensors, dead after step 2: "
          f"{d2 or 'none'}")


# ---------------------------------------------------------------- 6. tying
def test_weight_tying():
    torch.manual_seed(5)
    a = LoomLM(VOCAB, k_max=1)
    torch.manual_seed(5)
    b = LoomLM(VOCAB, k_max=8)
    vocab_mats = [n for n, p in b.named_parameters()
                  if p.dim() == 2 and VOCAB in p.shape]
    check("recursion adds zero parameters", n_params(a) == n_params(b),
          f"k_max=1: {n_params(a):,} params == k_max=8: {n_params(b):,}")
    check("output head tied to embedding", vocab_mats == ['tok.weight'],
          f"vocab-sized matrices: {vocab_mats}")


# ---------------------------------------------------------------- 7. halting
def test_halting_controls_depth():
    torch.manual_seed(9)
    x = torch.randint(0, VOCAB, (4, 48))
    means = {}
    for bias in (+6.0, -6.0):
        m = LoomLM(VOCAB, k_max=4)
        with torch.no_grad():
            m.halt.bias.fill_(bias)
        m.eval()
        _, _, st = m(x, targets=x)
        means[bias] = st['steps'].mean().item()
    # ACT halts when CUMULATIVE p exceeds 1-eps = 0.99, so a one-step halt
    # needs sigmoid(bias) > 0.99, i.e. bias > 4.6. sigmoid(6) = 0.9975.
    # sigmoid(-6) = 0.0025 -> never crosses 0.99 -> forced halt at k_max.
    ok = means[+6.0] == 1.0 and means[-6.0] == 4.0
    check("halting head controls depth", ok,
          f"bias +6 -> mean steps {means[+6.0]:.2f} (predicted 1.00); "
          f"bias -6 -> mean steps {means[-6.0]:.2f} (predicted 4.00, k_max=4)")


# ---------------------------------------------------------------- 8. overfit
def test_overfit_one_batch():
    torch.manual_seed(21)
    m = LoomLM(VOCAB, k_max=4, block_size=48)
    x = torch.randint(0, VOCAB, (2, 48))
    y = torch.randint(0, VOCAB, (2, 48))          # arbitrary fixed mapping
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    m.train()
    t0, ce = time.time(), float('inf')
    for step in range(400):
        opt.zero_grad(set_to_none=True)
        _, loss, st = m(x, targets=y)
        loss.backward()
        opt.step()
        ce = st['ce'].item()
        if ce < 0.10:
            break
    check("end-to-end overfit (memorise 96 random targets)", ce < 0.15,
          f"CE {ce:.4f} after {step + 1} steps ({time.time() - t0:.0f}s); "
          f"chance = ln(65) = 4.17")


if __name__ == "__main__":
    t0 = time.time()
    test_pkm_exact()
    test_act_invariants()
    test_hypernet_zero_init_identity()
    test_causality()
    test_gradient_flow()
    test_weight_tying()
    test_halting_controls_depth()
    test_overfit_one_batch()
    n_ok = sum(ok for _, ok, _ in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} proofs passed in {time.time()-t0:.0f}s")
