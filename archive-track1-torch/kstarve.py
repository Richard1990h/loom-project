"""
Knowledge-starving experiment — the decisive test of the separation thesis.

Claim under test: if the reasoning core is too small to memorise a fact set,
a model WITH product-key memory will store the facts in memory (high recall,
collapsing when memory is ablated), while an identical core WITHOUT memory
saturates at its capacity ceiling.

Falsifiable predictions, stated before running:
  P1  model A (with memory): recall high (>= 0.90)
  P2  model A with memory DISABLED at eval: recall collapses (< 0.05)
      -> the facts live in the store, not the core
  P3  model B (identical core, memory off for all of training): recall
      substantially below A -> the core alone cannot hold the set
KILL CRITERION: if A <= B, memory bought nothing under starvation; the
separation thesis fails at this scale.

Facts: 6,000 random triples (3-token subject -> 2-token object) over a
54-symbol content alphabet. Information content ~ 6000 x 11.5 bits = 69 kbit.
Core (excluding memory): ~25k params. By the ~2 bits/param storage heuristic
the core alone should sit near ~50 kbit — deliberately below the fact set.

Sequence per fact: [Q s1 s2 s3 A o1 o2], next-token training on the full
sequence. Recall = argmax prediction of o1 then (teacher-forced) o2; a fact
counts only if BOTH are correct. Chance = (1/54)^2 = 0.034%.

Stages (resumable): A trains the memory model, B trains the no-memory twin,
C evaluates both + ablation and writes kstarve.json. Each stage checkpoints
on a time guard so no sandbox window is exceeded.
"""

import json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/claude/loom')
from loom import LoomLM, n_params

torch.manual_seed(7)
torch.set_num_threads(1)

ROOT = '/home/claude/loom'
CK = f'{ROOT}/kck'
os.makedirs(CK, exist_ok=True)

VOCAB, Q_TOK, A_TOK, LO = 64, 1, 2, 10          # content symbols: 10..63 (54)
N_FACTS, STEPS_MAX, BATCH, LR = 6000, 9000, 256, 5e-3
TIME_GUARD_S = 200.0

# ------------------------------------------------- deterministic fact table
g = torch.Generator().manual_seed(123)
subs = set()
rows = []
while len(rows) < N_FACTS:                       # unique 3-token subjects
    s = tuple(torch.randint(LO, VOCAB, (3,), generator=g).tolist())
    if s in subs:
        continue
    subs.add(s)
    o = torch.randint(LO, VOCAB, (2,), generator=g).tolist()
    rows.append([Q_TOK, *s, A_TOK, *o])
FACTS = torch.tensor(rows)                       # (N, 7)


def make_model():
    torch.manual_seed(31337)
    return LoomLM(VOCAB, d_model=32, n_head=4, n_blocks=1, ffn_mult=2,
                  k_max=2, ponder_tau=0.01, block_size=8,
                  mem_n_sub=64, mem_d_key=16, mem_topk=8, hyper_rank=2)


@torch.no_grad()
def recall(model, disable_memory, n=None):
    model.eval()
    f = FACTS if n is None else FACTS[:n]
    ok = torch.ones(len(f), dtype=torch.bool)
    for chunk in range(0, len(f), 1024):
        fc = f[chunk:chunk + 1024]
        l1, _, _ = model(fc[:, :5], disable_memory=disable_memory)
        p1 = l1[:, 4, :].argmax(-1)
        l2, _, _ = model(fc[:, :6], disable_memory=disable_memory)
        p2 = l2[:, 5, :].argmax(-1)
        ok[chunk:chunk + 1024] = (p1 == fc[:, 5]) & (p2 == fc[:, 6])
    model.train()
    return ok.float().mean().item()


def train(tag, disable_memory):
    path = f'{CK}/{tag}.pt'
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            betas=(0.9, 0.95), weight_decay=0.01)
    start, gen_state = 0, torch.Generator().manual_seed(555)
    if os.path.exists(path):
        ck = torch.load(path, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start = ck['step']
        gen_state.set_state(ck['gen'])
        if ck.get('done'):
            print(f"[{tag}] already trained ({start} steps)")
            return
    model.train()
    t0 = time.time()
    for step in range(start, STEPS_MAX):
        ix = torch.randint(N_FACTS, (BATCH,), generator=gen_state)
        seq = FACTS[ix]
        x, y = seq[:, :-1], seq[:, 1:]
        opt.zero_grad(set_to_none=True)
        # Masked objective (declared): subject tokens are random noise and
        # unpredictable by construction; loss is computed only on the
        # answerable positions A, o1, o2 (target indices 3,4,5). The ACT
        # ponder cost is omitted in this experiment (recall is the metric,
        # not compute efficiency).
        logits, _, _ = model(x, disable_memory=disable_memory)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                             reduction='none').view(len(ix), -1)
        loss = ce[:, 3:].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == STEPS_MAX - 1:
            r = recall(model, disable_memory, n=1024)
            print(f"[{tag}] step {step:4d}  loss {loss.item():.3f}  "
                  f"recall@1024 {r:.3f}", flush=True)
            if r > 0.995:
                torch.save({'model': model.state_dict(),
                            'opt': opt.state_dict(), 'step': step + 1,
                            'gen': gen_state.get_state(), 'done': True}, path)
                print(f"[{tag}] converged at step {step}")
                return
        if time.time() - t0 > TIME_GUARD_S:
            torch.save({'model': model.state_dict(),
                        'opt': opt.state_dict(), 'step': step + 1,
                        'gen': gen_state.get_state(), 'done': False}, path)
            print(f"[{tag}] time guard at step {step}; resumable")
            sys.exit(0)
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                'step': STEPS_MAX, 'gen': gen_state.get_state(),
                'done': True}, path)
    print(f"[{tag}] finished {STEPS_MAX} steps")


def stage_done(tag):
    p = f'{CK}/{tag}.pt'
    return os.path.exists(p) and torch.load(p, weights_only=False).get('done')


# ------------------------------------------------------------- run stages
if not stage_done('A_mem'):
    m = make_model()
    core = n_params(m) - sum(p.numel() for p in m.memory.parameters())
    print(f"core (non-memory) params: {core:,}  |  "
          f"memory params: {sum(p.numel() for p in m.memory.parameters()):,}"
          f"  |  facts: {N_FACTS} (~69 kbit)")
    del m
    train('A_mem', disable_memory=False)
    if not stage_done('A_mem'):
        sys.exit(0)

if not stage_done('B_nomem'):
    train('B_nomem', disable_memory=True)
    if not stage_done('B_nomem'):
        sys.exit(0)

# ------------------------------------------------------------- stage C
mA, mB = make_model(), make_model()
mA.load_state_dict(torch.load(f'{CK}/A_mem.pt', weights_only=False)['model'])
mB.load_state_dict(torch.load(f'{CK}/B_nomem.pt', weights_only=False)['model'])

res = {
    'facts': N_FACTS,
    'chance_both_tokens': (1 / 54) ** 2,
    'core_params_non_memory':
        n_params(mA) - sum(p.numel() for p in mA.memory.parameters()),
    'memory_params': sum(p.numel() for p in mA.memory.parameters()),
    'A_with_memory_recall': recall(mA, disable_memory=False),
    'A_memory_ablated_recall': recall(mA, disable_memory=True),
    'B_never_memory_recall': recall(mB, disable_memory=True),
    'steps_A': torch.load(f'{CK}/A_mem.pt', weights_only=False)['step'],
    'steps_B': torch.load(f'{CK}/B_nomem.pt', weights_only=False)['step'],
}

# where do the facts live? slot usage of the trained memory model
with torch.no_grad():
    mA.eval()
    _, _, st = mA(FACTS[:, :6])
    counts = torch.bincount(st['mem_indices'], minlength=4096)
    res['memory_slots_used'] = int((counts > 0).sum())

p1 = res['A_with_memory_recall'] >= 0.90
p2 = res['A_memory_ablated_recall'] < 0.05
p3 = res['A_with_memory_recall'] - res['B_never_memory_recall'] > 0.20
res['P1_memory_model_high_recall'] = bool(p1)
res['P2_ablation_collapses'] = bool(p2)
res['P3_core_alone_far_below'] = bool(p3)
res['verdict'] = ('SEPARATION THESIS SUPPORTED at this scale'
                  if (p1 and p2 and p3) else
                  'KILL CRITERION TRIGGERED — thesis fails at this scale'
                  if res['A_with_memory_recall'] <= res['B_never_memory_recall']
                  else 'MIXED — partial support, see individual predictions')

with open(f'{ROOT}/kstarve.json', 'w') as fp:
    json.dump(res, fp, indent=2)
print(json.dumps(res, indent=2))
