#!/usr/bin/env bash
# verify.sh — the project's self-enforcing guardrail (INSTRUMENT, not artifact).
# Fingerprint -> compile day1/2/3 with documented flags -> run -> parse their
# outputs against reference.json -> purity scan -> ZERO.md append-only check.
# Exits nonzero with a plain-language reason on any failure. CPU-only; GPU
# checks print SKIPPED-NO-GPU (covered by on-machine runs + sandbox review).
set -u
cd "$(dirname "$0")" || exit 2
FAIL=0
red(){ echo "FAIL: $*"; FAIL=1; }
ok(){ echo "ok   $*"; }

echo "== [1/6] fingerprint: required files =="
REQ="HANDOFF.md ZERO.md CLAUDE.md reference.json hw.json \
     zero/day1.c zero/day2.c zero/day3.c zero/day4.c \
     archive-track1-torch/README.md loom-architecture.html verify.sh Makefile"
for f in $REQ; do
  if [ -f "$f" ]; then ok "$f"; else red "missing required file: $f"; fi
done
if ! grep -q 'the loom weaves what the weaver wills' zero/day1.c; then
  red "zero/day1.c is missing its sentinel string"; else ok "day1.c sentinel present"; fi

echo "== [2/6] compile day1/2/3 (documented flags) =="
cd zero || exit 2
gcc -O2 -std=c11 -Wall -o day1 day1.c -lm || red "day1 did not compile"
gcc -O2 -std=c11 -Wall -Wno-misleading-indentation -o day2 day2.c -lm || red "day2 did not compile"
gcc -O2 -std=c11 -Wall -Wno-misleading-indentation -o day3 day3.c -lm || red "day3 did not compile"
cd ..

echo "== [3/6] run day1/2/3 =="
./zero/day1 > /tmp/d1.out 2>&1 || red "day1 crashed at runtime"
./zero/day2 > /tmp/d2.out 2>&1 || red "day2 crashed at runtime"
./zero/day3 > /tmp/d3.out 2>&1 || red "day3 crashed at runtime"

echo "== [4/6] check outputs vs reference.json =="
python3 - "$@" <<'PY' || FAIL=1
import json,re,sys
ref=json.load(open("reference.json"))
d1=open("/tmp/d1.out").read(); d2=open("/tmp/d2.out").read(); d3=open("/tmp/d3.out").read()
fails=[]
def jline(txt):
    for ln in txt.splitlines():
        if ln.strip().startswith("JSON "): return json.loads(ln.strip()[5:])
    return None
# ---- day1 (text output) ----
m=re.search(r'max relative error vs finite differences = ([0-9.eE+-]+)', d1)
if not m: fails.append("day1: could not find gradcheck value")
else:
    v=float(m.group(1)); mx=ref["day1"]["gradcheck_max_rel"]["max"]
    print(f"  day1 gradcheck {v:.2e} < {mx:.0e} ?", "ok" if v<mx else "FAIL")
    if not v<mx: fails.append(f"day1 gradcheck {v:.2e} !< {mx:.0e}")
m=re.search(r'step 3000\s+loss ([0-9.]+)', d1)
if not m: fails.append("day1: could not find step-3000 loss")
else:
    v=float(m.group(1)); c=ref["day1"]["final_loss"]
    good=abs(v-c["value"])<=c["tol"]
    print(f"  day1 loss {v:.4f} within {c['value']}±{c['tol']} ?", "ok" if good else "FAIL")
    if not good: fails.append(f"day1 loss {v:.4f} outside {c['value']}±{c['tol']}")
# ---- day2 (JSON) ----
j=jline(d2)
if j is None: fails.append("day2: no JSON line")
else:
    v=j["gradcheck"]; mx=ref["day2"]["gradcheck_max_rel"]["max"]
    print(f"  day2 gradcheck {v:.2e} < {mx:.0e} ?", "ok" if v<mx else "FAIL")
    if not v<mx: fails.append(f"day2 gradcheck {v:.2e} !< {mx:.0e}")
    qref=ref["day2"]["quad"]; qmap={1:"kappa1",100:"kappa100",10000:"kappa10000"}
    for row in j["quad"]:
        r=qref[qmap[int(row["kappa"])]]
        okgd = row["gd_oracle_steps"]==r["gd"]; okours = row["ours_steps"]==r["ours"]
        print(f"  day2 kappa={int(row['kappa'])}: gd {row['gd_oracle_steps']}=={r['gd']} ours {row['ours_steps']}=={r['ours']} ?",
              "ok" if okgd and okours else "FAIL")
        if not(okgd and okours): fails.append(f"day2 kappa={int(row['kappa'])} steps mismatch")
    c=ref["day2"]["charlm"]; tol=c["tol"]
    okg=abs(j["charlm"]["gd_steps"]-c["gd_steps"])<=tol
    oko=abs(j["charlm"]["ours_steps"]-c["ours_steps"])<=tol
    print(f"  day2 charLM gd {j['charlm']['gd_steps']}~{c['gd_steps']} ours {j['charlm']['ours_steps']}~{c['ours_steps']} (±{tol}) ?",
          "ok" if okg and oko else "FAIL")
    if not(okg and oko): fails.append("day2 charLM steps outside tolerance")
# ---- day3 (JSON) ----
j=jline(d3)
if j is None: fails.append("day3: no JSON line")
else:
    v=j["gradcheck_abs"]; mx=ref["day3"]["gradcheck_abs"]["max"]
    print(f"  day3 gradcheck_abs {v:.2e} < {mx:.0e} ?", "ok" if v<mx else "FAIL")
    if not v<mx: fails.append(f"day3 gradcheck_abs {v:.2e} !< {mx:.0e}")
    cb=j["causality_bitexact"]; print("  day3 causality_bitexact == true ?", "ok" if cb is True else "FAIL")
    if cb is not True: fails.append("day3 causality not bit-exact")
    v=j["rope_rel_maxerr"]; mx=ref["day3"]["rope_rel_maxerr"]["max"]
    print(f"  day3 rope_rel_maxerr {v:.2e} < {mx:.0e} ?", "ok" if v<mx else "FAIL")
    if not v<mx: fails.append(f"day3 rope {v:.2e} !< {mx:.0e}")
    v=j["acc_attention"]; mn=ref["day3"]["acc_attention"]["min"]
    print(f"  day3 acc_attention {v:.3f} >= {mn} ?", "ok" if v>=mn else "FAIL")
    if not v>=mn: fails.append(f"day3 acc_attention {v:.3f} < {mn}")
    c=ref["day3"]["acc_baseline"]; good=abs(j["acc_baseline"]-c["value"])<=c["tol"]
    print(f"  day3 acc_baseline {j['acc_baseline']:.3f} within {c['value']}±{c['tol']} ?", "ok" if good else "FAIL")
    if not good: fails.append(f"day3 acc_baseline outside {c['value']}±{c['tol']}")
if fails:
    print("REFERENCE CHECK FAILED:")
    for f in fails: print("  - "+f)
    sys.exit(1)
print("  all reference checks passed")
PY

echo "== [5/6] purity scan (no ML/numerics libs in artifact code) =="
python3 - <<'PY' || FAIL=1
import re,subprocess,sys
FORB=re.compile(r'\b(torch|cublas|cudnn|thrust|openblas|mkl|numpy)\b', re.I)
files=subprocess.check_output(['git','ls-files']).decode().split()
bad=[]
for f in files:
    if f.startswith('archive-track1-torch/'): continue
    if not re.search(r'\.(c|cu|h|cuh)$', f): continue     # scan source code only
    src=open(f,encoding='utf-8',errors='replace').read()
    src=re.sub(r'/\*.*?\*/','',src,flags=re.S)            # strip block comments
    for i,line in enumerate(src.splitlines(),1):
        code=line.split('//',1)[0]                        # strip line comment
        if FORB.search(code): bad.append(f"{f}:{i}: {line.strip()}")
if bad:
    print("PURITY FAIL: forbidden library token in non-comment artifact code:")
    for b in bad: print("  "+b)
    sys.exit(1)
print("  clean: no torch/cublas/cudnn/thrust/openblas/mkl/numpy in artifact code")
PY

echo "== [6/6] ZERO.md append-only vs origin/main =="
git fetch -q origin main 2>/dev/null || true
if git rev-parse --verify -q origin/main >/dev/null 2>&1; then
  git show origin/main:ZERO.md > /tmp/zero_old 2>/dev/null || : > /tmp/zero_old
  if [ -s /tmp/zero_old ]; then
    ob=$(wc -c < /tmp/zero_old)
    head -c "$ob" ZERO.md > /tmp/zero_prefix
    if cmp -s /tmp/zero_prefix /tmp/zero_old; then
      ok "ZERO.md append-only preserved (old content unchanged)"
    else
      red "ZERO.md is NOT append-only: content before the last pushed version was edited or deleted"
    fi
  else ok "origin ZERO.md empty/new; nothing to compare"; fi
else
  echo "NOTE: origin/main unavailable; skipping append-only check (covered on next push)"
fi

echo "== GPU =="
if command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "GPU present: on-machine kernel/parity checks run separately (see STATUS.md)"
else
  echo "SKIPPED-NO-GPU: GPU kernel/parity checks not run here (CPU environment); covered by on-machine runs + sandbox review"
fi

echo
if [ "$FAIL" -eq 0 ]; then echo "VERIFY: GREEN — all CPU checks passed"; exit 0
else echo "VERIFY: RED — see FAIL lines above"; exit 1; fi
