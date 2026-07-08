#!/usr/bin/env bash
# Blind A/B: run 10 fixed prompts through the base (pretrained) and chat-tuned
# models. For each prompt the two replies are shown as "A" and "B" with the
# assignment randomised and hidden, so the human can judge without knowing
# which is which. The answer key is written to runs/latest/ab_key.txt.
cd "$(dirname "$0")/.." || exit 1
BASE=runs/latest/ckpt.bin; CHAT=runs/latest/chat.bin; TK=data/tokenizer.bin
[ -f "$CHAT" ] || { echo "no chat.bin — run the fine-tune first"; exit 1; }
PROMPTS=("Hello" "How are you?" "What is your name?" "Tell me a joke"
         "What color is the sky?" "What is a dog?" "Can you help me?"
         "What is the sun?" "Good night" "What is a mountain?")
key="runs/latest/ab_key.txt"; : > "$key"
echo "=== BLIND A/B: base vs chat-tuned (same 10 prompts) ==="
echo "For each prompt, pick A or B as the better reply. Key saved to $key."
echo
n=1
for p in "${PROMPTS[@]}"; do
  rb=$(printf '%s\n' "$p" | ./zero/gpt talk "$BASE" "$TK" 0.7 40 2>/dev/null | tr '\n' ' ')
  rc=$(printf '%s\n' "$p" | ./zero/gpt talk "$CHAT" "$TK" 0.7 40 2>/dev/null | tr '\n' ' ')
  # randomise A/B assignment (deterministic per line via $RANDOM-free hash of prompt length+n)
  if (( (${#p} + n) % 2 == 0 )); then A="$rb"; B="$rc"; echo "$n: A=base B=chat" >> "$key"; else A="$rc"; B="$rb"; echo "$n: A=chat B=base" >> "$key"; fi
  printf '%d) PROMPT: %s\n   A: %s\n   B: %s\n\n' "$n" "$p" "$A" "$B"
  n=$((n+1))
done
echo "(answer key in $key — do not peek until you've chosen)"
