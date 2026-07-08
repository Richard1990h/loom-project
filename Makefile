# ZERO project — build + verify entry points.
# Instruments (verify) and artifact builds. No ML/numerics libraries anywhere.

CC      ?= gcc
CFLAGS  ?= -O2 -std=c11 -Wall
CFLAGS2 ?= -O2 -std=c11 -Wall -Wno-misleading-indentation

NVCC    ?= nvcc
NVFLAGS ?= -O2 -arch=sm_89

.PHONY: verify all day1 day2 day3 day4 gpt talk clean

verify:
	@bash verify.sh

all: day1 day2 day3 day4

day1:
	$(CC) $(CFLAGS) -o zero/day1 zero/day1.c -lm
day2:
	$(CC) $(CFLAGS2) -o zero/day2 zero/day2.c -lm
day3:
	$(CC) $(CFLAGS2) -o zero/day3 zero/day3.c -lm
day4:
	$(CC) $(CFLAGS2) -o zero/day4 zero/day4.c -lm

# GPU transformer trainer + chat REPL (needs nvcc + an NVIDIA GPU)
gpt:
	$(NVCC) $(NVFLAGS) -o zero/gpt zero/gpt.cu
	$(CC) $(CFLAGS) -o zero/bpe zero/bpe.c
# `make talk` — the human's live chat REPL (loads the latest checkpoint).
talk: gpt
	@echo "loom chat REPL — type a message, Ctrl-D to quit."
	./zero/gpt talk runs/latest/ckpt.bin data/tokenizer.bin 0.8 40

clean:
	rm -f zero/day1 zero/day2 zero/day3 zero/day4 zero/day4c zero/day5 zero/gpt zero/bpe zero/*.o
