# ZERO project — build + verify entry points.
# Instruments (verify) and artifact builds. No ML/numerics libraries anywhere.

CC      ?= gcc
CFLAGS  ?= -O2 -std=c11 -Wall
CFLAGS2 ?= -O2 -std=c11 -Wall -Wno-misleading-indentation

.PHONY: verify all day1 day2 day3 day4 clean

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

clean:
	rm -f zero/day1 zero/day2 zero/day3 zero/day4 zero/*.o
