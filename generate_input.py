#!/usr/bin/env python3
"""
Generate random test input for the Stable Marriage (Gale-Shapley) CUDA programs.

Usage:
    python generate_input.py <n>           # prints to stdout
    python generate_input.py <n> > input.txt  # saves to file

Input format produced (1-indexed, row 0 is the person's own ID):
    n
    <man 1 id> <pref 1> <pref 2> ... <pref n>    (men's preference lists)
    ...
    <woman 1 id> <pref 1> <pref 2> ... <pref n>  (women's preference lists)
"""

import sys
import random


def generate_input(n):
    print(n)

    # Men's preferences: each man lists women 1..n in random order
    for i in range(1, n + 1):
        prefs = list(range(1, n + 1))
        random.shuffle(prefs)
        print(f"{i} " + " ".join(map(str, prefs)))

    # Women's preferences: each woman lists men 1..n in random order
    for i in range(1, n + 1):
        prefs = list(range(1, n + 1))
        random.shuffle(prefs)
        print(f"{i} " + " ".join(map(str, prefs)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <n>", file=sys.stderr)
        sys.exit(1)

    n = int(sys.argv[1])
    if n < 1 or n > 1024:
        print("Error: n must be between 1 and 1024", file=sys.stderr)
        sys.exit(1)

    generate_input(n)
