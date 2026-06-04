#!/usr/bin/env python3
"""
Verify the output of the Stable Marriage CUDA programs.

Checks:
  1. The matching is a valid bijection (each man matched to exactly one woman, and vice versa)
  2. The matching is stable (no blocking pairs)

Usage:
    python verify_matching.py <input_file> <output_file>

Where:
    input_file  = the input fed to the CUDA program (n, men's prefs, women's prefs)
    output_file = the program's output (pairs of "man woman" lines, ignoring timing lines)
"""

import sys


def parse_input(filename):
    with open(filename) as f:
        lines = f.read().split('\n')

    idx = 0
    n = int(lines[idx].strip())
    idx += 1

    # Men's preference lists (1-indexed)
    men_prefs = {}  # men_prefs[m] = [w1, w2, ...] in order of preference
    for i in range(n):
        parts = list(map(int, lines[idx].strip().split()))
        man_id = parts[0]
        prefs = parts[1:]
        men_prefs[man_id] = prefs
        idx += 1

    # Women's preference lists (1-indexed)
    women_prefs = {}
    for i in range(n):
        parts = list(map(int, lines[idx].strip().split()))
        woman_id = parts[0]
        prefs = parts[1:]
        women_prefs[woman_id] = prefs
        idx += 1

    return n, men_prefs, women_prefs


def parse_output(filename):
    matching = {}  # man -> woman
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip timing lines
            if 'time' in line.lower() or 'read' in line.lower() or 'us' in line.lower():
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    m, w = int(parts[0]), int(parts[1])
                    matching[m] = w
                except ValueError:
                    continue
    return matching


def verify(n, men_prefs, women_prefs, matching):
    errors = []

    # Check 1: Valid bijection
    if len(matching) != n:
        errors.append(f"Expected {n} pairs, got {len(matching)}")

    women_matched = {}
    for m, w in matching.items():
        if w in women_matched:
            errors.append(f"Woman {w} matched to both man {women_matched[w]} and man {m}")
        women_matched[w] = m

    if errors:
        return False, errors

    # Build rank lookup tables
    # men_rank[m][w] = rank of woman w in man m's preference list (lower = better)
    men_rank = {}
    for m, prefs in men_prefs.items():
        men_rank[m] = {w: rank for rank, w in enumerate(prefs)}

    # women_rank[w][m] = rank of man m in woman w's preference list
    women_rank = {}
    for w, prefs in women_prefs.items():
        women_rank[w] = {m: rank for rank, m in enumerate(prefs)}

    # Check 2: Stability — no blocking pairs
    for m in range(1, n + 1):
        current_w = matching[m]
        # Check all women that m prefers over his current partner
        for w in men_prefs[m]:
            if w == current_w:
                break  # all remaining women are less preferred
            # m prefers w over current_w. Does w prefer m over her current partner?
            w_current_m = women_matched[w]
            if women_rank[w][m] < women_rank[w][w_current_m]:
                errors.append(
                    f"Blocking pair: man {m} and woman {w} "
                    f"(man {m} prefers w{w} over w{current_w}, "
                    f"woman {w} prefers m{m} over m{w_current_m})"
                )

    if errors:
        return False, errors

    return True, ["Matching is valid and stable!"]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    n, men_prefs, women_prefs = parse_input(sys.argv[1])
    matching = parse_output(sys.argv[2])

    ok, messages = verify(n, men_prefs, women_prefs, matching)
    for msg in messages:
        print(msg)

    sys.exit(0 if ok else 1)
