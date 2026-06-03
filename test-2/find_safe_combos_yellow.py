#!/usr/bin/env python3
"""
Find good 2-position combinations in yellow_shuffled dataset.
A combination is "safe" if no illegal position has code correlation >= 0.75 with either legal position.
"""
import sys, os
import numpy as np
import pandas as pd
import itertools

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_code(pos, probes):
    csv_file = os.path.join(project_root, "data", "15pro", "yellow_shuffled", f"{pos}.csv")
    df = pd.read_csv(csv_file)
    row_indices = [int((p/5)-1) for p in probes]
    mat = df.values[row_indices].astype(float)
    x = probes.astype(float)
    A = np.column_stack([x, np.ones_like(x)])
    coeffs = np.linalg.lstsq(A, mat, rcond=None)[0]
    trend = A @ coeffs
    residual = mat - trend
    _, _, Vt = np.linalg.svd(residual, full_matrices=False)
    w = Vt[0].copy()
    z = residual @ w
    if z[0] < 0:
        w = -w
        z = -z
    code = np.where(z >= 0, 1, -1)
    return code

probes = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)

# Precompute all codes
all_codes = {}
for pos in range(1, 29):
    all_codes[pos] = get_code(pos, probes)

print(f"Scanning all C(28,2) = {28*27//2} combinations...")
print(f"Criteria: no illegal position has code correlation >= 0.75 with either legal position\n")

safe_combos = []
for combo in itertools.combinations(range(1, 29), 2):
    pos_a, pos_b = combo
    code_a = all_codes[pos_a]
    code_b = all_codes[pos_b]
    
    max_corr = 0
    worst_pos = -1
    for pos in range(1, 29):
        if pos in combo:
            continue
        code_il = all_codes[pos]
        corr_a = abs(float(np.mean(code_il * code_a)))
        corr_b = abs(float(np.mean(code_il * code_b)))
        max_c = max(corr_a, corr_b)
        if max_c > max_corr:
            max_corr = max_c
            worst_pos = pos
    
    if max_corr < 0.75:
        safe_combos.append((combo, max_corr, worst_pos))
        print(f"SAFE: {combo}, max_illegal_corr={max_corr:.4f}, worst_pos={worst_pos}")

print(f"\nTotal safe combinations: {len(safe_combos)}")
if safe_combos:
    print(f"\nBest 5 (lowest max correlation):")
    safe_combos.sort(key=lambda x: x[1])
    for combo, max_corr, worst_pos in safe_combos[:5]:
        print(f"  {combo}: max_corr={max_corr:.4f}, worst={worst_pos}")
else:
    print("\nNo completely safe combinations found with threshold 0.75")
    print("\nTop 10 closest-to-safe combinations:")
    all_combos = []
    for combo in itertools.combinations(range(1, 29), 2):
        pos_a, pos_b = combo
        code_a = all_codes[pos_a]
        code_b = all_codes[pos_b]
        max_corr = 0
        worst_pos = -1
        for pos in range(1, 29):
            if pos in combo:
                continue
            code_il = all_codes[pos]
            corr_a = abs(float(np.mean(code_il * code_a)))
            corr_b = abs(float(np.mean(code_il * code_b)))
            max_c = max(corr_a, corr_b)
            if max_c > max_corr:
                max_corr = max_c
                worst_pos = pos
        all_combos.append((combo, max_corr, worst_pos))
    all_combos.sort(key=lambda x: x[1])
    for combo, max_corr, worst_pos in all_combos[:10]:
        print(f"  {combo}: max_corr={max_corr:.4f}, worst={worst_pos}")

print("\nDone.")
