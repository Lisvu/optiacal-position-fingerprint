#!/usr/bin/env python3
"""
Diagnose: why does the worst illegal position have BER=0 for combo (5,13)?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yellow_shuffled_probe_search_2 import (
    build_legal_models,
    find_security_aware_hue_mapping,
    generate_exact_bit_blocks,
    corrected_ber,
)
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legal_positions = (5, 13)
probes = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)

print(f"Diagnosing combination {legal_positions}")
print(f"Probes: {list(probes)}\n")

legal_models = build_legal_models(project_root, legal_positions, probes)
result = find_security_aware_hue_mapping(project_root, legal_positions, probes)

print(f"Legal codes:")
for i, m in enumerate(legal_models):
    print(f"  Pos {legal_positions[i]}: {list(m.code)}")

print(f"\nHue mapping: {result['hue_mapping']}")
print(f"Worst illegal position: {result['worst_illegal_position']}")

# Get worst illegal model
worst_pos = result['worst_illegal_position']
csv_file = os.path.join(project_root, "data", "15pro", "yellow_shuffled", f"{worst_pos}.csv")
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
code_worst = np.where(z >= 0, 1, -1)

print(f"\nWorst illegal pos {worst_pos} code: {list(code_worst)}")
print(f"Correlation with Pos 5: {float(np.mean(code_worst * legal_models[0].code)):.4f}")
print(f"Correlation with Pos 13: {float(np.mean(code_worst * legal_models[1].code)):.4f}")

# Check each symbol combination
bit_blocks = generate_exact_bit_blocks(2)
legal_codes = [m.code for m in legal_models]

print(f"\nPer-symbol-combination analysis:")
for bits_pm in bit_blocks:
    _, symbol_combinations = __import__('yellow_shuffled_probe_search_2').test.build_symbol_sequence(bits_pm, legal_codes)
    key = tuple(symbol_combinations[0])
    hue = result['hue_mapping'][key]
    probe_idx = list(probes).index(hue)
    
    # Legal decodings
    dec_legals = []
    for li in range(2):
        gamma = int(legal_codes[li][probe_idx]) * symbol_combinations[0][li]
        dec_legals.append(1 if gamma > 0 else -1)
    
    # Illegal decoding
    gamma_il = [int(code_worst[probe_idx]) * symbol_combinations[0][li] for li in range(2)]
    dec_il = [1 if g > 0 else -1 for g in gamma_il]
    
    match = [dec_legals[i] == dec_il[i] for i in range(2)]
    
    print(f"  Bits {list(bits_pm)} -> key {key} -> hue {hue} (probe idx {probe_idx})")
    print(f"    Legal signs: {symbol_combinations[0]}")
    print(f"    Legal codes at probe: pos5={legal_codes[0][probe_idx]}, pos13={legal_codes[1][probe_idx]}")
    print(f"    Illegal code at probe: {code_worst[probe_idx]}")
    print(f"    Legal dec: {dec_legals}, Illegal dec: {dec_il}, Match: {match}")

print("\nDone.")
