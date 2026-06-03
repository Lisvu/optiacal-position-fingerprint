#!/usr/bin/env python3
"""
Diagnose: print all illegal position codes and their correlation to legal positions
for yellow_shuffled dataset.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yellow_shuffled_probe_search_2 import build_legal_models
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legal_positions = (1, 2)
probes = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)

legal_models = build_legal_models(project_root, legal_positions, probes)

print(f"Legal positions: {legal_positions}")
print(f"Probes: {list(probes)}")
print(f"\n{'非法位置':>8} | {'Code':>20} | {'Corr Pos1':>10} | {'Corr Pos2':>10} | {'Best match':>10}")
print("-" * 70)

for pos in range(1, 29):
    if pos in legal_positions:
        continue
    csv_file = os.path.join(project_root, "data", "15pro", "yellow_shuffled", f"{pos}.csv")
    df = pd.read_csv(csv_file)
    row_indices = [int((p/5)-1) for p in probes]
    mat = df.values[row_indices].astype(float)
    
    # Simple SVD to get code
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
    
    corr1 = float(np.mean(code * legal_models[0].code))
    corr2 = float(np.mean(code * legal_models[1].code))
    best = max(abs(corr1), abs(corr2))
    
    print(f"{pos:>8} | {str(list(code)):>20} | {corr1:>10.4f} | {corr2:>10.4f} | {best:>10.4f}")

print(f"\nLegal Pos 1 code: {list(legal_models[0].code)}")
print(f"Legal Pos 2 code: {list(legal_models[1].code)}")

# Check if any illegal code exactly matches legal code
print(f"\nExact matches:")
for pos in range(1, 29):
    if pos in legal_positions:
        continue
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
    
    if np.array_equal(code, legal_models[0].code):
        print(f"  Pos {pos} EXACTLY matches Pos 1")
    if np.array_equal(code, legal_models[1].code):
        print(f"  Pos {pos} EXACTLY matches Pos 2")

print("\nDone.")
