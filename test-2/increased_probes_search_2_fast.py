#!/usr/bin/env python3
"""
Optimized increased probe count search for 2-position system on yellow_shuffled.

Key optimizations:
1. Pre-compute all 28 position codes for all 72 probes once
2. Fast prescreen using lookup tables only (no file I/O or SVD)
3. Reduce search space by early termination
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yellow_shuffled_probe_search_2 import (
    build_legal_models,
    find_security_aware_hue_mapping,
    LIGHT_CONDITION,
)
import numpy as np
import pandas as pd
import itertools

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
all_probes = np.arange(5, 366, 5, dtype=float)

# ============================================================================
# 1. Pre-compute all codes for all positions and all 72 probes
# ============================================================================
print("Pre-computing all codes...")
all_codes = {}  # pos -> code array of length 72
for pos in range(1, 29):
    csv_file = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION, f"{pos}.csv")
    df = pd.read_csv(csv_file)
    mat = df.values.astype(float)  # 73 x 4
    x = all_probes.astype(float)
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
    all_codes[pos] = code
print("Done.\n")


def get_codes_for_indices(pos, indices):
    """Fast lookup of codes at specific probe indices"""
    return all_codes[pos][np.array(indices)]


def fast_prescreen(probe_indices, pos_a, pos_b):
    """
    Fast prescreen: check if there exists a 4-tuple of probe indices such that
    for every illegal position attacking each legal position,
    at least one of the two probes for bit=-1 or bit=+1 differs.
    """
    code_a = all_codes[pos_a][probe_indices]
    code_b = all_codes[pos_b][probe_indices]
    n = len(probe_indices)
    all_illegal = [p for p in range(1, 29) if p not in (pos_a, pos_b)]
    
    # Pre-get all illegal codes
    illegal_codes = {p: all_codes[p][probe_indices] for p in all_illegal}
    
    for quad in itertools.combinations(range(n), 4):
        i1, i2, i3, i4 = quad
        
        best_failure = float('inf')
        for perm in itertools.permutations(quad):
            p_m11, p_m1p1, p_p1m1, p_p1p1 = perm
            
            total_failures = 0
            for pos_il in all_illegal:
                code_il = illegal_codes[pos_il]
                
                # Attack pos_a: bit=-1 via p_m11, p_m1p1; bit=+1 via p_p1m1, p_p1p1
                match_m1_a = (code_il[p_m11] == code_a[p_m11]) and (code_il[p_m1p1] == code_a[p_m1p1])
                match_p1_a = (code_il[p_p1m1] == code_a[p_p1m1]) and (code_il[p_p1p1] == code_a[p_p1p1])
                if match_m1_a or match_p1_a:
                    total_failures += 1
                
                # Attack pos_b: bit=-1 via p_m11, p_p1m1; bit=+1 via p_m1p1, p_p1p1
                match_m1_b = (code_il[p_m11] == code_b[p_m11]) and (code_il[p_p1m1] == code_b[p_p1m1])
                match_p1_b = (code_il[p_m1p1] == code_b[p_m1p1]) and (code_il[p_p1p1] == code_b[p_p1p1])
                if match_m1_b or match_p1_b:
                    total_failures += 1
            
            if total_failures < best_failure:
                best_failure = total_failures
        
        if best_failure == 0:
            return True, 0
    
    return False, best_failure


# ============================================================================
# 2. Search with increased probe counts
# ============================================================================
def search_combo(pos_a, pos_b, probe_counts, max_trials):
    print(f"\nSearching ({pos_a}, {pos_b})")
    rng = np.random.RandomState(42)
    best_min_illegal = -1
    best_result = None
    
    for probe_count in probe_counts:
        print(f"  probe_count={probe_count}")
        passed = 0
        
        for trial in range(max_trials):
            probe_indices = sorted(rng.choice(72, size=probe_count, replace=False).tolist())
            probes = all_probes[probe_indices]
            
            is_safe, failure_count = fast_prescreen(probe_indices, pos_a, pos_b)
            if not is_safe:
                continue
            
            passed += 1
            
            # Full evaluation
            try:
                result = find_security_aware_hue_mapping(project_root, (pos_a, pos_b), probes)
                if result['legal_ber'] > 0.001:
                    continue
                
                min_il = result['min_illegal_ber']
                if min_il > best_min_illegal:
                    best_min_illegal = min_il
                    best_result = result.copy()
                    best_result['probes'] = probes.copy()
                    print(f"    Trial {trial+1}: min_il={min_il:.4f}, avg_il={result['average_illegal_ber']:.4f}")
                    
                    if min_il > 0.3:
                        print(f"    *** TARGET ACHIEVED ***")
                        return best_result
            except Exception as e:
                continue
        
        print(f"  Passed prescreen: {passed}/{max_trials}")
    
    return best_result


safe_combos = [(5, 13), (5, 16), (5, 23), (6, 14), (13, 16), (13, 23), (16, 23)]

print("="*70)
print("Increased Probe Count Search for 2-Position System")
print(f"Dataset: {LIGHT_CONDITION}")
print(f"Target: legal BER <= 0.001, min illegal BER > 0.3")
print("="*70)

best_overall = None
best_overall_min = -1

for pos_a, pos_b in safe_combos:
    result = search_combo(pos_a, pos_b, probe_counts=[12, 14, 16], max_trials=30)
    if result and result['min_illegal_ber'] > best_overall_min:
        best_overall_min = result['min_illegal_ber']
        best_overall = result
        best_overall['positions'] = (pos_a, pos_b)

print("\n" + "="*70)
print("FINAL RESULT:")
if best_overall:
    print(f"  Positions: {best_overall['positions']}")
    print(f"  Probes: {list(best_overall['probes'])}")
    print(f"  Min illegal BER: {best_overall['min_illegal_ber']:.4f}")
    print(f"  Legal BER: {best_overall['legal_ber']:.4f}")
    print(f"  Avg illegal BER: {best_overall['average_illegal_ber']:.4f}")
else:
    print("  No result found with min illegal BER > 0.3")
    if best_overall:
        print(f"  Best found: min={best_overall_min:.4f}")
print("="*70)
print("\nDone.")
