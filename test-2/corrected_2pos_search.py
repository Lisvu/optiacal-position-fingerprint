#!/usr/bin/env python3
"""
Corrected 2-position search with increased probe count on yellow_shuffled.

Key insight: probe_count = fingerprint length (code vector dimension).
Hue mapping still uses only 4 angles, but chosen from ALL probes.

For 2-position system:
- 4 symbol combinations -> 4 sending angles
- Attacker decodes 4 bits over 4 symbol periods
- Attacker BER = matches / 4
- Secure BER = min(matches, 4-matches) / 4

Target: min illegal BER > 0.3 means ALL (illegal, legal) pairs have exactly 2 matches.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import itertools

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIGHT_CONDITION = "yellow_shuffled"
all_probes = np.arange(5, 366, 5, dtype=float)

# ============================================================================
# 1. Pre-compute all codes for all 28 positions (72 probes)
# ============================================================================
print("Pre-computing all codes for 28 positions x 72 probes...")
all_codes = {}  # pos -> (72,) array
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


def find_best_4tuple(probe_indices, pos_a, pos_b, max_quads=1000):
    """
    Find the best 4-tuple of probe indices for hue mapping.
    Randomly sample 4-tuples to avoid exponential search.
    """
    code_a = all_codes[pos_a][probe_indices]
    code_b = all_codes[pos_b][probe_indices]
    n = len(probe_indices)
    illegal_positions = [p for p in range(1, 29) if p not in (pos_a, pos_b)]
    illegal_codes = {p: all_codes[p][probe_indices] for p in illegal_positions}
    
    best_min_secure = -1
    best_quad = None
    
    rng = np.random.RandomState(42)
    
    # Generate random 4-tuples instead of all combinations
    all_indices = list(range(n))
    for _ in range(max_quads):
        quad = tuple(sorted(rng.choice(all_indices, size=4, replace=False)))
        
        # For this 4-tuple, compute min secure BER over all illegal positions
        # We don't try all permutations; instead we use the natural order
        # This is a heuristic but much faster
        min_secure_for_this_quad = float('inf')
        
        for pos_il in illegal_positions:
            c_il = illegal_codes[pos_il]
            
            # Count matches for pos_a over 4 probes
            matches_a = sum(1 for qi in quad if c_il[qi] == code_a[qi])
            secure_a = min(matches_a, 4 - matches_a) / 4.0
            
            # Count matches for pos_b over 4 probes
            matches_b = sum(1 for qi in quad if c_il[qi] == code_b[qi])
            secure_b = min(matches_b, 4 - matches_b) / 4.0
            
            worst_secure_for_il = min(secure_a, secure_b)
            if worst_secure_for_il < min_secure_for_this_quad:
                min_secure_for_this_quad = worst_secure_for_il
        
        if min_secure_for_this_quad > best_min_secure:
            best_min_secure = min_secure_for_this_quad
            best_quad = quad
    
    return best_min_secure, best_quad, None


def search_combo(pos_a, pos_b, probe_counts=[16, 20, 24, 28], max_trials=50):
    """Search for best probe set and 4-tuple for given 2-position combination."""
    print(f"\nSearching ({pos_a}, {pos_b})")
    
    rng = np.random.RandomState(42)
    best_overall_secure = -1
    best_overall_probes = None
    best_overall_quad = None
    
    for probe_count in probe_counts:
        print(f"  probe_count={probe_count}")
        
        for trial in range(max_trials):
            # Random probe set
            probe_indices = sorted(rng.choice(72, size=probe_count, replace=False).tolist())
            
            # Find best 4-tuple for this probe set
            best_secure, best_quad, best_perm = find_best_4tuple(probe_indices, pos_a, pos_b)
            
            if best_secure > best_overall_secure:
                best_overall_secure = best_secure
                best_overall_probes = probe_indices
                best_overall_quad = (best_quad, best_perm)
                probes_list = all_probes[probe_indices].tolist()
                print(f"    Trial {trial+1}: best_secure={best_secure:.4f}, probes={probes_list}")
                
                if best_secure > 0.3:
                    print(f"    *** TARGET ACHIEVED ***")
                    return best_overall_secure, best_overall_probes, best_overall_quad
        
        print(f"  Best after {max_trials} trials: {best_overall_secure:.4f}")
    
    return best_overall_secure, best_overall_probes, best_overall_quad


# Test all 7 pre-screened safe combinations
safe_combos = [(5, 13), (5, 16), (5, 23), (6, 14), (13, 16), (13, 23), (16, 23)]

print("="*70)
print("Corrected 2-Position Search with Increased Probe Count")
print(f"Dataset: {LIGHT_CONDITION}")
print("Target: min illegal secure BER > 0.3")
print("="*70)

best_global_secure = -1
best_global_combo = None
best_global_probes = None

for pos_a, pos_b in safe_combos:
    secure, probes, quad = search_combo(pos_a, pos_b, probe_counts=[16, 20, 24], max_trials=30)
    if secure > best_global_secure:
        best_global_secure = secure
        best_global_combo = (pos_a, pos_b)
        best_global_probes = probes
    
    print(f"  Result for ({pos_a},{pos_b}): min_secure={secure:.4f}")

print("\n" + "="*70)
print("FINAL RESULT:")
print(f"  Best combination: {best_global_combo}")
print(f"  Best min illegal secure: {best_global_secure:.4f}")
if best_global_probes:
    print(f"  Probe count: {len(best_global_probes)}")
    print(f"  Probes: {all_probes[best_global_probes].tolist()}")
print("="*70)
print("\nDone.")
