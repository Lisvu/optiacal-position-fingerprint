#!/usr/bin/env python3
"""
Brute force small search for combo (5, 13) on yellow_shuffled.
Try many small probe sets (6-10 probes) and evaluate min illegal BER.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yellow_shuffled_probe_search_2 import (
    build_legal_models,
    find_security_aware_hue_mapping,
    generate_exact_bit_blocks,
    evaluate_illegal_mapping,
    evaluate_legal_mapping,
    corrected_ber,
)
import numpy as np
import itertools

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legal_positions = (5, 13)
all_probes = np.arange(5, 361, 5, dtype=float)

print(f"Brute force search for {legal_positions} on yellow_shuffled")
print(f"Testing probe counts 6-10, random sampling...")
print()

best_min_illegal = -1
best_result = None
best_probes = None

rng = np.random.RandomState(42)

for probe_count in [6, 7, 8, 9, 10]:
    print(f"Probe count = {probe_count}")
    for trial in range(50):
        probes = np.sort(rng.choice(all_probes, size=probe_count, replace=False))
        
        try:
            result = find_security_aware_hue_mapping(project_root, legal_positions, probes)
            
            if result['legal_ber'] > 0.001:
                continue
            
            min_il = result['min_illegal_ber']
            if min_il > best_min_illegal:
                best_min_illegal = min_il
                best_result = result
                best_probes = probes.copy()
                print(f"  NEW BEST: probes={list(probes)}, min_illegal={min_il:.4f}, avg={result['average_illegal_ber']:.4f}")
                
                if min_il > 0.3:
                    print(f"  *** TARGET ACHIEVED! ***")
                    break
        except Exception as e:
            continue
    
    if best_min_illegal > 0.3:
        break

print(f"\n{'='*60}")
print(f"FINAL RESULT:")
print(f"  Best probes: {list(best_probes) if best_probes is not None else 'None'}")
print(f"  Best min illegal BER: {best_min_illegal:.4f}")
if best_result:
    print(f"  Legal BER: {best_result['legal_ber']:.4f}")
    print(f"  Avg illegal BER: {best_result['average_illegal_ber']:.4f}")
    print(f"  Worst illegal pos: {best_result['worst_illegal_position']}")
    print(f"  Security satisfied: {best_result['security_satisfied']}")
print(f"{'='*60}")

print("\nDone.")
