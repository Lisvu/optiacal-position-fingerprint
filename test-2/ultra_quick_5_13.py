#!/usr/bin/env python3
"""
Ultra quick: test just 3 hand-picked probe sets for combo (5, 13)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yellow_shuffled_probe_search_2 import (
    build_legal_models,
    find_security_aware_hue_mapping,
)
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legal_positions = (5, 13)

probe_sets = [
    np.array([5, 50, 100, 150, 200, 250, 300, 350], dtype=float),
    np.array([5, 55, 110, 165, 220, 275, 330], dtype=float),
    np.array([10, 60, 120, 180, 240, 300, 360], dtype=float),
]

print(f"Ultra quick test for {legal_positions}")
print(f"Dataset: yellow_shuffled\n")

for probes in probe_sets:
    print(f"Probes: {list(probes)}")
    try:
        result = find_security_aware_hue_mapping(project_root, legal_positions, probes)
        print(f"  Legal BER: {result['legal_ber']:.4f}")
        print(f"  Min illegal BER: {result['min_illegal_ber']:.4f}")
        print(f"  Avg illegal BER: {result['average_illegal_ber']:.4f}")
        print(f"  Worst illegal pos: {result['worst_illegal_position']}")
        print(f"  Security: {result['security_satisfied']}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

print("Done.")
