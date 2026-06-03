#!/usr/bin/env python3
"""
Quick test: evaluate fixed probe set for safe combination (5, 13)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yellow_shuffled_probe_search_2 import (
    build_legal_models,
    find_security_aware_hue_mapping,
    evaluate_illegal_mapping,
    generate_exact_bit_blocks,
    evaluate_legal_mapping,
)
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legal_positions = (5, 13)

# Try a few different probe sets
probe_sets = [
    np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float),
    np.array([5, 45, 90, 135, 180, 225, 270, 315], dtype=float),
    np.array([10, 50, 100, 150, 200, 250, 300, 350], dtype=float),
]

print(f"Testing combination {legal_positions}")
print(f"Dataset: yellow_shuffled\n")

bit_blocks = generate_exact_bit_blocks(2)

for probes in probe_sets:
    print(f"Probes: {list(probes)}")
    try:
        result = find_security_aware_hue_mapping(project_root, legal_positions, probes)
        print(f"  Legal BER: {result['legal_ber']:.4f}")
        print(f"  Min illegal BER: {result['min_illegal_ber']:.4f}")
        print(f"  Avg illegal BER: {result['average_illegal_ber']:.4f}")
        print(f"  Worst illegal pos: {result['worst_illegal_position']}")
        print(f"  Security satisfied: {result['security_satisfied']}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

print("Done.")
