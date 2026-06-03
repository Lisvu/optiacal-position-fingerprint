#!/usr/bin/env python3
"""
Super quick test: evaluate one fixed probe set for one 2-position combination
on yellow_shuffled dataset.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yellow_shuffled_probe_search_2 import (
    build_legal_models,
    evaluate_legal_mapping,
    evaluate_illegal_mapping,
    find_security_aware_hue_mapping,
    generate_exact_bit_blocks,
)
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Test combination (1, 2) with a fixed probe set
legal_positions = (1, 2)
probes = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)

print(f"Testing positions {legal_positions}")
print(f"Probes: {list(probes)}")
print(f"Dataset: yellow_shuffled")
print()

legal_models = build_legal_models(project_root, legal_positions, probes)
bit_blocks = generate_exact_bit_blocks(2)

print(f"Legal model codes:")
for i, m in enumerate(legal_models):
    print(f"  Pos {legal_positions[i]}: {m.code}")

# Build hue mapping
result = find_security_aware_hue_mapping(project_root, legal_positions, probes)

print(f"\nLegal BER: {result['legal_ber']:.4f}")
print(f"Min illegal BER: {result['min_illegal_ber']:.4f}")
print(f"Avg illegal BER: {result['average_illegal_ber']:.4f}")
print(f"Worst illegal position: {result['worst_illegal_position']}")
print(f"Worst legal position: {result['worst_legal_position']}")
print(f"Security satisfied: {result['security_satisfied']}")

print("\nDone.")
