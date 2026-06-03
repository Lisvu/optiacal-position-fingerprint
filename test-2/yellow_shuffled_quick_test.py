#!/usr/bin/env python3
"""
Quick test: run a few 2-position combinations on yellow_shuffled dataset
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yellow_shuffled_probe_search_2 import (
    search_security_aware_probes,
    generate_position_combinations,
    LIGHT_CONDITION,
    TARGET_LEGAL_BER,
    MIN_ILLEGAL_BER,
)
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
combinations = generate_position_combinations(project_root)

print(f"Dataset: {LIGHT_CONDITION}")
print(f"Target: legal BER <= {TARGET_LEGAL_BER}, min illegal BER > {MIN_ILLEGAL_BER}")
print(f"Total combinations: {len(combinations)}")
print()

# Test first 5 combinations
rng = random.Random(42)
for legal_positions in combinations[:5]:
    print(f"Testing {legal_positions}...")
    result = search_security_aware_probes(project_root, legal_positions, rng)
    print(f"  legal_ber={result['legal_ber']:.4f}")
    print(f"  min_illegal_ber={result['min_illegal_ber']:.4f}")
    print(f"  avg_illegal_ber={result['average_illegal_ber']:.4f}")
    print(f"  security_satisfied={result['security_satisfied']}")
    print()

print("Done.")
