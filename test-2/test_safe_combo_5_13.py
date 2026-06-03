#!/usr/bin/env python3
"""
Test a SAFE combination (5, 13) on yellow_shuffled with proper probe search.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yellow_shuffled_probe_search_2 import (
    search_security_aware_probes,
    evaluate_candidate,
    LIGHT_CONDITION,
    TARGET_LEGAL_BER,
    MIN_ILLEGAL_BER,
)
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legal_positions = (5, 13)

print(f"Testing SAFE combination {legal_positions}")
print(f"Dataset: {LIGHT_CONDITION}")
print(f"Target: legal BER <= {TARGET_LEGAL_BER}, min illegal BER > {MIN_ILLEGAL_BER}")
print()

rng = random.Random(42)
result = search_security_aware_probes(project_root, legal_positions, rng)

print(f"\n{'='*60}")
print(f"RESULT for {legal_positions}:")
print(f"  Probes: {list(result['probes'])}")
print(f"  Probe count: {result['probe_count']}")
print(f"  Legal BER: {result['legal_ber']:.6f}")
print(f"  Min illegal BER: {result['min_illegal_ber']:.6f}")
print(f"  Avg illegal BER: {result['average_illegal_ber']:.6f}")
print(f"  Worst illegal pos: {result['worst_illegal_position']}")
print(f"  Security satisfied: {result['security_satisfied']}")
print(f"{'='*60}")

print("\nDone.")
