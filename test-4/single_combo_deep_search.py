#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证：对单个已知easy组合进行深度搜索
"""
import sys, os
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
positions = (1, 3, 12, 22)

print(f"Testing positions {positions}")
print(f"N_COMPONENTS={test.N_COMPONENTS}, ALPHA={test.ALPHA}")

csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition="white")

best_probes = None
best_ber = float("inf")

for num_probes in [8, 9, 10, 11, 12, 13]:
    print(f"\nTrying num_probes={num_probes}")
    rng = random.Random(42 + num_probes)
    try:
        probes, ber = test.staged_beam_probe_selection(
            csv_files=csv_files,
            num_probes=num_probes,
            num_bits=5000,
            min_interval=30,
            coarse_bits=800,
            mapping_eval_bits=300,
            mapping_top_k=3,
            neighborhood_samples=16,
            local_rounds=4,
            beam_width=16,
            initial_sample_size=40,
            expansion_sample_size=20,
            finalist_count=10,
            sa_iterations=40,
            repeat_eval=2,
            candidate_pool_size=50,
            base_seed=10000 + num_probes,
            rng=rng,
        )
        print(f"  Coarse BER: {ber:.6f}")
        
        # Exact eval
        matrices = test.load_selected_rows(csv_files, probes)
        models = [test.extract_fingerprint(probes, mat, force_positive_first=True) for mat in matrices]
        models = test.align_model_directions(models)
        hue_mapping = test.build_hue_mapping(models, probes, mapping_eval_bits=300, top_k_per_combination=3, rng=rng)
        
        blocks = test.generate_all_bit_blocks(4)
        exact_ber = test.evaluate_blocks_ber(models, blocks, hue_mapping)
        print(f"  Exact BER: {exact_ber:.6f}")
        
        if exact_ber < best_ber:
            best_ber = exact_ber
            best_probes = probes
            
        if exact_ber <= 0.005:
            print(f"  *** TARGET REACHED! ***")
            break
            
    except Exception as e:
        print(f"  Error: {e}")

print(f"\nBest result: exact_ber={best_ber:.6f}")
if best_probes is not None:
    print(f"Best probes: {list(best_probes)}")
