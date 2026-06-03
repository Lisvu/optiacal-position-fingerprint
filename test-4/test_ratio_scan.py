#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同 min_gamma_ratio 参数下的性能权衡
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
positions = (1, 3, 12, 22)
probes = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)

print("=" * 80)
print("min_gamma_ratio 参数扫描")
print("=" * 80)

csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition='white')

for ratio in [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
    print(f"\n--- min_gamma_ratio = {ratio} ---")
    
    # 设置全局参数
    test.MIN_GAMMA_RATIO = ratio
    
    matrices = test.load_selected_rows(csv_files, probes)
    models = [test.extract_fingerprint(probes, mat, force_positive_first=True) for mat in matrices]
    models = test.align_model_directions(models)
    
    # 手动调用幅度感知版本
    hue_mapping = test.build_hue_mapping(
        models, probes,
        mapping_eval_bits=500, top_k_per_combination=3,
        use_amplitude_aware=True,
    )
    
    # Exact eval
    blocks = test.generate_all_bit_blocks(4)
    legal_ber = test.evaluate_blocks_ber(models, blocks, hue_mapping)
    
    # FEC eval
    rng = test.random.Random(42)
    info_bits = test.generate_random_information_bits(2000, 4, rng=rng)
    fec_ber = test.evaluate_blocks_ber_with_convolutional_fec(models, info_bits, hue_mapping)
    
    # Illegal check
    probe_to_row = test.build_probe_to_row(probes)
    legal_codes = [m.eff_code if m.eff_code is not None else m.code for m in models]
    all_positions = sorted([int(f.replace('.csv','')) for f in os.listdir(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white') if f.endswith('.csv')])
    
    worst_illegal = 1.0
    for pos in all_positions:
        if pos in positions:
            continue
        csv_file = os.path.join(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white', str(pos) + '.csv')
        mat = test.load_csv_matrix(csv_file)
        row_indices = [int((p/5)-1) for p in probes]
        illegal_mat = mat[row_indices].astype(float, copy=False)
        illegal_model = test.extract_fingerprint(probes, illegal_mat, force_positive_first=True)
        
        corr = test.calculate_correlation(models[0].eff_code if models[0].eff_code is not None else models[0].code,
                                          illegal_model.eff_code if illegal_model.eff_code is not None else illegal_model.code)
        if corr < 0:
            if illegal_model.W is not None and illegal_model.W.ndim > 1:
                illegal_model.W = -illegal_model.W
                illegal_model.Z = -illegal_model.Z
                illegal_model.multi_code = -illegal_model.multi_code
                illegal_model.eff_w = -illegal_model.eff_w
                illegal_model.eff_code = -illegal_model.eff_code
            else:
                illegal_model.w = -illegal_model.w
                illegal_model.z = -illegal_model.z
                illegal_model.code = -illegal_model.code
        
        errors = np.zeros(4)
        totals = np.zeros(4)
        for bits_pm in blocks:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            obs = test.observe_block_from_measured_matrix(hue_seq, illegal_model.Y, probe_to_row)
            if illegal_model.eff_w is not None and illegal_model.eff_code is not None:
                dec = test.decode_local_block(obs, illegal_model.eff_w, illegal_model.eff_code)
            else:
                dec = test.decode_local_block(obs, illegal_model.w, illegal_model.code)
            true_bits = test.pm1_to_bin(bits_pm)
            for li in range(4):
                totals[li] += 1
                if dec.bit_hat_bin != true_bits[li]:
                    errors[li] += 1
        raw_bers = errors / totals
        secure_bers = np.minimum(raw_bers, 1-raw_bers)
        if secure_bers.min() < worst_illegal:
            worst_illegal = secure_bers.min()
    
    print(f"  Legal exact={legal_ber:.4f}, FEC={fec_ber:.4f}, Illegal worst={worst_illegal:.3f}")

print("\nDone.")
