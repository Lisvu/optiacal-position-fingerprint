#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证幅度感知 hue mapping 的效果
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 测试组合
positions = (1, 3, 12, 22)
probes = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)

print("=" * 80)
print("幅度感知 Hue Mapping 验证")
print("=" * 80)
print(f"Positions: {positions}")
print(f"Probes: {list(probes)}")
print(f"N_COMPONENTS={test.N_COMPONENTS}, ALPHA={test.ALPHA}")

csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition='white')

for use_aware in [False, True]:
    label = "幅度感知" if use_aware else "原始版本"
    print(f"\n{'=' * 80}")
    print(f"测试: {label}")
    print("=" * 80)
    
    models, hue_mapping = test.build_models_from_probes(
        csv_files, probes,
        mapping_eval_bits=500,
        mapping_top_k=3,
        use_amplitude_aware=use_aware,
    )
    
    # 检查每个 combination 的 gamma 分布
    print("\n各 symbol combination 的 min_gamma 分布:")
    z_list = []
    for m in models:
        if m.eff_z is not None:
            z_list.append(np.asarray(m.eff_z, dtype=float))
        elif m.Z is not None and m.Z.ndim > 1:
            k = m.Z.shape[1]
            alpha = test.ALPHA[:k] if hasattr(test, 'ALPHA') else np.ones(k)
            alpha = np.asarray(alpha, dtype=float)
            z_list.append(np.asarray(m.Z @ alpha, dtype=float))
        else:
            z_list.append(np.asarray(m.z, dtype=float))
    
    for combination, probe in sorted(hue_mapping.items()):
        probe_idx = list(probes).index(probe)
        gammas = []
        for pos_idx, sign in enumerate(combination):
            gamma = int(sign) * z_list[pos_idx][probe_idx]
            gammas.append(gamma)
        min_gamma = min(gammas)
        avg_gamma = sum(gammas) / len(gammas)
        if min_gamma <= 0:
            print(f"  {combination}: probe={probe}, min_gamma={min_gamma:.1f} [FAIL] 符号不匹配!")
        else:
            print(f"  {combination}: probe={probe}, min_gamma={min_gamma:.1f}, avg={avg_gamma:.1f}")
    
    # Exact eval
    blocks = test.generate_all_bit_blocks(4)
    legal_ber = test.evaluate_blocks_ber(models, blocks, hue_mapping)
    print(f"\nExact legal BER: {legal_ber:.6f}")
    
    # FEC eval
    rng = test.random.Random(42)
    info_bits = test.generate_random_information_bits(5000, 4, rng=rng)
    fec_ber = test.evaluate_blocks_ber_with_convolutional_fec(models, info_bits, hue_mapping)
    print(f"FEC legal BER: {fec_ber:.6f}")
    
    # Illegal check (sample 5 positions)
    print("\n非法位置安全性检查 (exact eval):")
    probe_to_row = test.build_probe_to_row(probes)
    legal_codes = [m.eff_code if m.eff_code is not None else m.code for m in models]
    all_positions = sorted([int(f.replace('.csv','')) for f in os.listdir(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white') if f.endswith('.csv')])
    
    worst_illegal = 1.0
    for pos in [2, 4, 5, 7, 8]:
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
        print(f"  Pos {pos}: secure_min={secure_bers.min():.3f}")
        if secure_bers.min() < worst_illegal:
            worst_illegal = secure_bers.min()
    
    print(f"\n最差非法位置 secure BER: {worst_illegal:.3f}")

print("\n" + "=" * 80)
print("Done.")
