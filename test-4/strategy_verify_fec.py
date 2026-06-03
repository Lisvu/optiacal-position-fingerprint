#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证分阶段策略：放宽raw BER -> 保证security -> FEC纠错提升准确性

策略：
  1. 找到probe set使 raw legal BER <= 0.05
  2. 保证 min illegal secure BER >= 0.2
  3. 用卷积FEC(2,1,3) + Viterbi将legal BER从0.05降到<0.001
"""
import sys, os
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 测试组合
positions = (1, 3, 12, 22)
probes = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)

print("=" * 80)
print("分阶段策略验证：Raw BER放宽 + FEC纠错")
print("=" * 80)
print(f"Positions: {positions}")
print(f"Probes: {list(probes)}")
print(f"N_COMPONENTS={test.N_COMPONENTS}")

# 尝试不同ALPHA配置，找到满足 raw_legal<=0.05 且 min_illegal>=0.2 的组合
alpha_candidates = [
    [0.95, 0.04, 0.01],
    [0.90, 0.08, 0.02],
    [0.85, 0.12, 0.03],
    [0.80, 0.16, 0.04],
    [0.75, 0.20, 0.05],
    [0.70, 0.24, 0.06],
    [0.65, 0.28, 0.07],
    [0.60, 0.32, 0.08],
]

best_config = None
best_score = -1

for alpha in alpha_candidates:
    test.ALPHA = np.array(alpha)
    test.BETA = np.array(alpha)
    
    csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition='white')
    models, hue_mapping = test.build_models_from_probes(csv_files, probes, mapping_eval_bits=500, mapping_top_k=3)
    
    # Raw legal exact BER
    blocks = test.generate_all_bit_blocks(4)
    raw_legal = test.evaluate_blocks_ber(models, blocks, hue_mapping)
    
    # Check illegal positions (exact, all positions)
    probe_to_row = test.build_probe_to_row(probes)
    legal_codes = [m.eff_code if m.eff_code is not None else m.code for m in models]
    all_positions = sorted([int(f.replace('.csv','')) for f in os.listdir(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white') if f.endswith('.csv')])
    
    worst_illegal = 1.0
    worst_pos = None
    
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
            worst_pos = pos
    
    print(f"\nALPHA={alpha}: raw_legal={raw_legal:.4f}, worst_illegal={worst_illegal:.3f} (pos {worst_pos})")
    
    # 评估是否满足策略要求
    if raw_legal <= 0.05 and worst_illegal >= 0.2:
        print(f"  *** 满足分阶段策略要求! ***")
        score = worst_illegal - raw_legal
        if score > best_score:
            best_score = score
            best_config = {
                'alpha': alpha,
                'raw_legal': raw_legal,
                'worst_illegal': worst_illegal,
                'worst_pos': worst_pos,
                'models': models,
                'hue_mapping': hue_mapping,
            }

print("\n" + "=" * 80)
if best_config:
    print("找到满足分阶段策略的配置!")
    print(f"ALPHA={best_config['alpha']}")
    print(f"Raw Legal BER={best_config['raw_legal']:.4f}")
    print(f"Worst Illegal secure BER={best_config['worst_illegal']:.3f}")
    
    # Phase 2: FEC验证
    print("\n" + "=" * 80)
    print("Phase 2: FEC纠错验证")
    print("=" * 80)
    
    test.ALPHA = np.array(best_config['alpha'])
    test.BETA = np.array(best_config['alpha'])
    models = best_config['models']
    hue_mapping = best_config['hue_mapping']
    
    for num_bits in [1000, 5000, 10000]:
        rng = random.Random(42)
        info_bits = test.generate_random_information_bits(num_bits, 4, rng=rng)
        fec_ber = test.evaluate_blocks_ber_with_convolutional_fec(models, info_bits, hue_mapping)
        print(f"FEC ({num_bits} bits): BER = {fec_ber:.6f}")
    
    # 同时测试非法位置在FEC下的表现（攻击者知道FEC结构）
    print("\n--- 非法位置FEC攻击测试 ---")
    all_positions = sorted([int(f.replace('.csv','')) for f in os.listdir(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white') if f.endswith('.csv')])
    probe_to_row = test.build_probe_to_row(probes)
    legal_codes = [m.eff_code if m.eff_code is not None else m.code for m in models]
    
    for pos in [2, 4, 5, 7, 8, 9, 10]:
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
        
        # FEC攻击：攻击者知道FEC结构，用Viterbi解码
        rng = random.Random(42)
        info_bits = test.generate_random_information_bits(5000, 4, rng=rng)
        bit_blocks_pm = test.build_convolutional_bit_blocks(info_bits)
        
        received = [[] for _ in range(4)]
        for bits_pm in bit_blocks_pm:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            obs = test.observe_block_from_measured_matrix(hue_seq, illegal_model.Y, probe_to_row)
            if illegal_model.eff_w is not None and illegal_model.eff_code is not None:
                dec = test.decode_local_block(obs, illegal_model.eff_w, illegal_model.eff_code)
            else:
                dec = test.decode_local_block(obs, illegal_model.w, illegal_model.code)
            for li in range(4):
                received[li].append(dec.bit_hat_bin)
        
        for li in range(4):
            decoded = test.viterbi_decode_hard(received[li])
            ref = info_bits[:, li]
            cl = min(len(decoded), len(ref))
            if cl > 0:
                raw_ber = float(np.mean(decoded[:cl] != ref[:cl]))
                secure_ber = min(raw_ber, 1.0 - raw_ber)
                if li == 0:
                    print(f"Pos {pos}: FEC secure BER = {secure_ber:.4f}")
else:
    print("未找到同时满足 raw_legal<=0.05 且 worst_illegal>=0.2 的配置")
    print("\n需要进一步放宽raw BER目标或调整多主成分权重")

print("\nDone.")
