#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单主成分发送 + 多主成分攻击评估

保持发送端为单主成分（保证 legal BER=0），
但让非法位置尝试用多主成分方向做最优攻击。
"""
import sys, os
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 临时切换回单主成分（发送端）
original_N = test.N_COMPONENTS
test.N_COMPONENTS = 1

positions = (1, 3, 12, 22)
probes = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)

print(f"Testing positions {positions} (single-component TX)")

csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition='white')
matrices = test.load_selected_rows(csv_files, probes)
models_single = [test.extract_fingerprint(probes, mat, force_positive_first=True) for mat in matrices]
models_single = test.align_model_directions(models_single)

# 单主成分 hue mapping
hue_mapping_single = test.build_hue_mapping(models_single, probes, mapping_eval_bits=500, top_k_per_combination=3)

# 验证 legal BER=0
blocks = test.generate_all_bit_blocks(4)
legal_ber = test.evaluate_blocks_ber(models_single, blocks, hue_mapping_single)
print(f"Single-comp legal BER: {legal_ber:.6f}")

# 恢复多主成分（用于提取多主成分信息）
test.N_COMPONENTS = original_N

probe_to_row = test.build_probe_to_row(probes)
legal_codes = [m.code for m in models_single]  # 单主成分发送

all_positions = sorted([int(f.replace('.csv','')) for f in os.listdir(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white') if f.endswith('.csv')])

print("\n--- Illegal position multi-component attack ---")
for pos in all_positions[:15]:
    if pos in positions:
        continue
    
    csv_file = os.path.join(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white', str(pos) + '.csv')
    mat = test.load_csv_matrix(csv_file)
    row_indices = [int((p/5)-1) for p in probes]
    illegal_mat = mat[row_indices].astype(float, copy=False)
    
    # 提取多主成分模型
    illegal_multi = test.extract_fingerprint(probes, illegal_mat, force_positive_first=True)
    
    # 对齐方向
    corr = test.calculate_correlation(models_single[0].code, illegal_multi.code)
    if corr < 0:
        if illegal_multi.W is not None and illegal_multi.W.ndim > 1:
            illegal_multi.W = -illegal_multi.W
            illegal_multi.Z = -illegal_multi.Z
            illegal_multi.multi_code = -illegal_multi.multi_code
            illegal_multi.eff_w = -illegal_multi.eff_w
            illegal_multi.eff_code = -illegal_multi.eff_code
        else:
            illegal_multi.w = -illegal_multi.w
            illegal_multi.z = -illegal_multi.z
            illegal_multi.code = -illegal_multi.code
    
    # 单主成分攻击（基准）
    errors_1d = np.zeros(4)
    totals = np.zeros(4)
    for bits_pm in blocks:
        _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
        hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping_single)
        obs = test.observe_block_from_measured_matrix(hue_seq, illegal_multi.Y, probe_to_row)
        dec = test.decode_local_block(obs, illegal_multi.w, illegal_multi.code)
        true_bits = test.pm1_to_bin(bits_pm)
        for li in range(4):
            totals[li] += 1
            if dec.bit_hat_bin != true_bits[li]:
                errors_1d[li] += 1
    raw_1d = errors_1d / totals
    secure_1d = np.minimum(raw_1d, 1-raw_1d)
    
    # 多主成分攻击：尝试不同权重组合寻找最佳攻击方向
    best_secure = secure_1d.min()
    best_alpha = None
    
    if illegal_multi.W is not None and illegal_multi.W.ndim > 1:
        for a1 in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
            for a2 in [0.0, 0.05, 0.1, 0.2, 0.3]:
                a3 = max(0, 1 - a1 - a2)
                alpha = np.array([a1, a2, a3])
                if alpha.sum() <= 0:
                    continue
                alpha = alpha / alpha.sum()
                
                eff_z = illegal_multi.Z @ alpha
                eff_code = np.sign(eff_z).astype(int)
                eff_w = illegal_multi.W @ alpha
                
                errors_mc = np.zeros(4)
                for bits_pm in blocks:
                    _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
                    hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping_single)
                    obs = test.observe_block_from_measured_matrix(hue_seq, illegal_multi.Y, probe_to_row)
                    dec = test.decode_local_block(obs, eff_w, eff_code)
                    true_bits = test.pm1_to_bin(bits_pm)
                    for li in range(4):
                        if dec.bit_hat_bin != true_bits[li]:
                            errors_mc[li] += 1
                raw_mc = errors_mc / totals
                secure_mc = np.minimum(raw_mc, 1-raw_mc)
                if secure_mc.min() < best_secure:
                    best_secure = secure_mc.min()
                    best_alpha = alpha.copy()
    
    print(f"Pos {pos:2d}: 1D_attack={secure_1d.min():.3f}, best_MC_attack={best_secure:.3f}, alpha={best_alpha}")

# 恢复原始设置
test.N_COMPONENTS = original_N
print("\nDone.")
