#!/usr/bin/env python3
"""
精确验证：单主成分基线对每个非法位置的 secure BER
目标：找出 min illegal BER（worst-case 漏洞）
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

# 强制单主成分
test.N_COMPONENTS = 1
test.ALPHA = np.array([1.0])
test.BETA = np.array([1.0])
test.USE_AMPLITUDE_AWARE = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
positions = (1, 3, 12, 22)
probes = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)

csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition='white')
matrices = test.load_selected_rows(csv_files, probes)
models = [test.extract_fingerprint(probes, mat, force_positive_first=True) for mat in matrices]
models = test.align_model_directions(models)

codes = [m.code for m in models]
probe_to_row = test.build_probe_to_row(probes)

hue_mapping = test.build_hue_mapping(models, probes, mapping_eval_bits=500, top_k_per_combination=3, use_amplitude_aware=False)
blocks = test.generate_all_bit_blocks(4)

all_positions = list(range(1, 29))
illegal_positions = [p for p in all_positions if p not in positions]

print(f"合法位置: {positions}")
print(f"探针: {list(probes)}")
print(f"\n{'非法位置':>8} | {'Total raw BER':>12} | {'Secure BER':>12} | {'Closest legal':>14} | {'Code corr':>10}")
print("-" * 75)

results = []
for pos in illegal_positions:
    csv_file = os.path.join(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white', str(pos) + '.csv')
    mat_full = test.load_csv_matrix(csv_file)
    row_indices = [int((p/5)-1) for p in probes]
    mat = mat_full[row_indices].astype(float, copy=False)
    m_il = test.extract_fingerprint(probes, mat, force_positive_first=True)
    
    # 对齐到 models[0]
    corr = test.calculate_correlation(models[0].code, m_il.code)
    if corr < 0:
        m_il.w = -m_il.w; m_il.z = -m_il.z; m_il.code = -m_il.code
    
    # 找最近的合法位置（code 相关性最高）
    best_corr = -1
    best_legal = -1
    for li in range(4):
        c = abs(test.calculate_correlation(models[li].code, m_il.code))
        if c > best_corr:
            best_corr = c
            best_legal = positions[li]
    
    errors = np.zeros(4)
    totals = np.zeros(4)
    for bits_pm in blocks:
        _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes)
        hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
        bits_tx = test.pm1_to_bin(bits_pm)
        for li in range(4):
            Y_obs = test.observe_block_from_measured_matrix(hue_seq, m_il.Y, probe_to_row)
            dec = test.decode_local_block(Y_obs, m_il.w, m_il.code)
            totals[li] += 1
            if dec.bit_hat_bin != bits_tx[li]:
                errors[li] += 1
    
    raw = errors / np.maximum(totals, 1)
    secure_per_pos = np.minimum(raw, 1 - raw)
    total_raw = np.mean(raw)
    total_secure = np.mean(secure_per_pos)
    
    results.append((total_secure, pos, total_raw, best_legal, best_corr))
    print(f"{pos:>8} | {total_raw:>12.4f} | {total_secure:>12.4f} | Pos {best_legal:>10} | {best_corr:>10.4f}")

results.sort()
print(f"\nWorst-case (最低 secure BER):")
for sec, pos, raw, ble, bc in results[:5]:
    print(f"  Pos {pos}: secure={sec:.4f}, raw={raw:.4f}, closest=Pos{ble}, corr={bc:.4f}")

print(f"\nBest-case (最高 secure BER):")
for sec, pos, raw, ble, bc in results[-5:]:
    print(f"  Pos {pos}: secure={sec:.4f}, raw={raw:.4f}, closest=Pos{ble}, corr={bc:.4f}")

print(f"\nMin illegal secure BER: {results[0][0]:.4f} (Pos {results[0][1]})")
print(f"Max illegal secure BER: {results[-1][0]:.4f} (Pos {results[-1][1]})")
print(f"Mean illegal secure BER: {np.mean([r[0] for r in results]):.4f}")

# 同时输出每个合法位置各自的 secure BER（看攻击者针对特定位置的难度）
print(f"\n按合法位置分解 (每个非法位置攻击每个合法位置的平均 secure BER):")
for li in range(4):
    per_legals = []
    for pos in illegal_positions:
        csv_file = os.path.join(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white', str(pos) + '.csv')
        mat_full = test.load_csv_matrix(csv_file)
        row_indices = [int((p/5)-1) for p in probes]
        mat = mat_full[row_indices].astype(float, copy=False)
        m_il = test.extract_fingerprint(probes, mat, force_positive_first=True)
        corr = test.calculate_correlation(models[0].code, m_il.code)
        if corr < 0:
            m_il.w = -m_il.w; m_il.z = -m_il.z; m_il.code = -m_il.code
        
        errors = 0
        totals = 0
        for bits_pm in blocks:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            bits_tx = test.pm1_to_bin(bits_pm)
            Y_obs = test.observe_block_from_measured_matrix(hue_seq, m_il.Y, probe_to_row)
            dec = test.decode_local_block(Y_obs, m_il.w, m_il.code)
            totals += 1
            if dec.bit_hat_bin != bits_tx[li]:
                errors += 1
        raw = errors / max(totals, 1)
        per_legals.append(min(raw, 1-raw))
    
    print(f"  攻击合法位置 {positions[li]}: min={np.min(per_legals):.4f}, mean={np.mean(per_legals):.4f}, max={np.max(per_legals):.4f}")

print("\nDone.")
