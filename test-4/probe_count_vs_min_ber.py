#!/usr/bin/env python3
"""
验证：增加探针数量对固定4位置的 min illegal BER 的影响
位置固定: (1, 3, 12, 22)
探针数: 从8逐步增加到20，随机采样
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

test.N_COMPONENTS = 1
test.ALPHA = np.array([1.0])
test.BETA = np.array([1.0])
test.USE_AMPLITUDE_AWARE = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
positions = (1, 3, 12, 22)
all_probes = np.arange(5, 361, 5, dtype=float)

all_positions = list(range(1, 29))
illegal_positions = [p for p in all_positions if p not in positions]

csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition='white')

def eval_illegal_for_probes(probes, models, codes, hue_mapping, probe_to_row):
    min_secure = float('inf')
    worst_pos = -1
    for pos in illegal_positions:
        csv_file = os.path.join(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white', str(pos) + '.csv')
        mat_full = test.load_csv_matrix(csv_file)
        row_indices = [int((p/5)-1) for p in probes]
        mat = mat_full[row_indices].astype(float, copy=False)
        m_il = test.extract_fingerprint(probes, mat, force_positive_first=True)
        corr = test.calculate_correlation(models[0].code, m_il.code)
        if corr < 0:
            m_il.w = -m_il.w; m_il.z = -m_il.z; m_il.code = -m_il.code
        
        blocks = test.generate_all_bit_blocks(4)
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
        secure_per = np.minimum(raw, 1 - raw)
        mean_secure = np.mean(secure_per)
        if mean_secure < min_secure:
            min_secure = mean_secure
            worst_pos = pos
    return min_secure, worst_pos

print(f"固定位置: {positions}")
print(f"扫描不同探针数量对 min illegal BER 的影响")
print(f"\n{'Probe count':>12} | {'Best min illegal':>18} | {'Worst pos':>10} | {'Legal BER':>12}")
print("-" * 65)

rng = np.random.RandomState(42)
for probe_count in [8, 10, 12, 14, 16, 18, 20]:
    best_min_illegal = -1
    best_probes = None
    best_worst_pos = -1
    best_legal = float('inf')
    
    # 随机采样100组探针集合
    for trial in range(100):
        probes = np.sort(rng.choice(all_probes, size=probe_count, replace=False))
        matrices = test.load_selected_rows(csv_files, probes)
        models = [test.extract_fingerprint(probes, mat, force_positive_first=True) for mat in matrices]
        models = test.align_model_directions(models)
        
        codes = [m.code for m in models]
        probe_to_row = test.build_probe_to_row(probes)
        
        # 快速评估 legal BER (用少量随机 blocks)
        test_blocks = [test.generate_random_bit_blocks(100, 4, rng=test.random.Random(trial)) for _ in range(1)][0]
        legal_errors = np.zeros(4)
        legal_totals = np.zeros(4)
        for bits_pm in test_blocks:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes)
            try:
                hue_mapping = test.build_hue_mapping(models, probes, mapping_eval_bits=100, top_k_per_combination=2, use_amplitude_aware=False, rng=test.random.Random(trial))
            except:
                continue
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            bits_tx = test.pm1_to_bin(bits_pm)
            for li in range(4):
                Y_obs = test.observe_block_from_measured_matrix(hue_seq, models[li].Y, probe_to_row)
                dec = test.decode_local_block(Y_obs, models[li].w, models[li].code)
                legal_totals[li] += 1
                if dec.bit_hat_bin != bits_tx[li]:
                    legal_errors[li] += 1
        
        if np.sum(legal_totals) < 100:
            continue
        legal_raw = legal_errors / np.maximum(legal_totals, 1)
        legal_secure = np.mean(np.minimum(legal_raw, 1 - legal_raw))
        
        if legal_secure > 0.05:  # legal BER 太高，跳过
            continue
        
        # 评估 illegal min BER
        try:
            hue_mapping = test.build_hue_mapping(models, probes, mapping_eval_bits=200, top_k_per_combination=2, use_amplitude_aware=False, rng=test.random.Random(trial))
        except:
            continue
        min_il, wpos = eval_illegal_for_probes(probes, models, codes, hue_mapping, probe_to_row)
        
        if min_il > best_min_illegal:
            best_min_illegal = min_il
            best_probes = probes.copy()
            best_worst_pos = wpos
            best_legal = legal_secure
    
    print(f"{probe_count:>12} | {best_min_illegal:>18.4f} | Pos {best_worst_pos:>6} | {best_legal:>12.4f}")

print("\nDone.")
