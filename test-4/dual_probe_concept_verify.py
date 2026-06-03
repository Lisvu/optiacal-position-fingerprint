#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双探针概念验证：用已知 zero-BER probe set 作为 P_a，
找互补的 P_b，测试双探针联合解码效果。
"""
import sys, os
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 已知 zero-BER 组合和 probes
positions = (1, 3, 12, 22)
probes_a = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)

print("=" * 80)
print("双探针概念验证")
print("=" * 80)
print(f"Positions: {positions}")
print(f"P_a (known zero-BER): {list(probes_a)}")

all_probes = (5 + np.arange(73) * 5).astype(float).tolist()
remaining = [p for p in all_probes if p not in probes_a]
print(f"Available probes for P_b: {len(remaining)}")

csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition='white')

# Build P_a models
matrices_a = test.load_selected_rows(csv_files, probes_a)
models_a = [test.extract_fingerprint(probes_a, mat, force_positive_first=True) for mat in matrices_a]
models_a = test.align_model_directions(models_a)
probe_to_row_a = test.build_probe_to_row(probes_a)

# Test multiple P_b candidates
print("\nTesting P_b candidates...")
print("=" * 80)

best_result = None
best_joint_ber = float("inf")

for i in range(50):
    # Random P_b from remaining probes
    count_b = random.choice([7, 8, 9, 10])
    probes_b = np.sort(np.asarray(random.sample(remaining, count_b), dtype=float))
    if not test.is_valid_probe_set(probes_b, 30):
        continue
    
    try:
        if i == 0:
            print(f"  Trying first sample with P_b={list(probes_b)}")
        
        # Build P_b models
        matrices_b = test.load_selected_rows(csv_files, probes_b)
        models_b = [test.extract_fingerprint(probes_b, mat, force_positive_first=True) for mat in matrices_b]
        models_b = test.align_model_directions(models_b)
        probe_to_row_b = test.build_probe_to_row(probes_b)
        
        # Hue mapping based on P_a
        hue_mapping = test.build_hue_mapping(
            models_a, probes_a,
            mapping_eval_bits=500, top_k_per_combination=3,
            use_amplitude_aware=True,
        )
        
        # Evaluate legal (exact)
        blocks = test.generate_all_bit_blocks(4)
        codes_a = [m.eff_code if m.eff_code is not None else m.code for m in models_a]
        
        errors_a = np.zeros(4)
        errors_b = np.zeros(4)
        errors_joint = np.zeros(4)
        totals = np.zeros(4)
        
        for bits_pm in blocks:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes_a)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            bits_tx = test.pm1_to_bin(bits_pm)
            
            for li in range(4):
                # P_a decode
                Y_obs_a = test.observe_block_from_measured_matrix(hue_seq, models_a[li].Y, probe_to_row_a)
                if models_a[li].eff_w is not None and models_a[li].eff_code is not None:
                    dec_a = test.decode_local_block(Y_obs_a, models_a[li].eff_w, models_a[li].eff_code)
                else:
                    dec_a = test.decode_local_block(Y_obs_a, models_a[li].w, models_a[li].code)
                
                # P_b decode
                Y_obs_b = test.observe_block_from_measured_matrix(hue_seq, models_b[li].Y, probe_to_row_b)
                if models_b[li].eff_w is not None and models_b[li].eff_code is not None:
                    dec_b = test.decode_local_block(Y_obs_b, models_b[li].eff_w, models_b[li].eff_code)
                else:
                    dec_b = test.decode_local_block(Y_obs_b, models_b[li].w, models_b[li].code)
                
                totals[li] += 1
                if dec_a.bit_hat_bin != bits_tx[li]:
                    errors_a[li] += 1
                if dec_b.bit_hat_bin != bits_tx[li]:
                    errors_b[li] += 1
                # Joint: error if either disagrees with truth OR they disagree with each other
                if dec_a.bit_hat_bin != bits_tx[li] or dec_b.bit_hat_bin != bits_tx[li] or dec_a.bit_hat_bin != dec_b.bit_hat_bin:
                    errors_joint[li] += 1
        
        raw_a = errors_a / totals
        raw_b = errors_b / totals
        raw_joint = errors_joint / totals
        secure_a = np.minimum(raw_a, 1 - raw_a)
        secure_b = np.minimum(raw_b, 1 - raw_b)
        secure_joint = np.minimum(raw_joint, 1 - raw_joint)
        
        ber_a = float(np.mean(secure_a))
        ber_b = float(np.mean(secure_b))
        ber_joint = float(np.mean(secure_joint))
        
        if ber_joint < best_joint_ber:
            best_joint_ber = ber_joint
            best_result = {
                'probes_b': probes_b.copy(),
                'ber_a': ber_a,
                'ber_b': ber_b,
                'ber_joint': ber_joint,
                'models_b': models_b,
                'probe_to_row_b': probe_to_row_b,
            }
            print(f"  Sample {i}: P_b count={count_b}, ber_a={ber_a:.4f}, ber_b={ber_b:.4f}, joint={ber_joint:.4f}")
        
    except Exception as e:
        if i % 10 == 0:
            print(f"  Sample {i} error: {e}")
        continue

if best_result:
    print(f"\n{'=' * 80}")
    print(f"Best P_b found:")
    print(f"  Probes: {list(best_result['probes_b'])}")
    print(f"  P_a only BER: {best_result['ber_a']:.4f}")
    print(f"  P_b only BER: {best_result['ber_b']:.4f}")
    print(f"  Joint BER: {best_result['ber_joint']:.4f}")
    
    # Evaluate illegal positions
    print(f"\nIllegal position security test:")
    all_positions = sorted([int(f.replace('.csv','')) for f in os.listdir(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white') if f.endswith('.csv')])
    
    worst_joint = 1.0
    worst_pos = None
    
    for pos in all_positions:
        if pos in positions:
            continue
        
        csv_file = os.path.join(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white', str(pos) + '.csv')
        mat = test.load_csv_matrix(csv_file)
        
        # P_a illegal model
        row_indices_a = [int((p/5)-1) for p in probes_a]
        illegal_mat_a = mat[row_indices_a].astype(float, copy=False)
        illegal_model_a = test.extract_fingerprint(probes_a, illegal_mat_a, force_positive_first=True)
        
        # P_b illegal model
        probes_b = best_result['probes_b']
        row_indices_b = [int((p/5)-1) for p in probes_b]
        illegal_mat_b = mat[row_indices_b].astype(float, copy=False)
        illegal_model_b = test.extract_fingerprint(probes_b, illegal_mat_b, force_positive_first=True)
        
        # Align
        corr_a = test.calculate_correlation(models_a[0].eff_code if models_a[0].eff_code is not None else models_a[0].code,
                                            illegal_model_a.eff_code if illegal_model_a.eff_code is not None else illegal_model_a.code)
        if corr_a < 0:
            if illegal_model_a.W is not None and illegal_model_a.W.ndim > 1:
                illegal_model_a.W = -illegal_model_a.W
                illegal_model_a.Z = -illegal_model_a.Z
                illegal_model_a.multi_code = -illegal_model_a.multi_code
                illegal_model_a.eff_w = -illegal_model_a.eff_w
                illegal_model_a.eff_code = -illegal_model_a.eff_code
            else:
                illegal_model_a.w = -illegal_model_a.w
                illegal_model_a.z = -illegal_model_a.z
                illegal_model_a.code = -illegal_model_a.code
        
        corr_b = test.calculate_correlation(models_a[0].eff_code if models_a[0].eff_code is not None else models_a[0].code,
                                            illegal_model_b.eff_code if illegal_model_b.eff_code is not None else illegal_model_b.code)
        if corr_b < 0:
            if illegal_model_b.W is not None and illegal_model_b.W.ndim > 1:
                illegal_model_b.W = -illegal_model_b.W
                illegal_model_b.Z = -illegal_model_b.Z
                illegal_model_b.multi_code = -illegal_model_b.multi_code
                illegal_model_b.eff_w = -illegal_model_b.eff_w
                illegal_model_b.eff_code = -illegal_model_b.eff_code
            else:
                illegal_model_b.w = -illegal_model_b.w
                illegal_model_b.z = -illegal_model_b.z
                illegal_model_b.code = -illegal_model_b.code
        
        errors_a = np.zeros(4)
        errors_b = np.zeros(4)
        errors_joint = np.zeros(4)
        totals = np.zeros(4)
        
        for bits_pm in blocks:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes_a)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            bits_tx = test.pm1_to_bin(bits_pm)
            
            for li in range(4):
                Y_obs_a = test.observe_block_from_measured_matrix(hue_seq, illegal_model_a.Y, probe_to_row_a)
                if illegal_model_a.eff_w is not None and illegal_model_a.eff_code is not None:
                    dec_a = test.decode_local_block(Y_obs_a, illegal_model_a.eff_w, illegal_model_a.eff_code)
                else:
                    dec_a = test.decode_local_block(Y_obs_a, illegal_model_a.w, illegal_model_a.code)
                
                Y_obs_b = test.observe_block_from_measured_matrix(hue_seq, illegal_model_b.Y, best_result['probe_to_row_b'])
                if illegal_model_b.eff_w is not None and illegal_model_b.eff_code is not None:
                    dec_b = test.decode_local_block(Y_obs_b, illegal_model_b.eff_w, illegal_model_b.eff_code)
                else:
                    dec_b = test.decode_local_block(Y_obs_b, illegal_model_b.w, illegal_model_b.code)
                
                totals[li] += 1
                if dec_a.bit_hat_bin != bits_tx[li]:
                    errors_a[li] += 1
                if dec_b.bit_hat_bin != bits_tx[li]:
                    errors_b[li] += 1
                if dec_a.bit_hat_bin != bits_tx[li] or dec_b.bit_hat_bin != bits_tx[li] or dec_a.bit_hat_bin != dec_b.bit_hat_bin:
                    errors_joint[li] += 1
        
        raw_a = errors_a / totals
        raw_b = errors_b / totals
        raw_joint = errors_joint / totals
        secure_a = np.minimum(raw_a, 1 - raw_a)
        secure_b = np.minimum(raw_b, 1 - raw_b)
        secure_joint = np.minimum(raw_joint, 1 - raw_joint)
        
        print(f"  Pos {pos}: P_a_only={secure_a.min():.3f}, P_b_only={secure_b.min():.3f}, joint={secure_joint.min():.3f}")
        
        if secure_joint.min() < worst_joint:
            worst_joint = secure_joint.min()
            worst_pos = pos
    
    print(f"\nWorst illegal position: Pos {worst_pos}, joint_secure={worst_joint:.3f}")
else:
    print("No valid P_b found!")

print("\nDone.")
