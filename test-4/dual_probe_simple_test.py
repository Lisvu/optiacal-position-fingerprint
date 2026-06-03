#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双探针概念验证（简化版）：直接测试一个固定的 P_b
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
positions = (1, 3, 12, 22)
probes_a = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)
probes_b = np.array([5, 40, 80, 155, 185, 265, 330, 360], dtype=float)

print("Dual-Probe Concept Verification (Fixed P_b)")
print(f"P_a: {list(probes_a)}")
print(f"P_b: {list(probes_b)}")

csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition='white')

# Build P_a models
print("Building P_a models...")
matrices_a = test.load_selected_rows(csv_files, probes_a)
models_a = [test.extract_fingerprint(probes_a, mat, force_positive_first=True) for mat in matrices_a]
models_a = test.align_model_directions(models_a)
probe_to_row_a = test.build_probe_to_row(probes_a)

# Build P_b models
print("Building P_b models...")
matrices_b = test.load_selected_rows(csv_files, probes_b)
models_b = [test.extract_fingerprint(probes_b, mat, force_positive_first=True) for mat in matrices_b]
models_b = test.align_model_directions(models_b)
probe_to_row_b = test.build_probe_to_row(probes_b)

# Hue mapping based on P_a, but restricted to probes in BOTH P_a and P_b
print("Building hue mapping (restricted to common probes)...")
common_probes = np.array(sorted(set(probes_a.tolist()) & set(probes_b.tolist())), dtype=float)
print(f"Common probes: {list(common_probes)}")

# Temporarily restrict hue mapping to common probes
# We need to modify the models to only use common probes for hue mapping
# But keep full models for decoding

# Create mapping models from models_a but restricted to common probes for observation.
# Keep full decoding directions (eff_w, eff_code) so mapping is consistent with decoder.
models_for_mapping = []
for m in models_a:
    # Find indices of common probes in models_a's probe list
    probe_list_a = m.probes.tolist()
    indices = [probe_list_a.index(p) for p in common_probes if p in probe_list_a]
    m_copy = test.FingerprintModel(
        probes=common_probes.copy(),
        Y=m.Y[indices].copy(),
        trend=m.trend[indices].copy() if m.trend is not None else None,
        residual=m.residual[indices].copy() if m.residual is not None else None,
        w=m.w.copy() if m.w is not None else None,
        z=m.z[indices].copy() if m.z is not None else None,
        code=m.code[indices].copy() if m.code is not None else None,
        W=m.W.copy() if m.W is not None else None,
        Z=m.Z[indices].copy() if m.Z is not None else None,
        multi_code=m.multi_code[indices].copy() if m.multi_code is not None else None,
        eff_code=m.eff_code[indices].copy() if m.eff_code is not None else None,
        eff_z=m.eff_z[indices].copy() if m.eff_z is not None else None,
        eff_w=m.eff_w.copy() if m.eff_w is not None else None,
    )
    models_for_mapping.append(m_copy)

hue_mapping = test.build_hue_mapping(
    models_for_mapping, common_probes,
    mapping_eval_bits=500, top_k_per_combination=3,
    use_amplitude_aware=True,
)

# Evaluate legal (exact)
print("Evaluating legal positions...")
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
        if dec_a.bit_hat_bin != bits_tx[li] or dec_b.bit_hat_bin != bits_tx[li] or dec_a.bit_hat_bin != dec_b.bit_hat_bin:
            errors_joint[li] += 1

raw_a = errors_a / totals
raw_b = errors_b / totals
raw_joint = errors_joint / totals
secure_a = np.minimum(raw_a, 1 - raw_a)
secure_b = np.minimum(raw_b, 1 - raw_b)
secure_joint = np.minimum(raw_joint, 1 - raw_joint)

print(f"\nLegal position results:")
print(f"  P_a only: raw={raw_a}, secure={secure_a}")
print(f"  P_b only: raw={raw_b}, secure={secure_b}")
print(f"  Joint:    raw={raw_joint}, secure={secure_joint}")
print(f"  Mean secure: P_a={np.mean(secure_a):.4f}, P_b={np.mean(secure_b):.4f}, Joint={np.mean(secure_joint):.4f}")

# Test illegal position
print(f"\nTesting illegal position (Pos 2)...")
pos = 2
csv_file = os.path.join(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white', str(pos) + '.csv')
mat = test.load_csv_matrix(csv_file)

row_indices_a = [int((p/5)-1) for p in probes_a]
illegal_mat_a = mat[row_indices_a].astype(float, copy=False)
illegal_model_a = test.extract_fingerprint(probes_a, illegal_mat_a, force_positive_first=True)

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
        
        Y_obs_b = test.observe_block_from_measured_matrix(hue_seq, illegal_model_b.Y, probe_to_row_b)
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

print(f"\nIllegal position (Pos 2) results:")
print(f"  P_a only: raw={raw_a}, secure={secure_a}")
print(f"  P_b only: raw={raw_b}, secure={secure_b}")
print(f"  Joint:    raw={raw_joint}, secure={secure_joint}")
print(f"  Mean secure: P_a={np.mean(secure_a):.4f}, P_b={np.mean(secure_b):.4f}, Joint={np.mean(secure_joint):.4f}")

print("\nDone.")
