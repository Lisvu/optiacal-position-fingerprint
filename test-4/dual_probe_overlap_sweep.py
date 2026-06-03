#!/usr/bin/env python3
"""
Quick diagnostic: sweep shared-probe count to characterize the reliability/security trade-off.
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
positions = (1, 3, 12, 22)
probes_a = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)

# Candidate P_b sets with varying overlap
candidates = [
    np.array([5, 40, 80, 155, 190, 260, 330, 360], dtype=float),   # 4 common
    np.array([5, 40, 80, 155, 185, 260, 330, 360], dtype=float),   # 5 common
    np.array([5, 40, 80, 155, 185, 265, 330, 360], dtype=float),   # 6 common
    np.array([5, 40, 80, 155, 185, 265, 335, 360], dtype=float),   # 7 common
    np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float),   # 8 common (identical)
]

csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition='white')

# Build P_a models once
matrices_a = test.load_selected_rows(csv_files, probes_a)
models_a = [test.extract_fingerprint(probes_a, mat, force_positive_first=True) for mat in matrices_a]
models_a = test.align_model_directions(models_a)
codes_a = [m.eff_code if m.eff_code is not None else m.code for m in models_a]
probe_to_row_a = test.build_probe_to_row(probes_a)
blocks = test.generate_all_bit_blocks(4)

# Illegal position setup (Pos 2)
pos_illegal = 2
csv_illegal = os.path.join(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white', str(pos_illegal) + '.csv')
mat_illegal_full = test.load_csv_matrix(csv_illegal)

print(f"{'Common':>6} | {'Legal mean secure':>18} | {'Illegal mean secure':>20} | {'Joint (illegal)':>16}")
print("-" * 70)

for probes_b in candidates:
    common_probes = np.array(sorted(set(probes_a.tolist()) & set(probes_b.tolist())), dtype=float)
    n_common = len(common_probes)

    # Build P_b models
    matrices_b = test.load_selected_rows(csv_files, probes_b)
    models_b = [test.extract_fingerprint(probes_b, mat, force_positive_first=True) for mat in matrices_b]
    models_b = test.align_model_directions(models_b)
    probe_to_row_b = test.build_probe_to_row(probes_b)

    # Build mapping models from models_a restricted to common probes
    models_for_mapping = []
    for m in models_a:
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

    # Evaluate legal
    errors_a = np.zeros(4)
    errors_b = np.zeros(4)
    errors_joint = np.zeros(4)
    totals = np.zeros(4)
    for bits_pm in blocks:
        _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes_a)
        hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
        bits_tx = test.pm1_to_bin(bits_pm)
        for li in range(4):
            Y_obs_a = test.observe_block_from_measured_matrix(hue_seq, models_a[li].Y, probe_to_row_a)
            dec_a = test.decode_local_block(Y_obs_a, models_a[li].eff_w, models_a[li].eff_code)
            Y_obs_b = test.observe_block_from_measured_matrix(hue_seq, models_b[li].Y, probe_to_row_b)
            dec_b = test.decode_local_block(Y_obs_b, models_b[li].eff_w, models_b[li].eff_code)
            totals[li] += 1
            if dec_a.bit_hat_bin != bits_tx[li]: errors_a[li] += 1
            if dec_b.bit_hat_bin != bits_tx[li]: errors_b[li] += 1
            if dec_a.bit_hat_bin != bits_tx[li] or dec_b.bit_hat_bin != bits_tx[li] or dec_a.bit_hat_bin != dec_b.bit_hat_bin:
                errors_joint[li] += 1
    raw_joint_legal = errors_joint / totals
    secure_joint_legal = np.minimum(raw_joint_legal, 1 - raw_joint_legal)
    mean_legal = np.mean(secure_joint_legal)

    # Evaluate illegal
    row_indices_a = [int((p/5)-1) for p in probes_a]
    illegal_mat_a = mat_illegal_full[row_indices_a].astype(float, copy=False)
    illegal_model_a = test.extract_fingerprint(probes_a, illegal_mat_a, force_positive_first=True)
    row_indices_b = [int((p/5)-1) for p in probes_b]
    illegal_mat_b = mat_illegal_full[row_indices_b].astype(float, copy=False)
    illegal_model_b = test.extract_fingerprint(probes_b, illegal_mat_b, force_positive_first=True)

    # Align illegal models to models_a[0]
    corr_a = test.calculate_correlation(models_a[0].eff_code, illegal_model_a.eff_code)
    if corr_a < 0:
        illegal_model_a.W = -illegal_model_a.W; illegal_model_a.Z = -illegal_model_a.Z
        illegal_model_a.multi_code = -illegal_model_a.multi_code
        illegal_model_a.eff_w = -illegal_model_a.eff_w; illegal_model_a.eff_code = -illegal_model_a.eff_code
    corr_b = test.calculate_correlation(models_a[0].eff_code, illegal_model_b.eff_code)
    if corr_b < 0:
        illegal_model_b.W = -illegal_model_b.W; illegal_model_b.Z = -illegal_model_b.Z
        illegal_model_b.multi_code = -illegal_model_b.multi_code
        illegal_model_b.eff_w = -illegal_model_b.eff_w; illegal_model_b.eff_code = -illegal_model_b.eff_code

    errors_a_il = np.zeros(4)
    errors_b_il = np.zeros(4)
    errors_joint_il = np.zeros(4)
    totals_il = np.zeros(4)
    for bits_pm in blocks:
        _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes_a)
        hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
        bits_tx = test.pm1_to_bin(bits_pm)
        for li in range(4):
            Y_obs_a = test.observe_block_from_measured_matrix(hue_seq, illegal_model_a.Y, probe_to_row_a)
            dec_a = test.decode_local_block(Y_obs_a, illegal_model_a.eff_w, illegal_model_a.eff_code)
            Y_obs_b = test.observe_block_from_measured_matrix(hue_seq, illegal_model_b.Y, probe_to_row_b)
            dec_b = test.decode_local_block(Y_obs_b, illegal_model_b.eff_w, illegal_model_b.eff_code)
            totals_il[li] += 1
            if dec_a.bit_hat_bin != bits_tx[li]: errors_a_il[li] += 1
            if dec_b.bit_hat_bin != bits_tx[li]: errors_b_il[li] += 1
            if dec_a.bit_hat_bin != bits_tx[li] or dec_b.bit_hat_bin != bits_tx[li] or dec_a.bit_hat_bin != dec_b.bit_hat_bin:
                errors_joint_il[li] += 1
    raw_joint_il = errors_joint_il / totals_il
    secure_joint_il = np.minimum(raw_joint_il, 1 - raw_joint_il)
    mean_illegal = np.mean(secure_joint_il)
    mean_illegal_indiv = np.mean(np.minimum(errors_a_il/totals_il, 1-errors_a_il/totals_il))

    print(f"{n_common:>6} | {mean_legal:>18.4f} | {mean_illegal_indiv:>20.4f} | {mean_illegal:>16.4f}")

print("\nDone.")
