#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人工噪声注入实验：单主成分 + 正交人工噪声

核心思想：
- 发送端在信号中注入人工噪声 n，满足 C^T n = 0（与所有合法位置指纹正交）
- 合法位置解码时噪声被抵消，非法位置受噪声干扰
- 发送端能力假设：可同时调光多个LED角度（发送探针加权组合）
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

# ============================================================================
# 强制单主成分模式
# ============================================================================
test.N_COMPONENTS = 1
test.ALPHA = np.array([1.0])
test.BETA = np.array([1.0])
test.USE_AMPLITUDE_AWARE = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
positions = (1, 3, 12, 22)
probes = np.array([5, 40, 80, 155, 185, 265, 335, 365], dtype=float)

print("=" * 70)
print("人工噪声注入实验 (Artificial Noise Injection)")
print("=" * 70)
print(f"合法位置: {positions}")
print(f"探针集合: {list(probes)}")
print(f"主成分数: {test.N_COMPONENTS}")

csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition='white')
matrices = test.load_selected_rows(csv_files, probes)
models = [test.extract_fingerprint(probes, mat, force_positive_first=True) for mat in matrices]
models = test.align_model_directions(models)

codes = [m.code for m in models]  # 单主成分下 eff_code = code
probe_to_row = test.build_probe_to_row(probes)

# ============================================================================
# 1. 基线：无噪声时的 Legal / Illegal BER
# ============================================================================
hue_mapping = test.build_hue_mapping(
    models, probes,
    mapping_eval_bits=500, top_k_per_combination=3,
    use_amplitude_aware=False,
)

blocks = test.generate_all_bit_blocks(4)

def evaluate_legal(models, codes, hue_mapping, probe_to_row, observe_fn):
    errors = np.zeros(len(models))
    totals = np.zeros(len(models))
    for bits_pm in blocks:
        _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes)
        hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
        bits_tx = test.pm1_to_bin(bits_pm)
        for li in range(len(models)):
            Y_obs = observe_fn(hue_seq, models[li].Y, probe_to_row)
            dec = test.decode_local_block(Y_obs, models[li].w, models[li].code)
            totals[li] += 1
            if dec.bit_hat_bin != bits_tx[li]:
                errors[li] += 1
    raw = errors / np.maximum(totals, 1)
    secure = np.minimum(raw, 1 - raw)
    return raw, secure

def baseline_observe(hue_seq, Y, probe_to_row):
    return test.observe_block_from_measured_matrix(hue_seq, Y, probe_to_row)

raw_legal_base, sec_legal_base = evaluate_legal(models, codes, hue_mapping, probe_to_row, baseline_observe)
print(f"\n[基线] Legal raw BER: {raw_legal_base}")
print(f"[基线] Legal secure BER: {sec_legal_base}, mean={np.mean(sec_legal_base):.4f}")

# ============================================================================
# 2. 构造人工噪声向量 n
# ============================================================================
# C: P x 4 矩阵，每列是一个合法位置的 code
C = np.column_stack([m.code.astype(float) for m in models])
print(f"\nCode 矩阵 C 形状: {C.shape}")

# 检查 code 平衡性
for i in range(4):
    bal = np.sum(C[:, i])
    print(f"  位置 {positions[i]}: code sum = {bal:+.0f} (|+1|-1| = {np.sum(C[:,i]==1)}|{np.sum(C[:,i]==-1)})")

# SVD 求零空间: C^T n = 0
U, s, Vt = np.linalg.svd(C.T)
rank = np.sum(s > 1e-10)
null_dim = C.shape[0] - rank
print(f"C 秩: {rank}, 零空间维度: {null_dim}")

if null_dim <= 0:
    print("ERROR: 无非平凡零空间，无法注入正交噪声！")
    sys.exit(1)

null_basis = Vt[rank:].T  # P x null_dim

# 优化噪声向量：在零空间中搜索使合法残余噪声小、非法干扰大的方向
# 先预计算每个位置的 z 向量（原始投影）
z_list = [m.z for m in models]

# 非法位置模型（用于评估）
all_positions = list(range(1, 29))
illegal_positions = [p for p in all_positions if p not in positions]

def build_illegal_model(pos):
    csv_file = os.path.join(r'E:\LuminaLink\Position_fingerprint_experiment\data\15pro\white', str(pos) + '.csv')
    mat_full = test.load_csv_matrix(csv_file)
    row_indices = [int((p/5)-1) for p in probes]
    mat = mat_full[row_indices].astype(float, copy=False)
    m = test.extract_fingerprint(probes, mat, force_positive_first=True)
    # 对齐到 models[0]
    corr = test.calculate_correlation(models[0].code, m.code)
    if corr < 0:
        m.w = -m.w; m.z = -m.z; m.code = -m.code
    return m

illegal_models = {pos: build_illegal_model(pos) for pos in illegal_positions}

# 预计算所有非法位置的 z
z_illegal = {pos: m.z for pos, m in illegal_models.items()}

# 搜索最优 n
best_n = None
best_score = -np.inf
print(f"\n在零空间中搜索最优噪声向量 (采样 10000 次)...")

rng = np.random.RandomState(42)
for trial in range(10000):
    coeffs = rng.randn(null_dim)
    n_vec = null_basis @ coeffs
    n_vec = n_vec / np.linalg.norm(n_vec)
    
    # 合法位置残余噪声指标: |n^T z_i|
    legal_residual = max(abs(np.dot(n_vec, z_list[i])) for i in range(4))
    
    # 非法位置干扰指标: 对多个非法位置采样，算平均 |n^T z_illegal|
    sample_illegal = rng.choice(illegal_positions, size=10, replace=False)
    illegal_interference = np.mean([abs(np.dot(n_vec, z_illegal[p])) for p in sample_illegal])
    
    # 评分：非法干扰大 / (合法残余 + epsilon)
    score = illegal_interference / (legal_residual + 1e-6)
    if score > best_score:
        best_score = score
        best_n = n_vec.copy()

n = best_n
print(f"最优 n found, score={best_score:.4f}")
print(f"n = {np.round(n, 4)}")
print(f"C^T n (应≈0): {C.T @ n}")
print(f"合法位置 |n^T z|: {[abs(np.dot(n, z_list[i])) for i in range(4)]}")

# ============================================================================
# 3. 带噪声的观测函数
# ============================================================================
# 对每个位置，预计算 Y_noise = Σ_p n_p * Y_p (D维光谱)
Y_noise_per_pos = {}
for li, m in enumerate(models):
    Y_noise = np.zeros(m.Y.shape[1])
    for p_idx, p_val in enumerate(probes):
        row_idx = probe_to_row[int(p_val)]
        Y_noise += n[p_idx] * m.Y[row_idx]
    Y_noise_per_pos[li] = Y_noise

Y_noise_illegal = {}
for pos, m in illegal_models.items():
    Y_noise = np.zeros(m.Y.shape[1])
    for p_idx, p_val in enumerate(probes):
        row_idx = probe_to_row[int(p_val)]
        Y_noise += n[p_idx] * m.Y[row_idx]
    Y_noise_illegal[pos] = Y_noise

def observe_with_noise(hue_seq, Y, probe_to_row, Y_noise, alpha):
    rows = []
    for hue in hue_seq:
        hue = int(hue)
        row_idx = probe_to_row[hue]
        rows.append(Y[row_idx] + alpha * Y_noise)
    return np.asarray(rows, dtype=float)

# ============================================================================
# 4. 扫描 alpha，评估 Legal / Illegal BER
# ============================================================================
print(f"\n{'Alpha':>8} | {'Legal mean secure':>18} | {'Min illegal secure':>19} | {'Avg illegal secure':>19}")
print("-" * 75)

results = []
for alpha in [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    # Legal
    err_legal = np.zeros(4)
    tot_legal = np.zeros(4)
    for bits_pm in blocks:
        _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes)
        hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
        bits_tx = test.pm1_to_bin(bits_pm)
        for li in range(4):
            Y_obs = observe_with_noise(hue_seq, models[li].Y, probe_to_row, Y_noise_per_pos[li], alpha)
            dec = test.decode_local_block(Y_obs, models[li].w, models[li].code)
            tot_legal[li] += 1
            if dec.bit_hat_bin != bits_tx[li]:
                err_legal[li] += 1
    raw_legal = err_legal / np.maximum(tot_legal, 1)
    sec_legal = np.mean(np.minimum(raw_legal, 1 - raw_legal))
    
    # Illegal (all 24 illegal positions)
    sec_illegal_list = []
    for pos in illegal_positions:
        err_il = np.zeros(4)
        tot_il = np.zeros(4)
        m_il = illegal_models[pos]
        for bits_pm in blocks:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            bits_tx = test.pm1_to_bin(bits_pm)
            for li in range(4):
                Y_obs = observe_with_noise(hue_seq, m_il.Y, probe_to_row, Y_noise_illegal[pos], alpha)
                dec = test.decode_local_block(Y_obs, m_il.w, m_il.code)
                tot_il[li] += 1
                if dec.bit_hat_bin != bits_tx[li]:
                    err_il[li] += 1
        raw_il = err_il / np.maximum(tot_il, 1)
        sec_il = np.mean(np.minimum(raw_il, 1 - raw_il))
        sec_illegal_list.append(sec_il)
    
    min_illegal = np.min(sec_illegal_list)
    avg_illegal = np.mean(sec_illegal_list)
    
    print(f"{alpha:>8.2f} | {sec_legal:>18.4f} | {min_illegal:>19.4f} | {avg_illegal:>19.4f}")
    results.append((alpha, sec_legal, min_illegal, avg_illegal))

# ============================================================================
# 5. 找最优 alpha 并打印详细结果
# ============================================================================
# 目标: legal <= 0.005 且 min_illegal >= 0.2
feasible = [(a, l, mi, ma) for a, l, mi, ma in results if l <= 0.005 and mi >= 0.2]
if feasible:
    print(f"\n[PASS] 找到可行解！最优 alpha = {feasible[0][0]}")
else:
    print(f"\n[FAIL] 未找到满足 legal <= 0.005 且 min_illegal >= 0.2 的 alpha")
    # 找最接近的
    best = min(results, key=lambda x: max(x[1] - 0.005, 0) + max(0.2 - x[2], 0))
    print(f"   最接近目标: alpha={best[0]}, legal={best[1]:.4f}, min_illegal={best[2]:.4f}")

# 对每个 alpha 打印 worst illegal positions
print(f"\n{'Alpha':>8} | {'Worst 3 illegal positions (secure BER)':>50}")
print("-" * 65)
for alpha, _, _, _ in results:
    pos_scores = []
    for pos in illegal_positions:
        err_il = np.zeros(4)
        tot_il = np.zeros(4)
        m_il = illegal_models[pos]
        for bits_pm in blocks:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            bits_tx = test.pm1_to_bin(bits_pm)
            for li in range(4):
                Y_obs = observe_with_noise(hue_seq, m_il.Y, probe_to_row, Y_noise_illegal[pos], alpha)
                dec = test.decode_local_block(Y_obs, m_il.w, m_il.code)
                tot_il[li] += 1
                if dec.bit_hat_bin != bits_tx[li]:
                    err_il[li] += 1
        raw_il = err_il / np.maximum(tot_il, 1)
        sec_il = np.mean(np.minimum(raw_il, 1 - raw_il))
        pos_scores.append((sec_il, pos))
    pos_scores.sort()
    worst = pos_scores[:3]
    print(f"{alpha:>8.2f} | {', '.join([f'Pos{p}:{s:.3f}' for s,p in worst]):>50}")

print("\nDone.")
