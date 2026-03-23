from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd  # 新增：用于读取CSV文件
import random  # 新增：用于固定随机种子

# 固定随机种子，保证可复现
random.seed(0)
np.random.seed(0)

Array = np.ndarray


@dataclass
class FingerprintModel:
    probes: Array  # shape: (P,) 探针
    Y: Array  # shape: (P, D)实测矩阵
    trend: Array  # shape: (P, D)拟合趋势
    residual: Array  # shape: (P, D)Y-trend
    W: Array  # shape: (D, k)指纹方向，多个特征向量
    Z: Array  # shape: (P, k)投影值
    code: Array  # shape: (P,)地址码
    k: int = 2  # 使用的特征向量数量
    singular_values: Array = None  # 前k个奇异值


@dataclass
class DecodeResult:
    Y_obs: Array  # shape: (L, D)观测矩阵
    mean_vec: Array  # shape: (D,)均值向量
    Y_centered: Array  # shape: (L, D)去中心化矩阵  Y_obs-mean
    u: Array  # shape: (L,)投影值
    gamma: float  # 相关检测值
    bit_hat_pm: int  # +1 / -1
    bit_hat_bin: int  # 1 / 0
    confidence: float = 0.0  # 置信度（归一化相关系数）
    is_discarded: bool = False  # 是否被丢弃


# =========================
# Core math utilities
# =========================

def fit_linear_trend(x: Array, Y: Array) -> Tuple[Array, Array]:
    """
    对 Y 的每一列做 y ~ a x + b 拟合。

    Parameters
    ----------
    x : shape (P,)
    Y : shape (P, D)

    Returns
    -------
    coeffs : shape (D, 2), each row is [a, b]
    trend  : shape (P, D)
    """
    x = np.asarray(x, dtype=float)
    Y = np.asarray(Y, dtype=float)
    P, D = Y.shape
    A = np.column_stack([x, np.ones_like(x)])
    coeffs = np.zeros((D, 2), dtype=float)
    trend = np.zeros_like(Y, dtype=float)
    for j in range(D):
        ab, *_ = np.linalg.lstsq(A, Y[:, j], rcond=None)
        coeffs[j] = ab
        trend[:, j] = A @ ab
    return coeffs, trend


def extract_fingerprint(x: Array, Y: Array, force_positive_first: bool = True, k: int = 2) -> FingerprintModel:
    """
    从实测响应矩阵 Y 中提取位置指纹模型。
    
    Parameters
    ----------
    x : shape (P,)
    Y : shape (P, D)
    force_positive_first : 是否强制第一个投影值为正
    k : 使用的特征向量数量
    """
    # 数据预处理：去噪
    Y_denoised = np.zeros_like(Y)
    for ch in range(Y.shape[1]):
        # 使用移动平均滤波去噪
        window_size = min(3, len(Y))
        if window_size > 1:
            Y_denoised[:, ch] = np.convolve(Y[:, ch], np.ones(window_size)/window_size, mode='same')
        else:
            Y_denoised[:, ch] = Y[:, ch]

    coeffs, trend = fit_linear_trend(x, Y_denoised)
    residual = Y_denoised - trend
    
    # SVD: residual = U S V^T
    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
    # 提取前k个特征向量
    W = Vt[:k].T  # shape: (D, k)
    Z = residual @ W  # shape: (P, k)

    # 使用第一个特征向量的投影值生成地址码
    z = Z[:, 0]
    if force_positive_first and z[0] < 0:
        # 翻转所有特征向量的方向
        W = -W
        Z = -Z
        z = -z

    # 使用中位数作为阈值，提高稳健性
    median_z = np.median(z)
    code = np.where(z >= median_z, 1, -1)

    return FingerprintModel(
        probes=np.asarray(x, dtype=float),
        Y=np.asarray(Y, dtype=float),
        trend=trend,
        residual=residual,
        W=W,
        Z=Z,
        code=code,
        k=k,
        singular_values=S[:k],  # 存储前k个奇异值
    )


def pm1_to_bin(bits_pm: Array) -> Array:
    return np.where(np.asarray(bits_pm) > 0, 1, 0)


def bin_to_pm1(bits_bin: List[int]) -> Array:
    bits_bin = np.asarray(bits_bin, dtype=int)
    return np.where(bits_bin > 0, 1, -1)


def calculate_correlation(c1: Array, c2: Array) -> float:
    """
    计算两个地址码之间的互相关指数
    
    Parameters
    ----------
    c1 : 第一个地址码数组
    c2 : 第二个地址码数组
    
    Returns
    -------
    float : 互相关指数
    """
    # 确保两个数组长度相同
    assert len(c1) == len(c2), "地址码长度必须相同"
    
    # 使用点积除以长度计算相关性
    correlation = np.dot(c1, c2) / len(c1)
    return correlation


def find_optimal_probe_count(csv_file1: str, csv_file2: str, csv_file3: str, min_probes: int = 5, max_probes: int = 15) -> Tuple[int, np.ndarray]:
    """
    遍历探针数量，找到地址码最大互相关指数最小的探针数量
    
    Parameters
    ----------
    csv_file1 : 第一个CSV文件路径
    csv_file2 : 第二个CSV文件路径
    csv_file3 : 第三个CSV文件路径
    min_probes : 最小探针数量
    max_probes : 最大探针数量
    
    Returns
    -------
    Tuple[int, np.ndarray] : (最优探针数量, 最优探针组合)
    """
    best_probe_count = min_probes
    min_max_correlation = float('inf')
    best_probes = None
    best_codes = None
    
    for num_probes in range(min_probes, max_probes + 1):
        # 读取数据
        df1 = pd.read_csv(csv_file1)
        df2 = pd.read_csv(csv_file2)
        df3 = pd.read_csv(csv_file3)
        Y1 = df1.values.astype(float)
        Y2 = df2.values.astype(float)
        Y3 = df3.values.astype(float)
        
        # 生成所有可能的探针位置
        all_probes = 5 + np.arange(len(df1)) * 5
        
        # 临时变量，用于当前探针数量的最优解
        current_best_probes = None
        current_min_max_correlation = float('inf')
        current_best_codes = None
        
        # 直接随机采样，避免生成所有组合
        import random
        max_candidates = 1000
        for i in range(max_candidates):
            # 随机选择num_probes个探针（非等间距）
            probes = np.array(random.sample(all_probes.tolist(), num_probes), dtype=float)
            
            # 计算探针之间的最小间隔
            sorted_probes = np.sort(probes)
            min_interval = np.min(np.diff(sorted_probes))
            
            # 确保探针之间有足够的间隔
            if min_interval < 30:
                continue  # 跳过间隔太小的探针组合
            
            # 计算行索引
            target_row_indices = []
            for probe in probes:
                row_index = int((probe / 5) - 1)
                target_row_indices.append(row_index)
            
            # 提取数据
            Y1_subset = Y1[target_row_indices]
            Y2_subset = Y2[target_row_indices]
            Y3_subset = Y3[target_row_indices]
            
            # 提取指纹模型
            m1 = extract_fingerprint(probes, Y1_subset)
            m2 = extract_fingerprint(probes, Y2_subset)
            m3 = extract_fingerprint(probes, Y3_subset)
            
            # 计算互相关指数
            rho12 = abs(calculate_correlation(m1.code, m2.code))
            rho13 = abs(calculate_correlation(m1.code, m3.code))
            rho23 = abs(calculate_correlation(m2.code, m3.code))
            max_correlation = max(rho12, rho13, rho23)
            
            # 更新当前探针数量的最优值
            if max_correlation < current_min_max_correlation:
                current_min_max_correlation = max_correlation
                current_best_probes = probes
                current_best_codes = (m1.code, m2.code, m3.code)
        
        # 更新全局最优值
        if current_min_max_correlation < min_max_correlation:
            min_max_correlation = current_min_max_correlation
            best_probe_count = num_probes
            best_probes = current_best_probes
            best_codes = current_best_codes
    
    # 只打印最终的最优结果
    print(f"最优探针数量: {best_probe_count}, 最小ρmax: {min_max_correlation:.4f}")
    if best_probes is not None and best_codes is not None:
        print(f"暴力搜索最优探针: {best_probes}")
        print(f"地址码1: {best_codes[0]}")
        print(f"地址码2: {best_codes[1]}")
        print(f"地址码3: {best_codes[2]}")
    
    return best_probe_count, best_probes


# =========================
# Encoding / observation
# =========================

def build_symbol_sequence(bits_pm: Array, codes: List[Array]) -> Tuple[Array, List[List[int]]]:
    """s = sum_i b_i c_i，同时返回符号组合序列"""
    bits_pm = np.asarray(bits_pm, dtype=int)
    out = np.zeros_like(codes[0], dtype=int)
    symbol_combinations = []
    
    # 对每个探针位置计算符号组合
    for i in range(len(codes[0])):
        combination = []
        for b, c in zip(bits_pm, codes):
            contribution = int(b) * int(c[i])
            combination.append(contribution)
            out[i] += contribution
        symbol_combinations.append(combination)
    
    return out, symbol_combinations


def map_symbol_to_hue(symbol_seq: Array, symbol_combinations: List[List[int]], hue_mapping: Dict[tuple, int]) -> Array:
    """根据符号组合映射到hue值"""
    hue_seq = []
    for i, combination in enumerate(symbol_combinations):
        # 将组合转换为元组作为字典键
        key = tuple(combination)
        if key in hue_mapping:
            hue_seq.append(hue_mapping[key])
        else:
            # 如果没有找到对应的组合，使用默认值
            hue_seq.append(0)
    return np.array(hue_seq, dtype=int)


def build_probe_to_row(probes: Array) -> Dict[int, int]:
    return {int(v): i for i, v in enumerate(np.asarray(probes).tolist())}


def observe_block_from_measured_matrix(hue_seq: Array, Y: Array, probe_to_row: Dict[int, int]) -> Array:
    """
    根据 hue 序列，从实测矩阵按行查表构造局部观测块。

    如果 hue 不在 probe_to_row 中，使用最接近的可用探针值。
    使用邻域平均提高观测质量。
    返回 shape (L, D) 的观测矩阵。
    """
    rows = []
    available_hues = sorted(probe_to_row.keys())
    
    for hue in hue_seq:
        hue = int(hue)
        if hue not in probe_to_row:
            # 找到最接近的可用探针值
            closest_hue = min(available_hues, key=lambda x: abs(x - hue))
            idx = probe_to_row[closest_hue]
        else:
            idx = probe_to_row[hue]
        
        # 使用邻域平均（±1行）
        neighbors = []
        for j in [idx-1, idx, idx+1]:
            if 0 <= j < len(Y):
                neighbors.append(Y[j])
        
        if neighbors:
            # 计算邻域平均值
            avg_row = np.mean(neighbors, axis=0)
            rows.append(avg_row)
        else:
            # fallback
            rows.append(Y[idx])
    
    return np.asarray(rows, dtype=float)


# =========================
# Local decoding
# =========================

def decode_local_block(Y_obs: Array, W: Array, code: Array, singular_values: Array = None, threshold: float = 0.0) -> DecodeResult:
    """
    软判决解码流程（使用多个特征向量）：
    1) 对当前块按列去中心化
    2) U = Y_centered @ W  # 投影到多个特征方向
    3) gamma = sum(w_i * (code @ U[:, i]))  # 加权计算多维度的相关检测值
    4) 计算置信度 confidence = gamma / (||code|| * ||U||)
    5) 计算信道噪声水平，动态调整阈值
    6) 根据自适应阈值判决
    """
    Y_obs = np.asarray(Y_obs, dtype=float)
    W = np.asarray(W, dtype=float)
    code = np.asarray(code, dtype=float)

    mean_vec = Y_obs.mean(axis=0)
    Y_centered = Y_obs - mean_vec
    U = Y_centered @ W  # shape: (L, k)
    
    # 计算多维度的相关检测值（加权）
    if singular_values is not None and len(singular_values) == U.shape[1]:
        # 使用奇异值作为权重
        weights = singular_values / np.sum(singular_values)
        gamma = 0
        for i in range(U.shape[1]):
            gamma += weights[i] * np.dot(code, U[:, i])
    else:
        #  fallback: 等权重
        gamma = np.mean([np.dot(code, U[:, i]) for i in range(U.shape[1])])
    
    # 计算置信度（归一化相关性）
    norm_code = np.linalg.norm(code)
    norm_U = np.linalg.norm(U)
    if norm_code > 0 and norm_U > 0:
        confidence = gamma / (norm_code * norm_U)
    else:
        confidence = 0.0
    
    # 计算信道噪声水平，动态调整阈值
    noise_level = np.var(Y_obs)
    
    # 优化的阈值调整策略
    base_threshold = max(0.0, threshold)
    
    # 更精细的噪声水平划分
    if noise_level < 50:
        # 低噪声环境，使用较低的阈值
        adaptive_threshold = base_threshold * 0.8
    elif noise_level < 150:
        # 中等噪声环境，使用中等阈值
        adaptive_threshold = base_threshold * (1 + noise_level / 100)
    else:
        # 高噪声环境，使用较高的阈值
        adaptive_threshold = base_threshold * (1 + min(noise_level / 80, 3.0))
    
    # 限制阈值范围，根据噪声水平动态调整
    min_threshold = 0.05 if noise_level < 50 else 0.1
    max_threshold = 0.6 if noise_level < 50 else 0.8
    adaptive_threshold = max(min_threshold, min(adaptive_threshold, max_threshold))
    
    # 优化的判决策略
    is_discarded = False
    
    # 高置信度区域
    if confidence > adaptive_threshold + 0.1:
        # 非常确定为1
        bit_hat_pm = 1
        bit_hat_bin = 1
    elif confidence < -adaptive_threshold - 0.1:
        # 非常确定为0
        bit_hat_pm = -1
        bit_hat_bin = 0
    # 中等置信度区域
    elif confidence > adaptive_threshold:
        # 较确定为1
        bit_hat_pm = 1
        bit_hat_bin = 1
    elif confidence < -adaptive_threshold:
        # 较确定为0
        bit_hat_pm = -1
        bit_hat_bin = 0
    # 低置信度区域
    else:
        # 基于gamma值重新判决，加入噪声水平和置信度的综合考量
        if noise_level < 50:
            # 低噪声环境，主要依赖gamma
            bit_hat_pm = 1 if gamma > 0 else -1
            bit_hat_bin = 1 if bit_hat_pm > 0 else 0
        elif noise_level < 150:
            # 中等噪声环境，平衡gamma和置信度
            if abs(gamma) > np.std(code) * np.std(U) * 0.3:
                bit_hat_pm = 1 if gamma > 0 else -1
                bit_hat_bin = 1 if bit_hat_pm > 0 else 0
            else:
                # 当gamma较小时，依赖置信度
                bit_hat_pm = 1 if confidence > 0 else -1
                bit_hat_bin = 1 if bit_hat_pm > 0 else 0
        else:
            # 高噪声环境，需要更严格的gamma阈值
            threshold_gamma = np.std(code) * np.std(U) * 0.6
            if gamma > threshold_gamma:
                bit_hat_pm = 1
                bit_hat_bin = 1
            elif gamma < -threshold_gamma:
                bit_hat_pm = -1
                bit_hat_bin = 0
            else:
                # 极不确定的情况，基于置信度符号判决
                bit_hat_pm = 1 if confidence > 0 else -1
                bit_hat_bin = 1 if bit_hat_pm > 0 else 0
    
    return DecodeResult(
        Y_obs=Y_obs,
        mean_vec=mean_vec,
        Y_centered=Y_centered,
        u=U[:, 0],  # 保持向后兼容，返回第一个特征向量的投影
        gamma=gamma,
        bit_hat_pm=bit_hat_pm,
        bit_hat_bin=bit_hat_bin,
        confidence=confidence,
        is_discarded=is_discarded,
    )


# =========================
# End-to-end experiment
# =========================

def simulate_blocks(
        models: List[FingerprintModel],
        bit_blocks_pm: List[Array],
        hue_mapping: Dict[int, int],
        threshold: float = 0.0,
) -> List[dict]:
    """
    针对多个发送块做端到端仿真。

    Parameters
    ----------
    models : 每个位置一个 FingerprintModel
    bit_blocks_pm : list of shape-(N_pos,) arrays, each entry in {+1, -1}
    hue_mapping : e.g. {-2:150, 0:100, 2:300}

    Returns
    -------
    一个列表，每个元素对应一个 block，包含：
      - bits_pm
      - bits_bin
      - symbol_seq
      - hue_seq
      - per_pos decode result
    """
    codes = [m.code for m in models]
    probe_to_row = build_probe_to_row(models[0].probes)
    results = []

    for bits_pm in bit_blocks_pm:
        bits_pm = np.asarray(bits_pm, dtype=int)
        symbol_seq, symbol_combinations = build_symbol_sequence(bits_pm, codes)
        hue_seq = map_symbol_to_hue(symbol_seq, symbol_combinations, hue_mapping)

        block_info = {
            "bits_pm": bits_pm,
            "bits_bin": pm1_to_bin(bits_pm),
            "symbol_seq": symbol_seq,
            "hue_seq": hue_seq,
            "per_position": [],
        }

        for pos_idx, model in enumerate(models):
            Y_obs = observe_block_from_measured_matrix(hue_seq, model.Y, probe_to_row)
            dec = decode_local_block(Y_obs, model.W, model.code, model.singular_values, threshold)
            block_info["per_position"].append(dec)

        results.append(block_info)
    return results


# =========================
# Pretty print helpers
# =========================

def arr_str(a: Array, precision: int = 2) -> str:
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.integer):
        return np.array2string(a, separator=', ')
    return np.array2string(a, precision=precision, suppress_small=False, separator=', ')


def calculate_ber_with_threshold(results: List[dict], threshold: float) -> float:
    """
    计算给定阈值下的误码率，只统计未被丢弃的比特
    
    Parameters
    ----------
    results : 仿真结果
    threshold : 判决阈值
    
    Returns
    -------
    float : 误码率
    """
    total_bits = 0
    error_bits = 0
    
    for res in results:
        bits_tx = res["bits_bin"]
        for p, dec in enumerate(res["per_position"]):
            if not dec.is_discarded:
                total_bits += 1
                if dec.bit_hat_bin != bits_tx[p]:
                    error_bits += 1
    
    if total_bits == 0:
        return 0.0
    
    return error_bits / total_bits


def find_optimal_threshold(models: List[FingerprintModel], bit_blocks_pm: List[Array], hue_mapping: Dict[tuple, int]) -> float:
    """
    网格搜索最优阈值T
    
    Parameters
    ----------
    models : 每个位置一个 FingerprintModel
    bit_blocks_pm : list of shape-(N_pos,) arrays, each entry in {+1, -1}
    hue_mapping : 颜色映射字典
    
    Returns
    -------
    float : 最优阈值T
    """
    # 优化的阈值搜索范围和步长
    thresholds = np.linspace(-0.2, 0.8, 51)  # -0.2, -0.18, ..., 0.8
    best_threshold = 0.0
    min_ber = float('inf')
    
    for threshold in thresholds:
        # 重新计算解码结果
        new_results = []
        codes = [m.code for m in models]
        probe_to_row = build_probe_to_row(models[0].probes)
        
        for bits_pm in bit_blocks_pm:
            bits_pm = np.asarray(bits_pm, dtype=int)
            symbol_seq, symbol_combinations = build_symbol_sequence(bits_pm, codes)
            hue_seq = map_symbol_to_hue(symbol_seq, symbol_combinations, hue_mapping)
            
            block_info = {
                "bits_pm": bits_pm,
                "bits_bin": pm1_to_bin(bits_pm),
                "symbol_seq": symbol_seq,
                "hue_seq": hue_seq,
                "per_position": [],
            }
            
            for pos_idx, model in enumerate(models):
                Y_obs = observe_block_from_measured_matrix(hue_seq, model.Y, probe_to_row)
                dec = decode_local_block(Y_obs, model.W, model.code, model.singular_values, threshold)
                block_info["per_position"].append(dec)
            
            new_results.append(block_info)
        
        # 计算误码率
        ber = calculate_ber_with_threshold(new_results, threshold)
        
        if ber < min_ber:
            min_ber = ber
            best_threshold = threshold
    
    print(f"Grid search optimal threshold: {best_threshold:.4f}, Minimum BER: {min_ber:.6f}")
    return best_threshold



# =========================
# 从CSV文件读取数据（修改部分）
# =========================

def generate_probes(num_probes: int, max_row_index: int) -> np.ndarray:
    """
    根据探针数量生成固定间隔的探针数组

    Parameters
    ----------
    num_probes : 探针数量
    max_row_index : 最大行索引（CSV文件行数-1）

    Returns
    -------
    probes : 固定间隔的探针数组，从5开始，间隔为360/(num_probes-1)的最近5的倍数
    """
    if num_probes < 2:
        raise ValueError("探针数量至少为2")
    
    # 计算理论间隔
    theoretical_interval = 360 / (num_probes - 1)
    
    # 找到最近的5的倍数
    interval = round(theoretical_interval / 5) * 5
    
    # 生成探针数组，从5开始，间隔为计算得到的interval
    probes = []
    for i in range(num_probes):
        probe = 5 + i * interval
        # 确保探针值对应的行索引不超过max_row_index
        max_probe = (max_row_index + 1) * 5
        if probe > max_probe:
            probe = max_probe
        probes.append(probe)
    
    return np.array(probes, dtype=float)

def brute_force_probe_selection(csv_file1: str, csv_file2: str, csv_file3: str, num_probes: int, max_candidates: int = 1000) -> np.ndarray:
    """
    暴力搜索最优探针组合（使用随机采样避免内存不足）

    Parameters
    ----------
    csv_file1 : 第一个CSV文件路径
    csv_file2 : 第二个CSV文件路径
    csv_file3 : 第三个CSV文件路径
    num_probes : 探针数量
    max_candidates : 最大候选组合数

    Returns
    -------
    np.ndarray : 最优探针组合
    """
    # 读取数据
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    df3 = pd.read_csv(csv_file3)
    Y1 = df1.values.astype(float)
    Y2 = df2.values.astype(float)
    Y3 = df3.values.astype(float)
    
    # 生成所有可能的探针位置
    all_probes = 5 + np.arange(len(df1)) * 5
    
    best_probes = None
    min_max_correlation = float('inf')
    
    # 直接随机采样，避免生成所有组合
    import random
    for i in range(max_candidates):
        # 随机选择num_probes个探针（非等间距）
        probes = np.array(random.sample(all_probes.tolist(), num_probes), dtype=float)
        
        # 计算探针之间的最小间隔
        sorted_probes = np.sort(probes)
        min_interval = np.min(np.diff(sorted_probes))
        
        # 确保探针之间有足够的间隔
        if min_interval < 30:
            continue  # 跳过间隔太小的探针组合
        
        # 计算行索引
        target_row_indices = []
        for probe in probes:
            row_index = int((probe / 5) - 1)
            target_row_indices.append(row_index)
        
        # 提取数据
        Y1_subset = Y1[target_row_indices]
        Y2_subset = Y2[target_row_indices]
        Y3_subset = Y3[target_row_indices]
        
        # 提取指纹模型
        m1 = extract_fingerprint(probes, Y1_subset)
        m2 = extract_fingerprint(probes, Y2_subset)
        m3 = extract_fingerprint(probes, Y3_subset)
        
        # 计算互相关指数
        rho12 = abs(calculate_correlation(m1.code, m2.code))
        rho13 = abs(calculate_correlation(m1.code, m3.code))
        rho23 = abs(calculate_correlation(m2.code, m3.code))
        max_correlation = max(rho12, rho13, rho23)
        
        # 更新最优值
        if max_correlation < min_max_correlation:
            min_max_correlation = max_correlation
            best_probes = probes
    
    # 如果没有找到最优探针，使用随机选择的探针
    if best_probes is None:
        # 尝试不考虑间隔限制，确保找到探针组合
        for i in range(max_candidates):
            probes = np.array(random.sample(all_probes.tolist(), num_probes), dtype=float)
            target_row_indices = [int((probe / 5) - 1) for probe in probes]
            Y1_subset = Y1[target_row_indices]
            Y2_subset = Y2[target_row_indices]
            Y3_subset = Y3[target_row_indices]
            m1 = extract_fingerprint(probes, Y1_subset)
            m2 = extract_fingerprint(probes, Y2_subset)
            m3 = extract_fingerprint(probes, Y3_subset)
            rho12 = abs(calculate_correlation(m1.code, m2.code))
            rho13 = abs(calculate_correlation(m1.code, m3.code))
            rho23 = abs(calculate_correlation(m2.code, m3.code))
            max_correlation = max(rho12, rho13, rho23)
            if max_correlation < min_max_correlation:
                min_max_correlation = max_correlation
                best_probes = probes
        
        # 如果仍然没有找到，使用随机选择的探针
        if best_probes is None:
            best_probes = np.array(random.sample(all_probes.tolist(), num_probes), dtype=float)
    
    # 打印搜索结果
    print(f"暴力搜索最优探针: {best_probes}, 最小ρmax: {min_max_correlation:.4f}")
    
    return best_probes


def load_data_from_csv(file_path1: str, file_path2: str, file_path3: str, num_probes: int, optimal_probes: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从三个CSV文件读取数据，根据探针数量提取对应行作为Y1、Y2和Y3矩阵

    Parameters
    ----------
    file_path1 : 第一个CSV文件路径（对应位置1）
    file_path2 : 第二个CSV文件路径（对应位置2）
    file_path3 : 第三个CSV文件路径（对应位置3）
    num_probes : 探针数量
    optimal_probes : 最优探针组合（如果提供）

    Returns
    -------
    probes : 探针数组
    Y1 : 位置1的实测矩阵
    Y2 : 位置2的实测矩阵
    Y3 : 位置3的实测矩阵
    """
    if optimal_probes is not None:
        print(f"使用预定义的最优探针组合")
        probes = optimal_probes
    else:
        print(f"正在搜索最优探针组合（数量: {num_probes}）...")
        # 暴力搜索最优探针（非等间距）
        probes = brute_force_probe_selection(file_path1, file_path2, file_path3, num_probes)
    
    # 计算要提取的行索引：(probe/5)-1
    target_row_indices = []
    for probe in probes:
        row_index = int((probe / 5) - 1)
        target_row_indices.append(row_index)

    # 读取第一个CSV文件（位置1）
    df1 = pd.read_csv(file_path1)
    # 提取指定行的数据，转换为numpy数组
    Y1 = df1.iloc[target_row_indices].values.astype(float)

    # 读取第二个CSV文件（位置2）
    df2 = pd.read_csv(file_path2)
    # 提取指定行的数据，转换为numpy数组
    Y2 = df2.iloc[target_row_indices].values.astype(float)

    # 读取第三个CSV文件（位置3）
    df3 = pd.read_csv(file_path3)
    # 提取指定行的数据，转换为numpy数组
    Y3 = df3.iloc[target_row_indices].values.astype(float)

    return probes, Y1, Y2, Y3


def get_data_from_csv(csv_file1: str, csv_file2: str, csv_file3: str, num_probes: int, optimal_probes: np.ndarray = None, test_split: float = 0.2) -> Tuple[List[FingerprintModel], List[Array], Dict[int, int], List[FingerprintModel], List[Array]]:
    """
    从CSV文件获取数据并创建FingerprintModel，替代原有的builtin_example

    Parameters
    ----------
    csv_file1 : 第一个CSV文件路径
    csv_file2 : 第二个CSV文件路径
    csv_file3 : 第三个CSV文件路径
    num_probes : 探针数量
    optimal_probes : 最优探针组合（如果提供）
    test_split : 测试集比例

    Returns
    -------
    models : 三个位置的FingerprintModel列表（训练集）
    [Y1, Y2, Y3] : 原始矩阵列表（训练集）
    hue_mapping : 颜色映射字典
    test_models : 三个位置的FingerprintModel列表（测试集）
    [test_Y1, test_Y2, test_Y3] : 原始矩阵列表（测试集）
    """
    # 直接从CSV文件加载数据
    probes, Y1, Y2, Y3 = load_data_from_csv(csv_file1, csv_file2, csv_file3, num_probes, optimal_probes)
    
    # 数据分离：训练集和测试集
    num_samples = len(Y1)
    test_size = int(num_samples * test_split)
    train_size = num_samples - test_size
    
    # 随机选择测试样本
    test_indices = np.random.choice(num_samples, test_size, replace=False)
    train_indices = [i for i in range(num_samples) if i not in test_indices]
    
    # 分割数据
    train_Y1 = Y1[train_indices]
    train_Y2 = Y2[train_indices]
    train_Y3 = Y3[train_indices]
    train_probes = probes[train_indices]
    
    test_Y1 = Y1[test_indices]
    test_Y2 = Y2[test_indices]
    test_Y3 = Y3[test_indices]
    test_probes = probes[test_indices]

    # 提取训练集的指纹模型
    m1 = extract_fingerprint(train_probes, train_Y1)
    m2 = extract_fingerprint(train_probes, train_Y2)
    m3 = extract_fingerprint(train_probes, train_Y3)
    
    # 提取测试集的指纹模型
    test_m1 = extract_fingerprint(test_probes, test_Y1)
    test_m2 = extract_fingerprint(test_probes, test_Y2)
    test_m3 = extract_fingerprint(test_probes, test_Y3)

    # 注意：由于探针数量是用户指定的，不再强制调整地址码方向
    # 保持原始提取的地址码方向

    # 计算每个探针的观测差异
    def calculate_observation_diff(probe_idx, Y):
        """计算探针的观测差异"""
        probe_data = Y[probe_idx]
        diff = 0
        for i in range(len(Y)):
            if i != probe_idx:
                diff += np.linalg.norm(probe_data - Y[i])
        return diff
    
    # 生成hue_mapping，从符号组合数组映射到probe
    hue_mapping = {}
    
    # 所有可能的符号组合
    possible_combinations = [
        (1, 1, 1),    # symbol 3
        (1, 1, -1),   # symbol 1
        (1, -1, 1),   # symbol 1
        (1, -1, -1),  # symbol -1
        (-1, 1, 1),   # symbol 1
        (-1, 1, -1),  # symbol -1
        (-1, -1, 1),  # symbol -1
        (-1, -1, -1)  # symbol -3
    ]
    
    # 对探针进行排序并排除头部和尾部的异常值
    sorted_probes = np.sort(train_probes)
    # 排除前5%和后5%的探针，避免异常值
    num_probes = len(sorted_probes)
    start_idx = int(num_probes * 0.05)
    end_idx = int(num_probes * 0.95)
    valid_probes = sorted_probes[start_idx:end_idx+1]
    
    # 为每个符号组合选择最优探针
    for combination in possible_combinations:
        c1, c2, c3 = combination
        
        # 计算每个探针的得分：观测差异 + 符号匹配度
        scores = []
        for i, probe in enumerate(valid_probes):
            # 直接使用训练集中的索引，而不是基于probe值计算
            row_idx = i
            
            # 确保row_idx在有效范围内
            if row_idx < len(train_Y1) and row_idx < len(m1.Z):
                # 计算观测差异
                obs_diff1 = calculate_observation_diff(row_idx, train_Y1)
                obs_diff2 = calculate_observation_diff(row_idx, train_Y2)
                obs_diff3 = calculate_observation_diff(row_idx, train_Y3)
                total_obs_diff = obs_diff1 + obs_diff2 + obs_diff3
                
                # 计算符号匹配度
                z1_val = m1.Z[row_idx, 0]
                z2_val = m2.Z[row_idx, 0]
                z3_val = m3.Z[row_idx, 0]
                
                sign_match = 0
                if (c1 > 0 and z1_val > 0) or (c1 < 0 and z1_val < 0):
                    sign_match += 1
                if (c2 > 0 and z2_val > 0) or (c2 < 0 and z2_val < 0):
                    sign_match += 1
                if (c3 > 0 and z3_val > 0) or (c3 < 0 and z3_val < 0):
                    sign_match += 1
                
                # 综合得分
                score = total_obs_diff * (1 + sign_match * 0.5)
                scores.append((score, probe))
        
        # 选择得分最高的探针
        if scores:
            scores.sort(reverse=True)
            best_probe = scores[0][1]
            hue_mapping[combination] = int(best_probe)
        else:
            # 如果没有有效得分，使用第一个有效探针
            if valid_probes:
                hue_mapping[combination] = int(valid_probes[0])
            else:
                # 如果没有有效探针，使用第一个探针
                hue_mapping[combination] = int(train_probes[0])

    models = [m1, m2, m3]
    test_models = [test_m1, test_m2, test_m3]
    
    return models, [Y1, Y2, Y3], hue_mapping, test_models, [test_Y1, test_Y2, test_Y3]


def main() -> None:
    csv_file1 = "data\15pro\mid\1.csv"
    csv_file2 = "data\15pro\mid\4.csv"
    csv_file3 = "data\15pro\mid\6.csv"  # 新增第三个位置的CSV文件
    
    # 寻找最优探针数量和探针组合
    best_probe_count, best_probes = find_optimal_probe_count(csv_file1, csv_file2, csv_file3, min_probes=5, max_probes=20)
    
    # 使用最优探针数量进行实验
    num_probes = best_probe_count
    
    
    
    # 从CSV文件获取数据（替代原有的get_builtin_example()）
    models, _, hue_mapping, test_models, _ = get_data_from_csv(csv_file1, csv_file2, csv_file3, num_probes, best_probes)


    # Example in the paper/document:
    # position1 sends 101, position2 sends 011, position3 sends 110
    # blocks: (+1,-1,+1), (-1,-1,-1), (+1,+1,-1)
    bit_blocks_pm = [
        np.array([-1, -1, +1]),  # 三个设备的发送值
        np.array([-1, +1, -1]),
        np.array([-1, +1, +1]),
        np.array([+1, -1, +1]),
        np.array([+1, -1, -1]),
        np.array([-1, +1, +1]),
        np.array([-1, +1, -1]),
        np.array([-1, +1, -1]),
        np.array([-1, +1, -1]),
        np.array([-1, -1, +1]),
    ]
    # 自动学习最优阈值
    optimal_threshold = find_optimal_threshold(models, bit_blocks_pm, hue_mapping)
    
    # 使用最优阈值运行仿真
    results = simulate_blocks(models, bit_blocks_pm, hue_mapping, threshold=optimal_threshold)


    # =============================
    # 按设备整理发送与接收数据（只执行一次）
    # =============================

    num_pos = len(models)

    # 发送端 bit 序列
    tx_bits = [[] for _ in range(num_pos)]

    # 收集 gamma 和 confidence 值
    gamma_all = [[] for _ in range(num_pos)]
    conf_all = [[] for _ in range(num_pos)]

    for res in results:

        # 发送端
        bits_tx = res["bits_bin"]

        # 收集 gamma 和 confidence
        for p, dec in enumerate(res["per_position"]):
            tx_bits[p].append(int(bits_tx[p]))
            gamma_all[p].append(dec.gamma)
            conf_all[p].append(dec.confidence)

    # 方法A：confidence融合
    final_bits = []
    for p in range(num_pos):
        Gamma = np.sum(conf_all[p])   # 软融合
        bit_hat = 1 if Gamma > 0 else 0
        final_bits.append(bit_hat)

    print("发送端：")
    for i in range(num_pos):
        print(f"position{i + 1}: {tx_bits[i]}")

    print("\n接收端（单block判决）：")
    for i in range(num_pos):
        # 计算每个位置的单block判决结果
        single_block_bits = [1 if g > 0 else 0 for g in gamma_all[i]]
        print(f"position{i + 1}: {single_block_bits}")

    print("\n接收端（多block软融合）：")
    for i in range(num_pos):
        print(f"position{i + 1}: {final_bits[i]}")




if __name__ == "__main__":
    main()