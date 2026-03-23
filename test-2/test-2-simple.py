#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位置自生指纹通信：最小可复现实验脚本

覆盖流程：
1. 对实测 Probe 响应做逐列线性去趋势
2. 对残差矩阵做 SVD，提取本地读出方向 w
3. 依据残差投影符号生成位置地址码 c
4. 将多位置 bit 叠加为抽象发送序列 s
5. 用 hue 映射生成发送 hue 序列
6. 基于实测矩阵逐行查表，构造各位置的局部观测矩阵
7. 对每个接收块做按列去中心化，再进行本地相关解码

当前脚本从CSV文件读取数据，两个矩阵分别从两个CSV文件的第2、11、21、31、41、51、61行获取数据
后续可替换CSV文件路径、hue_mapping / bit blocks，扩展到更多位置。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd  # 新增：用于读取CSV文件

Array = np.ndarray


@dataclass
class FingerprintModel:
    probes: Array  # shape: (P,) 探针
    Y: Array  # shape: (P, D)实测矩阵
    trend: Array  # shape: (P, D)拟合趋势
    residual: Array  # shape: (P, D)Y-trend
    w: Array  # shape: (D,)第一指纹方向，特征向量
    w2: Array  # shape: (D,)第二指纹方向，特征向量
    z: Array  # shape: (P,)第一特征向量投影值
    z2: Array  # shape: (P,)第二特征向量投影值
    code: Array  # shape: (P,)地址码


@dataclass
class DecodeResult:
    Y_obs: Array  # shape: (L, D)观测矩阵
    mean_vec: Array  # shape: (D,)均值向量
    Y_centered: Array  # shape: (L, D)去中心化矩阵  Y_obs-mean
    u: Array  # shape: (L,)投影值
    gamma: float  # 相关检测值
    bit_hat_pm: int  # +1 / -1
    bit_hat_bin: int  # 1 / 0


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


def extract_fingerprint(x: Array, Y: Array, force_positive_first: bool = True) -> FingerprintModel:
    """
    从实测响应矩阵 Y 中提取位置指纹模型。
    """

    coeffs, trend = fit_linear_trend(x, Y)
    residual = Y - trend

    

    # SVD: residual = U S V^T
    _, _, Vt = np.linalg.svd(residual, full_matrices=False)
    w = Vt[0].copy()
    z = residual @ w
    
    # 提取第二特征向量
    w2 = Vt[1].copy() if Vt.shape[0] > 1 else np.zeros_like(w)
    z2 = residual @ w2

    if force_positive_first and z[0] < 0:
        w = -w
        z = -z
        w2 = -w2
        z2 = -z2

    # 基于第一特征向量的投影生成地址码
    code = np.where(z >= 0, 1, -1)

    return FingerprintModel(
        probes=np.asarray(x, dtype=float),
        Y=np.asarray(Y, dtype=float),
        trend=trend,
        residual=residual,
        w=w,
        w2=w2,
        z=z,
        z2=z2,
        code=code,
    )


def pm1_to_bin(bits_pm: Array) -> Array:
    return np.where(np.asarray(bits_pm) > 0, 1, 0)


def bin_to_pm1(bits_bin: List[int]) -> Array:
    bits_bin = np.asarray(bits_bin, dtype=int)
    return np.where(bits_bin > 0, 1, -1)


# =========================
# Encoding / observation
# =========================

def build_symbol_sequence(bits_pm: Array, codes: List[Array], codes2: List[Array]) -> Tuple[Array, Array]:
    """s = sum_i b_i c_i，同时使用第一和第二特征向量的地址码"""
    bits_pm = np.asarray(bits_pm, dtype=int)
    out1 = np.zeros_like(codes[0], dtype=int)
    out2 = np.zeros_like(codes2[0], dtype=int)
    
    for b, c, c2 in zip(bits_pm, codes, codes2):
        out1 += int(b) * np.asarray(c, dtype=int)
        out2 += int(b) * np.asarray(c2, dtype=int)
    
    return out1, out2


def map_symbol_to_hue(symbol_seq1: Array, symbol_seq2: Array, hue_mapping: Dict[int, int]) -> Array:
    """根据两个特征向量的符号序列映射到hue值"""
    hue_seq = []
    for s1, s2 in zip(symbol_seq1, symbol_seq2):
        # 融合两个特征向量的符号，第一特征向量权重更高
        combined_symbol = int(0.7 * s1 + 0.3 * s2)
        # 确保符号值为-2, 0, 或 2
        if combined_symbol > 1:
            combined_symbol = 2
        elif combined_symbol < -1:
            combined_symbol = -2
        else:
            combined_symbol = 0
        
        if combined_symbol in hue_mapping:
            hue_seq.append(hue_mapping[combined_symbol])
        else:
            # 如果没有找到对应的符号，使用默认值
            hue_seq.append(0)
    return np.array(hue_seq, dtype=int)


def build_probe_to_row(probes: Array) -> Dict[int, int]:
    return {int(v): i for i, v in enumerate(np.asarray(probes).tolist())}


def observe_block_from_measured_matrix(hue_seq: Array, Y: Array, probe_to_row: Dict[int, int]) -> Array:
    """
    根据 hue 序列，从实测矩阵按行查表构造局部观测块。

    hue_seq[t] 必须恰好出现在 probe_to_row 中。
    返回 shape (L, D) 的观测矩阵。
    """
    rows = []
    for hue in hue_seq:
        hue = int(hue)
        if hue not in probe_to_row:
            raise KeyError(f"Hue {hue} not found in measured probes: {sorted(probe_to_row.keys())}")
        rows.append(Y[probe_to_row[hue]])
    return np.asarray(rows, dtype=float)


# =========================
# Local decoding
# =========================

def decode_local_block(Y_obs: Array, w: Array, w2: Array, code: Array) -> DecodeResult:
    """
    考虑第二特征向量的解码流程：
    1) 对当前块按列去中心化
    2) u1 = Y_centered @ w1（第一特征向量投影）
    3) u2 = Y_centered @ w2（第二特征向量投影）
    4) gamma1 = c^T u1（第一特征向量相关检测值）
    5) gamma2 = c^T u2（第二特征向量相关检测值）
    6) 融合两个特征向量的信息进行判决
    """
    Y_obs = np.asarray(Y_obs, dtype=float)
    w = np.asarray(w, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    code = np.asarray(code, dtype=float)

    mean_vec = Y_obs.mean(axis=0)
    Y_centered = Y_obs - mean_vec
    
    # 计算两个特征向量的投影
    u1 = Y_centered @ w
    u2 = Y_centered @ w2
    
    # 计算相关检测值
    gamma1 = float(code @ u1)
    gamma2 = float(code @ u2)
    
    # 融合两个gamma值，第一特征向量权重更高
    gamma = 0.7 * gamma1 + 0.3 * gamma2
    
    # 基于融合后的gamma值判决
    bit_hat_pm = 1 if gamma > 0 else -1
    bit_hat_bin = 1 if bit_hat_pm > 0 else 0
    
    return DecodeResult(
        Y_obs=Y_obs,
        mean_vec=mean_vec,
        Y_centered=Y_centered,
        u=u1,  # 保持向后兼容，返回第一特征向量的投影
        gamma=gamma,  # 返回融合后的gamma值
        bit_hat_pm=bit_hat_pm,
        bit_hat_bin=bit_hat_bin,
    )


# =========================
# End-to-end experiment
# =========================

def simulate_blocks(
        models: List[FingerprintModel],
        bit_blocks_pm: List[Array],
        hue_mapping: Dict[int, int],
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
    codes2 = [m.code for m in models]  # 使用相同的地址码，因为第二特征向量的投影也用于生成地址码
    probe_to_row = build_probe_to_row(models[0].probes)
    results = []

    for bits_pm in bit_blocks_pm:
        bits_pm = np.asarray(bits_pm, dtype=int)
        symbol_seq1, symbol_seq2 = build_symbol_sequence(bits_pm, codes, codes2)
        hue_seq = map_symbol_to_hue(symbol_seq1, symbol_seq2, hue_mapping)

        block_info = {
            "bits_pm": bits_pm,
            "bits_bin": pm1_to_bin(bits_pm),
            "symbol_seq": symbol_seq1,  # 保持向后兼容，存储第一特征向量的符号序列
            "symbol_seq2": symbol_seq2,  # 新增：存储第二特征向量的符号序列
            "hue_seq": hue_seq,
            "per_position": [],
        }

        for pos_idx, model in enumerate(models):
            Y_obs = observe_block_from_measured_matrix(hue_seq, model.Y, probe_to_row)
            dec = decode_local_block(Y_obs, model.w, model.w2, model.code)
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


def load_data_from_csv(file_path1: str, file_path2: str, num_probes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从两个CSV文件读取数据，根据探针数量提取对应行作为Y1和Y2矩阵

    Parameters
    ----------
    file_path1 : 第一个CSV文件路径（对应位置1）
    file_path2 : 第二个CSV文件路径（对应位置2）
    num_probes : 探针数量

    Returns
    -------
    probes : 探针数组
    Y1 : 位置1的实测矩阵
    Y2 : 位置2的实测矩阵
    """
    # 读取第一个CSV文件（位置1）获取行数
    df1 = pd.read_csv(file_path1)
    max_row_index = len(df1) - 1
    
    # 生成探针数组
    probes = generate_probes(num_probes, max_row_index)
    
    # 计算要提取的行索引：(probe/5)-1
    target_row_indices = []
    for probe in probes:
        row_index = int((probe / 5) - 1)
        target_row_indices.append(row_index)

    # 提取指定行的数据，转换为numpy数组
    Y1 = df1.iloc[target_row_indices].values.astype(float)

    # 读取第二个CSV文件（位置2）
    df2 = pd.read_csv(file_path2)
    # 提取指定行的数据，转换为numpy数组
    Y2 = df2.iloc[target_row_indices].values.astype(float)

    return probes, Y1, Y2


def get_data_from_csv(csv_file1: str, csv_file2: str, num_probes: int) -> Tuple[List[FingerprintModel], List[Array], Dict[int, int]]:
    """
    从CSV文件获取数据并创建FingerprintModel，替代原有的builtin_example

    Parameters
    ----------
    csv_file1 : 第一个CSV文件路径
    csv_file2 : 第二个CSV文件路径
    num_probes : 探针数量

    Returns
    -------
    models : 两个位置的FingerprintModel列表
    [Y1, Y2] : 原始矩阵列表
    hue_mapping : 颜色映射字典
    """
    # 直接从CSV文件加载数据
    probes, Y1, Y2 = load_data_from_csv(csv_file1, csv_file2, num_probes)

    # 提取两个位置的指纹模型
    m1 = extract_fingerprint(probes, Y1)
    m2 = extract_fingerprint(probes, Y2)

    # 注意：由于探针数量是用户指定的，不再强制调整地址码方向
    # 保持原始提取的地址码方向

    # 生成hue_mapping，根据两个位置的z值特征选择合适的探针值
    # -2: 两个位置的z值都是负值且绝对值更大的位置
    # 0: 两个位置的z值有正有负，绝对值接近于0的位置
    # 2: 两个位置的z值都大于0且绝对值更大的位置
    
    # 获取两个位置的z值
    z1 = m1.z
    z2 = m2.z
    
    # 计算每个位置的特征
    features = []
    for i in range(len(z1)):
        z1_val = z1[i]
        z2_val = z2[i]
        
        # 计算特征：
        # 1. 两个值是否都为负
        both_negative = z1_val < 0 and z2_val < 0
        # 2. 两个值是否都为正
        both_positive = z1_val > 0 and z2_val > 0
        # 3. 两个值是否有正有负
        mixed_sign = not (both_negative or both_positive)
        # 4. 绝对值之和（用于比较大小）
        abs_sum = abs(z1_val) + abs(z2_val)
        # 5. 绝对值之和的倒数（用于比较接近0的程度）
        near_zero_score = 1 / (abs_sum + 1e-10)  # 避免除零
        
        features.append({
            'index': i,
            'both_negative': both_negative,
            'both_positive': both_positive,
            'mixed_sign': mixed_sign,
            'abs_sum': abs_sum,
            'near_zero_score': near_zero_score
        })
    
    # 选择-2的映射：两个值都为负且绝对值之和最大的位置
    negative_candidates = [f for f in features if f['both_negative']]
    if negative_candidates:
        negative_candidates.sort(key=lambda x: x['abs_sum'], reverse=True)
        neg_index = negative_candidates[0]['index']
    else:
        # 如果没有符合条件的，选择第一个位置
        neg_index = 0
    
    # 选择2的映射：两个值都为正且绝对值之和最大的位置
    positive_candidates = [f for f in features if f['both_positive']]
    if positive_candidates:
        positive_candidates.sort(key=lambda x: x['abs_sum'], reverse=True)
        pos_index = positive_candidates[0]['index']
    else:
        # 如果没有符合条件的，选择最后一个位置
        pos_index = len(probes) - 1
    
    # 选择0的映射：两个值有正有负且绝对值之和最小的位置
    mixed_candidates = [f for f in features if f['mixed_sign']]
    if mixed_candidates:
        mixed_candidates.sort(key=lambda x: x['abs_sum'])
        zero_index = mixed_candidates[0]['index']
    else:
        # 如果没有符合条件的，选择中间位置
        zero_index = len(probes) // 2
    
    # 生成hue_mapping
    hue_mapping = {
        -2: int(probes[neg_index]),
        0: int(probes[zero_index]),
        2: int(probes[pos_index])
    }
    


    return [m1, m2], [Y1, Y2], hue_mapping


def main() -> None:
    csv_file1 = "data\\15pro\\yellow\\12.csv"
    csv_file2 = "data\\15pro\\yellow\\14.csv"
    
    # 指定探针数量
    num_probes = 15
    
    # 从CSV文件获取数据（替代原有的get_builtin_example()）
    models, _, hue_mapping = get_data_from_csv(csv_file1, csv_file2, num_probes)

    # Example in the paper/document:
    # position1 sends 101, position2 sends 011
    # blocks: (+1,-1), (-1,+1), (+1,+1)
    bit_blocks_pm = [
        np.array([+1, -1]),
        np.array([-1, -1]),
        np.array([-1, +1]),
        np.array([-1, -1]),
        np.array([-1, +1]),
        np.array([-1, +1]),
    ]

    results = simulate_blocks(models, bit_blocks_pm, hue_mapping)

    # =============================
    # 逐 block 打印
    # =============================
    
    # =============================
    # 按设备整理发送与接收数据（只执行一次）
    # =============================

    num_pos = len(models)

    # 发送端 bit 序列
    tx_bits = [[] for _ in range(num_pos)]

    # 接收端 bit 序列
    rx_bits = [[] for _ in range(num_pos)]

    for res in results:

        # 发送端
        bits_tx = res["bits_bin"]

        # 接收端
        bits_rx = [dec.bit_hat_bin for dec in res["per_position"]]

        for p in range(num_pos):
            tx_bits[p].append(int(bits_tx[p]))
            rx_bits[p].append(int(bits_rx[p]))



if __name__ == "__main__":
    main()