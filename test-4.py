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
    w: Array  # shape: (D,)指纹方向，特征向量
    z: Array  # shape: (P,)投影值
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

    # ===== 新增打印信息 =====
    print("\n========== 原始实测矩阵 Y ==========")
    print(Y)

    print("\n========== 线性回归公式 (最小二乘) ==========")
    for j, (a, b) in enumerate(coeffs):
        print(f"Channel {j}: y = {a:.6f} * x + {b:.6f}")

    print("\n========== 拟合趋势矩阵 trend ==========")
    print(np.round(trend, 2))

    print("\n========== 残差矩阵 residual = Y - trend ==========")
    print(np.round(residual, 2))
    # ===== 打印结束 =====

    # SVD: residual = U S V^T
    _, _, Vt = np.linalg.svd(residual, full_matrices=False)
    w = Vt[0].copy()
    z = residual @ w

    if force_positive_first and z[0] < 0:
        w = -w
        z = -z

    code = np.where(z >= 0, 1, -1)

    return FingerprintModel(
        probes=np.asarray(x, dtype=float),
        Y=np.asarray(Y, dtype=float),
        trend=trend,
        residual=residual,
        w=w,
        z=z,
        code=code,
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
    
    # 使用numpy的corrcoef函数计算相关系数
    # corrcoef返回一个2x2矩阵，[0,1]和[1,0]位置是两个数组的相关系数
    correlation_matrix = np.corrcoef(c1, c2)
    correlation = correlation_matrix[0, 1]
    
    return correlation


def find_optimal_probe_count(csv_file1: str, csv_file2: str, csv_file3: str, csv_file4: str, min_probes: int = 5, max_probes: int = 15) -> int:
    """
    遍历探针数量，找到地址码最大互相关指数最小的探针数量
    
    Parameters
    ----------
    csv_file1 : 第一个CSV文件路径
    csv_file2 : 第二个CSV文件路径
    csv_file3 : 第三个CSV文件路径
    csv_file4 : 第四个CSV文件路径
    min_probes : 最小探针数量
    max_probes : 最大探针数量
    
    Returns
    -------
    int : 最优探针数量
    """
    best_probe_count = min_probes
    min_max_correlation = float('inf')
    
    print("\n========== 寻找最优探针数量 ==========")
    
    for num_probes in range(min_probes, max_probes + 1):
        # 加载数据
        probes, Y1, Y2, Y3, Y4 = load_data_from_csv(csv_file1, csv_file2, csv_file3, csv_file4, num_probes)
        
        # 提取指纹模型
        m1 = extract_fingerprint(probes, Y1)
        m2 = extract_fingerprint(probes, Y2)
        m3 = extract_fingerprint(probes, Y3)
        m4 = extract_fingerprint(probes, Y4)
        
        # 获取地址码
        c1 = m1.code
        c2 = m2.code
        c3 = m3.code
        c4 = m4.code
        
        # 计算互相关指数
        rho12 = abs(calculate_correlation(c1, c2))
        rho13 = abs(calculate_correlation(c1, c3))
        rho14 = abs(calculate_correlation(c1, c4))
        rho23 = abs(calculate_correlation(c2, c3))
        rho24 = abs(calculate_correlation(c2, c4))
        rho34 = abs(calculate_correlation(c3, c4))
        
        # 计算最大互相关指数
        max_correlation = max(rho12, rho13, rho14, rho23, rho24, rho34)
        
        print(f"探针数量: {num_probes}, ρ12: {rho12:.4f}, ρ13: {rho13:.4f}, ρ14: {rho14:.4f}, ρ23: {rho23:.4f}, ρ24: {rho24:.4f}, ρ34: {rho34:.4f}, ρmax: {max_correlation:.4f}")
        
        # 更新最优值
        if max_correlation < min_max_correlation:
            min_max_correlation = max_correlation
            best_probe_count = num_probes
    
    print(f"\n最优探针数量: {best_probe_count}, 最小ρmax: {min_max_correlation:.4f}")
    return best_probe_count


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

def decode_local_block(Y_obs: Array, w: Array, code: Array) -> DecodeResult:
    """
    严格按论文/汇报中的流程：
    1) 对当前块按列去中心化
    2) u = Y_centered @ w
    3) gamma = c^T u
    4) gamma > 0 判 +1 / bit=1，否则判 -1 / bit=0
    """
    Y_obs = np.asarray(Y_obs, dtype=float)
    w = np.asarray(w, dtype=float)
    code = np.asarray(code, dtype=float)

    mean_vec = Y_obs.mean(axis=0)
    Y_centered = Y_obs - mean_vec
    u = Y_centered @ w
    gamma = float(code @ u)
    bit_hat_pm = 1 if gamma > 0 else -1
    bit_hat_bin = 1 if bit_hat_pm > 0 else 0
    return DecodeResult(
        Y_obs=Y_obs,
        mean_vec=mean_vec,
        Y_centered=Y_centered,
        u=u,
        gamma=gamma,
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
        auto_adjust_direction: bool = True,
) -> List[dict]:
    """
    针对多个发送块做端到端仿真。

    Parameters
    ----------
    models : 每个位置一个 FingerprintModel
    bit_blocks_pm : list of shape-(N_pos,) arrays, each entry in {+1, -1}
    hue_mapping : e.g. {-2:150, 0:100, 2:300}
    auto_adjust_direction : 是否自动调整方向

    Returns
    -------
    一个列表，每个元素对应一个 block，包含：
      - bits_pm
      - bits_bin
      - symbol_seq
      - hue_seq
      - per_pos decode result
      - direction_adjusted : 是否调整了方向
    """
    codes = [m.code for m in models]
    probe_to_row = build_probe_to_row(models[0].probes)
    results = []
    
    # 用于存储每个位置的方向调整状态
    direction_adjusted = [False] * len(models)
    # 用于存储每个位置的方向调整标志
    direction_flip = [1] * len(models)  # 1表示正常，-1表示需要反转

    for block_idx, bits_pm in enumerate(bit_blocks_pm):
        bits_pm = np.asarray(bits_pm, dtype=int)
        symbol_seq, symbol_combinations = build_symbol_sequence(bits_pm, codes)
        hue_seq = map_symbol_to_hue(symbol_seq, symbol_combinations, hue_mapping)

        block_info = {
            "bits_pm": bits_pm,
            "bits_bin": pm1_to_bin(bits_pm),
            "symbol_seq": symbol_seq,
            "hue_seq": hue_seq,
            "per_position": [],
            "direction_adjusted": False,
        }

        for pos_idx, model in enumerate(models):
            Y_obs = observe_block_from_measured_matrix(hue_seq, model.Y, probe_to_row)
            
            # 解码
            dec = decode_local_block(Y_obs, model.w, model.code)
            
            # 自动调整方向
            if auto_adjust_direction:
                reference_bit = pm1_to_bin(bits_pm)[pos_idx]
                
                # 对于第一个block，确定方向调整标志
                if block_idx == 0:
                    if dec.bit_hat_bin != reference_bit:
                        direction_flip[pos_idx] = -1
                        direction_adjusted[pos_idx] = True
                        block_info["direction_adjusted"] = True
                
                # 应用方向调整
                if direction_flip[pos_idx] == -1:
                    dec.bit_hat_pm = -dec.bit_hat_pm
                    dec.bit_hat_bin = 1 - dec.bit_hat_bin
            
            block_info["per_position"].append(dec)

        results.append(block_info)
    
    # 打印方向调整结果
    if auto_adjust_direction:
        print("\n========== 方向调整结果 ==========")
        for i, adjusted in enumerate(direction_adjusted):
            if adjusted:
                print(f"位置 {i+1} 的地址码方向已调整")
    
    return results


# =========================
# Pretty print helpers
# =========================

def arr_str(a: Array, precision: int = 2) -> str:
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.integer):
        return np.array2string(a, separator=', ')
    return np.array2string(a, precision=precision, suppress_small=False, separator=', ')


def print_model_summary(name: str, model: FingerprintModel) -> None:
    print(f"==== {name} ====")
    print("w   =", arr_str(model.w, 3))
    print("z   =", arr_str(model.z, 2))
    print("code=", arr_str(model.code))
    print()


def print_block_result(block_idx: int, result: dict) -> None:
    print(f"==== BLOCK {block_idx} ====")
    print("bits_pm   =", arr_str(result["bits_pm"]))
    print("bits_bin  =", arr_str(result["bits_bin"]))
    print("symbol_seq=", arr_str(result["symbol_seq"]))
    print("hue_seq   =", arr_str(result["hue_seq"]))
    print()
    for i, dec in enumerate(result["per_position"], start=1):
        print(f"-- position {i} --")
        print("Y_obs =")
        print(dec.Y_obs)
        print("mean  =", arr_str(dec.mean_vec, 2))
        print("u     =", arr_str(dec.u, 2))
        print("gamma =", round(dec.gamma, 2))
        print("bit   =", dec.bit_hat_bin)
        print()


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
    
    print(f"\n========== 探针生成 ==========")
    print(f"探针数量: {num_probes}")
    print(f"理论间隔: {theoretical_interval:.4f}")
    print(f"实际间隔: {interval}")
    print(f"最大行索引: {max_row_index}")
    print(f"最大允许探针值: {(max_row_index + 1) * 5}")
    print(f"生成的探针: {probes}")
    
    return np.array(probes, dtype=float)


def load_data_from_csv(file_path1: str, file_path2: str, file_path3: str, file_path4: str, num_probes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从四个CSV文件读取数据，根据探针数量提取对应行作为Y1、Y2、Y3和Y4矩阵

    Parameters
    ----------
    file_path1 : 第一个CSV文件路径（对应位置1）
    file_path2 : 第二个CSV文件路径（对应位置2）
    file_path3 : 第三个CSV文件路径（对应位置3）
    file_path4 : 第四个CSV文件路径（对应位置4）
    num_probes : 探针数量

    Returns
    -------
    probes : 探针数组
    Y1 : 位置1的实测矩阵
    Y2 : 位置2的实测矩阵
    Y3 : 位置3的实测矩阵
    Y4 : 位置4的实测矩阵
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

    # 读取第三个CSV文件（位置3）
    df3 = pd.read_csv(file_path3)
    # 提取指定行的数据，转换为numpy数组
    Y3 = df3.iloc[target_row_indices].values.astype(float)

    # 读取第四个CSV文件（位置4）
    df4 = pd.read_csv(file_path4)
    # 提取指定行的数据，转换为numpy数组
    Y4 = df4.iloc[target_row_indices].values.astype(float)

    print(f"成功从CSV文件读取数据：")
    print(f"探针数组：{probes}")
    print(f"提取的行索引：{target_row_indices}")
    print(f"第一个文件 {file_path1} 提取行数：{len(Y1)}，列数：{Y1.shape[1]}")
    print(f"第二个文件 {file_path2} 提取行数：{len(Y2)}，列数：{Y2.shape[1]}")
    print(f"第三个文件 {file_path3} 提取行数：{len(Y3)}，列数：{Y3.shape[1]}")
    print(f"第四个文件 {file_path4} 提取行数：{len(Y4)}，列数：{Y4.shape[1]}")

    return probes, Y1, Y2, Y3, Y4


def get_data_from_csv(csv_file1: str, csv_file2: str, csv_file3: str, csv_file4: str, num_probes: int) -> Tuple[List[FingerprintModel], List[Array], Dict[int, int]]:
    """
    从CSV文件获取数据并创建FingerprintModel，替代原有的builtin_example

    Parameters
    ----------
    csv_file1 : 第一个CSV文件路径
    csv_file2 : 第二个CSV文件路径
    csv_file3 : 第三个CSV文件路径
    csv_file4 : 第四个CSV文件路径
    num_probes : 探针数量

    Returns
    -------
    models : 四个位置的FingerprintModel列表
    [Y1, Y2, Y3, Y4] : 原始矩阵列表
    hue_mapping : 颜色映射字典
    """
    # 直接从CSV文件加载数据
    probes, Y1, Y2, Y3, Y4 = load_data_from_csv(csv_file1, csv_file2, csv_file3, csv_file4, num_probes)

    # 提取四个位置的指纹模型
    m1 = extract_fingerprint(probes, Y1)
    m2 = extract_fingerprint(probes, Y2)
    m3 = extract_fingerprint(probes, Y3)
    m4 = extract_fingerprint(probes, Y4)

    # 统一地址码方向：以位置1为参考，调整其他位置的地址码方向
    # 计算每个位置与位置1的地址码相关系数
    # 如果相关系数为负，说明方向相反，需要调整
    c1 = m1.code
    
    # 调整位置2的方向
    c2 = m2.code
    correlation_12 = calculate_correlation(c1, c2)
    if correlation_12 < 0:
        m2 = FingerprintModel(
            probes=m2.probes,
            Y=m2.Y,
            trend=m2.trend,
            residual=m2.residual,
            w=-m2.w,
            z=-m2.z,
            code=-m2.code,
        )
        print("已调整位置2的地址码方向")
    
    # 调整位置3的方向
    c3 = m3.code
    correlation_13 = calculate_correlation(c1, c3)
    if correlation_13 < 0:
        m3 = FingerprintModel(
            probes=m3.probes,
            Y=m3.Y,
            trend=m3.trend,
            residual=m3.residual,
            w=-m3.w,
            z=-m3.z,
            code=-m3.code,
        )
        print("已调整位置3的地址码方向")
    
    # 调整位置4的方向
    c4 = m4.code
    correlation_14 = calculate_correlation(c1, c4)
    if correlation_14 < 0:
        m4 = FingerprintModel(
            probes=m4.probes,
            Y=m4.Y,
            trend=m4.trend,
            residual=m4.residual,
            w=-m4.w,
            z=-m4.z,
            code=-m4.code,
        )
        print("已调整位置4的地址码方向")

    # 生成hue_mapping，根据四个位置的z值特征选择合适的探针值
    
    # 获取四个位置的z值
    z1 = m1.z
    z2 = m2.z
    z3 = m3.z
    z4 = m4.z
    
    # 生成hue_mapping，从符号组合数组映射到probe
    hue_mapping = {}
    
    # 所有可能的符号组合（这里只列举部分，实际应包含所有可能的4元素组合）
    possible_combinations = [
        (1, 1, 1, 1),    # symbol 4
        (1, 1, 1, -1),   # symbol 2
        (1, 1, -1, 1),   # symbol 2
        (1, 1, -1, -1),  # symbol 0
        (1, -1, 1, 1),   # symbol 2
        (1, -1, 1, -1),  # symbol 0
        (1, -1, -1, 1),  # symbol 0
        (1, -1, -1, -1), # symbol -2
        (-1, 1, 1, 1),   # symbol 2
        (-1, 1, 1, -1),  # symbol 0
        (-1, 1, -1, 1),  # symbol 0
        (-1, 1, -1, -1), # symbol -2
        (-1, -1, 1, 1),  # symbol 0
        (-1, -1, 1, -1), # symbol -2
        (-1, -1, -1, 1), # symbol -2
        (-1, -1, -1, -1) # symbol -4
    ]
    
    # 为每个组合选择合适的probe值
    for combination in possible_combinations:
        c1, c2, c3, c4 = combination
        
        # 计算每个位置的贡献
        # 对于c1=1，选择z1[i]为正且绝对值较大的位置
        # 对于c1=-1，选择z1[i]为负且绝对值较大的位置
        # 同样处理c2、c3和c4
        candidates = []
        for i in range(len(z1)):
            # 检查当前位置的z值符号是否与组合匹配
            if (c1 > 0 and z1[i] > 0) or (c1 < 0 and z1[i] < 0) or (c1 == 0):
                if (c2 > 0 and z2[i] > 0) or (c2 < 0 and z2[i] < 0) or (c2 == 0):
                    if (c3 > 0 and z3[i] > 0) or (c3 < 0 and z3[i] < 0) or (c3 == 0):
                        if (c4 > 0 and z4[i] > 0) or (c4 < 0 and z4[i] < 0) or (c4 == 0):
                            # 计算得分：符号匹配的位置绝对值越大越好
                            score = 0
                            if c1 != 0:
                                score += abs(z1[i]) * 2
                            if c2 != 0:
                                score += abs(z2[i]) * 2
                            if c3 != 0:
                                score += abs(z3[i]) * 2
                            if c4 != 0:
                                score += abs(z4[i]) * 2
                            candidates.append((score, i))
        
        if candidates:
            # 选择得分最高的位置
            candidates.sort(reverse=True)
            best_index = candidates[0][1]
            hue_mapping[combination] = int(probes[best_index])
        else:
            # 如果没有符合条件的，使用默认值
            hue_mapping[combination] = int(probes[len(probes)//2])
    
    # 打印hue_mapping的选择结果
    print("\n========== Hue Mapping 选择 ==========")
    for combination, probe in hue_mapping.items():
        print(f"组合 {combination} 映射到探针值: {probe}")
    print(f"hue_mapping: {hue_mapping}")

    return [m1, m2, m3, m4], [Y1, Y2, Y3, Y4], hue_mapping


def main() -> None:
    csv_file1 = "data\\mate40pro\\黄\\1.csv"
    csv_file2 = "data\\mate40pro\\黄\\7.csv"
    csv_file3 = "data\\mate40pro\\黄\\17.csv"  # 新增第三个位置的CSV文件
    csv_file4 = "data\\mate40pro\\黄\\27.csv"  # 新增第四个位置的CSV文件
    
    # 寻找最优探针数量
    best_probe_count = find_optimal_probe_count(csv_file1, csv_file2, csv_file3, csv_file4, min_probes=5, max_probes=25)
    
    # 使用最优探针数量进行实验
    num_probes = best_probe_count
    
    print(f"\n========== 使用最优探针数量 {num_probes} 进行实验 ==========")
    
    # 从CSV文件获取数据（替代原有的get_builtin_example()）
    models, _, hue_mapping = get_data_from_csv(csv_file1, csv_file2, csv_file3, csv_file4, num_probes)

    print_model_summary("position 1", models[0])
    print_model_summary("position 2", models[1])
    print_model_summary("position 3", models[2])  # 新增打印第三个位置的模型摘要
    print_model_summary("position 4", models[3])  # 新增打印第四个位置的模型摘要

    # Example in the paper/document:
    # position1 sends 101, position2 sends 011, position3 sends 110, position4 sends 001
    # blocks: (+1,-1,+1,+1), (-1,-1,-1,-1), (+1,+1,-1,+1)
    bit_blocks_pm = [
        np.array([-1, -1, +1, +1]),  # 四个设备的发送值
        np.array([-1, -1, -1, -1]),
        np.array([-1, +1, +1, -1]),
        np.array([+1, -1, -1, -1]),
        np.array([-1, -1, +1, -1]),
        np.array([-1, -1, +1, -1]),
        np.array([-1, +1, +1, -1]),
        np.array([-1, +1, -1, +1]),
        np.array([-1, -1, -1, -1]),
        np.array([-1, -1, +1, -1]),
        
    ]

    results = simulate_blocks(models, bit_blocks_pm, hue_mapping)

    # =============================
    # 逐 block 打印
    # =============================
    for idx, res in enumerate(results, start=1):
        print_block_result(idx, res)

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

    print("\n==============================")
    print("按设备统计通信结果")
    print("==============================\n")

    print("发送端：")
    for i in range(num_pos):
        print(f"position{i + 1}: {tx_bits[i]}")

    print("\n接收端：")
    for i in range(num_pos):
        print(f"position{i + 1}: {rx_bits[i]}")


if __name__ == "__main__":
    main()