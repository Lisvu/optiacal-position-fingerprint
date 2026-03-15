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


# =========================
# Encoding / observation
# =========================

def build_symbol_sequence(bits_pm: Array, codes: List[Array]) -> Array:
    """s = sum_i b_i c_i"""
    bits_pm = np.asarray(bits_pm, dtype=int)
    out = np.zeros_like(codes[0], dtype=int)
    for b, c in zip(bits_pm, codes):
        out += int(b) * np.asarray(c, dtype=int)
    return out


def map_symbol_to_hue(symbol_seq: Array, hue_mapping: Dict[int, int]) -> Array:
    return np.array([hue_mapping[int(v)] for v in symbol_seq], dtype=int)


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
        symbol_seq = build_symbol_sequence(bits_pm, codes)
        hue_seq = map_symbol_to_hue(symbol_seq, hue_mapping)

        block_info = {
            "bits_pm": bits_pm,
            "bits_bin": pm1_to_bin(bits_pm),
            "symbol_seq": symbol_seq,
            "hue_seq": hue_seq,
            "per_position": [],
        }

        for pos_idx, model in enumerate(models):
            Y_obs = observe_block_from_measured_matrix(hue_seq, model.Y, probe_to_row)
            dec = decode_local_block(Y_obs, model.w, model.code)
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

def load_data_from_csv(file_path1: str, file_path2: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从两个CSV文件读取数据，提取指定行作为Y1和Y2矩阵

    Parameters
    ----------
    file_path1 : 第一个CSV文件路径（对应位置1）
    file_path2 : 第二个CSV文件路径（对应位置2）

    Returns
    -------
    probes : 探针数组
    Y1 : 位置1的实测矩阵（第2、11、21、31、41、51、61行）
    Y2 : 位置2的实测矩阵（第2、11、21、31、41、51、61行）
    """
    # 定义要提取的行索引（第2、11、21、31、41、51、61行，Python索引从0开始，所以减1）
    target_row_indices = [0, 9, 19, 29, 39, 49, 59]

    # 读取第一个CSV文件（位置1）
    df1 = pd.read_csv(file_path1)
    # 提取指定行的数据，转换为numpy数组
    Y1 = df1.iloc[target_row_indices].values.astype(float)

    # 读取第二个CSV文件（位置2）
    df2 = pd.read_csv(file_path2)
    # 提取指定行的数据，转换为numpy数组
    Y2 = df2.iloc[target_row_indices].values.astype(float)

    # 探针数组（保持与原代码一致）
    probes = np.array([5, 50, 100, 150, 200, 250, 300], dtype=float)

    print(f"成功从CSV文件读取数据：")
    print(f"第一个文件 {file_path1} 提取行数：{len(Y1)}，列数：{Y1.shape[1]}")
    print(f"第二个文件 {file_path2} 提取行数：{len(Y2)}，列数：{Y2.shape[1]}")

    return probes, Y1, Y2


def get_data_from_csv(csv_file1: str, csv_file2: str) -> Tuple[List[FingerprintModel], List[Array], Dict[int, int]]:
    """
    从CSV文件获取数据并创建FingerprintModel，替代原有的builtin_example

    Parameters
    ----------
    csv_file1 : 第一个CSV文件路径
    csv_file2 : 第二个CSV文件路径

    Returns
    -------
    models : 两个位置的FingerprintModel列表
    [Y1, Y2] : 原始矩阵列表
    hue_mapping : 颜色映射字典
    """
    # 直接从CSV文件加载数据
    probes, Y1, Y2 = load_data_from_csv(csv_file1, csv_file2)

    # 提取两个位置的指纹模型
    m1 = extract_fingerprint(probes, Y1)
    m2 = extract_fingerprint(probes, Y2)

    # 为了与原文档中已固定的地址码方向保持一致，必要时翻转位置1方向
    target_c1 = np.array([1, 1, 1, -1, -1, -1, 1], dtype=int)
    if not np.array_equal(m1.code, target_c1):
        m1 = FingerprintModel(
            probes=m1.probes,
            Y=m1.Y,
            trend=m1.trend,
            residual=m1.residual,
            w=-m1.w,
            z=-m1.z,
            code=-m1.code,
        )

    # 保持hue_mapping与原代码一致
    hue_mapping = {-2: 200, 0: 5, 2: 100}

    return [m1, m2], [Y1, Y2], hue_mapping


# =========================
# 主函数（修改为使用CSV数据）
# =========================

def main() -> None:
    csv_file1 = "15pro/white/9.csv"
    csv_file2 = "15pro/white/18.csv"
    # 从CSV文件获取数据（替代原有的get_builtin_example()）
    models, _, hue_mapping = get_data_from_csv(csv_file1, csv_file2)

    print_model_summary("position 1", models[0])
    print_model_summary("position 2", models[1])

    # Example in the paper/document:
    # position1 sends 101, position2 sends 011
    # blocks: (+1,-1), (-1,+1), (+1,+1)
    bit_blocks_pm = [
        np.array([+1, -1]),
        np.array([-1, -1]),
        np.array([-1, +1]),
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