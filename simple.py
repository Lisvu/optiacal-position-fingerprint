#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

Array = np.ndarray


@dataclass
class FingerprintModel:
    probes: Array
    Y: Array
    trend: Array
    residual: Array
    w: Array
    z: Array
    code: Array


@dataclass
class DecodeResult:
    Y_obs: Array
    mean_vec: Array
    Y_centered: Array
    u: Array
    gamma: float
    bit_hat_pm: int
    bit_hat_bin: int


# =========================
# Core math utilities
# =========================

def fit_linear_trend(x: Array, Y: Array) -> Tuple[Array, Array]:
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


def extract_fingerprint(x: Array, Y: Array) -> FingerprintModel:
    _, trend = fit_linear_trend(x, Y)
    residual = Y - trend

    _, _, Vt = np.linalg.svd(residual, full_matrices=False)
    w = Vt[0].copy()
    z = residual @ w

    if z[0] < 0:
        w = -w
        z = -z

    code = np.where(z >= 0, 1, -1)

    return FingerprintModel(x, Y, trend, residual, w, z, code)


def pm1_to_bin(bits_pm: Array) -> Array:
    return np.where(bits_pm > 0, 1, 0)


def calculate_correlation(c1: Array, c2: Array) -> float:
    covariance = np.cov(c1, c2)[0, 1]
    std1 = np.std(c1)
    std2 = np.std(c2)
    if std1 == 0 or std2 == 0:
        return 0.0
    return covariance / (std1 * std2)


# =========================
# Probe search
# =========================

def generate_probes(num_probes: int, max_row_index: int) -> np.ndarray:
    interval = round((360 / (num_probes - 1)) / 5) * 5
    probes = [5 + i * interval for i in range(num_probes)]
    return np.array(probes, dtype=float)


def load_data_from_csv(f1, f2, f3, num_probes):
    df1 = pd.read_csv(f1)
    max_row_index = len(df1) - 1

    probes = generate_probes(num_probes, max_row_index)
    idx = [(int(p / 5) - 1) for p in probes]

    Y1 = df1.iloc[idx].values.astype(float)
    Y2 = pd.read_csv(f2).iloc[idx].values.astype(float)
    Y3 = pd.read_csv(f3).iloc[idx].values.astype(float)

    return probes, Y1, Y2, Y3


def find_optimal_probe_count(f1, f2, f3, min_p=5, max_p=20):
    best_p = min_p
    best_corr = float('inf')

    for p in range(min_p, max_p + 1):
        probes, Y1, Y2, Y3 = load_data_from_csv(f1, f2, f3, p)

        m1 = extract_fingerprint(probes, Y1)
        m2 = extract_fingerprint(probes, Y2)
        m3 = extract_fingerprint(probes, Y3)

        c1, c2, c3 = m1.code, m2.code, m3.code

        rho = max(
            abs(calculate_correlation(c1, c2)),
            abs(calculate_correlation(c1, c3)),
            abs(calculate_correlation(c2, c3)),
        )

        if rho < best_corr:
            best_corr = rho
            best_p = p

    print(f"最优探针数量: {best_p}, 最小互相关: {best_corr:.4f}")
    return best_p


# =========================
# Encoding / decoding
# =========================

def build_probe_to_row(probes: Array):
    return {int(v): i for i, v in enumerate(probes)}


def observe_block(hue_seq, Y, mapping):
    return np.array([Y[mapping[int(h)]] for h in hue_seq])


def decode_block(Y_obs, w, code):
    mean = Y_obs.mean(axis=0)
    Yc = Y_obs - mean
    u = Yc @ w
    gamma = code @ u
    bit = 1 if gamma > 0 else 0
    return bit


# =========================
# Main
# =========================

def main():
    f1 = "data\\mate40pro\\white\\15.csv"
    f2 = "data\\mate40pro\\white\\16.csv"
    f3 = "data\\mate40pro\\white\\17.csv"

    best_p = find_optimal_probe_count(f1, f2, f3)

    probes, Y1, Y2, Y3 = load_data_from_csv(f1, f2, f3, best_p)

    m1 = extract_fingerprint(probes, Y1)
    m2 = extract_fingerprint(probes, Y2)
    m3 = extract_fingerprint(probes, Y3)

    models = [m1, m2, m3]
    mapping = build_probe_to_row(probes)

    # 示例发送
    bit_blocks = [
        np.array([-1, -1, +1]),
        np.array([-1, -1, -1]),
        np.array([+1, +1, -1]),
    ]

    for bits in bit_blocks:
        symbol = sum(b * m.code for b, m in zip(bits, models))
        hue_seq = probes  # 简化

        for i, m in enumerate(models):
            Y_obs = observe_block(hue_seq, m.Y, mapping)
            bit = decode_block(Y_obs, m.w, m.code)
            print(f"设备{i+1} 解码: {bit}")


if __name__ == "__main__":
    main()