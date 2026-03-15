#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def arr_str(a, precision=2):
    if np.issubdtype(a.dtype, np.integer):
        return np.array2string(a, separator=', ')
    return np.array2string(a, precision=precision, suppress_small=False)


def fit_linear_trend(x, Y):
    """
    对每一列做线性拟合 y = ax + b
    """
    x = np.asarray(x, dtype=float)
    Y = np.asarray(Y, dtype=float)

    P, D = Y.shape
    A = np.column_stack([x, np.ones_like(x)])

    coeffs = np.zeros((D, 2))
    trend = np.zeros_like(Y)

    for j in range(D):
        ab, *_ = np.linalg.lstsq(A, Y[:, j], rcond=None)
        coeffs[j] = ab
        trend[:, j] = A @ ab

    return coeffs, trend


def extract_fingerprint(probes, Y):

    probes = np.asarray(probes, dtype=float)
    Y = np.asarray(Y, dtype=float)

    print("========== 原始实测矩阵 Y ==========")
    print(Y)
    print()

    # 线性回归
    coeffs, trend = fit_linear_trend(probes, Y)

    print("========== 线性回归公式 (最小二乘) ==========")
    for i, (a, b) in enumerate(coeffs):
        print(f"Channel {i}: y = {a:.6f} * x + {b:.6f}")
    print()

    print("========== 拟合趋势矩阵 trend ==========")
    print(arr_str(trend))
    print()

    residual = Y - trend

    print("========== 残差矩阵 residual = Y - trend ==========")
    print(arr_str(residual))
    print()

    # SVD
    _, _, Vt = np.linalg.svd(residual, full_matrices=False)

    w = Vt[0]
    z = residual @ w

    if z[0] < 0:
        w = -w
        z = -z

    code = np.where(z >= 0, 1, -1)

    print("========== 指纹提取结果 ==========")
    print("w   =", arr_str(w, 3))
    print("z   =", arr_str(z, 2))
    print("code=", arr_str(code))
    print()


if __name__ == "__main__":

    probes = np.array([5, 50, 100, 150, 200, 250, 300])

    Y = np.array([
        [352, 2, 0, 10],
        [20, 54, 44, 66],
        [78, 116, 100, 102],
        [124, 138, 136, 132],
        [238, 190, 206, 180],
        [256, 248, 248, 244],
        [306, 296, 302, 306],
    ], dtype=float)

    extract_fingerprint(probes, Y)