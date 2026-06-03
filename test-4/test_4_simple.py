#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Four-position fingerprint experiment script.

Features:
1. Search for the best probe count and probe set for a given 4-position combination.
2. Use random search bits to minimize BER during probe selection.
3. Re-evaluate the best probe set on a fresh random test bit sequence.
4. Return the best BER and the final test BER for use by batch_test.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import itertools
import os
import random
import sys

import numpy as np
import pandas as pd

Array = np.ndarray

# ============================================================================
# 多主成分指纹配置
# ============================================================================
N_COMPONENTS = 3
# 发送端：多主成分混合权重（用于生成 effective code）
# 保守设置：PC1 主导，PC2/PC3 辅助放大差异
ALPHA = np.array([0.75, 0.20, 0.05])
# 接收端：多主成分解码权重（可搜索优化）
# 合法位置用更高 PC1 权重，降低 PC2/PC3 噪声影响
BETA = np.array([0.70, 0.25, 0.05])
EXACT_ENUMERATION_MAX_POSITIONS = 8
CSV_MATRIX_CACHE: dict[str, Array] = {}
USE_CONVOLUTIONAL_FEC = True
USE_AMPLITUDE_AWARE = False  # 全局开关：是否启用幅度感知 hue mapping
MIN_GAMMA_RATIO = 0.05       # 幅度感知的最小 gamma 比例阈值
CONV_CODE_CONSTRAINT_LENGTH = 3
CONV_CODE_TAIL_BITS = CONV_CODE_CONSTRAINT_LENGTH - 1
CONV_CODE_NUM_STATES = 1 << CONV_CODE_TAIL_BITS


@dataclass
class FingerprintModel:
    probes: Array
    Y: Array
    trend: Array
    residual: Array
    w: Array
    z: Array
    code: Array
    # --- 多主成分扩展字段 ---
    W: Array = None          # D x k, 多主成分方向矩阵
    Z: Array = None          # P x k, 多主成分投影矩阵
    multi_code: Array = None # P x k, 多主成分code矩阵
    eff_code: Array = None   # P, 有效发送code（多主成分加权混合）
    eff_z: Array = None      # P, 有效投影（多主成分加权混合，用于hue mapping生成）
    eff_w: Array = None      # D, 有效接收方向（W @ alpha，一维解码）


@dataclass
class DecodeResult:
    bit_hat_pm: int
    bit_hat_bin: int


def fit_linear_trend(x: Array, Y: Array) -> Tuple[Array, Array]:
    x = np.asarray(x, dtype=float)
    Y = np.asarray(Y, dtype=float)
    _, d = Y.shape
    A = np.column_stack([x, np.ones_like(x)])
    coeffs = np.zeros((d, 2), dtype=float)
    trend = np.zeros_like(Y, dtype=float)

    for j in range(d):
        ab, *_ = np.linalg.lstsq(A, Y[:, j], rcond=None)
        coeffs[j] = ab
        trend[:, j] = A @ ab

    return coeffs, trend


def extract_fingerprint(x: Array, Y: Array, n_components: int = None, force_positive_first: bool = True) -> FingerprintModel:
    if n_components is None:
        n_components = getattr(sys.modules[__name__], 'N_COMPONENTS', 1)

    _, trend = fit_linear_trend(x, Y)
    residual = np.asarray(Y, dtype=float) - trend

    _, s, Vt = np.linalg.svd(residual, full_matrices=False)
    k = min(n_components, len(s))
    W = Vt[:k].T.copy()  # D x k
    Z_mat = residual @ W  # P x k

    if force_positive_first:
        for j in range(k):
            if Z_mat[0, j] < 0:
                W[:, j] = -W[:, j]
                Z_mat[:, j] = -Z_mat[:, j]

    code_mat = np.where(Z_mat >= 0, 1, -1)

    # 计算有效发送code和接收方向：多主成分加权混合
    alpha = getattr(sys.modules[__name__], 'ALPHA', None)
    if alpha is not None and k > 1:
        alpha = np.asarray(alpha[:k], dtype=float)
        eff_z = Z_mat @ alpha
        eff_code = np.sign(eff_z).astype(int)
        eff_w = W @ alpha  # 一维有效接收方向
    else:
        eff_z = Z_mat[:, 0].copy() if k > 0 else np.zeros(len(np.asarray(x, dtype=float)), dtype=float)
        eff_code = code_mat[:, 0].copy() if k > 0 else np.ones(len(np.asarray(x, dtype=float)), dtype=int)
        eff_w = W[:, 0].copy() if k > 0 else np.zeros(Y.shape[1], dtype=float)

    return FingerprintModel(
        probes=np.asarray(x, dtype=float),
        Y=np.asarray(Y, dtype=float),
        trend=trend,
        residual=residual,
        w=W[:, 0],
        z=Z_mat[:, 0],
        code=code_mat[:, 0],
        W=W,
        Z=Z_mat,
        multi_code=code_mat,
        eff_code=eff_code,
        eff_z=eff_z,
        eff_w=eff_w,
    )


def pm1_to_bin(bits_pm: Array) -> Array:
    return np.where(np.asarray(bits_pm) > 0, 1, 0)


def bin_to_pm1(bits_bin: Array) -> Array:
    return np.where(np.asarray(bits_bin, dtype=int) > 0, 1, -1)


def should_use_exact_block_evaluation(num_positions: int) -> bool:
    return num_positions <= EXACT_ENUMERATION_MAX_POSITIONS


def generate_all_bit_blocks(num_positions: int) -> List[Array]:
    return [
        np.asarray(bits_pm, dtype=int)
        for bits_pm in itertools.product([-1, 1], repeat=num_positions)
    ]


def convolutional_encode_bits(bits_bin: Sequence[int]) -> Array:
    memory_1 = 0
    memory_2 = 0
    outputs: list[int] = []

    for bit in list(np.asarray(bits_bin, dtype=int).tolist()) + [0] * CONV_CODE_TAIL_BITS:
        bit = int(bit)
        outputs.append(bit ^ memory_1 ^ memory_2)
        outputs.append(bit ^ memory_2)
        memory_2 = memory_1
        memory_1 = bit

    return np.asarray(outputs, dtype=int)


def viterbi_decode_hard(coded_bits: Sequence[int]) -> Array:
    coded = np.asarray(coded_bits, dtype=int)
    if coded.size % 2 != 0:
        raise ValueError("Convolutional coded bit stream length must be even")

    num_steps = coded.size // 2
    path_metrics = np.full(CONV_CODE_NUM_STATES, np.inf, dtype=float)
    path_metrics[0] = 0.0
    predecessors = np.full((num_steps, CONV_CODE_NUM_STATES), -1, dtype=int)
    decided_bits = np.zeros((num_steps, CONV_CODE_NUM_STATES), dtype=int)

    for step in range(num_steps):
        rx_0 = int(coded[2 * step])
        rx_1 = int(coded[2 * step + 1])
        next_metrics = np.full(CONV_CODE_NUM_STATES, np.inf, dtype=float)

        for state in range(CONV_CODE_NUM_STATES):
            if not np.isfinite(path_metrics[state]):
                continue

            memory_1 = (state >> 1) & 1
            memory_2 = state & 1
            for input_bit in (0, 1):
                out_0 = input_bit ^ memory_1 ^ memory_2
                out_1 = input_bit ^ memory_2
                next_state = (input_bit << 1) | memory_1
                branch_metric = float((out_0 != rx_0) + (out_1 != rx_1))
                metric = path_metrics[state] + branch_metric
                if metric < next_metrics[next_state]:
                    next_metrics[next_state] = metric
                    predecessors[step, next_state] = state
                    decided_bits[step, next_state] = input_bit

        path_metrics = next_metrics

    state = 0
    if not np.isfinite(path_metrics[state]):
        state = int(np.argmin(path_metrics))

    decoded = np.zeros(num_steps, dtype=int)
    for step in range(num_steps - 1, -1, -1):
        decoded[step] = decided_bits[step, state]
        prev_state = predecessors[step, state]
        if prev_state < 0:
            prev_state = 0
        state = prev_state

    if CONV_CODE_TAIL_BITS == 0:
        return decoded
    return decoded[:-CONV_CODE_TAIL_BITS]


def generate_random_information_bits(
    num_bits: int,
    num_positions: int,
    rng: random.Random | None = None,
) -> Array:
    if rng is None:
        rng = random.Random()
    return np.asarray(
        [[rng.choice([0, 1]) for _ in range(num_positions)] for _ in range(num_bits)],
        dtype=int,
    )


def build_convolutional_bit_blocks(info_bits_bin: Array) -> List[Array]:
    info_bits = np.asarray(info_bits_bin, dtype=int)
    if info_bits.ndim != 2:
        raise ValueError("info_bits_bin must have shape (num_bits, num_positions)")

    num_positions = info_bits.shape[1]
    encoded_streams = [convolutional_encode_bits(info_bits[:, pos_idx]) for pos_idx in range(num_positions)]
    encoded_length = len(encoded_streams[0])
    if any(len(stream) != encoded_length for stream in encoded_streams):
        raise ValueError("Encoded streams must have the same length")

    return [
        bin_to_pm1(np.asarray([stream[step_idx] for stream in encoded_streams], dtype=int))
        for step_idx in range(encoded_length)
    ]


def calculate_correlation(c1: Array, c2: Array) -> float:
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    if len(c1) != len(c2):
        raise ValueError("Address code lengths must match")

    std1 = np.std(c1)
    std2 = np.std(c2)
    if std1 == 0 or std2 == 0:
        return 0.0
    return float(np.cov(c1, c2)[0, 1] / (std1 * std2))


def align_model_directions(models: List[FingerprintModel]) -> List[FingerprintModel]:
    if not models:
        return models

    ref = models[0]
    # 使用 eff_code 做方向对齐（与发送端一致）
    ref_code = ref.eff_code if ref.eff_code is not None else ref.code
    for idx in range(1, len(models)):
        model_code = models[idx].eff_code if models[idx].eff_code is not None else models[idx].code
        corr = calculate_correlation(ref_code, model_code)
        if corr < 0:
            models[idx].w = -models[idx].w
            models[idx].z = -models[idx].z
            models[idx].code = -models[idx].code
            if models[idx].W is not None and models[idx].W.ndim > 1:
                models[idx].W = -models[idx].W
                models[idx].Z = -models[idx].Z
                models[idx].multi_code = -models[idx].multi_code
                models[idx].eff_z = -models[idx].eff_z
                models[idx].eff_w = -models[idx].eff_w
                models[idx].eff_code = -models[idx].eff_code
            else:
                if models[idx].eff_w is not None:
                    models[idx].eff_w = -models[idx].eff_w
                if models[idx].eff_code is not None:
                    models[idx].eff_code = -models[idx].eff_code
    return models


def build_symbol_sequence(bits_pm: Array, codes: List[Array]) -> Tuple[Array, List[List[int]]]:
    bits_pm = np.asarray(bits_pm, dtype=int)
    out = np.zeros_like(codes[0], dtype=int)
    symbol_combinations: List[List[int]] = []

    for i in range(len(codes[0])):
        combination = []
        for b, c in zip(bits_pm, codes):
            contribution = int(b) * int(c[i])
            combination.append(contribution)
            out[i] += contribution
        symbol_combinations.append(combination)

    return out, symbol_combinations


def map_symbol_to_hue(symbol_combinations: List[List[int]], hue_mapping: Dict[Tuple[int, ...], int]) -> Array:
    hue_seq = []
    for combination in symbol_combinations:
        key = tuple(combination)
        if key not in hue_mapping:
            raise KeyError(f"Symbol combination {key} is not in hue_mapping")
        hue_seq.append(hue_mapping[key])
    return np.asarray(hue_seq, dtype=int)


def build_probe_to_row(probes: Array) -> Dict[int, int]:
    return {int(v): i for i, v in enumerate(np.asarray(probes).tolist())}


def observe_block_from_measured_matrix(hue_seq: Array, Y: Array, probe_to_row: Dict[int, int]) -> Array:
    rows = []
    for hue in hue_seq:
        hue = int(hue)
        if hue not in probe_to_row:
            raise KeyError(f"Hue {hue} not found in measured probes: {sorted(probe_to_row.keys())}")
        rows.append(Y[probe_to_row[hue]])
    return np.asarray(rows, dtype=float)


def decode_local_block(Y_obs: Array, w_or_W: Array, code_or_multi_code: Array, weights: Array = None) -> DecodeResult:
    Y_obs = np.asarray(Y_obs, dtype=float)
    W = np.asarray(w_or_W, dtype=float)
    code = np.asarray(code_or_multi_code, dtype=float)

    # 兼容旧的一维输入
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    if code.ndim == 1:
        code = code.reshape(-1, 1)

    k = W.shape[1]

    if weights is None:
        weights = getattr(sys.modules[__name__], 'BETA', None)
    if weights is None:
        weights = np.ones(k) / k
    weights = np.asarray(weights, dtype=float)

    mean_vec = Y_obs.mean(axis=0)
    Y_centered = Y_obs - mean_vec
    U = Y_centered @ W  # L x k

    gamma = 0.0
    for j in range(k):
        gamma += weights[j] * float(code[:, j] @ U[:, j])

    bit_hat_pm = 1 if gamma > 0 else -1
    bit_hat_bin = 1 if bit_hat_pm > 0 else 0
    return DecodeResult(bit_hat_pm=bit_hat_pm, bit_hat_bin=bit_hat_bin)


def simulate_blocks(
    models: List[FingerprintModel],
    bit_blocks_pm: List[Array],
    hue_mapping: Dict[Tuple[int, ...], int],
) -> List[dict]:
    # 发送端使用有效code（多主成分混合后的code）
    codes = [m.eff_code if m.eff_code is not None else m.code for m in models]
    probe_to_row = build_probe_to_row(models[0].probes)
    results = []

    for bits_pm in bit_blocks_pm:
        bits_pm = np.asarray(bits_pm, dtype=int)
        _, symbol_combinations = build_symbol_sequence(bits_pm, codes)
        hue_seq = map_symbol_to_hue(symbol_combinations, hue_mapping)

        block_info = {
            "bits_pm": bits_pm,
            "bits_bin": pm1_to_bin(bits_pm),
            "hue_seq": hue_seq,
            "per_position": [],
        }

        for model in models:
            Y_obs = observe_block_from_measured_matrix(hue_seq, model.Y, probe_to_row)
            # 接收端使用一维有效方向解码（与发送端匹配）
            if model.eff_w is not None and model.eff_code is not None:
                dec = decode_local_block(Y_obs, model.eff_w, model.eff_code)
            else:
                dec = decode_local_block(Y_obs, model.w, model.code)
            block_info["per_position"].append(dec)

        results.append(block_info)

    return results


def evaluate_blocks_ber(
    models: List[FingerprintModel],
    bit_blocks_pm: Sequence[Array],
    hue_mapping: Dict[Tuple[int, ...], int],
) -> float:
    if not bit_blocks_pm:
        return 0.0

    codes = [m.eff_code if m.eff_code is not None else m.code for m in models]
    probe_to_row = build_probe_to_row(models[0].probes)
    position_errors = np.zeros(len(models), dtype=float)
    position_total = np.zeros(len(models), dtype=float)

    for bits_pm in bit_blocks_pm:
        bits_pm = np.asarray(bits_pm, dtype=int)
        _, symbol_combinations = build_symbol_sequence(bits_pm, codes)
        hue_seq = map_symbol_to_hue(symbol_combinations, hue_mapping)
        bits_tx = pm1_to_bin(bits_pm)

        for model_idx, model in enumerate(models):
            Y_obs = observe_block_from_measured_matrix(hue_seq, model.Y, probe_to_row)
            if model.eff_w is not None and model.eff_code is not None:
                dec = decode_local_block(Y_obs, model.eff_w, model.eff_code)
            else:
                dec = decode_local_block(Y_obs, model.w, model.code)
            position_total[model_idx] += 1
            if bits_tx[model_idx] != dec.bit_hat_bin:
                position_errors[model_idx] += 1

    corrected_position_bers = np.minimum(position_errors / np.maximum(position_total, 1.0), 1.0)
    corrected_position_bers = np.minimum(corrected_position_bers, 1.0 - corrected_position_bers)
    corrected_error_bits = float(np.sum(corrected_position_bers * position_total))
    total_bits = float(np.sum(position_total))
    return corrected_error_bits / total_bits if total_bits > 0 else 0.0


def evaluate_blocks_ber_with_convolutional_fec(
    models: List[FingerprintModel],
    info_bits_bin: Array,
    hue_mapping: Dict[Tuple[int, ...], int],
) -> float:
    info_bits = np.asarray(info_bits_bin, dtype=int)
    if info_bits.size == 0:
        return 0.0

    codes = [m.eff_code if m.eff_code is not None else m.code for m in models]
    probe_to_row = build_probe_to_row(models[0].probes)
    bit_blocks_pm = build_convolutional_bit_blocks(info_bits)
    received_streams: list[list[int]] = [[] for _ in models]

    for bits_pm in bit_blocks_pm:
        bits_pm = np.asarray(bits_pm, dtype=int)
        _, symbol_combinations = build_symbol_sequence(bits_pm, codes)
        hue_seq = map_symbol_to_hue(symbol_combinations, hue_mapping)

        for model_idx, model in enumerate(models):
            Y_obs = observe_block_from_measured_matrix(hue_seq, model.Y, probe_to_row)
            if model.eff_w is not None and model.eff_code is not None:
                dec = decode_local_block(Y_obs, model.eff_w, model.eff_code)
            else:
                dec = decode_local_block(Y_obs, model.w, model.code)
            received_streams[model_idx].append(int(dec.bit_hat_bin))

    position_errors = np.zeros(len(models), dtype=float)
    position_total = np.zeros(len(models), dtype=float)
    for model_idx in range(len(models)):
        decoded_bits = viterbi_decode_hard(received_streams[model_idx])
        reference_bits = info_bits[:, model_idx]
        compare_len = min(len(decoded_bits), len(reference_bits))
        position_total[model_idx] = compare_len
        if compare_len <= 0:
            continue
        position_errors[model_idx] = float(np.sum(decoded_bits[:compare_len] != reference_bits[:compare_len]))

    corrected_position_bers = np.minimum(position_errors / np.maximum(position_total, 1.0), 1.0)
    corrected_position_bers = np.minimum(corrected_position_bers, 1.0 - corrected_position_bers)
    corrected_error_bits = float(np.sum(corrected_position_bers * position_total))
    total_bits = float(np.sum(position_total))
    return corrected_error_bits / total_bits if total_bits > 0 else 0.0


def calculate_ber(results: List[dict]) -> float:
    if not results:
        return 0.0

    total_bits = 0
    position_errors = [0] * len(results[0]["per_position"])
    position_total = [0] * len(results[0]["per_position"])

    for res in results:
        bits_tx = res["bits_bin"]
        bits_rx = [dec.bit_hat_bin for dec in res["per_position"]]

        for i, (tx, rx) in enumerate(zip(bits_tx, bits_rx)):
            total_bits += 1
            position_total[i] += 1
            if tx != rx:
                position_errors[i] += 1

    position_bers = [e / t if t > 0 else 0.0 for e, t in zip(position_errors, position_total)]
    corrected_position_bers = [min(ber, 1.0 - ber) for ber in position_bers]
    corrected_error_bits = sum(ber * total for ber, total in zip(corrected_position_bers, position_total))
    return float(corrected_error_bits / total_bits) if total_bits > 0 else 0.0


def generate_mapping_candidates(
    models: List[FingerprintModel],
    probes: Array,
    top_k_per_combination: int = 3,
) -> Dict[Tuple[int, ...], List[int]]:
    # 使用有效z值（多主成分混合后的投影，与发送端 eff_code 对应）
    z_list = []
    for m in models:
        if m.eff_z is not None:
            z_list.append(np.asarray(m.eff_z, dtype=float))
        elif m.Z is not None and m.Z.ndim > 1:
            k = m.Z.shape[1]
            alpha = getattr(sys.modules[__name__], 'ALPHA', None)
            if alpha is not None and k > 1:
                alpha = np.asarray(alpha[:k], dtype=float)
                z_list.append(np.asarray(m.Z @ alpha, dtype=float))
            else:
                z_list.append(np.asarray(m.Z[:, 0], dtype=float))
        else:
            z_list.append(np.asarray(m.z, dtype=float))

    possible_combinations = list(itertools.product([1, -1], repeat=len(models)))
    candidate_map: Dict[Tuple[int, ...], List[int]] = {}

    for combination in possible_combinations:
        scored_candidates = []
        for i in range(len(probes)):
            match = True
            score = 0.0
            for pos_idx, sign in enumerate(combination):
                z_val = z_list[pos_idx][i]
                if (sign > 0 and z_val <= 0) or (sign < 0 and z_val >= 0):
                    match = False
                    break
                score += abs(z_val)
            if match:
                scored_candidates.append((score, int(probes[i])))

        if scored_candidates:
            scored_candidates.sort(reverse=True)
            unique_probes: List[int] = []
            for _, probe in scored_candidates:
                if probe not in unique_probes:
                    unique_probes.append(probe)
                if len(unique_probes) >= top_k_per_combination:
                    break
            candidate_map[combination] = unique_probes
        else:
            candidate_map[combination] = [int(probes[len(probes) // 2])]

    return candidate_map


def generate_mapping_candidates_amplitude_aware(
    models: List[FingerprintModel],
    probes: Array,
    top_k_per_combination: int = 3,
    min_gamma_ratio: float = None,
) -> Dict[Tuple[int, ...], List[int]]:
    if min_gamma_ratio is None:
        min_gamma_ratio = getattr(sys.modules[__name__], 'MIN_GAMMA_RATIO', 0.05)
    """
    幅度感知版本的 hue mapping 候选生成。
    
    不仅要求符号匹配，还要求每个位置的 gamma = sign * z > threshold，
    保证解码时有足够的信噪比。
    """
    z_list = []
    z_max_list = []
    for m in models:
        if m.eff_z is not None:
            z = np.asarray(m.eff_z, dtype=float)
        elif m.Z is not None and m.Z.ndim > 1:
            k = m.Z.shape[1]
            alpha = getattr(sys.modules[__name__], 'ALPHA', None)
            if alpha is not None and k > 1:
                alpha = np.asarray(alpha[:k], dtype=float)
                z = np.asarray(m.Z @ alpha, dtype=float)
            else:
                z = np.asarray(m.Z[:, 0], dtype=float)
        else:
            z = np.asarray(m.z, dtype=float)
        z_list.append(z)
        z_max_list.append(np.max(np.abs(z)))

    possible_combinations = list(itertools.product([1, -1], repeat=len(models)))
    candidate_map: Dict[Tuple[int, ...], List[int]] = {}

    for combination in possible_combinations:
        scored_candidates = []
        fallback_candidates = []
        
        for i in range(len(probes)):
            gammas = []
            all_positive = True
            min_gamma_ratio_actual = float("inf")
            
            for pos_idx, sign in enumerate(combination):
                z_val = z_list[pos_idx][i]
                gamma = int(sign) * float(z_val)
                gammas.append(gamma)
                if gamma <= 0:
                    all_positive = False
                else:
                    ratio = gamma / z_max_list[pos_idx]
                    if ratio < min_gamma_ratio_actual:
                        min_gamma_ratio_actual = ratio
            
            if all_positive and min_gamma_ratio_actual >= min_gamma_ratio:
                # 幅度保证通过，用 min_gamma 评分
                min_gamma = min(gammas)
                avg_gamma = sum(gammas) / len(gammas)
                score = min_gamma * 10.0 + avg_gamma
                scored_candidates.append((score, int(probes[i]), gammas))
            elif all_positive:
                # 符号匹配但幅度不够，fallback
                min_gamma = min(gammas)
                avg_gamma = sum(gammas) / len(gammas)
                score = min_gamma * 10.0 + avg_gamma
                fallback_candidates.append((score, int(probes[i]), gammas))
        
        # 优先用幅度保证通过的
        if scored_candidates:
            scored_candidates.sort(reverse=True, key=lambda x: x[0])
            unique_probes: List[int] = []
            for _, probe, _ in scored_candidates:
                if probe not in unique_probes:
                    unique_probes.append(probe)
                if len(unique_probes) >= top_k_per_combination:
                    break
            candidate_map[combination] = unique_probes
        elif fallback_candidates:
            fallback_candidates.sort(reverse=True, key=lambda x: x[0])
            unique_probes: List[int] = []
            for _, probe, _ in fallback_candidates:
                if probe not in unique_probes:
                    unique_probes.append(probe)
                if len(unique_probes) >= top_k_per_combination:
                    break
            candidate_map[combination] = unique_probes
        else:
            # 最后 fallback：找符号匹配最多的
            scored_candidates = []
            for i in range(len(probes)):
                matched = sum(1 for pos_idx, sign in enumerate(combination) 
                             if int(sign) * z_list[pos_idx][i] > 0)
                margin_sum = sum(abs(int(sign) * z_list[pos_idx][i]) 
                                for pos_idx, sign in enumerate(combination))
                scored_candidates.append((matched * 1000.0 + margin_sum, int(probes[i])))
            scored_candidates.sort(reverse=True)
            candidate_map[combination] = [scored_candidates[0][1]]

    return candidate_map


def build_hue_mapping(
    models: List[FingerprintModel],
    probes: Array,
    mapping_eval_bits: int = 500,
    top_k_per_combination: int = 3,
    use_amplitude_aware: bool = False,
    rng: random.Random | None = None,
) -> Dict[Tuple[int, ...], int]:
    if rng is None:
        rng = random.Random()

    if use_amplitude_aware:
        candidate_map = generate_mapping_candidates_amplitude_aware(
            models=models,
            probes=probes,
            top_k_per_combination=top_k_per_combination,
            min_gamma_ratio=None,  # 使用全局 MIN_GAMMA_RATIO
        )
    else:
        candidate_map = generate_mapping_candidates(
            models=models,
            probes=probes,
            top_k_per_combination=top_k_per_combination,
        )

    hue_mapping = {combination: candidates[0] for combination, candidates in candidate_map.items()}
    if should_use_exact_block_evaluation(len(models)):
        bit_blocks_pm = generate_all_bit_blocks(len(models))
    else:
        bit_blocks_pm = generate_random_bit_blocks(mapping_eval_bits, len(models), rng=rng)

    for combination, candidates in candidate_map.items():
        best_probe = hue_mapping[combination]
        best_ber = float("inf")
        for probe in candidates:
            trial_mapping = dict(hue_mapping)
            trial_mapping[combination] = probe
            ber = evaluate_blocks_ber(models, bit_blocks_pm, trial_mapping)
            if ber < best_ber:
                best_ber = ber
                best_probe = probe
        hue_mapping[combination] = best_probe

    return hue_mapping


def load_csv_matrix(csv_file: str) -> Array:
    if csv_file not in CSV_MATRIX_CACHE:
        CSV_MATRIX_CACHE[csv_file] = pd.read_csv(csv_file).values.astype(float)
    return CSV_MATRIX_CACHE[csv_file]


def load_selected_rows(csv_files: Sequence[str], probes: Array) -> List[Array]:
    row_indices = [int((probe / 5) - 1) for probe in probes]
    matrices = []
    for csv_file in csv_files:
        matrix = load_csv_matrix(csv_file)
        matrices.append(matrix[row_indices].astype(float, copy=False))
    return matrices


def build_models_from_probes(
    csv_files: Sequence[str],
    probes: Array,
    mapping_eval_bits: int = 500,
    mapping_top_k: int = 3,
    use_amplitude_aware: bool = None,
    rng: random.Random | None = None,
) -> Tuple[List[FingerprintModel], Dict[Tuple[int, ...], int]]:
    if use_amplitude_aware is None:
        use_amplitude_aware = getattr(sys.modules[__name__], 'USE_AMPLITUDE_AWARE', False)
    matrices = load_selected_rows(csv_files, probes)
    models = [extract_fingerprint(probes, mat, force_positive_first=True) for mat in matrices]
    models = align_model_directions(models)
    hue_mapping = build_hue_mapping(
        models,
        probes,
        mapping_eval_bits=mapping_eval_bits,
        top_k_per_combination=mapping_top_k,
        use_amplitude_aware=use_amplitude_aware,
        rng=rng,
    )
    return models, hue_mapping


def generate_random_bit_blocks(num_blocks: int, num_positions: int, rng: random.Random | None = None) -> List[Array]:
    if rng is None:
        rng = random.Random()
    return [np.asarray([rng.choice([-1, 1]) for _ in range(num_positions)], dtype=int) for _ in range(num_blocks)]


def evaluate_probe_combination(
    csv_files: Sequence[str],
    probes: Array,
    num_bits: int = 10000,
    mapping_eval_bits: int = 500,
    mapping_top_k: int = 3,
    rng: random.Random | None = None,
    force_random_bits: bool = False,
) -> float:
    models, hue_mapping = build_models_from_probes(
        csv_files,
        probes,
        mapping_eval_bits=mapping_eval_bits,
        mapping_top_k=mapping_top_k,
        rng=rng,
    )
    if USE_CONVOLUTIONAL_FEC:
        info_bits_bin = generate_random_information_bits(num_bits, len(csv_files), rng=rng)
        return evaluate_blocks_ber_with_convolutional_fec(models, info_bits_bin, hue_mapping)

    if should_use_exact_block_evaluation(len(csv_files)) and not force_random_bits:
        bit_blocks_pm = generate_all_bit_blocks(len(csv_files))
    else:
        bit_blocks_pm = generate_random_bit_blocks(num_bits, len(csv_files), rng=rng)
    return evaluate_blocks_ber(models, bit_blocks_pm, hue_mapping)


def validate_zero_ber_candidate(
    csv_files: Sequence[str],
    probes: Sequence[float],
    mapping_eval_bits: int,
    mapping_top_k: int,
    validation_bits: int,
    seed: int,
) -> float:
    return evaluate_probe_combination(
        csv_files=csv_files,
        probes=np.asarray(probes, dtype=float),
        num_bits=validation_bits,
        mapping_eval_bits=mapping_eval_bits,
        mapping_top_k=mapping_top_k,
        rng=random.Random(seed),
        force_random_bits=True,
    )


def is_valid_probe_set(probes: Array, min_interval: int) -> bool:
    sorted_probes = np.sort(np.asarray(probes, dtype=float))
    return len(sorted_probes) <= 1 or np.min(np.diff(sorted_probes)) >= min_interval


def create_eval_cache_key(
    probes: Sequence[float],
    num_positions: int,
    num_bits: int,
    mapping_eval_bits: int,
    mapping_top_k: int,
    seed: int,
) -> tuple:
    if USE_CONVOLUTIONAL_FEC:
        return (
            canonicalize_probe_set(probes),
            int(num_positions),
            int(num_bits),
            int(mapping_eval_bits),
            int(mapping_top_k),
            int(seed),
            "conv_fec",
        )
    if should_use_exact_block_evaluation(num_positions):
        return (
            canonicalize_probe_set(probes),
            int(num_positions),
            "exact",
            int(mapping_top_k),
        )
    return (
        canonicalize_probe_set(probes),
        int(num_positions),
        int(num_bits),
        int(mapping_eval_bits),
        int(mapping_top_k),
        int(seed),
    )


def canonicalize_probe_set(probes: Sequence[float]) -> tuple[float, ...]:
    return tuple(sorted(float(v) for v in probes))


def evaluate_probe_combination_cached(
    csv_files: Sequence[str],
    probes: Sequence[float],
    num_bits: int,
    mapping_eval_bits: int,
    mapping_top_k: int,
    seed: int,
    cache: dict[tuple, float],
) -> float:
    key = create_eval_cache_key(probes, len(csv_files), num_bits, mapping_eval_bits, mapping_top_k, seed)
    if key not in cache:
        cache[key] = evaluate_probe_combination(
            csv_files=csv_files,
            probes=np.asarray(probes, dtype=float),
            num_bits=num_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            rng=random.Random(seed),
        )
    return float(cache[key])


def evaluate_probe_combination_average(
    csv_files: Sequence[str],
    probes: Sequence[float],
    num_bits: int,
    mapping_eval_bits: int,
    mapping_top_k: int,
    base_seed: int,
    repeats: int,
    cache: dict[tuple, float],
) -> float:
    scores = [
        evaluate_probe_combination_cached(
            csv_files=csv_files,
            probes=probes,
            num_bits=num_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            seed=base_seed + idx,
            cache=cache,
        )
        for idx in range(repeats)
    ]
    return float(sum(scores) / len(scores))


def score_single_probe_candidates(
    csv_files: Sequence[str],
    candidate_probes: Sequence[float],
) -> List[float]:
    matrices = load_selected_rows(csv_files, np.asarray(candidate_probes, dtype=float))
    scores: List[float] = []

    for row_idx, _probe in enumerate(candidate_probes):
        probe_vectors = [mat[row_idx] for mat in matrices]
        pairwise_distances = []
        for left in range(len(probe_vectors)):
            for right in range(left + 1, len(probe_vectors)):
                distance = float(np.linalg.norm(probe_vectors[left] - probe_vectors[right]))
                pairwise_distances.append(distance)
        if pairwise_distances:
            scores.append(min(pairwise_distances) + 0.2 * float(np.mean(pairwise_distances)))
        else:
            scores.append(0.0)

    return scores


def select_candidate_probe_pool(
    csv_files: Sequence[str],
    all_probes: Sequence[float],
    min_interval: int,
    pool_size: int,
) -> List[float]:
    scores = score_single_probe_candidates(csv_files, all_probes)
    ranked_pairs = sorted(zip(scores, all_probes), key=lambda item: item[0], reverse=True)

    selected: List[float] = []
    for _score, probe in ranked_pairs:
        if all(abs(float(probe) - existing) >= min_interval for existing in selected):
            selected.append(float(probe))
        if len(selected) >= pool_size:
            break

    for _score, probe in ranked_pairs:
        if len(selected) >= pool_size:
            break
        if float(probe) not in selected:
            selected.append(float(probe))

    return sorted(selected)


def sample_valid_probe_sets(
    all_probes: Sequence[float],
    num_probes: int,
    min_interval: int,
    sample_count: int,
    rng: random.Random,
) -> List[np.ndarray]:
    seen: set[tuple[float, ...]] = set()
    sampled: List[np.ndarray] = []
    max_attempts = max(sample_count * 20, 50)

    for _ in range(max_attempts):
        if len(sampled) >= sample_count:
            break
        probes = np.array(rng.sample(list(all_probes), num_probes), dtype=float)
        probes = np.sort(probes)
        key = canonicalize_probe_set(probes)
        if key in seen or not is_valid_probe_set(probes, min_interval):
            continue
        seen.add(key)
        sampled.append(probes)

    return sampled


def keep_best_candidates(
    csv_files: Sequence[str],
    candidates: Sequence[Array],
    eval_bits: int,
    mapping_eval_bits: int,
    mapping_top_k: int,
    beam_width: int,
    base_seed: int,
    repeat_eval: int,
    cache: dict[tuple, float],
) -> List[Tuple[float, np.ndarray]]:
    scored: List[Tuple[float, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()

    for candidate_idx, candidate in enumerate(candidates):
        key = canonicalize_probe_set(candidate)
        if key in seen:
            continue
        seen.add(key)
        ber = evaluate_probe_combination_average(
            csv_files=csv_files,
            probes=np.asarray(candidate, dtype=float),
            num_bits=eval_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            base_seed=base_seed + candidate_idx * 17,
            repeats=repeat_eval,
            cache=cache,
        )
        scored.append((float(ber), np.sort(np.asarray(candidate, dtype=float))))

    scored.sort(key=lambda item: item[0])
    return scored[: min(beam_width, len(scored))]


def beam_search_probe_selection(
    csv_files: Sequence[str],
    num_probes: int,
    all_probes: Sequence[float],
    min_interval: int,
    coarse_bits: int,
    mapping_eval_bits: int,
    mapping_top_k: int,
    beam_width: int,
    initial_sample_size: int,
    expansion_sample_size: int,
    rng: random.Random,
    base_seed: int,
    repeat_eval: int,
    cache: dict[tuple, float],
) -> List[Tuple[float, np.ndarray]]:
    seed_size = min(4, num_probes)

    if num_probes <= seed_size:
        initial_candidates = sample_valid_probe_sets(
            all_probes=all_probes,
            num_probes=num_probes,
            min_interval=min_interval,
            sample_count=initial_sample_size,
            rng=rng,
        )
        return keep_best_candidates(
            csv_files=csv_files,
            candidates=initial_candidates,
            eval_bits=coarse_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            beam_width=beam_width,
            base_seed=base_seed,
            repeat_eval=repeat_eval,
            cache=cache,
        )

    initial_candidates = sample_valid_probe_sets(
        all_probes=all_probes,
        num_probes=seed_size,
        min_interval=min_interval,
        sample_count=initial_sample_size,
        rng=rng,
    )
    beam = keep_best_candidates(
        csv_files=csv_files,
        candidates=initial_candidates,
        eval_bits=coarse_bits,
        mapping_eval_bits=mapping_eval_bits,
        mapping_top_k=mapping_top_k,
        beam_width=beam_width,
        base_seed=base_seed,
        repeat_eval=repeat_eval,
        cache=cache,
    )

    for target_size in range(seed_size + 1, num_probes + 1):
        expanded_candidates: List[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()

        for _, base_probes in beam:
            pool = [float(p) for p in all_probes if float(p) not in set(base_probes.tolist())]
            if not pool:
                continue
            sample_count = min(expansion_sample_size, len(pool))
            for candidate_probe in rng.sample(pool, sample_count):
                trial = np.sort(np.append(base_probes, float(candidate_probe)))
                key = canonicalize_probe_set(trial)
                if key in seen or not is_valid_probe_set(trial, min_interval):
                    continue
                seen.add(key)
                expanded_candidates.append(trial)

        if not expanded_candidates:
            fallback = sample_valid_probe_sets(
                all_probes=all_probes,
                num_probes=target_size,
                min_interval=min_interval,
                sample_count=max(beam_width, 4),
                rng=rng,
            )
            expanded_candidates.extend(fallback)

        beam = keep_best_candidates(
            csv_files=csv_files,
            candidates=expanded_candidates,
            eval_bits=coarse_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            beam_width=beam_width,
            base_seed=base_seed + target_size * 101,
            repeat_eval=repeat_eval,
            cache=cache,
        )

    return beam


def local_refine_probes(
    csv_files: Sequence[str],
    initial_probes: Array,
    all_probes: Sequence[float],
    coarse_bits: int,
    mapping_eval_bits: int,
    mapping_top_k: int,
    min_interval: int,
    neighborhood_samples: int,
    rounds: int,
    rng: random.Random,
    base_seed: int,
    repeat_eval: int,
    cache: dict[tuple, float],
) -> Tuple[np.ndarray, float]:
    current = np.sort(np.asarray(initial_probes, dtype=float))
    current_ber = evaluate_probe_combination_average(
        csv_files=csv_files,
        probes=current,
        num_bits=coarse_bits,
        mapping_eval_bits=mapping_eval_bits,
        mapping_top_k=mapping_top_k,
        base_seed=base_seed,
        repeats=repeat_eval,
        cache=cache,
    )

    available = [float(v) for v in all_probes]
    for _ in range(rounds):
        improved = False
        for idx in range(len(current)):
            pool = [p for p in available if p not in current]
            if not pool:
                continue
            sample_size = min(neighborhood_samples, len(pool))
            for candidate in rng.sample(pool, sample_size):
                trial = current.copy()
                trial[idx] = candidate
                trial = np.sort(trial)
                if not is_valid_probe_set(trial, min_interval):
                    continue
                trial_ber = evaluate_probe_combination_average(
                    csv_files=csv_files,
                    probes=trial,
                    num_bits=coarse_bits,
                    mapping_eval_bits=mapping_eval_bits,
                    mapping_top_k=mapping_top_k,
                    base_seed=base_seed + idx * 257,
                    repeats=repeat_eval,
                    cache=cache,
                )
                if trial_ber < current_ber:
                    current = trial
                    current_ber = trial_ber
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return current, float(current_ber)


def simulated_annealing_refine_probes(
    csv_files: Sequence[str],
    initial_probes: Array,
    all_probes: Sequence[float],
    eval_bits: int,
    mapping_eval_bits: int,
    mapping_top_k: int,
    min_interval: int,
    iterations: int,
    start_temp: float,
    cooling_rate: float,
    base_seed: int,
    repeat_eval: int,
    rng: random.Random,
    cache: dict[tuple, float],
) -> Tuple[np.ndarray, float]:
    current = np.sort(np.asarray(initial_probes, dtype=float))
    current_ber = evaluate_probe_combination_average(
        csv_files=csv_files,
        probes=current,
        num_bits=eval_bits,
        mapping_eval_bits=mapping_eval_bits,
        mapping_top_k=mapping_top_k,
        base_seed=base_seed,
        repeats=repeat_eval,
        cache=cache,
    )
    best = current.copy()
    best_ber = current_ber
    available = [float(v) for v in all_probes]
    temperature = start_temp

    for step in range(iterations):
        idx = rng.randrange(len(current))
        pool = [probe for probe in available if probe not in current]
        if not pool:
            break

        candidate = float(rng.choice(pool))
        trial = current.copy()
        trial[idx] = candidate
        trial = np.sort(trial)
        if not is_valid_probe_set(trial, min_interval):
            temperature = max(temperature * cooling_rate, 1e-4)
            continue

        trial_ber = evaluate_probe_combination_average(
            csv_files=csv_files,
            probes=trial,
            num_bits=eval_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            base_seed=base_seed + 1000 + step * 31,
            repeats=repeat_eval,
            cache=cache,
        )
        delta = trial_ber - current_ber
        accept = delta <= 0 or rng.random() < np.exp(-delta / max(temperature, 1e-6))
        if accept:
            current = trial
            current_ber = trial_ber
            if current_ber < best_ber:
                best = current.copy()
                best_ber = current_ber

        temperature = max(temperature * cooling_rate, 1e-4)

    return best, float(best_ber)


def staged_beam_probe_selection(
    csv_files: Sequence[str],
    num_probes: int,
    num_bits: int = 10000,
    min_interval: int = 30,
    coarse_bits: int = 800,
    mapping_eval_bits: int = 300,
    mapping_top_k: int = 3,
    neighborhood_samples: int = 8,
    local_rounds: int = 2,
    beam_width: int = 10,
    initial_sample_size: int = 24,
    expansion_sample_size: int = 12,
    finalist_count: int = 6,
    sa_iterations: int = 24,
    repeat_eval: int = 1,
    candidate_pool_size: int = 30,
    base_seed: int = 0,
    rng: random.Random | None = None,
) -> Tuple[np.ndarray, float]:
    if rng is None:
        rng = random.Random()

    first_df = pd.read_csv(csv_files[0])
    all_probes = (5 + np.arange(len(first_df)) * 5).tolist()
    candidate_pool = select_candidate_probe_pool(
        csv_files=csv_files,
        all_probes=all_probes,
        min_interval=min_interval,
        pool_size=max(candidate_pool_size, num_probes + 6),
    )
    eval_cache: dict[tuple, float] = {}
    beam_results = beam_search_probe_selection(
        csv_files=csv_files,
        num_probes=num_probes,
        all_probes=candidate_pool,
        min_interval=min_interval,
        coarse_bits=coarse_bits,
        mapping_eval_bits=mapping_eval_bits,
        mapping_top_k=mapping_top_k,
        beam_width=beam_width,
        initial_sample_size=initial_sample_size,
        expansion_sample_size=expansion_sample_size,
        rng=rng,
        base_seed=base_seed,
        repeat_eval=repeat_eval,
        cache=eval_cache,
    )

    if not beam_results:
        fallback_candidates = sample_valid_probe_sets(
            all_probes=candidate_pool,
            num_probes=num_probes,
            min_interval=min_interval,
            sample_count=1,
            rng=rng,
        )
        fallback = fallback_candidates[0] if fallback_candidates else np.array(rng.sample(candidate_pool, num_probes), dtype=float)
        fallback = np.sort(np.asarray(fallback, dtype=float))
        final_ber = evaluate_probe_combination_average(
            csv_files=csv_files,
            probes=fallback,
            num_bits=num_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            base_seed=base_seed + 7000,
            repeats=max(2, repeat_eval),
            cache=eval_cache,
        )
        return fallback, float(final_ber)

    finalists = beam_results[: min(finalist_count, len(beam_results))]

    best_probes = None
    min_ber = float("inf")
    for _, probes in finalists:
        refined_probes, _ = local_refine_probes(
            csv_files=csv_files,
            initial_probes=probes,
            all_probes=candidate_pool,
            coarse_bits=coarse_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            min_interval=min_interval,
            neighborhood_samples=neighborhood_samples,
            rounds=local_rounds,
            rng=rng,
            base_seed=base_seed + 2000,
            repeat_eval=repeat_eval,
            cache=eval_cache,
        )
        annealed_probes, annealed_ber = simulated_annealing_refine_probes(
            csv_files=csv_files,
            initial_probes=refined_probes,
            all_probes=candidate_pool,
            eval_bits=max(coarse_bits, min(2500, num_bits)),
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            rng=rng,
            min_interval=min_interval,
            iterations=sa_iterations,
            start_temp=0.03,
            cooling_rate=0.92,
            base_seed=base_seed + 4000,
            repeat_eval=repeat_eval,
            cache=eval_cache,
        )
        final_ber = evaluate_probe_combination_average(
            csv_files=csv_files,
            probes=annealed_probes,
            num_bits=num_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            base_seed=base_seed + 6000,
            repeats=max(2, repeat_eval + 1),
            cache=eval_cache,
        )
        if annealed_ber < final_ber:
            final_ber = min(final_ber, annealed_ber)
        if final_ber < min_ber:
            min_ber = final_ber
            best_probes = annealed_probes.copy()

    return np.sort(best_probes), float(min_ber)


def generate_stage_probe_counts(min_probes: int, max_probes: int) -> List[int]:
    coarse_counts = list(range(min_probes, max_probes + 1, 2))
    if coarse_counts[-1] != max_probes:
        coarse_counts.append(max_probes)
    return sorted(set(coarse_counts))


def scaled_search_value(value: int, search_intensity: float, minimum: int) -> int:
    return max(minimum, int(round(float(value) * float(search_intensity))))


def find_optimal_probe_count(
    csv_files: Sequence[str],
    min_probes: int = 5,
    max_probes: int = 20,
    num_bits: int = 10000,
    max_candidates: int = 1000,
    min_interval: int = 30,
    search_intensity: float = 1.0,
    rng: random.Random | None = None,
) -> Tuple[int, np.ndarray, float]:
    if rng is None:
        rng = random.Random()

    zero_validation_bits = max(2000, min(10000, num_bits))

    stage_one_counts = generate_stage_probe_counts(min_probes, max_probes)
    coarse_results: List[Tuple[float, int, np.ndarray]] = []

    for num_probes in stage_one_counts:
        probes, ber = staged_beam_probe_selection(
            csv_files=csv_files,
            num_probes=num_probes,
            num_bits=max(3000, num_bits // 2),
            min_interval=min_interval,
            coarse_bits=min(500, max(200, num_bits // 20)),
            mapping_eval_bits=200,
            mapping_top_k=3,
            neighborhood_samples=scaled_search_value(4, search_intensity, 3),
            local_rounds=scaled_search_value(1, search_intensity, 1),
            beam_width=scaled_search_value(6, search_intensity, 4),
            initial_sample_size=scaled_search_value(12, search_intensity, 8),
            expansion_sample_size=scaled_search_value(8, search_intensity, 6),
            finalist_count=scaled_search_value(3, search_intensity, 2),
            sa_iterations=scaled_search_value(8, search_intensity, 6),
            repeat_eval=scaled_search_value(1, search_intensity, 1),
            candidate_pool_size=min(
                scaled_search_value(20, search_intensity, 16),
                max(scaled_search_value(16, search_intensity, 12), max_candidates // 60),
            ),
            base_seed=10000 + num_probes * 113,
            rng=rng,
        )
        print(f"  stage-1 probe count: {num_probes}, min BER: {ber:.6f}")
        if ber <= 0.0:
            validation_ber = validate_zero_ber_candidate(
                csv_files=csv_files,
                probes=probes,
                mapping_eval_bits=200,
                mapping_top_k=3,
                validation_bits=zero_validation_bits,
                seed=200000 + num_probes * 1009,
            )
            print(f"    zero-BER validation BER: {validation_ber:.6f}")
            if validation_ber <= 0.0:
                best_probes = np.sort(np.asarray(probes, dtype=float))
                print("    validation stayed at BER=0, stopping search early.")
                print(f"  best probe count: {num_probes}, min BER: 0.000000")
                print(f"  best probes: {best_probes}")
                return num_probes, best_probes, 0.0
        coarse_results.append((float(ber), num_probes, probes))

    coarse_results.sort(key=lambda item: item[0])
    top_stage_one = coarse_results[: min(3, len(coarse_results))]
    selected_counts = {num_probes for _, num_probes, _ in top_stage_one}
    for _, num_probes, _ in top_stage_one:
        if num_probes - 1 >= min_probes:
            selected_counts.add(num_probes - 1)
        if num_probes + 1 <= max_probes:
            selected_counts.add(num_probes + 1)

    best_stage_one_ber, best_stage_one_count, best_stage_one_probes = coarse_results[0]

    best_probe_count = best_stage_one_count
    best_probes = np.sort(np.asarray(best_stage_one_probes, dtype=float))
    best_ber = float(best_stage_one_ber)

    for num_probes in sorted(selected_counts):
        probes, ber = staged_beam_probe_selection(
            csv_files=csv_files,
            num_probes=num_probes,
            num_bits=num_bits,
            min_interval=min_interval,
            coarse_bits=min(1000, max(300, num_bits // 10)),
            mapping_eval_bits=350,
            mapping_top_k=3,
            neighborhood_samples=scaled_search_value(6, search_intensity, 4),
            local_rounds=scaled_search_value(2, search_intensity, 1),
            beam_width=scaled_search_value(8, search_intensity, 5),
            initial_sample_size=scaled_search_value(18, search_intensity, 10),
            expansion_sample_size=scaled_search_value(10, search_intensity, 6),
            finalist_count=scaled_search_value(4, search_intensity, 3),
            sa_iterations=scaled_search_value(16, search_intensity, 10),
            repeat_eval=scaled_search_value(1, search_intensity, 1),
            candidate_pool_size=min(
                scaled_search_value(24, search_intensity, 18),
                max(scaled_search_value(18, search_intensity, 14), max_candidates // 40),
            ),
            base_seed=30000 + num_probes * 211,
            rng=rng,
        )
        print(f"  stage-2 probe count: {num_probes}, min BER: {ber:.6f}")
        if ber <= 0.0:
            validation_ber = validate_zero_ber_candidate(
                csv_files=csv_files,
                probes=probes,
                mapping_eval_bits=350,
                mapping_top_k=3,
                validation_bits=zero_validation_bits,
                seed=300000 + num_probes * 1013,
            )
            print(f"    zero-BER validation BER: {validation_ber:.6f}")
            if validation_ber <= 0.0:
                best_probes = np.sort(np.asarray(probes, dtype=float))
                print("    validation stayed at BER=0, stopping search early.")
                print(f"  best probe count: {num_probes}, min BER: 0.000000")
                print(f"  best probes: {best_probes}")
                return num_probes, best_probes, 0.0
        if ber < best_ber:
            best_ber = ber
            best_probe_count = num_probes
            best_probes = np.sort(np.asarray(probes, dtype=float))

    if best_ber > 0.12:
        print("  stage-3 rescue search: deepening search for a difficult position combination...")
        rescue_counts = {best_probe_count}
        if best_probe_count - 1 >= min_probes:
            rescue_counts.add(best_probe_count - 1)
        if best_probe_count + 1 <= max_probes:
            rescue_counts.add(best_probe_count + 1)

        for num_probes in sorted(rescue_counts):
            probes, ber = staged_beam_probe_selection(
                csv_files=csv_files,
                num_probes=num_probes,
                num_bits=max(num_bits, 10000),
                min_interval=min_interval,
                coarse_bits=min(1200, max(400, num_bits // 9)),
                mapping_eval_bits=400,
                mapping_top_k=4,
                neighborhood_samples=scaled_search_value(8, search_intensity, 5),
                local_rounds=scaled_search_value(3, search_intensity, 2),
                beam_width=scaled_search_value(10, search_intensity, 6),
                initial_sample_size=scaled_search_value(20, search_intensity, 12),
                expansion_sample_size=scaled_search_value(12, search_intensity, 8),
                finalist_count=scaled_search_value(5, search_intensity, 3),
                sa_iterations=scaled_search_value(20, search_intensity, 12),
                repeat_eval=scaled_search_value(1, search_intensity, 1),
                candidate_pool_size=min(
                    scaled_search_value(28, search_intensity, 20),
                    max(scaled_search_value(20, search_intensity, 16), max_candidates // 36),
                ),
                base_seed=50000 + num_probes * 307,
                rng=rng,
            )
            print(f"  stage-3 probe count: {num_probes}, min BER: {ber:.6f}")
            if ber <= 0.0:
                validation_ber = validate_zero_ber_candidate(
                    csv_files=csv_files,
                    probes=probes,
                    mapping_eval_bits=400,
                    mapping_top_k=4,
                    validation_bits=zero_validation_bits,
                    seed=500000 + num_probes * 1019,
                )
                print(f"    zero-BER validation BER: {validation_ber:.6f}")
                if validation_ber <= 0.0:
                    best_probes = np.sort(np.asarray(probes, dtype=float))
                    print("    validation stayed at BER=0, stopping search early.")
                    print(f"  best probe count: {num_probes}, min BER: 0.000000")
                    print(f"  best probes: {best_probes}")
                    return num_probes, best_probes, 0.0
            if ber < best_ber:
                best_ber = ber
                best_probe_count = num_probes
                best_probes = np.sort(np.asarray(probes, dtype=float))

    if best_probes is None or best_probe_count is None:
        raise RuntimeError("Failed to find a valid probe set")

    print(f"  best probe count: {best_probe_count}, min BER: {best_ber:.6f}")
    print(f"  best probes: {best_probes}")

    return best_probe_count, best_probes, float(best_ber)


def run_position_experiment(
    csv_files: Sequence[str],
    search_bits: int = 10000,
    test_bits: int = 10000,
    min_probes: int = 5,
    max_probes: int = 20,
    max_candidates: int = 1000,
    seed: int | None = None,
    search_restarts: int = 3,
    min_interval: int = 30,
    search_intensity: float = 1.0,
) -> dict:
    result = run_position_experiment_collect_candidates(
        csv_files=csv_files,
        search_bits=search_bits,
        test_bits=test_bits,
        min_probes=min_probes,
        max_probes=max_probes,
        max_candidates=max_candidates,
        seed=seed,
        search_restarts=search_restarts,
        min_interval=min_interval,
        search_intensity=search_intensity,
        candidate_ber_threshold=None,
        max_collected_candidates=1,
    )
    return result["best_result"]


def run_position_experiment_collect_candidates(
    csv_files: Sequence[str],
    search_bits: int = 10000,
    test_bits: int = 10000,
    min_probes: int = 5,
    max_probes: int = 20,
    max_candidates: int = 1000,
    seed: int | None = None,
    search_restarts: int = 3,
    min_interval: int = 30,
    search_intensity: float = 1.0,
    candidate_ber_threshold: float | None = 0.01,
    max_collected_candidates: int = 12,
) -> dict:
    root_rng = random.Random(seed) if seed is not None else random.Random()

    print("Searching for the best probe combination...")

    best_probe_count = None
    best_probes = None
    best_ber = float("inf")
    collected_candidates: list[dict] = []
    seen_candidates: set[tuple[float, ...]] = set()

    for restart_idx in range(max(1, search_restarts)):
        restart_seed = root_rng.randrange(0, 2**31)
        print(f"  search restart {restart_idx + 1}/{max(1, search_restarts)}")
        probe_count, probes, ber = find_optimal_probe_count(
            csv_files=csv_files,
            min_probes=min_probes,
            max_probes=max_probes,
            num_bits=search_bits,
            max_candidates=max_candidates,
            min_interval=min_interval,
            search_intensity=search_intensity,
            rng=random.Random(restart_seed),
        )
        probes = np.sort(np.asarray(probes, dtype=float))
        candidate_key = canonicalize_probe_set(probes)
        test_rng = random.Random(root_rng.randrange(0, 2**31))
        test_ber = evaluate_probe_combination(
            csv_files=csv_files,
            probes=probes,
            num_bits=test_bits,
            rng=test_rng,
            force_random_bits=True,
        )
        candidate_result = {
            "best_probe_count": probe_count,
            "best_probes": probes,
            "best_ber": float(ber),
            "test_ber": float(test_ber),
            "restart_index": restart_idx + 1,
        }
        if (
            candidate_ber_threshold is not None
            and float(ber) <= float(candidate_ber_threshold)
            and candidate_key not in seen_candidates
        ):
            seen_candidates.add(candidate_key)
            collected_candidates.append(candidate_result)
        if ber < best_ber:
            best_probe_count = probe_count
            best_probes = probes
            best_ber = float(ber)
        if best_ber <= 0.0:
            break

    if best_probe_count is None or best_probes is None:
        raise RuntimeError("Failed to find a valid probe set across search restarts")

    if not any(canonicalize_probe_set(item["best_probes"]) == canonicalize_probe_set(best_probes) for item in collected_candidates):
        test_rng = random.Random(root_rng.randrange(0, 2**31))
        best_test_ber = evaluate_probe_combination(
            csv_files=csv_files,
            probes=best_probes,
            num_bits=test_bits,
            rng=test_rng,
            force_random_bits=True,
        )
    else:
        best_test_ber = next(
            item["test_ber"]
            for item in collected_candidates
            if canonicalize_probe_set(item["best_probes"]) == canonicalize_probe_set(best_probes)
        )

    collected_candidates.sort(key=lambda item: (float(item["best_ber"]), float(item["test_ber"])))
    if max_collected_candidates > 0:
        collected_candidates = collected_candidates[:max_collected_candidates]

    best_result = {
        "best_probe_count": best_probe_count,
        "best_probes": np.asarray(best_probes, dtype=float),
        "best_ber": float(best_ber),
        "test_ber": float(best_test_ber),
    }
    return {
        "best_result": best_result,
        "candidate_results": collected_candidates,
    }


def build_csv_files_for_positions(project_root: str, positions: Sequence[int], light_condition: str = "white") -> List[str]:
    return [
        os.path.join(project_root, "data", "15pro", light_condition, f"{pos}.csv")
        for pos in positions
    ]


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    positions = (1, 4, 5, 7)
    csv_files = build_csv_files_for_positions(project_root, positions, light_condition="white")

    result = run_position_experiment(csv_files)
    print(f"Position combination: {positions}")
    print(f"Best probe count: {result['best_probe_count']}")
    print(f"Probe combination: {result['best_probes']}")
    print(f"Best BER: {result['best_ber']:.6f}")
    print(f"Test BER: {result['test_ber']:.6f}")


if __name__ == "__main__":
    main()
