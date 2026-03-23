#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位置指纹通信实验脚本。

功能：
1. 对给定的位置组合搜索最优探针数量与探针组合。
2. 搜索标准为：随机发送 10000 个 bit block 时 BER 最小。
3. 用搜索得到的最优探针组合，再重新随机生成 10000 个 bit block 做测试。
4. 返回最优 BER 和测试 BER，供 batch_test.py 批量调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import itertools
import os
import random

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


def extract_fingerprint(x: Array, Y: Array, force_positive_first: bool = True) -> FingerprintModel:
    _, trend = fit_linear_trend(x, Y)
    residual = np.asarray(Y, dtype=float) - trend

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


def calculate_correlation(c1: Array, c2: Array) -> float:
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    if len(c1) != len(c2):
        raise ValueError("地址码长度必须相同")

    std1 = np.std(c1)
    std2 = np.std(c2)
    if std1 == 0 or std2 == 0:
        return 0.0
    return float(np.cov(c1, c2)[0, 1] / (std1 * std2))


def align_model_directions(models: List[FingerprintModel]) -> List[FingerprintModel]:
    if not models:
        return models

    ref = models[0]
    for idx in range(1, len(models)):
        corr = calculate_correlation(ref.code, models[idx].code)
        if corr < 0:
            models[idx].w = -models[idx].w
            models[idx].z = -models[idx].z
            models[idx].code = -models[idx].code
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
            raise KeyError(f"符号组合 {key} 不在 hue_mapping 中")
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


def decode_local_block(Y_obs: Array, w: Array, code: Array) -> DecodeResult:
    Y_obs = np.asarray(Y_obs, dtype=float)
    w = np.asarray(w, dtype=float)
    code = np.asarray(code, dtype=float)

    mean_vec = Y_obs.mean(axis=0)
    Y_centered = Y_obs - mean_vec
    u = Y_centered @ w
    gamma = float(code @ u)
    bit_hat_pm = 1 if gamma > 0 else -1
    bit_hat_bin = 1 if bit_hat_pm > 0 else 0
    return DecodeResult(bit_hat_pm=bit_hat_pm, bit_hat_bin=bit_hat_bin)


def simulate_blocks(
    models: List[FingerprintModel],
    bit_blocks_pm: List[Array],
    hue_mapping: Dict[Tuple[int, ...], int],
) -> List[dict]:
    codes = [m.code for m in models]
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
            dec = decode_local_block(Y_obs, model.w, model.code)
            block_info["per_position"].append(dec)

        results.append(block_info)

    return results


def calculate_ber(results: List[dict]) -> float:
    if not results:
        return 0.0

    total_bits = 0
    error_bits = 0
    position_errors = [0] * len(results[0]["per_position"])
    position_total = [0] * len(results[0]["per_position"])

    for res in results:
        bits_tx = res["bits_bin"]
        bits_rx = [dec.bit_hat_bin for dec in res["per_position"]]

        for i, (tx, rx) in enumerate(zip(bits_tx, bits_rx)):
            total_bits += 1
            position_total[i] += 1
            if tx != rx:
                error_bits += 1
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
    z_list = [m.z for m in models]
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


def build_hue_mapping(
    models: List[FingerprintModel],
    probes: Array,
    mapping_eval_bits: int = 500,
    top_k_per_combination: int = 3,
    rng: random.Random | None = None,
) -> Dict[Tuple[int, ...], int]:
    if rng is None:
        rng = random.Random()

    candidate_map = generate_mapping_candidates(
        models=models,
        probes=probes,
        top_k_per_combination=top_k_per_combination,
    )

    # 先按静态分数取每个组合的首选 probe，再用小批量 BER 做贪心细化。
    hue_mapping = {combination: candidates[0] for combination, candidates in candidate_map.items()}
    bit_blocks_pm = generate_random_bit_blocks(mapping_eval_bits, len(models), rng=rng)

    for combination, candidates in candidate_map.items():
        best_probe = hue_mapping[combination]
        best_ber = float("inf")
        for probe in candidates:
            trial_mapping = dict(hue_mapping)
            trial_mapping[combination] = probe
            results = simulate_blocks(models, bit_blocks_pm, trial_mapping)
            ber = calculate_ber(results)
            if ber < best_ber:
                best_ber = ber
                best_probe = probe
        hue_mapping[combination] = best_probe

    return hue_mapping


def load_selected_rows(csv_files: Sequence[str], probes: Array) -> List[Array]:
    row_indices = [int((probe / 5) - 1) for probe in probes]
    matrices = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        matrices.append(df.iloc[row_indices].values.astype(float))
    return matrices


def build_models_from_probes(
    csv_files: Sequence[str],
    probes: Array,
    mapping_eval_bits: int = 500,
    mapping_top_k: int = 3,
    rng: random.Random | None = None,
) -> Tuple[List[FingerprintModel], Dict[Tuple[int, ...], int]]:
    matrices = load_selected_rows(csv_files, probes)
    models = [extract_fingerprint(probes, mat, force_positive_first=True) for mat in matrices]
    models = align_model_directions(models)
    hue_mapping = build_hue_mapping(
        models,
        probes,
        mapping_eval_bits=mapping_eval_bits,
        top_k_per_combination=mapping_top_k,
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
) -> float:
    models, hue_mapping = build_models_from_probes(
        csv_files,
        probes,
        mapping_eval_bits=mapping_eval_bits,
        mapping_top_k=mapping_top_k,
        rng=rng,
    )
    bit_blocks_pm = generate_random_bit_blocks(num_bits, len(csv_files), rng=rng)
    results = simulate_blocks(models, bit_blocks_pm, hue_mapping)
    return calculate_ber(results)


def is_valid_probe_set(probes: Array, min_interval: int) -> bool:
    sorted_probes = np.sort(np.asarray(probes, dtype=float))
    return len(sorted_probes) <= 1 or np.min(np.diff(sorted_probes)) >= min_interval


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
) -> Tuple[np.ndarray, float]:
    current = np.sort(np.asarray(initial_probes, dtype=float))
    current_ber = evaluate_probe_combination(
        csv_files,
        current,
        num_bits=coarse_bits,
        mapping_eval_bits=mapping_eval_bits,
        mapping_top_k=mapping_top_k,
        rng=rng,
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
                trial_ber = evaluate_probe_combination(
                    csv_files,
                    trial,
                    num_bits=coarse_bits,
                    mapping_eval_bits=mapping_eval_bits,
                    mapping_top_k=mapping_top_k,
                    rng=rng,
                )
                if trial_ber < current_ber:
                    current = trial
                    current_ber = trial_ber
                    improved = True
        if not improved:
            break

    return current, float(current_ber)


def brute_force_probe_selection(
    csv_files: Sequence[str],
    num_probes: int,
    num_bits: int = 10000,
    max_candidates: int = 1000,
    min_interval: int = 30,
    coarse_bits: int = 800,
    coarse_keep: int = 20,
    mapping_eval_bits: int = 300,
    mapping_top_k: int = 3,
    neighborhood_samples: int = 8,
    local_rounds: int = 2,
    rng: random.Random | None = None,
) -> Tuple[np.ndarray, float]:
    if rng is None:
        rng = random.Random()

    first_df = pd.read_csv(csv_files[0])
    all_probes = (5 + np.arange(len(first_df)) * 5).tolist()
    coarse_results: List[Tuple[float, np.ndarray]] = []

    for _ in range(max_candidates):
        probes = np.array(rng.sample(all_probes, num_probes), dtype=float)
        if not is_valid_probe_set(probes, min_interval):
            continue

        ber = evaluate_probe_combination(
            csv_files,
            probes,
            num_bits=coarse_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            rng=rng,
        )
        coarse_results.append((float(ber), np.sort(probes.copy())))

    if not coarse_results:
        fallback = np.array(rng.sample(all_probes, num_probes), dtype=float)
        fallback = np.sort(fallback)
        final_ber = evaluate_probe_combination(
            csv_files,
            fallback,
            num_bits=num_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            rng=rng,
        )
        return fallback, float(final_ber)

    coarse_results.sort(key=lambda item: item[0])
    finalists = coarse_results[: min(coarse_keep, len(coarse_results))]

    best_probes = None
    min_ber = float("inf")
    for _, probes in finalists:
        refined_probes, _ = local_refine_probes(
            csv_files=csv_files,
            initial_probes=probes,
            all_probes=all_probes,
            coarse_bits=coarse_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            min_interval=min_interval,
            neighborhood_samples=neighborhood_samples,
            rounds=local_rounds,
            rng=rng,
        )
        final_ber = evaluate_probe_combination(
            csv_files,
            refined_probes,
            num_bits=num_bits,
            mapping_eval_bits=mapping_eval_bits,
            mapping_top_k=mapping_top_k,
            rng=rng,
        )
        if final_ber < min_ber:
            min_ber = final_ber
            best_probes = refined_probes.copy()

    return np.sort(best_probes), float(min_ber)


def find_optimal_probe_count(
    csv_files: Sequence[str],
    min_probes: int = 5,
    max_probes: int = 20,
    num_bits: int = 10000,
    max_candidates: int = 1000,
    rng: random.Random | None = None,
) -> Tuple[int, np.ndarray, float]:
    if rng is None:
        rng = random.Random()

    best_probe_count = min_probes
    best_probes = None
    best_ber = float("inf")

    for num_probes in range(min_probes, max_probes + 1):
        probes, ber = brute_force_probe_selection(
            csv_files,
            num_probes=num_probes,
            num_bits=num_bits,
            max_candidates=max_candidates,
            coarse_bits=min(800, max(200, num_bits // 10)),
            coarse_keep=20,
            mapping_eval_bits=300,
            mapping_top_k=3,
            neighborhood_samples=8,
            local_rounds=2,
            rng=rng,
        )
        print(f"  探针数量: {num_probes}, 最小BER: {ber:.6f}")
        if ber < best_ber:
            best_ber = ber
            best_probe_count = num_probes
            best_probes = probes

    if best_probes is None:
        raise RuntimeError("未能找到有效的探针组合")

    print(f"  最优探针数量: {best_probe_count}, 最小BER: {best_ber:.6f}")
    print(f"  最优探针组合: {best_probes}")

    return best_probe_count, best_probes, float(best_ber)


def run_position_experiment(
    csv_files: Sequence[str],
    search_bits: int = 10000,
    test_bits: int = 10000,
    min_probes: int = 5,
    max_probes: int = 20,
    max_candidates: int = 1000,
    seed: int | None = None,
) -> dict:
    rng = random.Random(seed)

    print("开始搜索最优探针组合...")

    best_probe_count, best_probes, best_ber = find_optimal_probe_count(
        csv_files=csv_files,
        min_probes=min_probes,
        max_probes=max_probes,
        num_bits=search_bits,
        max_candidates=max_candidates,
        rng=rng,
    )

    test_rng = random.Random(None if seed is None else seed + 1)
    test_ber = evaluate_probe_combination(
        csv_files=csv_files,
        probes=best_probes,
        num_bits=test_bits,
        rng=test_rng,
    )

    return {
        "best_probe_count": best_probe_count,
        "best_probes": np.asarray(best_probes, dtype=float),
        "best_ber": float(best_ber),
        "test_ber": float(test_ber),
    }


def build_csv_files_for_positions(project_root: str, positions: Sequence[int], light_condition: str = "white") -> List[str]:
    return [
        os.path.join(project_root, "data", "15pro", light_condition, f"{pos}.csv")
        for pos in positions
    ]


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    positions = (1, 4, 5)
    csv_files = build_csv_files_for_positions(project_root, positions, light_condition="white")

    result = run_position_experiment(csv_files)
    print(f"位置组合: {positions}")
    print(f"最优探针数量: {result['best_probe_count']}")
    print(f"探针组合: {result['best_probes']}")
    print(f"最优BER: {result['best_ber']:.6f}")
    print(f"测试BER: {result['test_ber']:.6f}")


if __name__ == "__main__":
    main()
