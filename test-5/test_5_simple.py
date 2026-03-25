#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Five-position fingerprint experiment script.

Features:
1. Search for the best probe count and probe set for a given 5-position combination.
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

import numpy as np
import pandas as pd

Array = np.ndarray
EXACT_ENUMERATION_MAX_POSITIONS = 8
CSV_MATRIX_CACHE: dict[str, Array] = {}


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


def should_use_exact_block_evaluation(num_positions: int) -> bool:
    return num_positions <= EXACT_ENUMERATION_MAX_POSITIONS


def generate_all_bit_blocks(num_positions: int) -> List[Array]:
    return [
        np.asarray(bits_pm, dtype=int)
        for bits_pm in itertools.product([-1, 1], repeat=num_positions)
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


def evaluate_blocks_ber(
    models: List[FingerprintModel],
    bit_blocks_pm: Sequence[Array],
    hue_mapping: Dict[Tuple[int, ...], int],
) -> float:
    if not bit_blocks_pm:
        return 0.0

    codes = [m.code for m in models]
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
            dec = decode_local_block(Y_obs, model.w, model.code)
            position_total[model_idx] += 1
            if bits_tx[model_idx] != dec.bit_hat_bin:
                position_errors[model_idx] += 1

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
    force_random_bits: bool = False,
) -> float:
    models, hue_mapping = build_models_from_probes(
        csv_files,
        probes,
        mapping_eval_bits=mapping_eval_bits,
        mapping_top_k=mapping_top_k,
        rng=rng,
    )
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

    zero_validation_bits = max(2000, min(10000, num_bits))

    stage_one_counts = generate_stage_probe_counts(min_probes, max_probes)
    coarse_results: List[Tuple[float, int, np.ndarray]] = []

    for num_probes in stage_one_counts:
        probes, ber = staged_beam_probe_selection(
            csv_files=csv_files,
            num_probes=num_probes,
            num_bits=max(3000, num_bits // 2),
            min_interval=30,
            coarse_bits=min(500, max(200, num_bits // 20)),
            mapping_eval_bits=200,
            mapping_top_k=3,
            neighborhood_samples=4,
            local_rounds=1,
            beam_width=6,
            initial_sample_size=12,
            expansion_sample_size=8,
            finalist_count=3,
            sa_iterations=8,
            repeat_eval=1,
            candidate_pool_size=min(20, max(16, max_candidates // 60)),
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
            min_interval=30,
            coarse_bits=min(1000, max(300, num_bits // 10)),
            mapping_eval_bits=350,
            mapping_top_k=3,
            neighborhood_samples=6,
            local_rounds=2,
            beam_width=8,
            initial_sample_size=18,
            expansion_sample_size=10,
            finalist_count=4,
            sa_iterations=16,
            repeat_eval=1,
            candidate_pool_size=min(24, max(18, max_candidates // 40)),
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
                min_interval=30,
                coarse_bits=min(1200, max(400, num_bits // 9)),
                mapping_eval_bits=400,
                mapping_top_k=4,
                neighborhood_samples=8,
                local_rounds=3,
                beam_width=10,
                initial_sample_size=20,
                expansion_sample_size=12,
                finalist_count=5,
                sa_iterations=20,
                repeat_eval=1,
                candidate_pool_size=min(28, max(20, max_candidates // 36)),
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
) -> dict:
    root_rng = random.Random(seed) if seed is not None else random.Random()

    print("Searching for the best probe combination...")

    best_probe_count = None
    best_probes = None
    best_ber = float("inf")

    for restart_idx in range(max(1, search_restarts)):
        restart_seed = root_rng.randrange(0, 2**31)
        print(f"  search restart {restart_idx + 1}/{max(1, search_restarts)}")
        probe_count, probes, ber = find_optimal_probe_count(
            csv_files=csv_files,
            min_probes=min_probes,
            max_probes=max_probes,
            num_bits=search_bits,
            max_candidates=max_candidates,
            rng=random.Random(restart_seed),
        )
        if ber < best_ber:
            best_probe_count = probe_count
            best_probes = np.sort(np.asarray(probes, dtype=float))
            best_ber = float(ber)
        if best_ber <= 0.0:
            break

    if best_probe_count is None or best_probes is None:
        raise RuntimeError("Failed to find a valid probe set across search restarts")

    test_rng = random.Random(root_rng.randrange(0, 2**31))
    test_ber = evaluate_probe_combination(
        csv_files=csv_files,
        probes=best_probes,
        num_bits=test_bits,
        rng=test_rng,
        force_random_bits=True,
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
    positions = (1, 4, 5, 7, 9)
    csv_files = build_csv_files_for_positions(project_root, positions, light_condition="white")

    result = run_position_experiment(csv_files)
    print(f"Position combination: {positions}")
    print(f"Best probe count: {result['best_probe_count']}")
    print(f"Probe combination: {result['best_probes']}")
    print(f"Best BER: {result['best_ber']:.6f}")
    print(f"Test BER: {result['test_ber']:.6f}")


if __name__ == "__main__":
    main()

