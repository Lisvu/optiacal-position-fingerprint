#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple
import math

import numpy as np
import pandas as pd

Array = np.ndarray
VectorMapping = Dict[tuple, Tuple[int, int, int, int]]


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


@dataclass
class MappingSearchResult:
    hue_mapping: Dict[tuple, int]
    accuracy: float
    min_margin: float
    avg_margin: float
    per_block_margins: List[Tuple[Tuple[int, ...], List[float]]]


@dataclass
class CodeSubsetSearchResult:
    selected_indices: Tuple[int, ...]
    max_code_correlation: float
    min_symbol_distance: float
    balance_penalty: float


@dataclass
class MultiSVDModel:
    probes: np.ndarray
    Y: np.ndarray
    W: np.ndarray
    C: np.ndarray


@dataclass
class AdaptiveDecoderBundle:
    mode: str
    use_dim: int
    models: List[Tuple[np.ndarray, float, np.ndarray, np.ndarray]] | None
    calibration_ber: float
    baseline_calibration_ber: float
    logistic_top2_calibration_ber: float
    logistic_top3_calibration_ber: float


@dataclass
class GenericConvSpec:
    code_name: str
    generators: Tuple[int, ...]
    constraint_len: int


@dataclass
class RandomConvRunResult:
    k: int
    target: Tuple[int, ...]
    num_probes: int
    conv_code: str
    rate: float
    decoder_mode: str
    calibration_ber: float
    authorized_uncoded_ber: float
    authorized_postfec_ber: float
    worst_unauthorized_postfec_ber: float
    avg_unauthorized_postfec_ber: float
    median_unauthorized_postfec_ber: float
    leak_positions_le_0p05: int
    leak_positions_le_0p10: int
    evaluated_unauthorized_positions: int


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


def extract_fingerprint(
    x: Array,
    Y: Array,
    force_positive_first: bool = True,
    verbose: bool = False,
) -> FingerprintModel:
    coeffs, trend = fit_linear_trend(x, Y)
    residual = Y - trend
    if verbose:
        _ = coeffs
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
    assert len(c1) == len(c2), "Codes must have the same length."
    std1 = np.std(c1)
    std2 = np.std(c2)
    if std1 == 0 or std2 == 0:
        return 0.0
    return float(np.corrcoef(c1, c2)[0, 1])


def calculate_max_code_correlation(codes: List[Array]) -> float:
    correlations = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            correlations.append(abs(calculate_correlation(codes[i], codes[j])))
    return max(correlations) if correlations else 0.0


def calculate_code_balance_penalty(codes: List[Array]) -> float:
    penalties = []
    for code in codes:
        code = np.asarray(code, dtype=float)
        penalties.append(abs(float(np.sum(code))) / len(code))
    return float(np.mean(penalties)) if penalties else 0.0


def calculate_min_symbol_distance(codes: List[Array]) -> float:
    bit_combinations = list(product([1, -1], repeat=len(codes)))
    symbol_sequences = []
    for bits in bit_combinations:
        symbol = np.zeros_like(codes[0], dtype=int)
        for b, c in zip(bits, codes):
            symbol += int(b) * np.asarray(c, dtype=int)
        symbol_sequences.append(symbol)

    min_distance = float("inf")
    for i in range(len(symbol_sequences)):
        for j in range(i + 1, len(symbol_sequences)):
            distance = float(np.sum(np.abs(symbol_sequences[i] - symbol_sequences[j])))
            if distance < min_distance:
                min_distance = distance
    return min_distance if min_distance != float("inf") else 0.0


def _codes_have_duplicate_or_antipodal_pair(codes: List[Array]) -> bool:
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            if np.array_equal(codes[i], codes[j]) or np.array_equal(codes[i], -codes[j]):
                return True
    return False


def restrict_models_to_subset(models: List[FingerprintModel], selected_indices: Tuple[int, ...]) -> List[FingerprintModel]:
    subset_models: List[FingerprintModel] = []
    for model in models:
        subset_models.append(
            FingerprintModel(
                probes=model.probes[list(selected_indices)],
                Y=model.Y[list(selected_indices)],
                trend=model.trend[list(selected_indices)],
                residual=model.residual[list(selected_indices)],
                w=model.w,
                z=model.z[list(selected_indices)],
                code=model.code[list(selected_indices)],
            )
        )
    return subset_models


def search_best_probe_subset(
    models: List[FingerprintModel],
    target_subset_size: int = 8,
    show_progress: bool = False,
    progress_label: str = "",
) -> CodeSubsetSearchResult:
    total_probes = len(models[0].probes)
    if target_subset_size >= total_probes:
        codes = [m.code for m in models]
        return CodeSubsetSearchResult(
            selected_indices=tuple(range(total_probes)),
            max_code_correlation=calculate_max_code_correlation(codes),
            min_symbol_distance=calculate_min_symbol_distance(codes),
            balance_penalty=calculate_code_balance_penalty(codes),
        )

    best_result: CodeSubsetSearchResult | None = None
    best_key: Tuple[int, float, float, float, float] | None = None
    total_combinations = math.comb(total_probes, target_subset_size)
    progress_step = max(1, total_combinations // 5)
    if show_progress:
        prefix = f"{progress_label} " if progress_label else ""
        print(f"{prefix}probe subset search start: C({total_probes}, {target_subset_size})={total_combinations}")

    for idx, selected_indices in enumerate(combinations(range(total_probes), target_subset_size), start=1):
        selected_codes = [m.code[list(selected_indices)] for m in models]
        max_corr = calculate_max_code_correlation(selected_codes)
        min_symbol_distance = calculate_min_symbol_distance(selected_codes)
        balance_penalty = calculate_code_balance_penalty(selected_codes)
        duplicate_penalty = 0 if _codes_have_duplicate_or_antipodal_pair(selected_codes) else 1
        avg_abs_z = float(np.mean([np.mean(np.abs(m.z[list(selected_indices)])) for m in models]))

        key = (
            duplicate_penalty,
            min_symbol_distance,
            -max_corr,
            -balance_penalty,
            avg_abs_z,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_result = CodeSubsetSearchResult(
                selected_indices=tuple(selected_indices),
                max_code_correlation=max_corr,
                min_symbol_distance=min_symbol_distance,
                balance_penalty=balance_penalty,
            )
        if show_progress and (idx == 1 or idx % progress_step == 0 or idx == total_combinations):
            prefix = f"{progress_label} " if progress_label else ""
            print(
                f"{prefix}probe subset search progress {idx}/{total_combinations}, "
                f"current best max|corr|={best_result.max_code_correlation:.4f}, "
                f"min_sym_dist={best_result.min_symbol_distance:.2f}, "
                f"balance_penalty={best_result.balance_penalty:.4f}"
            )

    if best_result is None:
        raise RuntimeError("Failed to find a valid probe subset for code design.")
    return best_result


def build_symbol_sequence(bits_pm: Array, codes: List[Array]) -> Tuple[Array, List[List[int]]]:
    bits_pm = np.asarray(bits_pm, dtype=int)
    out = np.zeros_like(codes[0], dtype=int)
    symbol_combinations = []
    for i in range(len(codes[0])):
        combination = []
        for b, c in zip(bits_pm, codes):
            contribution = int(b) * int(c[i])
            combination.append(contribution)
            out[i] += contribution
        symbol_combinations.append(combination)
    return out, symbol_combinations


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
    return DecodeResult(
        Y_obs=Y_obs,
        mean_vec=mean_vec,
        Y_centered=Y_centered,
        u=u,
        gamma=gamma,
        bit_hat_pm=bit_hat_pm,
        bit_hat_bin=bit_hat_bin,
    )


def generate_probes(num_probes: int, max_row_index: int, verbose: bool = False) -> np.ndarray:
    if num_probes < 2:
        raise ValueError("num_probes must be at least 2.")
    theoretical_interval = 360 / (num_probes - 1)
    interval = round(theoretical_interval / 5) * 5
    probes = []
    for i in range(num_probes):
        probe = 5 + i * interval
        max_probe = (max_row_index + 1) * 5
        if probe > max_probe:
            probe = max_probe
        probes.append(probe)
    if verbose:
        print(f"generated probes: {probes}")
    return np.array(probes, dtype=float)


def generate_random_bit_blocks(
    num_positions: int,
    num_blocks: int,
    rng: np.random.Generator,
) -> List[Array]:
    bit_blocks_pm: List[Array] = []
    for _ in range(num_blocks):
        bits_bin = rng.integers(0, 2, size=num_positions)
        bits_pm = np.where(bits_bin > 0, 1, -1).astype(int)
        bit_blocks_pm.append(bits_pm)
    return bit_blocks_pm


def list_position_files(dataset_dir: str) -> List[Tuple[int, str]]:
    path = Path(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    position_files: List[Tuple[int, str]] = []
    for csv_path in path.glob("*.csv"):
        try:
            position_id = int(csv_path.stem)
        except ValueError:
            continue
        position_files.append((position_id, str(csv_path)))
    position_files.sort(key=lambda x: x[0])
    if len(position_files) < 3:
        raise ValueError(f"Need at least 3 csv files in dataset directory: {dataset_dir}")
    return position_files


def calculate_position_distance(pos1: int, pos2: int) -> float:
    row1 = (pos1 - 1) // 7
    col1 = (pos1 - 1) % 7
    row2 = (pos2 - 1) // 7
    col2 = (pos2 - 1) % 7
    return float(((row1 - row2) ** 2 + (col1 - col2) ** 2) ** 0.5)


def load_single_position_model(csv_file: str, probes: np.ndarray):
    df = pd.read_csv(csv_file)
    row_indices = [int((float(probe) / 5) - 1) for probe in probes]
    Y = df.iloc[row_indices].values.astype(float)
    if np.any(Y < 0):
        bad_rows = [int(probes[i]) for i in np.where(np.any(Y < 0, axis=1))[0]]
        raise ValueError(f"{csv_file} contains negative placeholder/invalid values at probes {bad_rows}")
    return extract_fingerprint(np.asarray(probes, dtype=float), Y, verbose=False)


def compute_sequence_ber(tx_bits: List[int], rx_bits: List[int]) -> Tuple[float, int, int]:
    total = len(tx_bits)
    errors = sum(int(int(tx) != int(rx)) for tx, rx in zip(tx_bits, rx_bits))
    ber = errors / total if total > 0 else 0.0
    return ber, errors, total


def compute_leakage_metrics_from_decoded_bits(
    decoded_bits: List[int],
    true_stream_bits: List[List[int]],
) -> Dict[str, object]:
    per_stream_ber: List[float] = []
    best_stream_index = -1
    best_stream_ber = float("inf")
    best_stream_inverted = False

    for stream_idx, tx_bits in enumerate(true_stream_bits, start=1):
        ber, _, _ = compute_sequence_ber(tx_bits, decoded_bits)
        ber_inverted = 1.0 - ber
        effective_ber = min(ber, ber_inverted)
        inverted = bool(ber_inverted < ber)
        per_stream_ber.append(effective_ber)

        if effective_ber < best_stream_ber:
            best_stream_ber = effective_ber
            best_stream_index = stream_idx
            best_stream_inverted = inverted

    avg_all_stream_ber = float(np.mean(per_stream_ber)) if per_stream_ber else float("nan")
    median_all_stream_ber = float(np.median(per_stream_ber)) if per_stream_ber else float("nan")
    min_stream_ber = float(np.min(per_stream_ber)) if per_stream_ber else float("nan")
    max_stream_ber = float(np.max(per_stream_ber)) if per_stream_ber else float("nan")

    return {
        "primary_metric_ber": avg_all_stream_ber,
        "all_stream_avg_ber": avg_all_stream_ber,
        "all_stream_median_ber": median_all_stream_ber,
        "min_stream_ber": min_stream_ber,
        "max_stream_ber": max_stream_ber,
        "best_stream_ber": float(best_stream_ber),
        "best_stream_index": int(best_stream_index),
        "best_stream_inverted": bool(best_stream_inverted),
        "per_stream_ber": per_stream_ber,
    }


def load_n_position_data(
    csv_files: Sequence[str],
    num_probes: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    first_df = pd.read_csv(csv_files[0])
    max_row_index = len(first_df) - 1
    probes = generate_probes(num_probes, max_row_index, verbose=False)
    if len(np.unique(probes)) != len(probes):
        raise ValueError(f"Duplicate probes generated: {probes.tolist()}")

    row_indices = [int((float(probe) / 5) - 1) for probe in probes]
    Y_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        Y = df.iloc[row_indices].values.astype(float)
        if np.any(Y < 0):
            bad_rows = [int(probes[i]) for i in np.where(np.any(Y < 0, axis=1))[0]]
            raise ValueError(f"{csv_file} contains invalid negative values at probes {bad_rows}")
        Y_list.append(Y)
    return np.asarray(probes, dtype=float), Y_list


def prepare_n_position_models(
    csv_files: Sequence[str],
    num_probes: int,
    target_subset_size: int,
) -> Tuple[List[FingerprintModel], Tuple[int, ...], np.ndarray]:
    probes, Y_list = load_n_position_data(csv_files, num_probes=num_probes)
    models = [extract_fingerprint(probes, Y, verbose=False) for Y in Y_list]
    if target_subset_size >= len(probes):
        selected_indices = tuple(range(len(probes)))
        subset_models = models
    else:
        subset_result = search_best_probe_subset(
            models,
            target_subset_size=min(target_subset_size, len(probes)),
            show_progress=False,
            progress_label="",
        )
        selected_indices = subset_result.selected_indices
        subset_models = restrict_models_to_subset(models, selected_indices)
    return subset_models, selected_indices, probes


def evaluate_n_position_mapping(
    models: List[FingerprintModel],
    hue_mapping: Dict[tuple, int],
    bit_blocks_pm: List[np.ndarray] | None = None,
) -> MappingSearchResult:
    n = len(models)
    if bit_blocks_pm is None:
        bit_blocks_pm = [np.array(bits, dtype=int) for bits in product([1, -1], repeat=n)]

    results = simulate_blocks_scalar(models, bit_blocks_pm, hue_mapping)
    correct = 0
    total = 0
    margins: List[float] = []
    per_block_margins = []

    for bits_pm, result in zip(bit_blocks_pm, results):
        current_margins = []
        for pos_idx, dec in enumerate(result["per_position"]):
            true_pm = int(bits_pm[pos_idx])
            margin = float(true_pm * dec.gamma)
            margins.append(margin)
            current_margins.append(margin)
            correct += int(dec.bit_hat_pm == true_pm)
            total += 1
        per_block_margins.append((tuple(int(v) for v in bits_pm.tolist()), current_margins))

    accuracy = correct / total if total > 0 else 0.0
    min_margin = min(margins) if margins else float("-inf")
    avg_margin = float(np.mean(margins)) if margins else float("-inf")
    return MappingSearchResult(
        hue_mapping=dict(hue_mapping),
        accuracy=accuracy,
        min_margin=min_margin,
        avg_margin=avg_margin,
        per_block_margins=per_block_margins,
    )


def build_top_probe_candidates_for_n_positions(
    models: List[FingerprintModel],
    top_k: int,
) -> Dict[tuple, List[int]]:
    probes = [int(v) for v in np.asarray(models[0].probes).tolist()]
    z_list = [m.z for m in models]
    possible_combinations = [tuple(bits) for bits in product([1, -1], repeat=len(models))]
    fallback_probe = int(probes[len(probes) // 2])
    candidate_dict: Dict[tuple, List[int]] = {}

    for combination in possible_combinations:
        scored: List[Tuple[float, int]] = []
        for idx, probe in enumerate(probes):
            signs_match = all(
                (
                    (combination[pos] > 0 and z_list[pos][idx] > 0)
                    or (combination[pos] < 0 and z_list[pos][idx] < 0)
                )
                for pos in range(len(models))
            )
            if signs_match:
                score = sum(abs(z_list[pos][idx]) for pos in range(len(models)))
                scored.append((float(score), int(probe)))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidates: List[int] = []
        for _, probe in scored:
            if probe not in candidates:
                candidates.append(probe)
            if len(candidates) >= top_k:
                break
        if not candidates:
            candidates = [fallback_probe]
        candidate_dict[combination] = candidates
    return candidate_dict


def optimize_mapping_by_coordinate_descent(
    models: List[FingerprintModel],
    candidate_dict: Dict[tuple, List[int]],
    rng: np.random.Generator,
    num_restarts: int = 4,
    max_passes: int = 2,
) -> MappingSearchResult:
    possible_combinations = list(candidate_dict.keys())
    bit_blocks_pm = [np.array(bits, dtype=int) for bits in possible_combinations]

    initial_mappings: List[Dict[tuple, int]] = []
    initial_mappings.append({key: candidate_dict[key][0] for key in possible_combinations})
    for _ in range(max(0, num_restarts - 1)):
        initial_mappings.append({
            key: int(rng.choice(candidate_dict[key]))
            for key in possible_combinations
        })

    best_result = None
    best_key = None
    for mapping in initial_mappings:
        current_mapping = dict(mapping)
        current_result = evaluate_n_position_mapping(models, current_mapping, bit_blocks_pm=bit_blocks_pm)
        current_key = (current_result.accuracy, current_result.min_margin, current_result.avg_margin)

        improved = True
        pass_count = 0
        while improved and pass_count < max_passes:
            improved = False
            pass_count += 1
            for combination in possible_combinations:
                local_best_mapping = dict(current_mapping)
                local_best_result = current_result
                local_best_key = current_key
                for probe in candidate_dict[combination]:
                    trial_mapping = dict(current_mapping)
                    trial_mapping[combination] = int(probe)
                    trial_result = evaluate_n_position_mapping(models, trial_mapping, bit_blocks_pm=bit_blocks_pm)
                    trial_key = (trial_result.accuracy, trial_result.min_margin, trial_result.avg_margin)
                    if trial_key > local_best_key:
                        local_best_mapping = trial_mapping
                        local_best_result = trial_result
                        local_best_key = trial_key
                if local_best_key > current_key:
                    current_mapping = local_best_mapping
                    current_result = local_best_result
                    current_key = local_best_key
                    improved = True

        if best_key is None or current_key > best_key:
            best_key = current_key
            best_result = current_result

    if best_result is None:
        raise RuntimeError("Failed to optimize hue mapping.")
    return best_result


def filter_position_files_valid_for_probes(
    position_files: List[Tuple[int, str]],
    num_probes: int,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, List[int]]]]:
    if not position_files:
        return [], []
    first_df = pd.read_csv(position_files[0][1])
    max_row_index = len(first_df) - 1
    probes = generate_probes(num_probes, max_row_index, verbose=False)
    row_indices = [int((float(probe) / 5) - 1) for probe in probes]

    valid = []
    invalid = []
    for position_id, csv_file in position_files:
        df = pd.read_csv(csv_file)
        Y = df.iloc[row_indices].values.astype(float)
        if np.any(Y < 0):
            bad_rows = [int(probes[i]) for i in np.where(np.any(Y < 0, axis=1))[0]]
            invalid.append((int(position_id), bad_rows))
        else:
            valid.append((int(position_id), csv_file))
    return valid, invalid


def sample_tuples(
    position_files: List[Tuple[int, str]],
    k: int,
    sample_size: int,
    min_position_distance: float,
    rng: np.random.Generator,
) -> List[Tuple[Tuple[int, str], ...]]:
    ids = [item[0] for item in position_files]
    file_map = {item[0]: item[1] for item in position_files}

    if min_position_distance > 0:
        valid = []
        for combo in combinations(ids, k):
            ok = True
            for i in range(k):
                for j in range(i + 1, k):
                    if calculate_position_distance(combo[i], combo[j]) < min_position_distance:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                valid.append(combo)
        if not valid:
            return []
        sample_size = min(sample_size, len(valid))
        pick = rng.choice(len(valid), size=sample_size, replace=False)
        return [tuple((p, file_map[p]) for p in valid[int(idx)]) for idx in pick.tolist()]

    sampled_ids = set()
    sampled = []
    max_attempts = sample_size * 50
    attempts = 0
    while len(sampled) < sample_size and attempts < max_attempts:
        attempts += 1
        chosen = tuple(sorted(rng.choice(ids, size=k, replace=False).tolist()))
        if chosen in sampled_ids:
            continue
        sampled_ids.add(chosen)
        sampled.append(tuple((p, file_map[p]) for p in chosen))
    return sampled


def build_multi_svd_model(base_model) -> MultiSVDModel:
    probes = np.asarray(base_model.probes, dtype=float)
    Y = np.asarray(base_model.Y, dtype=float)
    residual = np.asarray(base_model.residual, dtype=float)
    _, _, vt = np.linalg.svd(residual, full_matrices=False)
    w = vt.T.copy()
    z = residual @ w
    for j in range(z.shape[1]):
        if z[0, j] < 0:
            w[:, j] *= -1
            z[:, j] *= -1
    c = np.where(z >= 0, 1, -1)
    return MultiSVDModel(probes=probes, Y=Y, W=w, C=c)


def decode_gammas(Y_obs: np.ndarray, model: MultiSVDModel, top_r: int = 3) -> np.ndarray:
    Y_centered = np.asarray(Y_obs, dtype=float) - np.asarray(Y_obs, dtype=float).mean(axis=0)
    gammas: List[float] = []
    for j in range(min(top_r, model.W.shape[1])):
        u = Y_centered @ model.W[:, j]
        gammas.append(float(model.C[:, j] @ u))
    while len(gammas) < top_r:
        gammas.append(0.0)
    return np.asarray(gammas, dtype=float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic_regression_numpy(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    num_steps: int = 4000,
    l2: float = 1e-3,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Xn = (X - mean) / std

    w = np.zeros(Xn.shape[1], dtype=float)
    b = 0.0
    n = Xn.shape[0]
    for _ in range(num_steps):
        logits = Xn @ w + b
        p = _sigmoid(logits)
        grad_w = (Xn.T @ (p - y)) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, mean, std


def predict_logistic(X: np.ndarray, model_params: Tuple[np.ndarray, float, np.ndarray, np.ndarray]) -> np.ndarray:
    w, b, mean, std = model_params
    Xn = (np.asarray(X, dtype=float) - mean) / std
    probs = _sigmoid(Xn @ w + b)
    return (probs > 0.5).astype(int)


def score_logistic(X: np.ndarray, model_params: Tuple[np.ndarray, float, np.ndarray, np.ndarray]) -> np.ndarray:
    w, b, mean, std = model_params
    Xn = (np.asarray(X, dtype=float) - mean) / std
    return Xn @ w + b


def extract_truth_bits(results: List[dict], num_positions: int) -> List[List[int]]:
    return [[int(block["bits_bin"][stream_idx]) for block in results] for stream_idx in range(num_positions)]


def precompute_receiver_gammas(
    results: List[dict],
    models: Sequence,
    build_probe_to_row_fn: Callable[[np.ndarray], dict],
    observe_fn: Callable[[np.ndarray, np.ndarray, dict], np.ndarray],
    top_r: int = 3,
) -> List[np.ndarray]:
    multi_models = [build_multi_svd_model(model) for model in models]
    probe_to_rows = [build_probe_to_row_fn(model.probes) for model in multi_models]
    out: List[np.ndarray] = []
    for rx_idx, model in enumerate(multi_models):
        gammas = []
        for block in results:
            Y_obs = observe_fn(block["hue_seq"], model.Y, probe_to_rows[rx_idx])
            gammas.append(decode_gammas(Y_obs, model, top_r=top_r))
        out.append(np.asarray(gammas, dtype=float))
    return out


def _overall_ber_from_receiver_preds(
    pred_by_receiver: List[np.ndarray],
    truth_by_receiver: List[Sequence[int]],
) -> float:
    err = 0
    total = 0
    for pred, truth in zip(pred_by_receiver, truth_by_receiver):
        truth_arr = np.asarray(truth, dtype=int)
        pred_arr = np.asarray(pred, dtype=int)
        err += int(np.sum(pred_arr != truth_arr))
        total += int(len(truth_arr))
    return err / total if total > 0 else 0.0


def train_logistic_auto_decoder(
    cal_gammas: List[np.ndarray],
    cal_truth_bits: List[Sequence[int]],
) -> AdaptiveDecoderBundle:
    baseline_preds = [(gammas[:, 0] > 0).astype(int) for gammas in cal_gammas]
    baseline_ber = _overall_ber_from_receiver_preds(baseline_preds, cal_truth_bits)

    top2_models = []
    top3_models = []
    top2_preds = []
    top3_preds = []
    for rx_idx in range(len(cal_gammas)):
        X2 = cal_gammas[rx_idx][:, :2]
        X3 = cal_gammas[rx_idx][:, :3]
        y = np.asarray(cal_truth_bits[rx_idx], dtype=int)
        model2 = fit_logistic_regression_numpy(X2, y)
        model3 = fit_logistic_regression_numpy(X3, y)
        top2_models.append(model2)
        top3_models.append(model3)
        top2_preds.append(predict_logistic(X2, model2))
        top3_preds.append(predict_logistic(X3, model3))

    top2_ber = _overall_ber_from_receiver_preds(top2_preds, cal_truth_bits)
    top3_ber = _overall_ber_from_receiver_preds(top3_preds, cal_truth_bits)

    candidates = [
        ("baseline", 1, None, baseline_ber),
        ("logistic_top2", 2, top2_models, top2_ber),
        ("logistic_top3", 3, top3_models, top3_ber),
    ]
    mode, use_dim, models, cal_ber = min(candidates, key=lambda item: (item[3], item[1]))
    return AdaptiveDecoderBundle(
        mode=mode,
        use_dim=use_dim,
        models=models,
        calibration_ber=float(cal_ber),
        baseline_calibration_ber=float(baseline_ber),
        logistic_top2_calibration_ber=float(top2_ber),
        logistic_top3_calibration_ber=float(top3_ber),
    )


def decode_bits_with_bundle(
    gammas_by_receiver: List[np.ndarray],
    bundle: AdaptiveDecoderBundle,
) -> List[np.ndarray]:
    decoded: List[np.ndarray] = []
    if bundle.mode == "baseline" or bundle.models is None:
        for gammas in gammas_by_receiver:
            decoded.append((gammas[:, 0] > 0).astype(int))
        return decoded

    for rx_idx, gammas in enumerate(gammas_by_receiver):
        decoded.append(predict_logistic(gammas[:, :bundle.use_dim], bundle.models[rx_idx]))
    return decoded


def score_bits_with_bundle(
    gammas_by_receiver: List[np.ndarray],
    bundle: AdaptiveDecoderBundle,
) -> List[np.ndarray]:
    scores: List[np.ndarray] = []
    if bundle.mode == "baseline" or bundle.models is None:
        for gammas in gammas_by_receiver:
            scores.append(np.asarray(gammas[:, 0], dtype=float))
        return scores

    for rx_idx, gammas in enumerate(gammas_by_receiver):
        scores.append(score_logistic(gammas[:, :bundle.use_dim], bundle.models[rx_idx]))
    return scores


def simulate_blocks_scalar(
    models: List[FingerprintModel],
    bit_blocks_pm: List[Array],
    hue_mapping: Dict[tuple, int],
) -> List[dict]:
    codes = [m.code for m in models]
    probe_to_row = build_probe_to_row(models[0].probes)
    results = []

    for bits_pm in bit_blocks_pm:
        bits_pm = np.asarray(bits_pm, dtype=int)
        symbol_seq, symbol_combinations = build_symbol_sequence(bits_pm, codes)
        hue_seq = np.asarray([hue_mapping[tuple(comb)] for comb in symbol_combinations], dtype=int)

        block_info = {
            "bits_pm": bits_pm,
            "bits_bin": pm1_to_bin(bits_pm),
            "symbol_seq": symbol_seq,
            "hue_seq": hue_seq,
            "per_position": [],
        }

        for model in models:
            Y_obs = observe_block_from_measured_matrix(hue_seq, model.Y, probe_to_row)
            dec = decode_local_block(Y_obs, model.w, model.code)
            block_info["per_position"].append(dec)
        results.append(block_info)
    return results


def _build_symbol_combinations(bits_pm: np.ndarray, codes: List[np.ndarray]) -> List[tuple]:
    bits_pm = np.asarray(bits_pm, dtype=int)
    out: List[tuple] = []
    for t in range(len(codes[0])):
        combo = []
        for b, c in zip(bits_pm, codes):
            combo.append(int(b) * int(c[t]))
        out.append(tuple(combo))
    return out


def observe_block_from_vector_mapping(
    hue_vec_seq: np.ndarray,
    Y: np.ndarray,
    probe_to_row: Dict[int, int],
) -> np.ndarray:
    hue_vec_seq = np.asarray(hue_vec_seq, dtype=int)
    Y = np.asarray(Y, dtype=float)
    if hue_vec_seq.ndim != 2:
        raise ValueError(f"Expected vector hue sequence with ndim=2, got shape={hue_vec_seq.shape}")

    rows = np.zeros((hue_vec_seq.shape[0], Y.shape[1]), dtype=float)
    for t in range(hue_vec_seq.shape[0]):
        for d in range(Y.shape[1]):
            hue = int(hue_vec_seq[t, d])
            if hue not in probe_to_row:
                raise KeyError(f"Hue {hue} not found in measured probes: {sorted(probe_to_row.keys())}")
            rows[t, d] = float(Y[probe_to_row[hue], d])
    return rows


def simulate_blocks_vector(
    models: List[FingerprintModel],
    bit_blocks_pm: List[np.ndarray],
    hue_mapping: VectorMapping,
) -> List[dict]:
    codes = [np.asarray(m.code, dtype=int) for m in models]
    probe_to_rows = [build_probe_to_row(m.probes) for m in models]
    results: List[dict] = []

    for bits_pm in bit_blocks_pm:
        bits_pm = np.asarray(bits_pm, dtype=int)
        symbol_seq, _ = build_symbol_sequence(bits_pm, codes)
        symbol_combinations = _build_symbol_combinations(bits_pm, codes)
        hue_vec_seq = np.asarray([hue_mapping[key] for key in symbol_combinations], dtype=int)

        block_info = {
            "bits_pm": bits_pm,
            "bits_bin": pm1_to_bin(bits_pm),
            "symbol_seq": symbol_seq,
            "hue_seq_vector": hue_vec_seq,
            "per_position": [],
        }

        for rx_idx, model in enumerate(models):
            Y_obs = observe_block_from_vector_mapping(hue_vec_seq, model.Y, probe_to_rows[rx_idx])
            dec = decode_local_block(Y_obs, model.w, model.code)
            block_info["per_position"].append(dec)
        results.append(block_info)
    return results


def _build_vector_candidate_dict(
    models: List[FingerprintModel],
    top_k: int,
) -> Dict[tuple, List[Tuple[int, int, int, int]]]:
    probes = [int(v) for v in np.asarray(models[0].probes, dtype=int).tolist()]
    residuals = [np.asarray(m.residual, dtype=float) for m in models]
    weights = [np.asarray(m.w, dtype=float) for m in models]
    num_channels = residuals[0].shape[1]
    possible_combinations = [tuple(bits) for bits in product([1, -1], repeat=len(models))]
    fallback_probe = int(probes[len(probes) // 2])

    candidate_dict: Dict[tuple, List[Tuple[int, int, int, int]]] = {}
    for combination in possible_combinations:
        per_channel_candidates: List[List[int]] = []
        channel_score_lookup: List[Dict[int, float]] = []
        for d in range(num_channels):
            scored: List[Tuple[float, int]] = []
            for idx, probe in enumerate(probes):
                score = 0.0
                for pos_idx in range(len(models)):
                    score += float(combination[pos_idx]) * residuals[pos_idx][idx, d] * weights[pos_idx][d]
                scored.append((score, int(probe)))
            scored.sort(key=lambda item: item[0], reverse=True)

            candidates: List[int] = []
            lookup: Dict[int, float] = {}
            for score, probe in scored:
                if probe not in lookup:
                    lookup[probe] = float(score)
                if probe not in candidates:
                    candidates.append(probe)
                if len(candidates) >= top_k:
                    break
            if not candidates:
                candidates = [fallback_probe]
                lookup[fallback_probe] = 0.0
            per_channel_candidates.append(candidates)
            channel_score_lookup.append(lookup)

        vector_scored: List[Tuple[float, Tuple[int, int, int, int]]] = []
        for vector in product(*per_channel_candidates):
            vector_tuple = tuple(int(v) for v in vector)
            score = sum(channel_score_lookup[d].get(vector_tuple[d], 0.0) for d in range(num_channels))
            vector_scored.append((float(score), vector_tuple))
        vector_scored.sort(key=lambda item: item[0], reverse=True)

        unique_vectors: List[Tuple[int, int, int, int]] = []
        for _, vector in vector_scored:
            if vector not in unique_vectors:
                unique_vectors.append(vector)
        candidate_dict[combination] = unique_vectors
    return candidate_dict


def _precompute_receiver_gammas_vector(
    results: List[dict],
    models: Sequence[FingerprintModel],
    top_r: int = 3,
) -> List[np.ndarray]:
    multi_models = [build_multi_svd_model(model) for model in models]
    probe_to_rows = [build_probe_to_row(model.probes) for model in multi_models]
    out: List[np.ndarray] = []
    for rx_idx, model in enumerate(multi_models):
        gammas = []
        for block in results:
            Y_obs = observe_block_from_vector_mapping(block["hue_seq_vector"], model.Y, probe_to_rows[rx_idx])
            gammas.append(decode_gammas(Y_obs, model, top_r=top_r))
        out.append(np.asarray(gammas, dtype=float))
    return out


def _calculate_ber_with_bundle_vector(
    results: List[dict],
    models: Sequence[FingerprintModel],
    bundle: AdaptiveDecoderBundle,
    top_r: int = 3,
) -> Tuple[float, int, int]:
    gammas_by_receiver = _precompute_receiver_gammas_vector(results, models, top_r=top_r)
    truth_by_receiver = extract_truth_bits(results, num_positions=len(models))
    decoded_by_receiver = decode_bits_with_bundle(gammas_by_receiver, bundle)

    err = 0
    total = 0
    for pred, truth in zip(decoded_by_receiver, truth_by_receiver):
        truth_arr = np.asarray(truth, dtype=int)
        pred_arr = np.asarray(pred, dtype=int)
        err += int(np.sum(pred_arr != truth_arr))
        total += int(len(truth_arr))
    ber = err / total if total > 0 else 0.0
    return ber, err, total


def evaluate_unauthorized_model_leakage_vector(
    unauthorized_model: FingerprintModel,
    authorized_results: List[dict],
    true_stream_bits: List[List[int]],
) -> Dict[str, object]:
    probe_to_row = build_probe_to_row(unauthorized_model.probes)
    decoded_bits: List[int] = []
    for block_result in authorized_results:
        Y_obs = observe_block_from_vector_mapping(block_result["hue_seq_vector"], unauthorized_model.Y, probe_to_row)
        dec = decode_local_block(Y_obs, unauthorized_model.w, unauthorized_model.code)
        decoded_bits.append(int(dec.bit_hat_bin))
    return compute_leakage_metrics_from_decoded_bits(decoded_bits, true_stream_bits)


def _load_models_for_target(
    tuple_items: Tuple[Tuple[int, str], ...],
    num_probes: int,
    target_subset_size: int,
) -> List[FingerprintModel]:
    csv_files = [item[1] for item in tuple_items]
    models, _, _ = prepare_n_position_models(
        csv_files=csv_files,
        num_probes=num_probes,
        target_subset_size=target_subset_size,
    )
    return models


def _build_top1_vector_mapping(models: List[FingerprintModel]) -> VectorMapping:
    candidate_dict = _build_vector_candidate_dict(models, top_k=1)
    return {key: values[0] for key, values in candidate_dict.items()}


def _branch_table(generators: Sequence[int], constraint_len: int):
    num_states = 2 ** (constraint_len - 1)
    next_state = np.zeros((num_states, 2), dtype=int)
    output_bits = np.zeros((num_states, 2, len(generators)), dtype=int)
    mask = (1 << constraint_len) - 1
    for state in range(num_states):
        for bit in [0, 1]:
            reg = ((state << 1) | bit) & mask
            next_state[state, bit] = reg & (num_states - 1)
            for gi, g in enumerate(generators):
                output_bits[state, bit, gi] = int(bin(reg & g).count("1") % 2)
    return next_state, output_bits


def conv_encode_generic(bits: Sequence[int], spec: GenericConvSpec, terminate: bool = True) -> np.ndarray:
    bits = list(int(b) for b in bits)
    if terminate:
        bits = bits + [0] * (spec.constraint_len - 1)
    state = 0
    mask = (1 << spec.constraint_len) - 1
    num_states = 2 ** (spec.constraint_len - 1)
    out: List[int] = []
    for bit in bits:
        reg = ((state << 1) | bit) & mask
        for g in spec.generators:
            out.append(int(bin(reg & g).count("1") % 2))
        state = reg & (num_states - 1)
    return np.asarray(out, dtype=int)


def viterbi_decode_soft_generic(scores: Sequence[float], spec: GenericConvSpec, terminate: bool = True) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    n_out = len(spec.generators)
    if len(scores) % n_out != 0:
        raise ValueError(f"Soft sequence length must be multiple of {n_out}.")

    num_states = 2 ** (spec.constraint_len - 1)
    next_state, output_bits = _branch_table(spec.generators, spec.constraint_len)
    steps = len(scores) // n_out
    neg_inf = -1e18
    path_metric = np.full((steps + 1, num_states), neg_inf, dtype=float)
    prev_state = np.full((steps + 1, num_states), -1, dtype=int)
    prev_bit = np.full((steps + 1, num_states), -1, dtype=int)
    path_metric[0, 0] = 0.0

    recv = scores.reshape(-1, n_out)
    for t in range(steps):
        for state in range(num_states):
            if path_metric[t, state] <= neg_inf / 2:
                continue
            for bit in [0, 1]:
                ns = next_state[state, bit]
                out = output_bits[state, bit]
                out_pm = np.where(out > 0, 1.0, -1.0)
                metric = float(np.sum(out_pm * recv[t]))
                cand = path_metric[t, state] + metric
                if cand > path_metric[t + 1, ns]:
                    path_metric[t + 1, ns] = cand
                    prev_state[t + 1, ns] = state
                    prev_bit[t + 1, ns] = bit

    end_state = 0 if terminate else int(np.argmax(path_metric[steps]))
    decoded_rev: List[int] = []
    state = end_state
    for t in range(steps, 0, -1):
        bit = int(prev_bit[t, state])
        decoded_rev.append(bit)
        state = int(prev_state[t, state])
    decoded = list(reversed(decoded_rev))
    if terminate:
        decoded = decoded[: -(spec.constraint_len - 1)]
    return np.asarray(decoded, dtype=int)


def get_conv_config_generic(channel_blocks: int, spec: GenericConvSpec) -> Tuple[int, int]:
    n_out = len(spec.generators)
    steps = channel_blocks // n_out
    info_bits = max(1, steps - (spec.constraint_len - 1))
    actual_channel_blocks = n_out * (info_bits + (spec.constraint_len - 1))
    return info_bits, actual_channel_blocks


def _compute_ber(tx_list: List[np.ndarray], rx_list: List[np.ndarray]) -> float:
    err = 0
    total = 0
    for tx, rx in zip(tx_list, rx_list):
        tx = np.asarray(tx, dtype=int)
        rx = np.asarray(rx, dtype=int)
        err += int(np.sum(tx != rx))
        total += int(len(tx))
    return err / total if total > 0 else 0.0


def _evaluate_target_once(
    dataset_dir: str,
    tuple_items: Tuple[Tuple[int, str], ...],
    num_probes: int,
    target_subset_size: int,
    channel_blocks: int,
    calibration_blocks: int,
    conv_spec: GenericConvSpec,
    random_seed: int,
) -> RandomConvRunResult:
    target = tuple(item[0] for item in tuple_items)
    k = len(target)

    models = _load_models_for_target(
        tuple_items=tuple_items,
        num_probes=num_probes,
        target_subset_size=target_subset_size,
    )
    hue_mapping = _build_top1_vector_mapping(models)

    calibration_blocks_pm = generate_random_bit_blocks(
        num_positions=k,
        num_blocks=calibration_blocks,
        rng=np.random.default_rng(random_seed + 1),
    )
    calibration_results = simulate_blocks_vector(models, calibration_blocks_pm, hue_mapping)
    cal_gammas = _precompute_receiver_gammas_vector(
        calibration_results,
        models,
        top_r=3,
    )
    bundle = train_logistic_auto_decoder(
        cal_gammas,
        extract_truth_bits(calibration_results, num_positions=k),
    )

    uncoded_blocks_pm = generate_random_bit_blocks(
        num_positions=k,
        num_blocks=channel_blocks,
        rng=np.random.default_rng(random_seed + 2),
    )
    uncoded_results = simulate_blocks_vector(models, uncoded_blocks_pm, hue_mapping)
    authorized_uncoded_ber, _, _ = _calculate_ber_with_bundle_vector(
        uncoded_results,
        models,
        bundle,
        top_r=3,
    )

    info_bits_per_stream, actual_channel_blocks = get_conv_config_generic(channel_blocks, conv_spec)
    rng = np.random.default_rng(random_seed + 3)
    info_bits_by_stream = [
        rng.integers(0, 2, size=info_bits_per_stream, dtype=int)
        for _ in range(k)
    ]
    tx_bits_by_stream = [conv_encode_generic(bits, conv_spec, terminate=True) for bits in info_bits_by_stream]

    coded_blocks_pm = []
    for t in range(actual_channel_blocks):
        bits_bin = np.array([int(stream_bits[t]) for stream_bits in tx_bits_by_stream], dtype=int)
        bits_pm = np.where(bits_bin > 0, 1, -1).astype(int)
        coded_blocks_pm.append(bits_pm)

    coded_results = simulate_blocks_vector(models, coded_blocks_pm, hue_mapping)
    eval_gammas = _precompute_receiver_gammas_vector(
        coded_results,
        models,
        top_r=3,
    )
    soft_scores_by_stream = score_bits_with_bundle(eval_gammas, bundle)
    legit_decoded_info_bits = [
        viterbi_decode_soft_generic(scores, conv_spec, terminate=True)
        for scores in soft_scores_by_stream
    ]
    authorized_postfec_ber = _compute_ber(info_bits_by_stream, legit_decoded_info_bits)

    position_files = list_position_files(dataset_dir)
    probes = np.asarray(models[0].probes, dtype=float)
    unauthorized_postfec_bers = []
    for unauthorized_position, unauthorized_csv in position_files:
        if unauthorized_position in target:
            continue
        try:
            unauthorized_model = load_single_position_model(unauthorized_csv, probes=probes)
            unauth_gammas = _precompute_receiver_gammas_vector(
                coded_results,
                [unauthorized_model],
                top_r=3,
            )[0]
            unauth_scores = np.asarray(unauth_gammas[:, 0], dtype=float)
            decoded_info = viterbi_decode_soft_generic(unauth_scores, conv_spec, terminate=True)

            per_stream_ber = []
            for stream_bits in info_bits_by_stream:
                ber = float(np.mean(np.asarray(decoded_info, dtype=int) != np.asarray(stream_bits, dtype=int)))
                per_stream_ber.append(min(ber, 1.0 - ber))
            unauthorized_postfec_bers.append(float(np.mean(per_stream_ber)))
        except Exception:
            continue

    return RandomConvRunResult(
        k=k,
        target=target,
        num_probes=int(num_probes),
        conv_code=conv_spec.code_name,
        rate=info_bits_per_stream / actual_channel_blocks if actual_channel_blocks > 0 else 0.0,
        decoder_mode=str(bundle.mode),
        calibration_ber=float(bundle.calibration_ber),
        authorized_uncoded_ber=float(authorized_uncoded_ber),
        authorized_postfec_ber=float(authorized_postfec_ber),
        worst_unauthorized_postfec_ber=float(min(unauthorized_postfec_bers)) if unauthorized_postfec_bers else float("nan"),
        avg_unauthorized_postfec_ber=float(np.mean(unauthorized_postfec_bers)) if unauthorized_postfec_bers else float("nan"),
        median_unauthorized_postfec_ber=float(np.median(unauthorized_postfec_bers)) if unauthorized_postfec_bers else float("nan"),
        leak_positions_le_0p05=int(sum(v <= 0.05 + 1e-12 for v in unauthorized_postfec_bers)),
        leak_positions_le_0p10=int(sum(v <= 0.10 + 1e-12 for v in unauthorized_postfec_bers)),
        evaluated_unauthorized_positions=int(len(unauthorized_postfec_bers)),
    )


def run_random_conv_experiment(
    dataset_dir: str,
    k_min: int,
    k_max: int,
    sample_size_per_k: int,
    min_position_distance: float,
    num_probes: int,
    target_subset_size: int,
    channel_blocks: int,
    calibration_blocks: int,
    conv_spec: GenericConvSpec,
    random_seed: int,
) -> pd.DataFrame:
    position_files = list_position_files(dataset_dir)
    filtered_position_files, invalid_positions = filter_position_files_valid_for_probes(
        position_files=position_files,
        num_probes=num_probes,
    )
    if invalid_positions:
        print(f"filtered invalid positions for num_probes={num_probes}: {invalid_positions}")
    position_files = filtered_position_files

    rows = []
    for k in range(k_min, k_max + 1):
        sampled = sample_tuples(
            position_files=position_files,
            k=k,
            sample_size=sample_size_per_k,
            min_position_distance=min_position_distance,
            rng=np.random.default_rng(random_seed + k * 100),
        )
        print("\n" + "=" * 72)
        print(f"Random vector-hue + conv experiment: k={k}, sampled_sets={len(sampled)}")
        print("=" * 72)

        for idx, tuple_items in enumerate(sampled, start=1):
            target = tuple(item[0] for item in tuple_items)
            print(f"[k={k}] {idx:>3}/{len(sampled)} target={target}")
            try:
                result = _evaluate_target_once(
                    dataset_dir=dataset_dir,
                    tuple_items=tuple_items,
                    num_probes=num_probes,
                    target_subset_size=target_subset_size,
                    channel_blocks=channel_blocks,
                    calibration_blocks=calibration_blocks,
                    conv_spec=conv_spec,
                    random_seed=random_seed + k * 1000 + idx,
                )
                row = asdict(result)
                row["status"] = "ok"
                print(
                    f"  uncoded={result.authorized_uncoded_ber:.6f} "
                    f"postfec={result.authorized_postfec_ber:.6f} "
                    f"worst_unauth={result.worst_unauthorized_postfec_ber:.6f} "
                    f"avg_unauth={result.avg_unauthorized_postfec_ber:.6f} "
                    f"decoder={result.decoder_mode}"
                )
            except Exception as exc:
                row = {
                    "k": k,
                    "target": target,
                    "status": f"error: {exc}",
                }
                print(f"  error: {exc}")
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    ok_df = df[df["status"] == "ok"].copy()
    if ok_df.empty:
        return pd.DataFrame()

    summary_rows = []
    for k, g in ok_df.groupby("k"):
        decoder_counts = g.groupby("decoder_mode").size().sort_values(ascending=False).to_dict()
        summary_rows.append(
            {
                "k": int(k),
                "samples": int(len(g)),
                "mean_authorized_uncoded_ber": float(g["authorized_uncoded_ber"].mean()),
                "median_authorized_uncoded_ber": float(g["authorized_uncoded_ber"].median()),
                "mean_authorized_postfec_ber": float(g["authorized_postfec_ber"].mean()),
                "median_authorized_postfec_ber": float(g["authorized_postfec_ber"].median()),
                "mean_worst_unauthorized_postfec_ber": float(g["worst_unauthorized_postfec_ber"].mean()),
                "median_worst_unauthorized_postfec_ber": float(g["worst_unauthorized_postfec_ber"].median()),
                "mean_avg_unauthorized_postfec_ber": float(g["avg_unauthorized_postfec_ber"].mean()),
                "median_avg_unauthorized_postfec_ber": float(g["avg_unauthorized_postfec_ber"].median()),
                "mean_gap_worst_minus_auth_postfec": float((g["worst_unauthorized_postfec_ber"] - g["authorized_postfec_ber"]).mean()),
                "mean_gap_avg_minus_auth_postfec": float((g["avg_unauthorized_postfec_ber"] - g["authorized_postfec_ber"]).mean()),
                "mean_fec_gain": float((g["authorized_postfec_ber"] - g["authorized_uncoded_ber"]).mean()),
                "improved_count": int((g["authorized_postfec_ber"] < g["authorized_uncoded_ber"]).sum()),
                "worsened_count": int((g["authorized_postfec_ber"] > g["authorized_uncoded_ber"]).sum()),
                "mean_calibration_ber": float(g["calibration_ber"].mean()),
                "mean_leak_positions_le_0p05": float(g["leak_positions_le_0p05"].mean()),
                "mean_leak_positions_le_0p10": float(g["leak_positions_le_0p10"].mean()),
                "decoder_mode_counts": str(decoder_counts),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("k").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random vector-hue + convolutional code experiment for k=2..8.")
    parser.add_argument("--dataset-dir", default="data\\15pro\\high")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--sample-size-per-k", type=int, default=20)
    parser.add_argument("--min-position-distance", type=float, default=0.0)
    parser.add_argument("--num-probes", type=int, default=15)
    parser.add_argument("--target-subset-size", type=int, default=15)
    parser.add_argument("--channel-blocks", type=int, default=120)
    parser.add_argument("--calibration-blocks", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260510)
    parser.add_argument("--output-csv", default="vector_hue_random_conv_k2_to_k8_singlefile_runs.csv")
    parser.add_argument("--summary-csv", default="vector_hue_random_conv_k2_to_k8_singlefile_summary.csv")
    args = parser.parse_args()

    conv_spec = GenericConvSpec("conv_k3_r13", (0b111, 0b101, 0b011), 3)
    df = run_random_conv_experiment(
        dataset_dir=args.dataset_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        sample_size_per_k=args.sample_size_per_k,
        min_position_distance=args.min_position_distance,
        num_probes=args.num_probes,
        target_subset_size=args.target_subset_size,
        channel_blocks=args.channel_blocks,
        calibration_blocks=args.calibration_blocks,
        conv_spec=conv_spec,
        random_seed=args.random_seed,
    )
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"saved: {args.output_csv}")

    summary_df = summarize_results(df)
    summary_df.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    print(f"saved: {args.summary_csv}")
    if not summary_df.empty:
        print("\n" + "=" * 72)
        print("Random vector-hue + conv summary")
        print("=" * 72)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
