#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为存在信息泄露风险的三位置组合重新搜索探针。

约束：
1. 合法三位置的 BER 必须为 0。
2. 任意非法位置对任一合法位置的解码 BER 必须严格大于 0.2。
"""

from __future__ import annotations

import ast
import csv
import itertools
import os
import random
import sys
import types
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd


LIGHT_CONDITION = "white"
SOURCE_RANDOM_RESULTS_FILENAME = "random_20_batch_test_results.csv"
SOURCE_SECURITY_RESULTS_FILENAME = "security_illegal_position_results.csv"
OUTPUT_RESULTS_FILENAME = "batch_test_results_security_aware.csv"
SAFE_PROBE_RESULTS_FILENAME = "security_aware_safe_probes.csv"
LEGAL_NUM_BITS = 10000
SECURITY_NUM_BITS_COARSE = 2000
SECURITY_NUM_BITS_FINAL = 10000
MAPPING_EVAL_BITS = 500
MAPPING_TOP_K = 3
TARGET_LEGAL_BER = 0.02
SECURITY_THRESHOLD = 0.2
SECURITY_MAPPING_TOP_K = 3
RELAXED_MAPPING_TOP_K = 4
MAX_HUE_MAPPINGS_PER_PROBE = 4096
DETERMINISTIC_HUE_MAPPING_PREFIX = 512
SELECTION_SEED = 20260401
SEARCH_EPOCHS = 1
MAX_RANDOM_CANDIDATES_PER_COUNT = 20
TOP_K_CANDIDATES_PER_COUNT = 2
LOCAL_NEIGHBOR_SAMPLES = 8
LOCAL_ROUNDS = 2
MIN_PROBE_COUNT = 5
MAX_PROBE_COUNT = 30
DEFAULT_BASELINE_PROBE_COUNT = 8
ZERO_OR_INVERSE_EPSILON = 1e-12
MIN_TRUTH_TABLE_DISTANCE = 1
FALLBACK_RESULTS_FILES: dict[str, str] = {}


def load_test_module() -> types.ModuleType:
    module_name = "test_3_simple_runtime_security_search"
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_3_simple.py")
    with open(module_path, "r", encoding="utf-8-sig") as f:
        source = f.read().lstrip("\ufeff")

    module = types.ModuleType(module_name)
    module.__file__ = module_path
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(compile(source, module_path, "exec"), module.__dict__)
    return module


test = load_test_module()


def parse_position_combination(text: str) -> tuple[int, int, int]:
    value = ast.literal_eval(text)
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"无效的位置组合: {text}")
    return tuple(int(v) for v in value)


def parse_probes(text: str) -> list[float]:
    value = ast.literal_eval(text)
    if not isinstance(value, list) or not value:
        raise ValueError(f"无效的探针列表: {text}")
    return [float(v) for v in value]


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def format_optional_probes(probes: Sequence[float] | None) -> str:
    if not probes:
        return ""
    return format_probes(probes)


def get_available_positions(project_root: str, light_condition: str) -> list[int]:
    data_dir = os.path.join(project_root, "data", "15pro", light_condition)
    positions = []
    for entry in os.listdir(data_dir):
        if not entry.endswith(".csv"):
            continue
        stem = os.path.splitext(entry)[0]
        if stem.isdigit():
            positions.append(int(stem))
    return sorted(positions)


def generate_position_combinations(positions: Sequence[int]) -> list[tuple[int, int, int]]:
    return [tuple(int(v) for v in combination) for combination in itertools.combinations(positions, 3)]


def min_interval_for_probe_count(probe_count: int) -> int:
    if probe_count <= 12:
        return 30
    if probe_count <= 16:
        return 20
    if probe_count <= 22:
        return 15
    return 10


def load_flagged_combinations(random_results_file: str, security_results_file: str) -> list[dict]:
    config_by_combination: dict[str, dict] = {}
    with open(random_results_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            config_by_combination[row["position_combination"]] = {
                "position_combination": parse_position_combination(row["position_combination"]),
                "best_probe_count": int(row["best_probe_count"]),
                "best_probes": parse_probes(row["best_probes"]),
                "best_ber": float(row["best_ber"]),
                "test_ber": float(row["test_ber"]),
            }

    flagged: dict[str, dict] = {}
    with open(security_results_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bers = [
                float(row["ber_vs_legal_pos_1"]),
                float(row["ber_vs_legal_pos_2"]),
                float(row["ber_vs_legal_pos_3"]),
            ]
            min_illegal_ber = min(bers)
            if min_illegal_ber >= SECURITY_THRESHOLD:
                continue

            combination_key = row["legal_position_combination"]
            config = config_by_combination[combination_key]
            existing = flagged.get(combination_key)
            if existing is None:
                flagged[combination_key] = {
                    **config,
                    "worst_illegal_single_ber": min_illegal_ber,
                    "worst_illegal_position": int(row["illegal_position"]),
                }
            elif min_illegal_ber < existing["worst_illegal_single_ber"]:
                existing["worst_illegal_single_ber"] = min_illegal_ber
                existing["worst_illegal_position"] = int(row["illegal_position"])

    return [flagged[key] for key in sorted(flagged.keys())]


def load_baseline_configs(random_results_file: str) -> dict[str, dict]:
    if not os.path.exists(random_results_file):
        return {}

    configs: dict[str, dict] = {}
    with open(random_results_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            combination = parse_position_combination(row["position_combination"])
            configs[str(combination)] = {
                "position_combination": combination,
                "best_probe_count": int(row["best_probe_count"]),
                "best_probes": parse_probes(row["best_probes"]),
                "best_ber": float(row["best_ber"]),
                "test_ber": float(row["test_ber"]),
                "worst_illegal_single_ber": None,
                "worst_illegal_position": None,
            }
    return configs


def build_all_combination_configs(project_root: str, random_results_file: str) -> list[dict]:
    baseline_by_combination = load_baseline_configs(random_results_file)
    available_positions = get_available_positions(project_root, LIGHT_CONDITION)
    configs = []

    for combination in generate_position_combinations(available_positions):
        key = str(combination)
        if key in baseline_by_combination:
            configs.append(baseline_by_combination[key])
            continue

        configs.append({
            "position_combination": combination,
            "best_probe_count": DEFAULT_BASELINE_PROBE_COUNT,
            "best_probes": None,
            "best_ber": None,
            "test_ber": None,
            "worst_illegal_single_ber": None,
            "worst_illegal_position": None,
        })

    return configs


def evaluate_illegal_min_ber(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
    num_bits: int,
    rng: random.Random,
) -> tuple[float, int | None, list[float] | None]:
    legal_csv_files = test.build_csv_files_for_positions(
        project_root,
        legal_positions,
        light_condition=LIGHT_CONDITION,
    )
    probes_array = np.asarray(probes, dtype=float)
    legal_models, hue_mapping = test.build_models_from_probes(
        legal_csv_files,
        probes_array,
        mapping_eval_bits=MAPPING_EVAL_BITS,
        mapping_top_k=MAPPING_TOP_K,
        rng=rng,
    )
    bit_blocks_pm = test.generate_random_bit_blocks(num_bits, len(legal_positions), rng=rng)
    legal_codes = [model.code for model in legal_models]
    probe_to_row = test.build_probe_to_row(probes_array)
    available_positions = get_available_positions(project_root, LIGHT_CONDITION)

    global_min_ber = float("inf")
    global_min_illegal_position: int | None = None
    global_min_ber_vector: list[float] | None = None

    for illegal_position in available_positions:
        if illegal_position in legal_positions:
            continue

        illegal_csv_file = test.build_csv_files_for_positions(
            project_root,
            [illegal_position],
            light_condition=LIGHT_CONDITION,
        )[0]
        illegal_matrix = test.load_selected_rows([illegal_csv_file], probes_array)[0]
        illegal_model = test.extract_fingerprint(probes_array, illegal_matrix, force_positive_first=True)
        position_errors = np.zeros(len(legal_models), dtype=float)
        position_total = np.zeros(len(legal_models), dtype=float)

        for bits_pm in bit_blocks_pm:
            bits_pm = np.asarray(bits_pm, dtype=int)
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            illegal_observation = test.observe_block_from_measured_matrix(hue_seq, illegal_matrix, probe_to_row)
            illegal_dec = test.decode_local_block(illegal_observation, illegal_model.w, illegal_model.code)
            true_bits = test.pm1_to_bin(bits_pm)

            for idx in range(len(legal_models)):
                position_total[idx] += 1
                if illegal_dec.bit_hat_bin != true_bits[idx]:
                    position_errors[idx] += 1

        position_bers = (position_errors / np.maximum(position_total, 1.0)).tolist()
        min_position_ber = float(min(position_bers))
        if min_position_ber < global_min_ber:
            global_min_ber = min_position_ber
            global_min_illegal_position = illegal_position
            global_min_ber_vector = [float(v) for v in position_bers]

        if global_min_ber <= 0.0:
            break

    if global_min_ber == float("inf"):
        return 1.0, None, None
    return global_min_ber, global_min_illegal_position, global_min_ber_vector


def generate_exact_bit_blocks(num_positions: int) -> list[np.ndarray]:
    return [
        np.asarray(bits_pm, dtype=int)
        for bits_pm in itertools.product([-1, 1], repeat=num_positions)
    ]


def build_legal_models_only(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
) -> list[test.FingerprintModel]:
    legal_csv_files = test.build_csv_files_for_positions(
        project_root,
        legal_positions,
        light_condition=LIGHT_CONDITION,
    )
    probes_array = np.asarray(probes, dtype=float)
    matrices = test.load_selected_rows(legal_csv_files, probes_array)
    models = [test.extract_fingerprint(probes_array, mat, force_positive_first=True) for mat in matrices]
    return test.align_model_directions(models)


def calculate_position_bers(results: Sequence[dict]) -> list[float]:
    if not results:
        return []

    position_errors = np.zeros(len(results[0]["per_position"]), dtype=float)
    position_total = np.zeros(len(results[0]["per_position"]), dtype=float)

    for res in results:
        bits_tx = res["bits_bin"]
        bits_rx = [dec.bit_hat_bin for dec in res["per_position"]]
        for idx, (tx, rx) in enumerate(zip(bits_tx, bits_rx)):
            position_total[idx] += 1
            if tx != rx:
                position_errors[idx] += 1

    return (position_errors / np.maximum(position_total, 1.0)).tolist()


def calculate_corrected_total_ber(position_bers: Sequence[float]) -> float:
    if not position_bers:
        return 0.0
    secure_bers = [min(float(ber), 1.0 - float(ber)) for ber in position_bers]
    return float(sum(secure_bers) / len(secure_bers))


def format_truth_table(values: Sequence[int]) -> str:
    return "[" + ", ".join(str(int(value)) for value in values) + "]"


def evaluate_legal_mapping(
    legal_models: Sequence[test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    bit_blocks_pm: Sequence[np.ndarray],
) -> tuple[float, list[float]]:
    results = test.simulate_blocks(list(legal_models), list(bit_blocks_pm), hue_mapping)
    position_bers = calculate_position_bers(results)
    return calculate_corrected_total_ber(position_bers), [float(v) for v in position_bers]


def evaluate_illegal_mapping(
    project_root: str,
    legal_positions: Sequence[int],
    legal_models: Sequence[test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    probes: Sequence[float],
    bit_blocks_pm: Sequence[np.ndarray],
    stop_below: float | None = None,
) -> tuple[float, int | None, int | None, list[float] | None, bool, int | None, list[int] | None, str | None]:
    probes_array = np.asarray(probes, dtype=float)
    legal_codes = [model.code for model in legal_models]
    probe_to_row = test.build_probe_to_row(probes_array)
    available_positions = get_available_positions(project_root, LIGHT_CONDITION)

    global_min_secure_ber = float("inf")
    global_min_illegal_position: int | None = None
    global_min_legal_position: int | None = None
    global_min_ber_vector: list[float] | None = None
    global_min_truth_table_distance: int | None = None
    global_min_illegal_truth_table: list[int] | None = None
    global_min_truth_table_reference: str | None = None
    zero_or_inverse_detected = False

    for illegal_position in available_positions:
        if illegal_position in legal_positions:
            continue

        illegal_csv_file = test.build_csv_files_for_positions(
            project_root,
            [illegal_position],
            light_condition=LIGHT_CONDITION,
        )[0]
        illegal_matrix = test.load_selected_rows([illegal_csv_file], probes_array)[0]
        illegal_model = test.extract_fingerprint(probes_array, illegal_matrix, force_positive_first=True)
        position_errors = np.zeros(len(legal_models), dtype=float)
        position_total = np.zeros(len(legal_models), dtype=float)
        illegal_truth_table: list[int] = []
        true_truth_tables = [[] for _ in legal_models]

        for bits_pm in bit_blocks_pm:
            bits_pm = np.asarray(bits_pm, dtype=int)
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            illegal_observation = test.observe_block_from_measured_matrix(hue_seq, illegal_matrix, probe_to_row)
            illegal_dec = test.decode_local_block(illegal_observation, illegal_model.w, illegal_model.code)
            true_bits = test.pm1_to_bin(bits_pm)
            illegal_truth_table.append(int(illegal_dec.bit_hat_bin))

            for idx in range(len(legal_models)):
                true_truth_tables[idx].append(int(true_bits[idx]))
                position_total[idx] += 1
                if illegal_dec.bit_hat_bin != true_bits[idx]:
                    position_errors[idx] += 1

        position_bers = (position_errors / np.maximum(position_total, 1.0)).tolist()
        secure_bers = [min(float(ber), 1.0 - float(ber)) for ber in position_bers]
        truth_table_distances: list[tuple[int, int, str]] = []
        illegal_array = np.asarray(illegal_truth_table, dtype=int)
        for idx, true_truth_table in enumerate(true_truth_tables):
            true_array = np.asarray(true_truth_table, dtype=int)
            direct_distance = int(np.sum(illegal_array != true_array))
            inverse_distance = int(np.sum(illegal_array != (1 - true_array)))
            truth_table_distances.append((direct_distance, idx, "direct"))
            truth_table_distances.append((inverse_distance, idx, "inverse"))
        local_truth_table_distance, local_truth_idx, local_truth_mode = min(truth_table_distances)
        local_min_secure_ber = float(min(secure_bers))
        local_idx = int(np.argmin(secure_bers))

        if local_min_secure_ber <= ZERO_OR_INVERSE_EPSILON:
            zero_or_inverse_detected = True

        if local_min_secure_ber < global_min_secure_ber:
            global_min_secure_ber = local_min_secure_ber
            global_min_illegal_position = illegal_position
            global_min_legal_position = int(legal_positions[local_idx])
            global_min_ber_vector = [float(v) for v in position_bers]
            global_min_truth_table_distance = int(local_truth_table_distance)
            global_min_illegal_truth_table = [int(value) for value in illegal_truth_table]
            global_min_truth_table_reference = (
                f"{local_truth_mode}:legal_pos_{int(legal_positions[local_truth_idx])}"
            )

        if stop_below is not None and global_min_secure_ber <= stop_below:
            break

    if global_min_secure_ber == float("inf"):
        return 1.0, None, None, None, False, None, None, None
    return (
        float(global_min_secure_ber),
        global_min_illegal_position,
        global_min_legal_position,
        global_min_ber_vector,
        zero_or_inverse_detected,
        global_min_truth_table_distance,
        global_min_illegal_truth_table,
        global_min_truth_table_reference,
    )


def calculate_worst_code_correlation(
    project_root: str,
    legal_positions: Sequence[int],
    legal_models: Sequence[test.FingerprintModel],
    probes: Sequence[float],
    worst_illegal_position: int | None,
    worst_legal_position: int | None,
) -> tuple[float | None, float | None]:
    if worst_illegal_position is None or worst_legal_position is None:
        return None, None

    probes_array = np.asarray(probes, dtype=float)
    illegal_csv_file = test.build_csv_files_for_positions(
        project_root,
        [worst_illegal_position],
        light_condition=LIGHT_CONDITION,
    )[0]
    illegal_matrix = test.load_selected_rows([illegal_csv_file], probes_array)[0]
    illegal_model = test.extract_fingerprint(probes_array, illegal_matrix, force_positive_first=True)

    legal_idx = list(legal_positions).index(int(worst_legal_position))
    illegal_code = np.asarray(illegal_model.code, dtype=float)
    legal_code = np.asarray(legal_models[legal_idx].code, dtype=float)
    if np.std(illegal_code) == 0 or np.std(legal_code) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(illegal_code, legal_code)[0, 1])
    return corr, abs(corr)


def format_hue_mapping(hue_mapping: dict[tuple[int, ...], int] | None) -> str:
    if not hue_mapping:
        return ""
    items = sorted(hue_mapping.items())
    return "{" + ", ".join(f"{key}: {value}" for key, value in items) + "}"


def generate_relaxed_mapping_candidates(
    legal_models: Sequence[test.FingerprintModel],
    probes: Sequence[float],
    top_k_per_combination: int,
) -> dict[tuple[int, ...], list[int]]:
    probes_array = np.asarray(probes, dtype=float)
    z_list = [np.asarray(model.z, dtype=float) for model in legal_models]
    possible_combinations = list(itertools.product([1, -1], repeat=len(legal_models)))
    strict_candidate_map = test.generate_mapping_candidates(
        list(legal_models),
        probes_array,
        top_k_per_combination=SECURITY_MAPPING_TOP_K,
    )
    candidate_map: dict[tuple[int, ...], list[int]] = {}

    for combination in possible_combinations:
        scored_candidates = []
        for probe_idx, probe in enumerate(probes_array):
            signed_margins = [
                int(sign) * float(z_values[probe_idx])
                for sign, z_values in zip(combination, z_list)
            ]
            matched_count = sum(1 for margin in signed_margins if margin > 0)
            margin_sum = sum(abs(margin) for margin in signed_margins)
            mismatch_penalty = sum(abs(margin) for margin in signed_margins if margin <= 0)
            score = matched_count * 1000.0 + margin_sum - mismatch_penalty * 2.0
            scored_candidates.append((score, int(probe)))

        scored_candidates.sort(reverse=True)
        candidates: list[int] = []
        for probe in strict_candidate_map.get(combination, []):
            if probe not in candidates:
                candidates.append(int(probe))
        for _, probe in scored_candidates:
            if probe not in candidates:
                candidates.append(int(probe))
            if len(candidates) >= top_k_per_combination:
                break
        candidate_map[combination] = candidates

    return candidate_map


def iter_security_aware_mappings(
    legal_models: Sequence[test.FingerprintModel],
    probes: Sequence[float],
) -> tuple[list[dict[tuple[int, ...], int]], int]:
    candidate_map = generate_relaxed_mapping_candidates(
        legal_models,
        probes,
        top_k_per_combination=RELAXED_MAPPING_TOP_K,
    )
    keys = sorted(candidate_map.keys())
    candidate_lists = [candidate_map[key] for key in keys]
    mappings = []
    seen: set[tuple[tuple[tuple[int, ...], int], ...]] = set()
    total_mapping_count = 1
    for candidates in candidate_lists:
        total_mapping_count *= len(candidates)

    for values in itertools.product(*candidate_lists):
        hue_mapping = {key: int(value) for key, value in zip(keys, values)}
        signature = tuple(sorted(hue_mapping.items()))
        if signature in seen:
            continue
        seen.add(signature)
        mappings.append(hue_mapping)
        if len(mappings) >= min(DETERMINISTIC_HUE_MAPPING_PREFIX, total_mapping_count):
            break

    if len(mappings) >= total_mapping_count or len(mappings) >= MAX_HUE_MAPPINGS_PER_PROBE:
        return mappings[:MAX_HUE_MAPPINGS_PER_PROBE], total_mapping_count

    sample_budget = MAX_HUE_MAPPINGS_PER_PROBE - len(mappings)
    seed = int(sum((idx + 1) * float(probe) for idx, probe in enumerate(np.sort(probes)))) + len(probes) * 1009
    rng = random.Random(seed)
    attempts = 0
    max_attempts = sample_budget * 20
    while len(mappings) < MAX_HUE_MAPPINGS_PER_PROBE and attempts < max_attempts:
        attempts += 1
        values = [rng.choice(candidates) for candidates in candidate_lists]
        hue_mapping = {key: int(value) for key, value in zip(keys, values)}
        signature = tuple(sorted(hue_mapping.items()))
        if signature in seen:
            continue
        seen.add(signature)
        mappings.append(hue_mapping)

    return mappings, total_mapping_count


def find_security_aware_hue_mapping(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
    security_stop_below: float | None,
) -> dict:
    legal_models = build_legal_models_only(project_root, legal_positions, probes)
    bit_blocks_pm = generate_exact_bit_blocks(len(legal_positions))
    mappings, total_mapping_candidate_count = iter_security_aware_mappings(legal_models, probes)

    best_result: dict | None = None
    for hue_mapping in mappings:
        legal_ber, legal_position_bers = evaluate_legal_mapping(
            legal_models=legal_models,
            hue_mapping=hue_mapping,
            bit_blocks_pm=bit_blocks_pm,
        )
        if legal_ber > TARGET_LEGAL_BER:
            candidate = {
                "legal_ber": legal_ber,
                "legal_position_bers": legal_position_bers,
                "hue_mapping": hue_mapping,
                "min_illegal_single_ber": None,
                "worst_illegal_position": None,
                "worst_legal_position": None,
                "worst_illegal_ber_vector": None,
                "zero_or_inverse_detected": None,
                "evaluated_mapping_count": len(mappings),
                "total_mapping_candidate_count": total_mapping_candidate_count,
                "worst_code_correlation": None,
                "worst_abs_code_correlation": None,
                "min_truth_table_distance": None,
                "worst_illegal_truth_table": None,
                "worst_truth_table_reference": None,
            }
        else:
            (
                min_illegal_single_ber,
                worst_illegal_position,
                worst_legal_position,
                worst_illegal_ber_vector,
                zero_or_inverse_detected,
                min_truth_table_distance,
                worst_illegal_truth_table,
                worst_truth_table_reference,
            ) = evaluate_illegal_mapping(
                project_root=project_root,
                legal_positions=legal_positions,
                legal_models=legal_models,
                hue_mapping=hue_mapping,
                probes=probes,
                bit_blocks_pm=bit_blocks_pm,
                stop_below=security_stop_below,
            )
            worst_code_correlation, worst_abs_code_correlation = calculate_worst_code_correlation(
                project_root=project_root,
                legal_positions=legal_positions,
                legal_models=legal_models,
                probes=probes,
                worst_illegal_position=worst_illegal_position,
                worst_legal_position=worst_legal_position,
            )
            candidate = {
                "legal_ber": legal_ber,
                "legal_position_bers": legal_position_bers,
                "hue_mapping": hue_mapping,
                "min_illegal_single_ber": min_illegal_single_ber,
                "worst_illegal_position": worst_illegal_position,
                "worst_legal_position": worst_legal_position,
                "worst_illegal_ber_vector": worst_illegal_ber_vector,
                "zero_or_inverse_detected": zero_or_inverse_detected,
                "evaluated_mapping_count": len(mappings),
                "total_mapping_candidate_count": total_mapping_candidate_count,
                "worst_code_correlation": worst_code_correlation,
                "worst_abs_code_correlation": worst_abs_code_correlation,
                "min_truth_table_distance": min_truth_table_distance,
                "worst_illegal_truth_table": worst_illegal_truth_table,
                "worst_truth_table_reference": worst_truth_table_reference,
            }

        candidate["security_satisfied"] = (
            float(candidate["legal_ber"]) <= TARGET_LEGAL_BER
            and candidate["min_illegal_single_ber"] is not None
            and float(candidate["min_illegal_single_ber"]) > SECURITY_THRESHOLD
            and (candidate.get("min_truth_table_distance") or 0) >= MIN_TRUTH_TABLE_DISTANCE
            and not bool(candidate["zero_or_inverse_detected"])
        )

        if is_better_candidate(candidate, best_result):
            best_result = candidate
            if candidate["security_satisfied"]:
                break

    if best_result is None:
        raise RuntimeError("No hue mapping candidates were generated")
    return best_result


def evaluate_candidate(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
    coarse_rng_seed: int,
) -> dict:
    result = find_security_aware_hue_mapping(
        project_root=project_root,
        legal_positions=legal_positions,
        probes=probes,
        security_stop_below=SECURITY_THRESHOLD,
    )
    result["security_satisfied"] = (
        float(result["legal_ber"]) <= TARGET_LEGAL_BER
        and result["min_illegal_single_ber"] is not None
        and float(result["min_illegal_single_ber"]) > SECURITY_THRESHOLD
        and (result.get("min_truth_table_distance") or 0) >= MIN_TRUTH_TABLE_DISTANCE
        and not bool(result["zero_or_inverse_detected"])
    )
    return result


def candidate_rank_key(candidate: dict) -> tuple[float, float, float, float, float, float, float, float, float]:
    satisfied = 1.0 if candidate.get("security_satisfied") else 0.0
    legal_ok = 1.0 if candidate["legal_ber"] <= TARGET_LEGAL_BER else 0.0
    zero_inverse_ok = 0.0 if candidate.get("zero_or_inverse_detected") else 1.0
    min_illegal = (
        float(candidate["min_illegal_single_ber"])
        if candidate["min_illegal_single_ber"] is not None
        else -1.0
    )
    truth_table_distance = float(candidate.get("min_truth_table_distance") or -1)
    worst_abs_corr = (
        float(candidate["worst_abs_code_correlation"])
        if candidate.get("worst_abs_code_correlation") is not None
        else 1.0
    )
    mapping_space = float(candidate.get("total_mapping_candidate_count") or 0)
    probe_count = float(candidate.get("probe_count") or len(candidate.get("probes", [])) or 0)
    return (
        satisfied,
        legal_ok,
        zero_inverse_ok,
        truth_table_distance,
        min_illegal,
        -worst_abs_corr,
        mapping_space,
        probe_count,
        -float(candidate["legal_ber"]),
    )


def is_better_candidate(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    return candidate_rank_key(candidate) > candidate_rank_key(incumbent)


def verify_candidate(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
    verify_seed: int,
) -> dict:
    result = find_security_aware_hue_mapping(
        project_root=project_root,
        legal_positions=legal_positions,
        probes=probes,
        security_stop_below=None,
    )
    result["security_satisfied"] = (
        float(result["legal_ber"]) <= TARGET_LEGAL_BER
        and result["min_illegal_single_ber"] is not None
        and float(result["min_illegal_single_ber"]) > SECURITY_THRESHOLD
        and (result.get("min_truth_table_distance") or 0) >= MIN_TRUTH_TABLE_DISTANCE
        and not bool(result["zero_or_inverse_detected"])
    )
    return result


def get_all_probes(project_root: str, legal_positions: Sequence[int]) -> list[float]:
    csv_file = test.build_csv_files_for_positions(project_root, legal_positions, light_condition=LIGHT_CONDITION)[0]
    first_df = pd.read_csv(csv_file)
    return (5 + np.arange(len(first_df)) * 5).astype(float).tolist()


def try_local_refinement(
    project_root: str,
    legal_positions: Sequence[int],
    initial_probes: Sequence[float],
    all_probes: Sequence[float],
    rng: random.Random,
    seed_prefix: int,
) -> tuple[np.ndarray, dict]:
    current = np.sort(np.asarray(initial_probes, dtype=float))
    current_eval = evaluate_candidate(project_root, legal_positions, current, seed_prefix)

    for round_idx in range(LOCAL_ROUNDS):
        improved = False
        for probe_idx in range(len(current)):
            pool = [p for p in all_probes if p not in current]
            if not pool:
                continue
            sample_size = min(LOCAL_NEIGHBOR_SAMPLES, len(pool))
            for candidate_probe in rng.sample(pool, sample_size):
                trial = current.copy()
                trial[probe_idx] = candidate_probe
                trial = np.sort(trial)
                if not test.is_valid_probe_set(trial, min_interval=min_interval_for_probe_count(len(trial))):
                    continue

                trial_seed = seed_prefix + round_idx * 1000 + probe_idx * 100 + int(candidate_probe)
                trial_eval = evaluate_candidate(project_root, legal_positions, trial, trial_seed)
                if is_better_candidate(trial_eval, current_eval):
                    current = trial
                    current_eval = trial_eval
                    improved = True
        if not improved:
            break

    return current, current_eval


def search_security_aware_probes(
    project_root: str,
    legal_positions: Sequence[int],
    baseline_probe_count: int,
    rng: random.Random,
    combination_index: int,
) -> dict | None:
    all_probes = get_all_probes(project_root, legal_positions)
    candidate_counts = sorted({
        max(MIN_PROBE_COUNT, baseline_probe_count - 3),
        max(MIN_PROBE_COUNT, baseline_probe_count - 2),
        max(MIN_PROBE_COUNT, baseline_probe_count - 1),
        baseline_probe_count,
        5,
        6,
        7,
        8,
    })
    candidate_counts = [count for count in candidate_counts if MIN_PROBE_COUNT <= count <= MAX_PROBE_COUNT]
    best_candidate: dict | None = None
    shortlisted_candidates: list[dict] = []

    for epoch in range(SEARCH_EPOCHS):
        print(f"  Search epoch {epoch + 1}/{SEARCH_EPOCHS}")
        for probe_count in candidate_counts:
            min_interval = min_interval_for_probe_count(probe_count)
            print(f"    Trying probe_count={probe_count}, min_interval={min_interval}")
            epoch_candidates: list[dict] = []
            for candidate_idx in range(MAX_RANDOM_CANDIDATES_PER_COUNT):
                probes = np.array(rng.sample(all_probes, probe_count), dtype=float)
                probes = np.sort(probes)
                if not test.is_valid_probe_set(probes, min_interval=min_interval):
                    continue

                coarse_seed = (
                    SELECTION_SEED
                    + combination_index * 1000000
                    + epoch * 100000
                    + probe_count * 1000
                    + candidate_idx
                )
                candidate_eval = evaluate_candidate(project_root, legal_positions, probes, coarse_seed)
                candidate_record = {
                    "probes": probes.copy(),
                    "probe_count": probe_count,
                    "min_interval": min_interval,
                    **candidate_eval,
                }
                epoch_candidates.append(candidate_record)

                if is_better_candidate(candidate_record, best_candidate):
                    best_candidate = candidate_record
                    if candidate_record["security_satisfied"]:
                        verify_eval = verify_candidate(
                            project_root=project_root,
                            legal_positions=legal_positions,
                            probes=probes,
                            verify_seed=coarse_seed + 500000,
                        )
                        verified_candidate = {
                            "probes": probes.copy(),
                            "probe_count": probe_count,
                            "min_interval": min_interval,
                            **verify_eval,
                        }
                        if verified_candidate["security_satisfied"]:
                            return verified_candidate

            epoch_candidates.sort(key=candidate_rank_key, reverse=True)
            shortlisted_candidates.extend(epoch_candidates[:TOP_K_CANDIDATES_PER_COUNT])

    shortlisted_candidates.sort(key=candidate_rank_key, reverse=True)

    for shortlist_idx, candidate in enumerate(shortlisted_candidates[: max(12, TOP_K_CANDIDATES_PER_COUNT * 2)], start=1):
        candidate_seed = (
            SELECTION_SEED
            + combination_index * 2000000
            + shortlist_idx * 10000
        )
        refined_probes, refined_eval = try_local_refinement(
            project_root=project_root,
            legal_positions=legal_positions,
            initial_probes=candidate["probes"],
            all_probes=all_probes,
            rng=rng,
            seed_prefix=candidate_seed,
        )

        refined_candidate = {
            "probes": refined_probes.copy(),
            "probe_count": len(refined_probes),
            "min_interval": min_interval_for_probe_count(len(refined_probes)),
            **refined_eval,
        }
        if is_better_candidate(refined_candidate, best_candidate):
            best_candidate = refined_candidate

        verify_eval = verify_candidate(
            project_root=project_root,
            legal_positions=legal_positions,
            probes=refined_probes,
            verify_seed=candidate_seed + 500000,
        )
        verified_candidate = {
            "probes": refined_probes.copy(),
            "probe_count": len(refined_probes),
            "min_interval": min_interval_for_probe_count(len(refined_probes)),
            **verify_eval,
        }
        if verified_candidate["security_satisfied"]:
            return verified_candidate
        if is_better_candidate(verified_candidate, best_candidate):
            best_candidate = verified_candidate

    return best_candidate


def write_results(results_file: str, rows: Sequence[dict]) -> None:
    fieldnames = [
        "position_combination",
        "baseline_probe_count",
        "baseline_probes",
        "baseline_best_ber",
        "baseline_test_ber",
        "baseline_worst_illegal_single_ber",
        "baseline_worst_illegal_position",
        "new_probe_count",
        "new_min_interval",
        "new_probes",
        "new_hue_mapping",
        "new_legal_ber",
        "new_legal_position_bers",
        "new_min_illegal_single_ber",
        "new_worst_illegal_position",
        "new_worst_legal_position",
        "new_worst_illegal_ber_vector",
        "zero_or_inverse_detected",
        "evaluated_mapping_count",
        "total_mapping_candidate_count",
        "worst_code_correlation",
        "worst_abs_code_correlation",
        "min_truth_table_distance",
        "worst_illegal_truth_table",
        "worst_truth_table_reference",
        "security_satisfied",
    ]
    writable_results_file = resolve_writable_results_file(results_file)
    with open(writable_results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_safe_probe_results(results_file: str, rows: Sequence[dict]) -> None:
    fieldnames = [
        "position_combination",
        "safe_probe_count",
        "safe_min_interval",
        "safe_probes",
        "safe_hue_mapping",
        "legal_ber",
        "legal_position_bers",
        "min_illegal_single_ber",
        "worst_illegal_position",
        "worst_legal_position",
        "worst_illegal_ber_vector",
        "zero_or_inverse_detected",
        "evaluated_mapping_count",
        "total_mapping_candidate_count",
        "worst_code_correlation",
        "worst_abs_code_correlation",
        "min_truth_table_distance",
        "worst_illegal_truth_table",
        "worst_truth_table_reference",
    ]
    safe_rows = []
    for row in rows:
        if row.get("security_satisfied") != "yes":
            continue
        safe_rows.append({
            "position_combination": row["position_combination"],
            "safe_probe_count": row["new_probe_count"],
            "safe_min_interval": row["new_min_interval"],
            "safe_probes": row["new_probes"],
            "safe_hue_mapping": row["new_hue_mapping"],
            "legal_ber": row["new_legal_ber"],
            "legal_position_bers": row["new_legal_position_bers"],
            "min_illegal_single_ber": row["new_min_illegal_single_ber"],
            "worst_illegal_position": row["new_worst_illegal_position"],
            "worst_legal_position": row["new_worst_legal_position"],
            "worst_illegal_ber_vector": row["new_worst_illegal_ber_vector"],
            "zero_or_inverse_detected": row["zero_or_inverse_detected"],
            "evaluated_mapping_count": row["evaluated_mapping_count"],
            "total_mapping_candidate_count": row["total_mapping_candidate_count"],
            "worst_code_correlation": row["worst_code_correlation"],
            "worst_abs_code_correlation": row["worst_abs_code_correlation"],
            "min_truth_table_distance": row["min_truth_table_distance"],
            "worst_illegal_truth_table": row["worst_illegal_truth_table"],
            "worst_truth_table_reference": row["worst_truth_table_reference"],
        })

    writable_results_file = resolve_writable_results_file(results_file)
    with open(writable_results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(safe_rows)


def safe_probe_fieldnames() -> list[str]:
    return [
        "position_combination",
        "safe_probe_count",
        "safe_min_interval",
        "safe_probes",
        "safe_hue_mapping",
        "legal_ber",
        "legal_position_bers",
        "min_illegal_single_ber",
        "worst_illegal_position",
        "worst_legal_position",
        "worst_illegal_ber_vector",
        "zero_or_inverse_detected",
        "evaluated_mapping_count",
        "total_mapping_candidate_count",
        "worst_code_correlation",
        "worst_abs_code_correlation",
        "min_truth_table_distance",
        "worst_illegal_truth_table",
        "worst_truth_table_reference",
    ]


def load_existing_safe_probe_rows(results_file: str) -> list[dict]:
    if not os.path.exists(results_file):
        return []
    with open(results_file, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_existing_output_rows(results_file: str) -> list[dict]:
    if not os.path.exists(results_file):
        return []
    with open(results_file, "r", newline="", encoding="utf-8-sig") as f:
        return [
            row
            for row in csv.DictReader(f)
            if row.get("position_combination")
        ]


def safe_probe_row_from_result(row: dict) -> dict:
    return {
        "position_combination": row["position_combination"],
        "safe_probe_count": row["new_probe_count"],
        "safe_min_interval": row["new_min_interval"],
        "safe_probes": row["new_probes"],
        "safe_hue_mapping": row["new_hue_mapping"],
        "legal_ber": row["new_legal_ber"],
        "legal_position_bers": row["new_legal_position_bers"],
        "min_illegal_single_ber": row["new_min_illegal_single_ber"],
        "worst_illegal_position": row["new_worst_illegal_position"],
        "worst_legal_position": row["new_worst_legal_position"],
        "worst_illegal_ber_vector": row["new_worst_illegal_ber_vector"],
        "zero_or_inverse_detected": row["zero_or_inverse_detected"],
        "evaluated_mapping_count": row["evaluated_mapping_count"],
        "total_mapping_candidate_count": row["total_mapping_candidate_count"],
        "worst_code_correlation": row["worst_code_correlation"],
        "worst_abs_code_correlation": row["worst_abs_code_correlation"],
        "min_truth_table_distance": row["min_truth_table_distance"],
        "worst_illegal_truth_table": row["worst_illegal_truth_table"],
        "worst_truth_table_reference": row["worst_truth_table_reference"],
    }


def upsert_safe_probe_row(safe_rows: list[dict], result_row: dict) -> list[dict]:
    if result_row.get("security_satisfied") != "yes":
        return safe_rows

    new_safe_row = safe_probe_row_from_result(result_row)
    combination = new_safe_row["position_combination"]
    filtered_rows = [row for row in safe_rows if row.get("position_combination") != combination]
    filtered_rows.append(new_safe_row)
    return filtered_rows


def write_safe_probe_rows(results_file: str, safe_rows: Sequence[dict]) -> None:
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=safe_probe_fieldnames())
        writer.writeheader()
        writer.writerows(safe_rows)


def resolve_writable_results_file(results_file: str) -> str:
    if results_file in FALLBACK_RESULTS_FILES:
        return FALLBACK_RESULTS_FILES[results_file]

    if can_open_for_overwrite(results_file):
        return results_file

    directory, filename = os.path.split(results_file)
    stem, extension = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback_file = os.path.join(directory, f"{stem}_{timestamp}{extension}")
    FALLBACK_RESULTS_FILES[results_file] = fallback_file
    print(f"Result file is locked; writing to fallback file: {fallback_file}")
    return fallback_file


def can_open_for_overwrite(results_file: str) -> bool:
    try:
        with open(results_file, "a", newline="", encoding="utf-8-sig"):
            pass
    except PermissionError:
        return False
    return True


def format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def run_security_aware_probe_search() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    random_results_file = os.path.join(project_root, "test-3", SOURCE_RANDOM_RESULTS_FILENAME)
    output_results_file = os.path.join(project_root, "test-3", OUTPUT_RESULTS_FILENAME)
    safe_probe_results_file = os.path.join(project_root, "test-3", SAFE_PROBE_RESULTS_FILENAME)
    if not can_open_for_overwrite(safe_probe_results_file):
        raise PermissionError(
            f"Safe probe file is locked. Close it before running: {safe_probe_results_file}"
        )

    search_configs = build_all_combination_configs(project_root, random_results_file)
    safe_probe_rows = load_existing_safe_probe_rows(safe_probe_results_file)
    result_rows = load_existing_output_rows(output_results_file)
    completed_safe_combinations = {
        row["position_combination"]
        for row in safe_probe_rows
        if row.get("position_combination")
    }
    attempted_combinations = {
        row["position_combination"]
        for row in result_rows
        if row.get("position_combination")
    }
    pending_count = sum(
        1
        for config in search_configs
        if str(config["position_combination"]) not in completed_safe_combinations
        and str(config["position_combination"]) not in attempted_combinations
    )
    print(
        f"Total combinations={len(search_configs)}, "
        f"safe_existing={len(completed_safe_combinations)}, "
        f"attempted_existing={len(attempted_combinations)}, "
        f"pending={pending_count}"
    )
    rng = random.Random(SELECTION_SEED)

    for idx, config in enumerate(search_configs, start=1):
        positions = config["position_combination"]
        if str(positions) in completed_safe_combinations:
            print(f"[{idx}/{len(search_configs)}] Skip {positions}: safe probes already exist.")
            continue
        if str(positions) in attempted_combinations:
            print(f"[{idx}/{len(search_configs)}] Skip {positions}: already attempted in output results.")
            continue

        baseline_worst = config.get("worst_illegal_single_ber")
        baseline_worst_text = "N/A" if baseline_worst is None else f"{float(baseline_worst):.6f}"
        print(
            f"[{idx}/{len(search_configs)}] Searching combination {positions}; "
            f"baseline_worst_illegal_ber={baseline_worst_text}"
        )
        best_result = search_security_aware_probes(
            project_root=project_root,
            legal_positions=positions,
            baseline_probe_count=config["best_probe_count"],
            rng=rng,
            combination_index=idx,
        )

        if best_result is None:
            row = {
                "position_combination": str(positions),
                "baseline_probe_count": config["best_probe_count"],
                "baseline_probes": format_optional_probes(config.get("best_probes")),
                "baseline_best_ber": format_optional_float(config.get("best_ber")),
                "baseline_test_ber": format_optional_float(config.get("test_ber")),
                "baseline_worst_illegal_single_ber": format_optional_float(config.get("worst_illegal_single_ber")),
                "baseline_worst_illegal_position": config.get("worst_illegal_position") or "",
                "new_probe_count": "",
                "new_min_interval": "",
                "new_probes": "",
                "new_hue_mapping": "",
                "new_legal_ber": "",
                "new_legal_position_bers": "",
                "new_min_illegal_single_ber": "",
                "new_worst_illegal_position": "",
                "new_worst_legal_position": "",
                "new_worst_illegal_ber_vector": "",
                "zero_or_inverse_detected": "",
                "evaluated_mapping_count": "",
                "total_mapping_candidate_count": "",
                "worst_code_correlation": "",
                "worst_abs_code_correlation": "",
                "min_truth_table_distance": "",
                "worst_illegal_truth_table": "",
                "worst_truth_table_reference": "",
                "security_satisfied": "no_result",
            }
        else:
            row = {
                "position_combination": str(positions),
                "baseline_probe_count": config["best_probe_count"],
                "baseline_probes": format_optional_probes(config.get("best_probes")),
                "baseline_best_ber": format_optional_float(config.get("best_ber")),
                "baseline_test_ber": format_optional_float(config.get("test_ber")),
                "baseline_worst_illegal_single_ber": format_optional_float(config.get("worst_illegal_single_ber")),
                "baseline_worst_illegal_position": config.get("worst_illegal_position") or "",
                "new_probe_count": best_result["probe_count"],
                "new_min_interval": best_result.get("min_interval", min_interval_for_probe_count(best_result["probe_count"])),
                "new_probes": format_probes(best_result["probes"]),
                "new_hue_mapping": format_hue_mapping(best_result.get("hue_mapping")),
                "new_legal_ber": f"{best_result['legal_ber']:.6f}",
                "new_legal_position_bers": (
                    ""
                    if best_result.get("legal_position_bers") is None
                    else "[" + ", ".join(f"{v:.6f}" for v in best_result["legal_position_bers"]) + "]"
                ),
                "new_min_illegal_single_ber": format_optional_float(best_result["min_illegal_single_ber"]),
                "new_worst_illegal_position": best_result["worst_illegal_position"],
                "new_worst_legal_position": best_result.get("worst_legal_position"),
                "new_worst_illegal_ber_vector": (
                    ""
                    if best_result["worst_illegal_ber_vector"] is None
                    else "[" + ", ".join(f"{v:.6f}" for v in best_result["worst_illegal_ber_vector"]) + "]"
                ),
                "zero_or_inverse_detected": "yes" if best_result.get("zero_or_inverse_detected") else "no",
                "evaluated_mapping_count": best_result.get("evaluated_mapping_count", ""),
                "total_mapping_candidate_count": best_result.get("total_mapping_candidate_count", ""),
                "worst_code_correlation": format_optional_float(best_result.get("worst_code_correlation")),
                "worst_abs_code_correlation": format_optional_float(best_result.get("worst_abs_code_correlation")),
                "min_truth_table_distance": best_result.get("min_truth_table_distance", ""),
                "worst_illegal_truth_table": format_truth_table(best_result.get("worst_illegal_truth_table") or []),
                "worst_truth_table_reference": best_result.get("worst_truth_table_reference", ""),
                "security_satisfied": "yes" if best_result["security_satisfied"] else "no",
            }
            print(
                f"  New result: legal_ber={best_result['legal_ber']:.6f}, "
                f"min_illegal_single_ber={format_optional_float(best_result['min_illegal_single_ber']) or 'N/A'}, "
                f"satisfied={best_result['security_satisfied']}"
            )

        if best_result is not None:
            print(
                f"  Truth-table diagnostics: min_distance={best_result.get('min_truth_table_distance', 'N/A')}, "
                f"reference={best_result.get('worst_truth_table_reference', '')}"
            )
            print(
                f"  Diagnostics: mapping={best_result.get('evaluated_mapping_count', '')}/"
                f"{best_result.get('total_mapping_candidate_count', '')}, "
                f"worst_abs_corr={format_optional_float(best_result.get('worst_abs_code_correlation')) or 'N/A'}"
            )

        result_rows.append(row)
        attempted_combinations.add(row["position_combination"])
        safe_probe_rows = upsert_safe_probe_row(safe_probe_rows, row)
        if row.get("security_satisfied") == "yes":
            completed_safe_combinations.add(row["position_combination"])
        write_results(output_results_file, result_rows)
        write_safe_probe_rows(safe_probe_results_file, safe_probe_rows)

    print(f"Results saved to: {output_results_file}")
    print(f"Safe probes saved to: {safe_probe_results_file}")
    return output_results_file


def main() -> None:
    run_security_aware_probe_search()


if __name__ == "__main__":
    main()
