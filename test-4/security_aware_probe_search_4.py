#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security-aware probe and hue-mapping search for 4 legal positions.

The script searches every 4-position combination in data/15pro/white and writes
successful probe sets to security_aware_safe_probes_4.csv. Existing successful
combinations in that CSV are skipped on restart.

Security target: 0.2 < min_illegal_ber < 0.8, where min_illegal_ber is the
secure BER min(BER, 1-BER) over all illegal positions and legal routes.
"""

from __future__ import annotations

import ast
import csv
import itertools
import os
import random
import sys
import types
from typing import Sequence

import numpy as np
import pandas as pd


LEGAL_POSITION_COUNT = 4
LIGHT_CONDITION = "white"
OUTPUT_RESULTS_FILENAME = "security_aware_safe_probes_4.csv"
TARGET_LEGAL_BER = 0.02
MIN_ILLEGAL_BER = 0.2
MAX_ILLEGAL_BER = 0.8
RELAXED_MAPPING_TOP_K = 4
MAX_HUE_MAPPINGS_PER_PROBE = 2048
DETERMINISTIC_HUE_MAPPING_PREFIX = 512
SEARCH_EPOCHS = 4
MAX_RANDOM_CANDIDATES_PER_COUNT = 320
TOP_K_CANDIDATES_PER_COUNT = 8
LOCAL_NEIGHBOR_SAMPLES = 16
LOCAL_ROUNDS = 4
MIN_PROBE_COUNT = 5
MAX_PROBE_COUNT = 30
DEFAULT_BASELINE_PROBE_COUNT = 8
SELECTION_SEED = 20260401


def load_test_module() -> types.ModuleType:
    module_name = "test_3_simple_runtime_security_search_4"
    module_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test-3",
        "test_3_simple.py",
    )
    with open(module_path, "r", encoding="utf-8-sig") as f:
        source = f.read().lstrip("\ufeff")

    module = types.ModuleType(module_name)
    module.__file__ = module_path
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(compile(source, module_path, "exec"), module.__dict__)
    return module


test = load_test_module()


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def format_hue_mapping(hue_mapping: dict[tuple[int, ...], int] | None) -> str:
    if not hue_mapping:
        return ""
    return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(hue_mapping.items())) + "}"


def get_available_positions(project_root: str) -> list[int]:
    data_dir = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION)
    positions = []
    for entry in os.listdir(data_dir):
        if entry.endswith(".csv") and os.path.splitext(entry)[0].isdigit():
            positions.append(int(os.path.splitext(entry)[0]))
    return sorted(positions)


def min_interval_for_probe_count(probe_count: int) -> int:
    if probe_count <= 12:
        return 30
    if probe_count <= 16:
        return 20
    if probe_count <= 22:
        return 15
    return 10


def get_all_probes(project_root: str, legal_positions: Sequence[int]) -> list[float]:
    csv_file = test.build_csv_files_for_positions(project_root, legal_positions, light_condition=LIGHT_CONDITION)[0]
    first_df = pd.read_csv(csv_file)
    return (5 + np.arange(len(first_df)) * 5).astype(float).tolist()


def build_legal_models(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
) -> list[test.FingerprintModel]:
    csv_files = test.build_csv_files_for_positions(project_root, legal_positions, light_condition=LIGHT_CONDITION)
    probes_array = np.asarray(probes, dtype=float)
    matrices = test.load_selected_rows(csv_files, probes_array)
    models = [test.extract_fingerprint(probes_array, matrix, force_positive_first=True) for matrix in matrices]
    return test.align_model_directions(models)


def generate_exact_bit_blocks(num_positions: int) -> list[np.ndarray]:
    return [np.asarray(bits_pm, dtype=int) for bits_pm in itertools.product([-1, 1], repeat=num_positions)]


def corrected_ber(ber: float) -> float:
    return min(float(ber), 1.0 - float(ber))


def calculate_position_bers(results: Sequence[dict]) -> list[float]:
    position_errors = np.zeros(len(results[0]["per_position"]), dtype=float)
    position_total = np.zeros(len(results[0]["per_position"]), dtype=float)
    for result in results:
        true_bits = result["bits_bin"]
        decoded_bits = [decode.bit_hat_bin for decode in result["per_position"]]
        for idx, (true_bit, decoded_bit) in enumerate(zip(true_bits, decoded_bits)):
            position_total[idx] += 1
            if int(true_bit) != int(decoded_bit):
                position_errors[idx] += 1
    return (position_errors / np.maximum(position_total, 1.0)).tolist()


def evaluate_legal_mapping(
    legal_models: Sequence[test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    bit_blocks_pm: Sequence[np.ndarray],
) -> tuple[float, list[float]]:
    results = test.simulate_blocks(list(legal_models), list(bit_blocks_pm), hue_mapping)
    raw_bers = calculate_position_bers(results)
    corrected_bers = [corrected_ber(ber) for ber in raw_bers]
    return float(np.mean(corrected_bers)), [float(v) for v in raw_bers]


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
        top_k_per_combination=min(3, top_k_per_combination),
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
    total_mapping_count = 1
    for candidates in candidate_lists:
        total_mapping_count *= len(candidates)

    mappings = []
    seen: set[tuple[tuple[tuple[int, ...], int], ...]] = set()
    for values in itertools.product(*candidate_lists):
        hue_mapping = {key: int(value) for key, value in zip(keys, values)}
        signature = tuple(sorted(hue_mapping.items()))
        if signature not in seen:
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
    while len(mappings) < MAX_HUE_MAPPINGS_PER_PROBE and attempts < sample_budget * 20:
        attempts += 1
        values = [rng.choice(candidates) for candidates in candidate_lists]
        hue_mapping = {key: int(value) for key, value in zip(keys, values)}
        signature = tuple(sorted(hue_mapping.items()))
        if signature in seen:
            continue
        seen.add(signature)
        mappings.append(hue_mapping)
    return mappings, total_mapping_count


def evaluate_illegal_mapping(
    project_root: str,
    legal_positions: Sequence[int],
    legal_models: Sequence[test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    probes: Sequence[float],
    bit_blocks_pm: Sequence[np.ndarray],
) -> dict:
    probes_array = np.asarray(probes, dtype=float)
    legal_codes = [model.code for model in legal_models]
    probe_to_row = test.build_probe_to_row(probes_array)
    available_positions = get_available_positions(project_root)

    global_min_secure_ber = float("inf")
    global_raw_ber_at_min = 0.0
    global_worst_illegal_position: int | None = None
    global_worst_legal_position: int | None = None
    global_worst_ber_vector: list[float] | None = None
    all_secure_bers: list[float] = []

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
                if int(illegal_dec.bit_hat_bin) != int(true_bits[idx]):
                    position_errors[idx] += 1
        raw_bers = (position_errors / np.maximum(position_total, 1.0)).tolist()
        secure_bers = [corrected_ber(ber) for ber in raw_bers]
        all_secure_bers.extend(secure_bers)
        local_min_secure_ber = float(min(secure_bers))
        local_idx = int(np.argmin(secure_bers))
        if local_min_secure_ber < global_min_secure_ber:
            global_min_secure_ber = local_min_secure_ber
            global_raw_ber_at_min = float(raw_bers[local_idx])
            global_worst_illegal_position = illegal_position
            global_worst_legal_position = int(legal_positions[local_idx])
            global_worst_ber_vector = [float(v) for v in raw_bers]

    return {
        "min_illegal_ber": float(global_min_secure_ber if all_secure_bers else 0.0),
        "raw_ber_at_min_illegal_ber": float(global_raw_ber_at_min),
        "average_illegal_ber": float(np.mean(all_secure_bers)) if all_secure_bers else 0.0,
        "worst_illegal_position": global_worst_illegal_position,
        "worst_legal_position": global_worst_legal_position,
        "worst_illegal_ber_vector": global_worst_ber_vector,
    }


def find_security_aware_hue_mapping(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
) -> dict:
    legal_models = build_legal_models(project_root, legal_positions, probes)
    bit_blocks_pm = generate_exact_bit_blocks(len(legal_positions))
    mappings, total_mapping_candidate_count = iter_security_aware_mappings(legal_models, probes)
    best_result: dict | None = None

    for hue_mapping in mappings:
        legal_ber, legal_position_bers = evaluate_legal_mapping(legal_models, hue_mapping, bit_blocks_pm)
        if legal_ber > TARGET_LEGAL_BER:
            candidate = {
                "legal_ber": legal_ber,
                "legal_position_bers": legal_position_bers,
                "hue_mapping": hue_mapping,
                "min_illegal_ber": None,
                "raw_ber_at_min_illegal_ber": None,
                "average_illegal_ber": None,
                "worst_illegal_position": None,
                "worst_legal_position": None,
                "worst_illegal_ber_vector": None,
                "evaluated_mapping_count": len(mappings),
                "total_mapping_candidate_count": total_mapping_candidate_count,
            }
        else:
            illegal_result = evaluate_illegal_mapping(
                project_root=project_root,
                legal_positions=legal_positions,
                legal_models=legal_models,
                hue_mapping=hue_mapping,
                probes=probes,
                bit_blocks_pm=bit_blocks_pm,
            )
            candidate = {
                "legal_ber": legal_ber,
                "legal_position_bers": legal_position_bers,
                "hue_mapping": hue_mapping,
                "evaluated_mapping_count": len(mappings),
                "total_mapping_candidate_count": total_mapping_candidate_count,
                **illegal_result,
            }
        candidate["security_satisfied"] = (
            float(candidate["legal_ber"]) <= TARGET_LEGAL_BER
            and candidate["min_illegal_ber"] is not None
            and MIN_ILLEGAL_BER < float(candidate["min_illegal_ber"]) < MAX_ILLEGAL_BER
        )
        if is_better_candidate(candidate, best_result):
            best_result = candidate
            if candidate["security_satisfied"]:
                break

    if best_result is None:
        raise RuntimeError("No hue mapping candidates were generated")
    return best_result


def candidate_rank_key(candidate: dict) -> tuple[float, float, float, float, float]:
    satisfied = 1.0 if candidate.get("security_satisfied") else 0.0
    legal_ok = 1.0 if candidate["legal_ber"] <= TARGET_LEGAL_BER else 0.0
    min_illegal = float(candidate["min_illegal_ber"]) if candidate.get("min_illegal_ber") is not None else -1.0
    distance_to_half = -abs(min_illegal - 0.5) if min_illegal >= 0 else -1.0
    mapping_space = float(candidate.get("total_mapping_candidate_count") or 0)
    return satisfied, legal_ok, min_illegal, distance_to_half, mapping_space


def is_better_candidate(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    return candidate_rank_key(candidate) > candidate_rank_key(incumbent)


def evaluate_candidate(project_root: str, legal_positions: Sequence[int], probes: Sequence[float]) -> dict:
    result = find_security_aware_hue_mapping(project_root, legal_positions, probes)
    result["security_satisfied"] = (
        float(result["legal_ber"]) <= TARGET_LEGAL_BER
        and result["min_illegal_ber"] is not None
        and MIN_ILLEGAL_BER < float(result["min_illegal_ber"]) < MAX_ILLEGAL_BER
    )
    return result


def try_local_refinement(
    project_root: str,
    legal_positions: Sequence[int],
    initial_probes: Sequence[float],
    all_probes: Sequence[float],
    rng: random.Random,
) -> tuple[np.ndarray, dict]:
    current = np.sort(np.asarray(initial_probes, dtype=float))
    current_eval = evaluate_candidate(project_root, legal_positions, current)
    for _ in range(LOCAL_ROUNDS):
        improved = False
        for probe_idx in range(len(current)):
            pool = [probe for probe in all_probes if probe not in current]
            if not pool:
                continue
            for candidate_probe in rng.sample(pool, min(LOCAL_NEIGHBOR_SAMPLES, len(pool))):
                trial = current.copy()
                trial[probe_idx] = candidate_probe
                trial = np.sort(trial)
                if not test.is_valid_probe_set(trial, min_interval=min_interval_for_probe_count(len(trial))):
                    continue
                trial_eval = evaluate_candidate(project_root, legal_positions, trial)
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
    rng: random.Random,
) -> dict:
    all_probes = get_all_probes(project_root, legal_positions)
    candidate_counts = [count for count in range(MIN_PROBE_COUNT, MAX_PROBE_COUNT + 1)]
    best_candidate: dict | None = None
    shortlisted_candidates: list[dict] = []
    for epoch in range(SEARCH_EPOCHS):
        print(f"  Search epoch {epoch + 1}/{SEARCH_EPOCHS}")
        for probe_count in candidate_counts:
            min_interval = min_interval_for_probe_count(probe_count)
            print(f"    Trying probe_count={probe_count}, min_interval={min_interval}")
            epoch_candidates: list[dict] = []
            for _ in range(MAX_RANDOM_CANDIDATES_PER_COUNT):
                probes = np.sort(np.asarray(rng.sample(all_probes, probe_count), dtype=float))
                if not test.is_valid_probe_set(probes, min_interval=min_interval):
                    continue
                candidate_eval = evaluate_candidate(project_root, legal_positions, probes)
                candidate_record = {
                    "probes": probes.copy(),
                    "probe_count": probe_count,
                    "min_interval": min_interval,
                    **candidate_eval,
                }
                epoch_candidates.append(candidate_record)
                if is_better_candidate(candidate_record, best_candidate):
                    best_candidate = candidate_record
            epoch_candidates.sort(key=candidate_rank_key, reverse=True)
            shortlisted_candidates.extend(epoch_candidates[:TOP_K_CANDIDATES_PER_COUNT])

    shortlisted_candidates.sort(key=candidate_rank_key, reverse=True)
    for candidate in shortlisted_candidates[: max(12, TOP_K_CANDIDATES_PER_COUNT * 2)]:
        refined_probes, refined_eval = try_local_refinement(
            project_root=project_root,
            legal_positions=legal_positions,
            initial_probes=candidate["probes"],
            all_probes=all_probes,
            rng=rng,
        )
        refined_candidate = {
            "probes": refined_probes.copy(),
            "probe_count": len(refined_probes),
            "min_interval": min_interval_for_probe_count(len(refined_probes)),
            **refined_eval,
        }
        if refined_candidate["security_satisfied"]:
            return refined_candidate
        if is_better_candidate(refined_candidate, best_candidate):
            best_candidate = refined_candidate
    if best_candidate is None:
        raise RuntimeError("No probe candidates were evaluated")
    return best_candidate


def generate_position_combinations(project_root: str) -> list[tuple[int, ...]]:
    return list(itertools.combinations(get_available_positions(project_root), LEGAL_POSITION_COUNT))


def load_existing_results(results_file: str) -> list[dict]:
    if not os.path.exists(results_file):
        return []
    with open(results_file, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_results(results_file: str, rows: Sequence[dict]) -> None:
    fieldnames = [
        "position_combination",
        "probe_count",
        "min_interval",
        "probes",
        "hue_mapping",
        "legal_ber",
        "legal_position_bers",
        "min_illegal_ber",
        "raw_ber_at_min_illegal_ber",
        "average_illegal_ber",
        "worst_illegal_position",
        "worst_legal_position",
        "worst_illegal_ber_vector",
        "evaluated_mapping_count",
        "total_mapping_candidate_count",
        "security_satisfied",
    ]
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_from_result(legal_positions: Sequence[int], result: dict) -> dict:
    return {
        "position_combination": str(tuple(legal_positions)),
        "probe_count": result["probe_count"],
        "min_interval": result["min_interval"],
        "probes": format_probes(result["probes"]),
        "hue_mapping": format_hue_mapping(result.get("hue_mapping")),
        "legal_ber": f"{result['legal_ber']:.6f}",
        "legal_position_bers": "[" + ", ".join(f"{v:.6f}" for v in result["legal_position_bers"]) + "]",
        "min_illegal_ber": "" if result.get("min_illegal_ber") is None else f"{result['min_illegal_ber']:.6f}",
        "raw_ber_at_min_illegal_ber": (
            "" if result.get("raw_ber_at_min_illegal_ber") is None else f"{result['raw_ber_at_min_illegal_ber']:.6f}"
        ),
        "average_illegal_ber": "" if result.get("average_illegal_ber") is None else f"{result['average_illegal_ber']:.6f}",
        "worst_illegal_position": result.get("worst_illegal_position") or "",
        "worst_legal_position": result.get("worst_legal_position") or "",
        "worst_illegal_ber_vector": (
            ""
            if result.get("worst_illegal_ber_vector") is None
            else "[" + ", ".join(f"{v:.6f}" for v in result["worst_illegal_ber_vector"]) + "]"
        ),
        "evaluated_mapping_count": result.get("evaluated_mapping_count", ""),
        "total_mapping_candidate_count": result.get("total_mapping_candidate_count", ""),
        "security_satisfied": "yes" if result.get("security_satisfied") else "no",
    }


def run_search() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(project_root, "test-4", OUTPUT_RESULTS_FILENAME)
    existing_rows = load_existing_results(results_file)
    completed = {
        row["position_combination"]
        for row in existing_rows
        if row.get("security_satisfied") == "yes"
    }
    combinations = generate_position_combinations(project_root)
    pending = [combination for combination in combinations if str(tuple(combination)) not in completed]
    print(f"Total combinations={len(combinations)}, safe_existing={len(completed)}, pending={len(pending)}")
    rng = random.Random(SELECTION_SEED)
    rows = existing_rows[:]
    for idx, legal_positions in enumerate(combinations, start=1):
        if str(tuple(legal_positions)) in completed:
            print(f"[{idx}/{len(combinations)}] Skip {legal_positions}: safe probes already exist.")
            continue
        print(f"[{idx}/{len(combinations)}] Searching combination {legal_positions}")
        result = search_security_aware_probes(project_root, legal_positions, rng)
        row = row_from_result(legal_positions, result)
        rows = [old for old in rows if old.get("position_combination") != str(tuple(legal_positions))]
        rows.append(row)
        if row["security_satisfied"] == "yes":
            completed.add(row["position_combination"])
        write_results(results_file, rows)
        print(
            f"  New result: legal_ber={row['legal_ber']}, "
            f"min_illegal_ber={row['min_illegal_ber'] or 'N/A'}, "
            f"satisfied={result['security_satisfied']}"
        )
    print(f"Results saved to: {results_file}")
    return results_file


def main() -> None:
    run_search()


if __name__ == "__main__":
    main()
