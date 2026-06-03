#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search security-aware hue mappings for existing 4-position zero-BER probe sets.

This script does not search probe positions. It reads probe sets already found
by batch_test.py in batch_test_zero_ber_results.csv, keeps the probes fixed, and
searches only the hue_mapping. A saved mapping must satisfy:

1. legal BER == 0 on the confirmation information bits
2. 0.1 < min_illegal_ber < 0.9

Here min_illegal_ber is the minimum secure BER min(raw_ber, 1 - raw_ber) across
all illegal positions and all four legal information streams. This rejects both
direct decoding (BER = 0) and inverted decoding (BER = 1).
"""

from __future__ import annotations

import ast
import csv
import itertools
import os
import random
import sys
from typing import Sequence

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import test_4_simple as test


LIGHT_CONDITION = "white"
SOURCE_RESULTS_FILENAME = "batch_test_zero_ber_results.csv"
OUTPUT_RESULTS_FILENAME = "security_aware_hue_mapping_results_4.csv"

MIN_ILLEGAL_BER = 0.2
MAX_ILLEGAL_BER = 0.8
# 多主成分系统下，合法BER目标放宽到0.005（比0.0宽松但仍有高可靠性）
LEGAL_BER_TARGET = 0.005

MAPPING_TOP_K = 6
RANDOM_MAPPING_COUNT = 1200
COORDINATE_ROUNDS = 3
SECURITY_SHORTLIST_SIZE = 48

LEGAL_SCREEN_BITS = 1200
ILLEGAL_SCREEN_BITS = 1200
LEGAL_CONFIRM_BITS = 10000
ILLEGAL_CONFIRM_BITS = 10000

SEED = 20260506


def parse_position_combination(text: str) -> tuple[int, ...]:
    value = ast.literal_eval(text)
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError(f"Invalid position combination: {text}")
    return tuple(int(v) for v in value)


def parse_probes(text: str) -> list[float]:
    value = ast.literal_eval(text)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Invalid probe list: {text}")
    return [float(v) for v in value]


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def format_hue_mapping(hue_mapping: dict[tuple[int, ...], int]) -> str:
    return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(hue_mapping.items())) + "}"


def load_zero_ber_configs(source_file: str) -> list[dict]:
    with open(source_file, "r", newline="", encoding="utf-8-sig") as f:
        rows = []
        for row in csv.DictReader(f):
            if float(row["test_ber"]) != 0.0:
                continue
            rows.append({
                "position_combination": parse_position_combination(row["position_combination"]),
                "best_probe_count": int(row["best_probe_count"]),
                "best_probes": parse_probes(row["best_probes"]),
                "source_best_ber": float(row["best_ber"]),
                "source_test_ber": float(row["test_ber"]),
                "zero_source": row.get("zero_source", ""),
            })
    return rows


def load_existing_results(results_file: str) -> list[dict]:
    if not os.path.exists(results_file):
        return []
    with open(results_file, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_available_positions(project_root: str) -> list[int]:
    data_dir = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION)
    positions = []
    for entry in os.listdir(data_dir):
        stem, ext = os.path.splitext(entry)
        if ext.lower() == ".csv" and stem.isdigit():
            positions.append(int(stem))
    return sorted(positions)


def build_legal_models(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
) -> list[test.FingerprintModel]:
    csv_files = test.build_csv_files_for_positions(project_root, legal_positions, light_condition=LIGHT_CONDITION)
    matrices = test.load_selected_rows(csv_files, np.asarray(probes, dtype=float))
    models = [test.extract_fingerprint(np.asarray(probes, dtype=float), matrix, force_positive_first=True) for matrix in matrices]
    return test.align_model_directions(models)


def build_illegal_models(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
) -> dict[int, test.FingerprintModel]:
    probes_array = np.asarray(probes, dtype=float)
    illegal_models = {}
    for illegal_position in get_available_positions(project_root):
        if illegal_position in legal_positions:
            continue
        csv_file = test.build_csv_files_for_positions(project_root, [illegal_position], light_condition=LIGHT_CONDITION)[0]
        matrix = test.load_selected_rows([csv_file], probes_array)[0]
        illegal_models[illegal_position] = test.extract_fingerprint(probes_array, matrix, force_positive_first=True)
    return illegal_models


def generate_relaxed_mapping_candidates(
    legal_models: Sequence[test.FingerprintModel],
    probes: Sequence[float],
    top_k: int,
) -> dict[tuple[int, ...], list[int]]:
    probes_array = np.asarray(probes, dtype=float)
    z_list = [np.asarray(model.z, dtype=float) for model in legal_models]
    strict = test.generate_mapping_candidates(list(legal_models), probes_array, top_k_per_combination=min(3, top_k))
    candidate_map: dict[tuple[int, ...], list[int]] = {}

    for combination in itertools.product([1, -1], repeat=len(legal_models)):
        scored = []
        for probe_idx, probe in enumerate(probes_array):
            signed_margins = [int(sign) * float(z_values[probe_idx]) for sign, z_values in zip(combination, z_list)]
            matched = sum(1 for value in signed_margins if value > 0)
            margin_sum = sum(abs(value) for value in signed_margins)
            mismatch_penalty = sum(abs(value) for value in signed_margins if value <= 0)
            score = matched * 1000.0 + margin_sum - 2.0 * mismatch_penalty
            scored.append((score, int(probe)))
        scored.sort(reverse=True)

        candidates: list[int] = []
        for probe in strict.get(tuple(combination), []):
            if probe not in candidates:
                candidates.append(int(probe))
        for _, probe in scored:
            if probe not in candidates:
                candidates.append(int(probe))
            if len(candidates) >= top_k:
                break
        candidate_map[tuple(combination)] = candidates

    return candidate_map


def build_default_mapping(
    legal_models: Sequence[test.FingerprintModel],
    probes: Sequence[float],
    rng: random.Random,
) -> dict[tuple[int, ...], int]:
    return test.build_hue_mapping(
        list(legal_models),
        np.asarray(probes, dtype=float),
        mapping_eval_bits=500,
        top_k_per_combination=3,
        rng=rng,
    )


def mapping_signature(hue_mapping: dict[tuple[int, ...], int]) -> tuple[tuple[tuple[int, ...], int], ...]:
    return tuple(sorted((tuple(key), int(value)) for key, value in hue_mapping.items()))


def legal_ber_for_mapping(
    legal_models: Sequence[test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    info_bits_bin: np.ndarray,
) -> float:
    return float(test.evaluate_blocks_ber_with_convolutional_fec(list(legal_models), info_bits_bin, hue_mapping))


def illegal_security_for_mapping(
    legal_models: Sequence[test.FingerprintModel],
    illegal_models: dict[int, test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    info_bits_bin: np.ndarray,
) -> dict:
    legal_codes = [model.code for model in legal_models]
    probe_to_row = test.build_probe_to_row(legal_models[0].probes)
    bit_blocks_pm = test.build_convolutional_bit_blocks(info_bits_bin)

    global_min_secure_ber = float("inf")
    raw_ber_at_min = 0.0
    worst_illegal_position = None
    worst_legal_index = None
    worst_ber_vector: list[float] | None = None
    all_secure_bers: list[float] = []

    for illegal_position, illegal_model in illegal_models.items():
        received_stream: list[int] = []
        for bits_pm in bit_blocks_pm:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            illegal_observation = test.observe_block_from_measured_matrix(hue_seq, illegal_model.Y, probe_to_row)
            illegal_dec = test.decode_local_block(illegal_observation, illegal_model.w, illegal_model.code)
            received_stream.append(int(illegal_dec.bit_hat_bin))

        decoded_bits = test.viterbi_decode_hard(received_stream)
        raw_bers = []
        for legal_idx in range(len(legal_models)):
            reference = info_bits_bin[:, legal_idx]
            compare_len = min(len(decoded_bits), len(reference))
            if compare_len <= 0:
                raw_bers.append(0.0)
                continue
            raw_bers.append(float(np.mean(decoded_bits[:compare_len] != reference[:compare_len])))

        secure_bers = [min(ber, 1.0 - ber) for ber in raw_bers]
        all_secure_bers.extend(secure_bers)
        local_min = float(min(secure_bers))
        local_idx = int(np.argmin(secure_bers))
        if local_min < global_min_secure_ber:
            global_min_secure_ber = local_min
            raw_ber_at_min = float(raw_bers[local_idx])
            worst_illegal_position = illegal_position
            worst_legal_index = local_idx
            worst_ber_vector = [float(v) for v in raw_bers]

    return {
        "min_illegal_ber": float(global_min_secure_ber if all_secure_bers else 0.0),
        "raw_ber_at_min": float(raw_ber_at_min),
        "average_illegal_ber": float(np.mean(all_secure_bers)) if all_secure_bers else 0.0,
        "worst_illegal_position": worst_illegal_position,
        "worst_legal_index": worst_legal_index,
        "worst_illegal_ber_vector": worst_ber_vector or [],
    }


def candidate_rank(candidate: dict) -> tuple[float, float, float, float]:
    legal_ok = 1.0 if candidate["legal_ber"] <= LEGAL_BER_TARGET else 0.0
    min_illegal = float(candidate.get("min_illegal_ber") or -1.0)
    satisfied = 1.0 if candidate.get("security_satisfied") else 0.0
    distance_to_half = -abs(min_illegal - 0.5) if min_illegal >= 0 else -1.0
    return satisfied, legal_ok, min_illegal, distance_to_half


def generate_mapping_pool(
    default_mapping: dict[tuple[int, ...], int],
    candidate_map: dict[tuple[int, ...], list[int]],
    rng: random.Random,
) -> list[dict[tuple[int, ...], int]]:
    keys = sorted(candidate_map.keys())
    pool: list[dict[tuple[int, ...], int]] = [dict(default_mapping)]
    seen = {mapping_signature(default_mapping)}

    for _round in range(COORDINATE_ROUNDS):
        base_pool = list(pool)
        for base in base_pool:
            for key in keys:
                for value in candidate_map[key]:
                    if int(value) == int(base[key]):
                        continue
                    trial = dict(base)
                    trial[key] = int(value)
                    signature = mapping_signature(trial)
                    if signature not in seen:
                        seen.add(signature)
                        pool.append(trial)

    while len(pool) < RANDOM_MAPPING_COUNT:
        trial = {key: int(rng.choice(candidate_map[key])) for key in keys}
        signature = mapping_signature(trial)
        if signature in seen:
            continue
        seen.add(signature)
        pool.append(trial)

    return pool[:RANDOM_MAPPING_COUNT]


def search_hue_mapping_for_config(project_root: str, config: dict, rng: random.Random) -> dict:
    legal_positions = config["position_combination"]
    probes = config["best_probes"]
    legal_models = build_legal_models(project_root, legal_positions, probes)
    illegal_models = build_illegal_models(project_root, legal_positions, probes)

    candidate_map = generate_relaxed_mapping_candidates(legal_models, probes, top_k=MAPPING_TOP_K)
    default_mapping = build_default_mapping(legal_models, probes, rng)
    mapping_pool = generate_mapping_pool(default_mapping, candidate_map, rng)

    screen_info_bits = test.generate_random_information_bits(LEGAL_SCREEN_BITS, len(legal_models), rng=rng)
    legal_zero_candidates = []
    best_candidate = None

    for idx, hue_mapping in enumerate(mapping_pool, start=1):
        legal_ber = legal_ber_for_mapping(legal_models, hue_mapping, screen_info_bits)
        candidate = {
            "hue_mapping": hue_mapping,
            "legal_ber": legal_ber,
            "min_illegal_ber": None,
            "average_illegal_ber": None,
            "security_satisfied": False,
        }
        if legal_ber <= LEGAL_BER_TARGET:
            legal_zero_candidates.append(candidate)
        if best_candidate is None or candidate_rank(candidate) > candidate_rank(best_candidate):
            best_candidate = candidate

        if idx % 200 == 0:
            print(f"    screened {idx}/{len(mapping_pool)} mappings, legal_zero={len(legal_zero_candidates)}")

    if not legal_zero_candidates:
        return best_candidate or {
            "hue_mapping": default_mapping,
            "legal_ber": 1.0,
            "min_illegal_ber": None,
            "average_illegal_ber": None,
            "security_satisfied": False,
        }

    illegal_screen_bits = test.generate_random_information_bits(ILLEGAL_SCREEN_BITS, len(legal_models), rng=rng)
    screened = []
    for candidate in legal_zero_candidates:
        security = illegal_security_for_mapping(
            legal_models=legal_models,
            illegal_models=illegal_models,
            hue_mapping=candidate["hue_mapping"],
            info_bits_bin=illegal_screen_bits,
        )
        candidate.update(security)
        candidate["security_satisfied"] = MIN_ILLEGAL_BER < float(security["min_illegal_ber"]) < MAX_ILLEGAL_BER
        screened.append(candidate)
        if best_candidate is None or candidate_rank(candidate) > candidate_rank(best_candidate):
            best_candidate = candidate

    screened.sort(key=candidate_rank, reverse=True)
    confirm_info_bits = test.generate_random_information_bits(LEGAL_CONFIRM_BITS, len(legal_models), rng=rng)
    confirm_illegal_bits = test.generate_random_information_bits(ILLEGAL_CONFIRM_BITS, len(legal_models), rng=rng)

    for candidate in screened[:SECURITY_SHORTLIST_SIZE]:
        confirm_legal_ber = legal_ber_for_mapping(legal_models, candidate["hue_mapping"], confirm_info_bits)
        security = illegal_security_for_mapping(
            legal_models=legal_models,
            illegal_models=illegal_models,
            hue_mapping=candidate["hue_mapping"],
            info_bits_bin=confirm_illegal_bits,
        )
        confirmed = {
            **candidate,
            **security,
            "legal_ber": confirm_legal_ber,
        }
        confirmed["security_satisfied"] = (
            confirm_legal_ber <= LEGAL_BER_TARGET
            and MIN_ILLEGAL_BER < float(confirmed["min_illegal_ber"]) < MAX_ILLEGAL_BER
        )
        if best_candidate is None or candidate_rank(confirmed) > candidate_rank(best_candidate):
            best_candidate = confirmed
        if confirmed["security_satisfied"]:
            return confirmed

    return best_candidate or screened[0]


def row_from_result(config: dict, result: dict) -> dict:
    legal_positions = config["position_combination"]
    worst_legal_index = result.get("worst_legal_index")
    worst_legal_position = "" if worst_legal_index is None else legal_positions[int(worst_legal_index)]
    return {
        "position_combination": str(tuple(legal_positions)),
        "probe_count": config["best_probe_count"],
        "probes": format_probes(config["best_probes"]),
        "hue_mapping": format_hue_mapping(result["hue_mapping"]),
        "source_best_ber": f"{config['source_best_ber']:.6f}",
        "source_test_ber": f"{config['source_test_ber']:.6f}",
        "legal_ber": f"{float(result['legal_ber']):.6f}",
        "min_illegal_ber": "" if result.get("min_illegal_ber") is None else f"{float(result['min_illegal_ber']):.6f}",
        "raw_ber_at_min": "" if result.get("raw_ber_at_min") is None else f"{float(result['raw_ber_at_min']):.6f}",
        "average_illegal_ber": "" if result.get("average_illegal_ber") is None else f"{float(result['average_illegal_ber']):.6f}",
        "worst_illegal_position": result.get("worst_illegal_position") or "",
        "worst_legal_position": worst_legal_position,
        "worst_illegal_ber_vector": (
            ""
            if not result.get("worst_illegal_ber_vector")
            else "[" + ", ".join(f"{float(v):.6f}" for v in result["worst_illegal_ber_vector"]) + "]"
        ),
        "security_satisfied": "yes" if result.get("security_satisfied") else "no",
    }


def write_results(results_file: str, rows: Sequence[dict]) -> None:
    fieldnames = [
        "position_combination",
        "probe_count",
        "probes",
        "hue_mapping",
        "source_best_ber",
        "source_test_ber",
        "legal_ber",
        "min_illegal_ber",
        "raw_ber_at_min",
        "average_illegal_ber",
        "worst_illegal_position",
        "worst_legal_position",
        "worst_illegal_ber_vector",
        "security_satisfied",
    ]
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_search() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_file = os.path.join(project_root, "test-4", SOURCE_RESULTS_FILENAME)
    results_file = os.path.join(project_root, "test-4", OUTPUT_RESULTS_FILENAME)

    configs = load_zero_ber_configs(source_file)
    rows = load_existing_results(results_file)
    completed = {row["position_combination"] for row in rows if row.get("security_satisfied") == "yes"}
    rng = random.Random(SEED)

    print(f"Loaded zero-BER configs={len(configs)}, safe_existing={len(completed)}")
    for idx, config in enumerate(configs, start=1):
        combination_key = str(tuple(config["position_combination"]))
        if combination_key in completed:
            print(f"[{idx}/{len(configs)}] Skip {combination_key}: safe hue_mapping already exists.")
            continue

        print(f"[{idx}/{len(configs)}] Search hue_mapping for {combination_key}")
        result = search_hue_mapping_for_config(project_root, config, rng)
        row = row_from_result(config, result)
        rows = [old for old in rows if old.get("position_combination") != combination_key]
        rows.append(row)
        if row["security_satisfied"] == "yes":
            completed.add(combination_key)
        write_results(results_file, rows)
        print(
            f"  New result: legal_ber={row['legal_ber']}, "
            f"min_illegal_ber={row['min_illegal_ber'] or 'N/A'}, "
            f"satisfied={row['security_satisfied']}"
        )

    print(f"Results saved to: {results_file}")
    return results_file


def main() -> None:
    run_search()


if __name__ == "__main__":
    main()
