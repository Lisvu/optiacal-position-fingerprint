#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate saved security-aware probe sets.

This script reads test-3/security_aware_safe_probes.csv and re-runs each saved
position combination with fresh random bit blocks. It uses the saved probes and
saved hue_mapping directly, then writes per-combination legal BER and illegal
BER summary metrics.
"""

from __future__ import annotations

import ast
import csv
import os
import random
import sys
import types
from typing import Sequence

import numpy as np


LIGHT_CONDITION = "white"
SAFE_PROBE_RESULTS_FILENAME = "security_aware_safe_probes.csv"
OUTPUT_RESULTS_FILENAME = "security_aware_safe_probe_validation_results.csv"
NUM_BITS = 10000
BASE_SEED = 20260506


def load_test_module() -> types.ModuleType:
    module_name = "test_3_simple_runtime_safe_probe_validation"
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
        raise ValueError(f"Invalid position combination: {text}")
    return tuple(int(v) for v in value)


def parse_probes(text: str) -> list[float]:
    value = ast.literal_eval(text)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Invalid probe list: {text}")
    return [float(v) for v in value]


def parse_hue_mapping(text: str) -> dict[tuple[int, ...], int]:
    value = ast.literal_eval(text)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Invalid hue mapping: {text}")

    mapping: dict[tuple[int, ...], int] = {}
    for key, hue in value.items():
        if not isinstance(key, tuple):
            raise ValueError(f"Invalid hue mapping key: {key}")
        mapping[tuple(int(v) for v in key)] = int(hue)
    return mapping


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def load_safe_probe_configs(source_file: str) -> list[dict]:
    with open(source_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        configs = []
        for row in reader:
            if not row.get("position_combination"):
                continue
            configs.append({
                "position_combination": parse_position_combination(row["position_combination"]),
                "safe_probe_count": int(row["safe_probe_count"]),
                "safe_probes": parse_probes(row["safe_probes"]),
                "safe_hue_mapping": parse_hue_mapping(row["safe_hue_mapping"]),
            })
        return configs


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


def build_legal_models(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
) -> list[test.FingerprintModel]:
    csv_files = test.build_csv_files_for_positions(
        project_root,
        legal_positions,
        light_condition=LIGHT_CONDITION,
    )
    probes_array = np.asarray(probes, dtype=float)
    matrices = test.load_selected_rows(csv_files, probes_array)
    models = [test.extract_fingerprint(probes_array, matrix, force_positive_first=True) for matrix in matrices]
    return test.align_model_directions(models)


def corrected_ber(ber: float) -> float:
    return min(float(ber), 1.0 - float(ber))


def evaluate_legal_positions(
    legal_models: Sequence[test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    bit_blocks_pm: Sequence[np.ndarray],
) -> tuple[list[float], list[float]]:
    results = test.simulate_blocks(list(legal_models), list(bit_blocks_pm), hue_mapping)
    position_errors = np.zeros(len(legal_models), dtype=float)
    position_total = np.zeros(len(legal_models), dtype=float)

    for result in results:
        true_bits = result["bits_bin"]
        decoded_bits = [decode.bit_hat_bin for decode in result["per_position"]]
        for idx, (true_bit, decoded_bit) in enumerate(zip(true_bits, decoded_bits)):
            position_total[idx] += 1
            if int(true_bit) != int(decoded_bit):
                position_errors[idx] += 1

    raw_bers = (position_errors / np.maximum(position_total, 1.0)).tolist()
    corrected_bers = [corrected_ber(ber) for ber in raw_bers]
    return [float(v) for v in raw_bers], [float(v) for v in corrected_bers]


def evaluate_illegal_positions(
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
    available_positions = get_available_positions(project_root, LIGHT_CONDITION)

    all_single_route_secure_bers: list[float] = []
    illegal_average_secure_bers: list[float] = []
    global_min_secure_ber = float("inf")
    global_raw_ber_at_min = 0.0
    global_worst_illegal_position: int | None = None
    global_worst_legal_position: int | None = None
    global_worst_ber_vector: list[float] | None = None

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
        all_single_route_secure_bers.extend(secure_bers)
        illegal_average_secure_bers.append(float(np.mean(secure_bers)))

        local_min_secure_ber = float(min(secure_bers))
        local_idx = int(np.argmin(secure_bers))
        if local_min_secure_ber < global_min_secure_ber:
            global_min_secure_ber = local_min_secure_ber
            global_raw_ber_at_min = float(raw_bers[local_idx])
            global_worst_illegal_position = illegal_position
            global_worst_legal_position = int(legal_positions[local_idx])
            global_worst_ber_vector = [float(v) for v in raw_bers]

    return {
        "illegal_position_count": len([pos for pos in available_positions if pos not in legal_positions]),
        "min_illegal_secure_ber": float(global_min_secure_ber if all_single_route_secure_bers else 0.0),
        "raw_ber_at_min_illegal_secure_ber": float(global_raw_ber_at_min),
        "average_illegal_secure_ber": float(np.mean(all_single_route_secure_bers)) if all_single_route_secure_bers else 0.0,
        "average_illegal_position_secure_ber": float(np.mean(illegal_average_secure_bers)) if illegal_average_secure_bers else 0.0,
        "worst_illegal_position": global_worst_illegal_position,
        "worst_legal_position": global_worst_legal_position,
        "worst_illegal_raw_ber_vector": global_worst_ber_vector,
    }


def write_results(results_file: str, rows: Sequence[dict]) -> None:
    fieldnames = [
        "position_combination",
        "probe_count",
        "probes",
        "num_bits",
        "legal_raw_ber_pos_1",
        "legal_raw_ber_pos_2",
        "legal_raw_ber_pos_3",
        "legal_corrected_ber_pos_1",
        "legal_corrected_ber_pos_2",
        "legal_corrected_ber_pos_3",
        "legal_average_corrected_ber",
        "illegal_position_count",
        "min_illegal_secure_ber",
        "raw_ber_at_min_illegal_secure_ber",
        "average_illegal_secure_ber",
        "average_illegal_position_secure_ber",
        "worst_illegal_position",
        "worst_legal_position",
        "worst_illegal_raw_ber_vector",
        "accuracy_satisfied",
        "security_satisfied",
    ]
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_validation() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_file = os.path.join(project_root, "test-3", SAFE_PROBE_RESULTS_FILENAME)
    results_file = os.path.join(project_root, "test-3", OUTPUT_RESULTS_FILENAME)
    configs = load_safe_probe_configs(source_file)
    rows = []

    for idx, config in enumerate(configs, start=1):
        legal_positions = config["position_combination"]
        probes = config["safe_probes"]
        hue_mapping = config["safe_hue_mapping"]
        rng = random.Random(BASE_SEED + idx)
        bit_blocks_pm = test.generate_random_bit_blocks(NUM_BITS, len(legal_positions), rng=rng)
        legal_models = build_legal_models(project_root, legal_positions, probes)

        legal_raw_bers, legal_corrected_bers = evaluate_legal_positions(
            legal_models=legal_models,
            hue_mapping=hue_mapping,
            bit_blocks_pm=bit_blocks_pm,
        )
        illegal_result = evaluate_illegal_positions(
            project_root=project_root,
            legal_positions=legal_positions,
            legal_models=legal_models,
            hue_mapping=hue_mapping,
            probes=probes,
            bit_blocks_pm=bit_blocks_pm,
        )

        accuracy_satisfied = max(legal_corrected_bers) <= 0.02
        security_satisfied = illegal_result["min_illegal_secure_ber"] > 0.1
        row = {
            "position_combination": str(tuple(legal_positions)),
            "probe_count": config["safe_probe_count"],
            "probes": format_probes(probes),
            "num_bits": NUM_BITS,
            "legal_raw_ber_pos_1": f"{legal_raw_bers[0]:.6f}",
            "legal_raw_ber_pos_2": f"{legal_raw_bers[1]:.6f}",
            "legal_raw_ber_pos_3": f"{legal_raw_bers[2]:.6f}",
            "legal_corrected_ber_pos_1": f"{legal_corrected_bers[0]:.6f}",
            "legal_corrected_ber_pos_2": f"{legal_corrected_bers[1]:.6f}",
            "legal_corrected_ber_pos_3": f"{legal_corrected_bers[2]:.6f}",
            "legal_average_corrected_ber": f"{float(np.mean(legal_corrected_bers)):.6f}",
            "illegal_position_count": illegal_result["illegal_position_count"],
            "min_illegal_secure_ber": f"{illegal_result['min_illegal_secure_ber']:.6f}",
            "raw_ber_at_min_illegal_secure_ber": f"{illegal_result['raw_ber_at_min_illegal_secure_ber']:.6f}",
            "average_illegal_secure_ber": f"{illegal_result['average_illegal_secure_ber']:.6f}",
            "average_illegal_position_secure_ber": f"{illegal_result['average_illegal_position_secure_ber']:.6f}",
            "worst_illegal_position": illegal_result["worst_illegal_position"],
            "worst_legal_position": illegal_result["worst_legal_position"],
            "worst_illegal_raw_ber_vector": (
                ""
                if illegal_result["worst_illegal_raw_ber_vector"] is None
                else "[" + ", ".join(f"{v:.6f}" for v in illegal_result["worst_illegal_raw_ber_vector"]) + "]"
            ),
            "accuracy_satisfied": "yes" if accuracy_satisfied else "no",
            "security_satisfied": "yes" if security_satisfied else "no",
        }
        rows.append(row)
        print(
            f"[{idx}/{len(configs)}] {legal_positions}: "
            f"legal_avg_corrected_ber={row['legal_average_corrected_ber']}, "
            f"min_illegal_secure_ber={row['min_illegal_secure_ber']}, "
            f"avg_illegal_secure_ber={row['average_illegal_secure_ber']}"
        )

    write_results(results_file, rows)
    print(f"Validation results saved to: {results_file}")
    return results_file


def main() -> None:
    run_validation()


if __name__ == "__main__":
    main()
