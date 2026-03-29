#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security test for four-position probe configurations.

For each legal 4-position combination recorded in batch_test_results_optimized_v2.csv:
1. Load its best probe set.
2. Build the legal models and hue mapping from the four legal positions.
3. For every other available position, treat it as an illegal device.
4. Let the illegal device build its own fingerprint model using the same probes.
5. Feed the illegal device the same transmitted sequence and decode it only with
   its own fingerprint information.
6. Compare that decoded bit stream against the true transmitted bits of the four
   legal positions and save BER results to CSV.
"""

from __future__ import annotations

import ast
import csv
import os
import random
import sys
from typing import Sequence

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import test_4_simple as test


LIGHT_CONDITION = "white"
SOURCE_RESULTS_FILENAME = "batch_test_results_optimized_v2.csv"
OUTPUT_RESULTS_FILENAME = "security_illegal_position_results.csv"
SUMMARY_RESULTS_FILENAME = "security_illegal_position_summary.csv"
NUM_BITS = 10000
MAPPING_EVAL_BITS = 500
MAPPING_TOP_K = 3
STABILITY_REPEAT_COUNT = 5


def parse_position_combination(text: str) -> tuple[int, int, int, int]:
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


def load_legal_configs(source_file: str) -> list[dict]:
    with open(source_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "position_combination": parse_position_combination(row["position_combination"]),
                "best_probe_count": int(row["best_probe_count"]),
                "best_probes": parse_probes(row["best_probes"]),
                "best_ber": float(row["best_ber"]),
                "test_ber": float(row["test_ber"]),
            })
    return rows


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


def evaluate_illegal_position_against_legal_bits(
    legal_models: Sequence[test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    illegal_csv_file: str,
    probes: Sequence[float],
    num_bits: int,
    rng: random.Random,
) -> tuple[list[float], float]:
    probes_array = np.asarray(probes, dtype=float)
    illegal_matrix = test.load_selected_rows([illegal_csv_file], probes_array)[0]
    illegal_model = test.extract_fingerprint(probes_array, illegal_matrix, force_positive_first=True)
    probe_to_row = test.build_probe_to_row(probes_array)
    legal_codes = [model.code for model in legal_models]
    bit_blocks_pm = test.generate_random_bit_blocks(num_bits, len(legal_models), rng=rng)

    position_errors = np.zeros(len(legal_models), dtype=float)
    position_total = np.zeros(len(legal_models), dtype=float)

    for bits_pm in bit_blocks_pm:
        bits_pm = np.asarray(bits_pm, dtype=int)
        _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
        hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
        illegal_observation = test.observe_block_from_measured_matrix(hue_seq, illegal_matrix, probe_to_row)
        true_bits = test.pm1_to_bin(bits_pm)
        illegal_dec = test.decode_local_block(illegal_observation, illegal_model.w, illegal_model.code)

        for idx in range(len(legal_models)):
            position_total[idx] += 1
            if illegal_dec.bit_hat_bin != true_bits[idx]:
                position_errors[idx] += 1

    position_bers = (position_errors / np.maximum(position_total, 1.0)).tolist()
    total_errors = float(np.sum(position_errors))
    total_bits = float(np.sum(position_total))
    total_ber = total_errors / total_bits if total_bits > 0 else 0.0
    return [float(v) for v in position_bers], float(total_ber)


def summarize_stability(total_bers: Sequence[float]) -> tuple[float, float, float]:
    values = [float(v) for v in total_bers]
    if not values:
        return 0.0, 0.0, 0.0
    return float(min(values)), float(sum(values) / len(values)), float(max(values))


def write_results(results_file: str, rows: Sequence[dict]) -> None:
    fieldnames = [
        "legal_position_combination",
        "illegal_position",
        "best_probe_count",
        "test_probes",
        "source_best_ber",
        "source_test_ber",
        "num_bits",
        "code_corr_legal_pos_1",
        "code_corr_legal_pos_2",
        "code_corr_legal_pos_3",
        "code_corr_legal_pos_4",
        "closest_legal_position",
        "closest_legal_position_index",
        "closest_code_correlation",
        "ber_vs_legal_pos_1",
        "ber_vs_legal_pos_2",
        "ber_vs_legal_pos_3",
        "ber_vs_legal_pos_4",
        "total_ber",
        "stability_repeat_count",
        "stability_min_total_ber",
        "stability_avg_total_ber",
        "stability_max_total_ber",
        "stability_run_1_total_ber",
        "stability_run_2_total_ber",
        "stability_run_3_total_ber",
        "stability_run_4_total_ber",
        "stability_run_5_total_ber",
    ]
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_summary_row(results_file: str, row: dict) -> None:
    fieldnames = [
        "legal_position_combination",
        "best_probe_count",
        "test_probes",
        "source_best_ber",
        "source_test_ber",
        "num_bits",
        "illegal_position_count",
        "min_illegal_position",
        "min_illegal_total_ber",
        "average_illegal_total_ber",
        "min_illegal_closest_legal_position",
        "min_illegal_closest_code_correlation",
        "min_illegal_stability_avg_total_ber",
        "min_illegal_stability_min_total_ber",
        "min_illegal_stability_max_total_ber",
    ]
    file_exists = os.path.exists(results_file)
    need_header = (not file_exists) or os.path.getsize(results_file) == 0
    with open(results_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow(row)


def run_security_validation() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_file = os.path.join(project_root, "test-4", SOURCE_RESULTS_FILENAME)
    results_file = os.path.join(project_root, "test-4", OUTPUT_RESULTS_FILENAME)
    summary_file = os.path.join(project_root, "test-4", SUMMARY_RESULTS_FILENAME)

    legal_configs = load_legal_configs(source_file)
    available_positions = get_available_positions(project_root, LIGHT_CONDITION)
    system_rng = random.SystemRandom()
    result_rows = []
    if os.path.exists(summary_file):
        os.remove(summary_file)

    for config_idx, config in enumerate(legal_configs, start=1):
        legal_positions = config["position_combination"]
        probes = config["best_probes"]
        legal_csv_files = test.build_csv_files_for_positions(
            project_root,
            legal_positions,
            light_condition=LIGHT_CONDITION,
        )

        missing_files = [path for path in legal_csv_files if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"Missing legal data files: {missing_files}")

        model_rng = random.Random(system_rng.randrange(0, 2**31))
        legal_models, hue_mapping = test.build_models_from_probes(
            legal_csv_files,
            np.asarray(probes, dtype=float),
            mapping_eval_bits=MAPPING_EVAL_BITS,
            mapping_top_k=MAPPING_TOP_K,
            rng=model_rng,
        )

        illegal_positions = [pos for pos in available_positions if pos not in legal_positions]
        print(
            f"[{config_idx}/{len(legal_configs)}] legal positions {legal_positions}, "
            f"testing {len(illegal_positions)} illegal positions"
        )

        illegal_total_ber_pairs: list[tuple[int, float]] = []
        illegal_diagnostics: dict[int, dict] = {}
        for illegal_position in illegal_positions:
            illegal_csv_file = test.build_csv_files_for_positions(
                project_root,
                [illegal_position],
                light_condition=LIGHT_CONDITION,
            )[0]
            if not os.path.exists(illegal_csv_file):
                raise FileNotFoundError(f"Missing illegal data file: {illegal_csv_file}")

            probes_array = np.asarray(probes, dtype=float)
            illegal_matrix = test.load_selected_rows([illegal_csv_file], probes_array)[0]
            illegal_model = test.extract_fingerprint(probes_array, illegal_matrix, force_positive_first=True)
            code_correlations = [
                float(test.calculate_correlation(illegal_model.code, legal_model.code))
                for legal_model in legal_models
            ]
            closest_idx = max(range(len(code_correlations)), key=lambda idx: abs(code_correlations[idx]))
            closest_legal_position = legal_positions[closest_idx]
            closest_code_correlation = code_correlations[closest_idx]

            repeated_runs: list[tuple[list[float], float]] = []
            for _ in range(STABILITY_REPEAT_COUNT):
                eval_rng = random.Random(system_rng.randrange(0, 2**31))
                repeated_runs.append(evaluate_illegal_position_against_legal_bits(
                    legal_models=legal_models,
                    hue_mapping=hue_mapping,
                    illegal_csv_file=illegal_csv_file,
                    probes=probes,
                    num_bits=NUM_BITS,
                    rng=eval_rng,
                ))
            position_bers, total_ber = repeated_runs[0]
            stability_total_bers = [total for _, total in repeated_runs]
            stability_min_total_ber, stability_avg_total_ber, stability_max_total_ber = summarize_stability(
                stability_total_bers
            )
            print(
                f"  illegal position {illegal_position}: "
                f"BERs = {[round(v, 6) for v in position_bers]}, total BER = {total_ber:.6f}, "
                f"closest legal = {closest_legal_position}, corr = {closest_code_correlation:.6f}, "
                f"stability avg = {stability_avg_total_ber:.6f}"
            )
            illegal_total_ber_pairs.append((illegal_position, float(total_ber)))
            illegal_diagnostics[illegal_position] = {
                "closest_legal_position": closest_legal_position,
                "closest_code_correlation": closest_code_correlation,
                "stability_min_total_ber": stability_min_total_ber,
                "stability_avg_total_ber": stability_avg_total_ber,
                "stability_max_total_ber": stability_max_total_ber,
            }

            result_rows.append({
                "legal_position_combination": str(legal_positions),
                "illegal_position": illegal_position,
                "best_probe_count": config["best_probe_count"],
                "test_probes": format_probes(probes),
                "source_best_ber": f"{config['best_ber']:.6f}",
                "source_test_ber": f"{config['test_ber']:.6f}",
                "num_bits": NUM_BITS,
                "code_corr_legal_pos_1": f"{code_correlations[0]:.6f}",
                "code_corr_legal_pos_2": f"{code_correlations[1]:.6f}",
                "code_corr_legal_pos_3": f"{code_correlations[2]:.6f}",
                "code_corr_legal_pos_4": f"{code_correlations[3]:.6f}",
                "closest_legal_position": closest_legal_position,
                "closest_legal_position_index": closest_idx + 1,
                "closest_code_correlation": f"{closest_code_correlation:.6f}",
                "ber_vs_legal_pos_1": f"{position_bers[0]:.6f}",
                "ber_vs_legal_pos_2": f"{position_bers[1]:.6f}",
                "ber_vs_legal_pos_3": f"{position_bers[2]:.6f}",
                "ber_vs_legal_pos_4": f"{position_bers[3]:.6f}",
                "total_ber": f"{total_ber:.6f}",
                "stability_repeat_count": STABILITY_REPEAT_COUNT,
                "stability_min_total_ber": f"{stability_min_total_ber:.6f}",
                "stability_avg_total_ber": f"{stability_avg_total_ber:.6f}",
                "stability_max_total_ber": f"{stability_max_total_ber:.6f}",
                "stability_run_1_total_ber": f"{stability_total_bers[0]:.6f}",
                "stability_run_2_total_ber": f"{stability_total_bers[1]:.6f}",
                "stability_run_3_total_ber": f"{stability_total_bers[2]:.6f}",
                "stability_run_4_total_ber": f"{stability_total_bers[3]:.6f}",
                "stability_run_5_total_ber": f"{stability_total_bers[4]:.6f}",
            })

        min_illegal_position = ""
        min_illegal_total_ber = 0.0
        if illegal_total_ber_pairs:
            min_illegal_position, min_illegal_total_ber = min(illegal_total_ber_pairs, key=lambda item: item[1])
        average_illegal_total_ber = (
            sum(total_ber for _, total_ber in illegal_total_ber_pairs) / len(illegal_total_ber_pairs)
            if illegal_total_ber_pairs else 0.0
        )
        min_illegal_diag = illegal_diagnostics[min_illegal_position] if illegal_total_ber_pairs else {
            "closest_legal_position": "",
            "closest_code_correlation": 0.0,
            "stability_avg_total_ber": 0.0,
            "stability_min_total_ber": 0.0,
            "stability_max_total_ber": 0.0,
        }
        append_summary_row(summary_file, {
            "legal_position_combination": str(legal_positions),
            "best_probe_count": config["best_probe_count"],
            "test_probes": format_probes(probes),
            "source_best_ber": f"{config['best_ber']:.6f}",
            "source_test_ber": f"{config['test_ber']:.6f}",
            "num_bits": NUM_BITS,
            "illegal_position_count": len(illegal_total_ber_pairs),
            "min_illegal_position": min_illegal_position,
            "min_illegal_total_ber": f"{min_illegal_total_ber:.6f}",
            "average_illegal_total_ber": f"{average_illegal_total_ber:.6f}",
            "min_illegal_closest_legal_position": min_illegal_diag["closest_legal_position"],
            "min_illegal_closest_code_correlation": f"{min_illegal_diag['closest_code_correlation']:.6f}",
            "min_illegal_stability_avg_total_ber": f"{min_illegal_diag['stability_avg_total_ber']:.6f}",
            "min_illegal_stability_min_total_ber": f"{min_illegal_diag['stability_min_total_ber']:.6f}",
            "min_illegal_stability_max_total_ber": f"{min_illegal_diag['stability_max_total_ber']:.6f}",
        })
        print(
            f"  summary saved: min illegal position = {min_illegal_position}, "
            f"min illegal total BER = {min_illegal_total_ber:.6f}, "
            f"average illegal total BER = {average_illegal_total_ber:.6f}, "
            f"closest legal = {min_illegal_diag['closest_legal_position']}, "
            f"stability avg = {min_illegal_diag['stability_avg_total_ber']:.6f}"
        )

    write_results(results_file, result_rows)
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    return results_file


def main() -> None:
    run_security_validation()


if __name__ == "__main__":
    main()
