#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate probe sets recorded in batch_test_zero_ber_results.csv.

For each 4-position combination and its stored probe set:
1. Run 5 independent random communication tests.
2. Each test uses 10000 random bits.
3. Save all BER results to a new CSV file.
"""

from __future__ import annotations

import ast
import csv
import os
import random
import sys
from typing import Sequence

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import test_4_simple as test


LIGHT_CONDITION = "white"
SOURCE_RESULTS_FILENAME = "batch_test_zero_ber_results.csv"
OUTPUT_RESULTS_FILENAME = "zero_ber_probe_validation_results.csv"
NUM_BITS = 10000
REPEAT_COUNT = 5


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


def load_probe_configs(source_file: str) -> list[dict]:
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
                "zero_source": row["zero_source"],
            })
    return rows


def write_results(results_file: str, rows: Sequence[dict]) -> None:
    fieldnames = [
        "position_combination",
        "best_probe_count",
        "best_probes",
        "source_best_ber",
        "source_test_ber",
        "zero_source",
        "num_bits",
        "repeat_count",
        "ber_run_1",
        "ber_run_2",
        "ber_run_3",
        "ber_run_4",
        "ber_run_5",
        "average_ber",
        "all_zero_5_runs",
    ]
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_zero_ber_probe_validation() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_file = os.path.join(project_root, "test-4", SOURCE_RESULTS_FILENAME)
    results_file = os.path.join(project_root, "test-4", OUTPUT_RESULTS_FILENAME)

    configs = load_probe_configs(source_file)
    result_rows = []
    system_rng = random.SystemRandom()

    for idx, config in enumerate(configs, start=1):
        positions = config["position_combination"]
        probes = config["best_probes"]
        csv_files = test.build_csv_files_for_positions(
            project_root,
            positions,
            light_condition=LIGHT_CONDITION,
        )

        missing_files = [path for path in csv_files if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"Missing data files: {missing_files}")

        ber_runs = []
        print(f"[{idx}/{len(configs)}] validating positions {positions}, probes {probes}")
        for repeat_idx in range(REPEAT_COUNT):
            rng = random.Random(system_rng.randrange(0, 2**31))
            ber = test.evaluate_probe_combination(
                csv_files=csv_files,
                probes=probes,
                num_bits=NUM_BITS,
                rng=rng,
                force_random_bits=True,
            )
            ber_runs.append(float(ber))
            print(f"  run {repeat_idx + 1}: BER = {ber:.6f}")

        average_ber = sum(ber_runs) / len(ber_runs)
        all_zero = all(ber <= 0.0 for ber in ber_runs)
        print(f"  average BER: {average_ber:.6f}")

        result_rows.append({
            "position_combination": str(positions),
            "best_probe_count": config["best_probe_count"],
            "best_probes": format_probes(probes),
            "source_best_ber": f"{config['best_ber']:.6f}",
            "source_test_ber": f"{config['test_ber']:.6f}",
            "zero_source": config["zero_source"],
            "num_bits": NUM_BITS,
            "repeat_count": REPEAT_COUNT,
            "ber_run_1": f"{ber_runs[0]:.6f}",
            "ber_run_2": f"{ber_runs[1]:.6f}",
            "ber_run_3": f"{ber_runs[2]:.6f}",
            "ber_run_4": f"{ber_runs[3]:.6f}",
            "ber_run_5": f"{ber_runs[4]:.6f}",
            "average_ber": f"{average_ber:.6f}",
            "all_zero_5_runs": "yes" if all_zero else "no",
        })

    write_results(results_file, result_rows)
    print(f"Results saved to: {results_file}")
    return results_file


def main() -> None:
    run_zero_ber_probe_validation()


if __name__ == "__main__":
    main()
