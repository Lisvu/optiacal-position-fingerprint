#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 random_20_batch_test_results.csv 中记录的位置组合和最优探针，
对每组位置执行 5 次独立随机测试，并将平均 BER 写入结果文件。
"""

from __future__ import annotations

import ast
import csv
import os
import random
import sys
from typing import Sequence

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import test_3_simple as test


LIGHT_CONDITION = "white"
SOURCE_RESULTS_FILENAME = "random_20_batch_test_results.csv"
OUTPUT_RESULTS_FILENAME = "specified_probe_test_results.csv"
NUM_BITS = 10000
REPEAT_COUNT = 5
BASE_SEED = 20260323


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


def load_probe_configs(source_file: str) -> list[dict]:
    with open(source_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "position_combination": parse_position_combination(row["position_combination"]),
                "best_probes": parse_probes(row["best_probes"]),
            })
        return rows


def write_results(results_file: str, rows: Sequence[dict]) -> None:
    fieldnames = [
        "position_combination",
        "best_probes",
        "num_bits",
        "repeat_count",
        "ber_run_1",
        "ber_run_2",
        "ber_run_3",
        "ber_run_4",
        "ber_run_5",
        "average_ber",
    ]
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_batch_probe_test() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_file = os.path.join(project_root, "test-3", SOURCE_RESULTS_FILENAME)
    results_file = os.path.join(project_root, "test-3", OUTPUT_RESULTS_FILENAME)

    configs = load_probe_configs(source_file)
    result_rows = []

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
            raise FileNotFoundError(f"以下数据文件不存在: {missing_files}")

        ber_runs = []
        print(f"[{idx}/{len(configs)}] 开始测试位置组合 {positions}，探针 {probes}")
        for repeat_idx in range(REPEAT_COUNT):
            seed = BASE_SEED + idx * 100 + repeat_idx
            rng = random.Random(seed)
            ber = test.evaluate_probe_combination(
                csv_files=csv_files,
                probes=probes,
                num_bits=NUM_BITS,
                rng=rng,
            )
            ber_runs.append(float(ber))
            print(f"  第 {repeat_idx + 1} 次 BER: {ber:.6f}")

        average_ber = sum(ber_runs) / len(ber_runs)
        print(f"  平均 BER: {average_ber:.6f}")

        result_rows.append({
            "position_combination": str(positions),
            "best_probes": format_probes(probes),
            "num_bits": NUM_BITS,
            "repeat_count": REPEAT_COUNT,
            "ber_run_1": f"{ber_runs[0]:.6f}",
            "ber_run_2": f"{ber_runs[1]:.6f}",
            "ber_run_3": f"{ber_runs[2]:.6f}",
            "ber_run_4": f"{ber_runs[3]:.6f}",
            "ber_run_5": f"{ber_runs[4]:.6f}",
            "average_ber": f"{average_ber:.6f}",
        })

    write_results(results_file, result_rows)
    print(f"结果已保存到: {results_file}")
    return results_file


def main() -> None:
    run_batch_probe_test()


if __name__ == "__main__":
    main()
