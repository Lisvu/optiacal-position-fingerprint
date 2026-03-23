#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量测试脚本：
1. 遍历所有 C(28,3) 位置组合。
2. 对每个位置组合搜索 5~20 个探针下的最优探针组合。
3. 搜索依据为随机 10000 个 bit block 时 BER 最小。
4. 使用最优探针组合重新随机生成 10000 个 bit block 进行测试。
5. 将结果写入 CSV 文件。
"""

from __future__ import annotations

import csv
import itertools
import os
import sys
from typing import Iterable, Sequence

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import test_3_simple as test


def generate_position_combinations(n: int, k: int) -> Iterable[tuple[int, ...]]:
    return itertools.combinations(range(1, n + 1), k)


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def load_existing_results(results_file: str) -> list[dict]:
    if not os.path.exists(results_file):
        return []

    with open(results_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def normalize_results_file(results_file: str) -> tuple[set[str], float, int]:
    existing_rows = load_existing_results(results_file)
    processed: set[str] = set()
    cumulative_sum = 0.0
    completed_count = 0

    if not existing_rows:
        with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "position_combination",
                "best_probe_count",
                "best_probes",
                "best_ber",
                "test_ber",
                "cumulative_test_ber",
            ])
        return processed, cumulative_sum, completed_count

    normalized_rows = []
    for row in existing_rows:
        position_combination = row["position_combination"]
        test_ber = float(row["test_ber"])
        cumulative_sum += test_ber
        completed_count += 1
        cumulative_test_ber = cumulative_sum / completed_count

        normalized_rows.append({
            "position_combination": position_combination,
            "best_probe_count": row["best_probe_count"],
            "best_probes": row["best_probes"],
            "best_ber": row["best_ber"],
            "test_ber": row["test_ber"],
            "cumulative_test_ber": f"{cumulative_test_ber:.6f}",
        })
        processed.add(position_combination)

    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "position_combination",
                "best_probe_count",
                "best_probes",
                "best_ber",
                "test_ber",
                "cumulative_test_ber",
            ],
        )
        writer.writeheader()
        writer.writerows(normalized_rows)

    return processed, cumulative_sum, completed_count


def run_batch_experiment() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(project_root, "test-3", "batch_test_results.csv")

    combinations = list(generate_position_combinations(28, 3))
    processed_combinations, cumulative_sum, completed_count = normalize_results_file(results_file)

    with open(results_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        for idx, combination in enumerate(combinations, start=1):
            combination_str = str(combination)
            if combination_str in processed_combinations:
                print(f"[{idx}/{len(combinations)}] 位置组合 {combination} 已完成，跳过")
                continue

            csv_files = test.build_csv_files_for_positions(project_root, combination, light_condition="white")

            if not all(os.path.exists(path) for path in csv_files):
                print(f"[{idx}/{len(combinations)}] 位置组合 {combination} 缺少数据文件，已跳过")
                continue

            print(f"[{idx}/{len(combinations)}] 开始处理位置组合: {combination}")
            result = test.run_position_experiment(
                csv_files=csv_files,
                search_bits=10000,
                test_bits=10000,
                min_probes=5,
                max_probes=20,
                max_candidates=1000,
            )

            print(f"[{idx}/{len(combinations)}] 位置组合: {combination}")
            print(f"  最优探针数量: {result['best_probe_count']}")
            print(f"  探针组合: {result['best_probes']}")
            print(f"  最优BER: {result['best_ber']:.6f}")
            print(f"  测试BER: {result['test_ber']:.6f}")

            completed_count += 1
            cumulative_sum += result["test_ber"]
            cumulative_test_ber = cumulative_sum / completed_count
            print(f"  累积误码率: {cumulative_test_ber:.6f}")

            writer.writerow([
                combination_str,
                result["best_probe_count"],
                format_probes(result["best_probes"]),
                f"{result['best_ber']:.6f}",
                f"{result['test_ber']:.6f}",
                f"{cumulative_test_ber:.6f}",
            ])
            f.flush()

    return results_file


def main() -> None:
    results_file = run_batch_experiment()
    print(f"结果已保存到: {results_file}")


if __name__ == "__main__":
    main()
