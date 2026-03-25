#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机抽取 20 个位置组合进行实验，并将结果保存到新的 CSV 文件。
默认使用固定随机种子，确保可复现且支持断点续跑。
"""

from __future__ import annotations

import os
import random
import sys
from typing import Iterable

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import batch_test
import test_3_simple as test


TOTAL_POSITIONS = 28
COMBINATION_SIZE = 3
SAMPLE_COUNT = 20
SELECTION_SEED = 20260323
LIGHT_CONDITION = "white"
RESULTS_FILENAME = "random_20_batch_test_results.csv"


def generate_random_position_combinations(
    n: int,
    k: int,
    sample_count: int,
    seed: int,
) -> list[tuple[int, ...]]:
    all_combinations = list(batch_test.generate_position_combinations(n, k))
    if sample_count > len(all_combinations):
        raise ValueError("sample_count exceeds the number of available combinations")

    rng = random.Random(seed)
    selected = rng.sample(all_combinations, sample_count)
    selected.sort()
    return selected


def run_random_batch_experiment() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(project_root, "test-3", RESULTS_FILENAME)
    combinations = generate_random_position_combinations(
        TOTAL_POSITIONS,
        COMBINATION_SIZE,
        SAMPLE_COUNT,
        SELECTION_SEED,
    )
    processed_combinations, cumulative_sum, completed_count = batch_test.normalize_results_file(results_file)

    print(f"随机种子: {SELECTION_SEED}")
    print(f"本次固定抽取 {len(combinations)} 个位置组合")

    with open(results_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = batch_test.csv.writer(f)

        for idx, combination in enumerate(combinations, start=1):
            combination_str = str(combination)
            if combination_str in processed_combinations:
                print(f"[{idx}/{len(combinations)}] 位置组合 {combination} 已完成，跳过")
                continue

            csv_files = test.build_csv_files_for_positions(project_root, combination, light_condition=LIGHT_CONDITION)
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
            print(f"  累计测试BER: {cumulative_test_ber:.6f}")

            writer.writerow([
                combination_str,
                result["best_probe_count"],
                batch_test.format_probes(result["best_probes"]),
                f"{result['best_ber']:.6f}",
                f"{result['test_ber']:.6f}",
                f"{cumulative_test_ber:.6f}",
            ])
            f.flush()

    return results_file


def main() -> None:
    results_file = run_random_batch_experiment()
    print(f"结果已保存到: {results_file}")


if __name__ == "__main__":
    main()
