#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch experiment for 20 random 5-position combinations.

For each position combination:
1. Search the best probe count and probe set.
2. Minimize BER on random search bits.
3. Re-test on fresh random test bits.
4. Write the result to CSV files under test-5.
"""

from __future__ import annotations

import csv
import itertools
import os
import random
from typing import Iterable, Sequence

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

import test_5_simple as test


TOTAL_POSITIONS = 28
COMBINATION_SIZE = 5
SAMPLE_COUNT = 20
RESULTS_FILENAME = "batch_test_results_optimized_v2.csv"
ZERO_BER_RESULTS_FILENAME = "batch_test_zero_ber_results.csv"
SEARCH_ATTEMPTS_PER_COMBINATION = 3


def generate_position_combinations(n: int, k: int) -> Iterable[tuple[int, ...]]:
    return itertools.combinations(range(1, n + 1), k)


def generate_unseen_random_position_combinations(
    n: int,
    k: int,
    sample_count: int,
    seed: int | None,
    processed_combinations: set[str],
) -> list[tuple[int, ...]]:
    remaining = [
        combination
        for combination in generate_position_combinations(n, k)
        if str(combination) not in processed_combinations
    ]
    if not remaining:
        return []

    rng = random.Random(seed)
    sample_size = min(sample_count, len(remaining))
    selected = rng.sample(remaining, sample_size)
    selected.sort()
    return selected


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


def initialize_zero_ber_results_file(results_file: str) -> set[tuple[str, str]]:
    existing_keys: set[tuple[str, str]] = set()
    if os.path.exists(results_file):
        with open(results_file, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add((row["position_combination"], row["best_probes"]))
        return existing_keys

    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "position_combination",
            "best_probe_count",
            "best_probes",
            "best_ber",
            "test_ber",
            "zero_source",
        ])
    return existing_keys


def run_batch_experiment() -> str:
    results_file = os.path.join(CURRENT_DIR, RESULTS_FILENAME)
    zero_ber_results_file = os.path.join(CURRENT_DIR, ZERO_BER_RESULTS_FILENAME)

    processed_combinations, cumulative_sum, completed_count = normalize_results_file(results_file)
    zero_ber_keys = initialize_zero_ber_results_file(zero_ber_results_file)
    combinations = generate_unseen_random_position_combinations(
        TOTAL_POSITIONS,
        COMBINATION_SIZE,
        SAMPLE_COUNT,
        None,
        processed_combinations,
    )

    print(f"Randomly selected {len(combinations)} new position combinations")

    with open(results_file, "a", newline="", encoding="utf-8-sig") as f, open(
        zero_ber_results_file,
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as zero_f:
        writer = csv.writer(f)
        zero_writer = csv.writer(zero_f)

        for idx, combination in enumerate(combinations, start=1):
            combination_str = str(combination)
            if combination_str in processed_combinations:
                print(f"[{idx}/{len(combinations)}] position combination {combination} already completed, skipping")
                continue

            csv_files = test.build_csv_files_for_positions(PROJECT_ROOT, combination, light_condition="white")

            if not all(os.path.exists(path) for path in csv_files):
                print(f"[{idx}/{len(combinations)}] position combination {combination} is missing data files, skipping")
                continue

            print(f"[{idx}/{len(combinations)}] processing position combination {combination}")
            best_result = None
            for attempt in range(1, SEARCH_ATTEMPTS_PER_COMBINATION + 1):
                print(f"  search attempt {attempt}/{SEARCH_ATTEMPTS_PER_COMBINATION}")
                result = test.run_position_experiment(
                    csv_files=csv_files,
                    search_bits=10000,
                    test_bits=10000,
                    min_probes=5,
                    max_probes=20,
                    max_candidates=1200,
                )
                if best_result is None or result["best_ber"] < best_result["best_ber"]:
                    best_result = result
                    print(f"  updated best BER in this batch search: {best_result['best_ber']:.6f}")

                if best_result["best_ber"] <= 0.0:
                    print("  found BER=0 probe set, stopping this position combination early.")
                    break

            if best_result is None:
                raise RuntimeError(f"Failed to obtain a search result for position combination {combination}")
            result = best_result

            print(f"[{idx}/{len(combinations)}] position combination: {combination}")
            print(f"  best probe count: {result['best_probe_count']}")
            print(f"  probe combination: {result['best_probes']}")
            print(f"  best BER: {result['best_ber']:.6f}")
            print(f"  test BER: {result['test_ber']:.6f}")

            completed_count += 1
            cumulative_sum += result["test_ber"]
            cumulative_test_ber = cumulative_sum / completed_count
            print(f"  cumulative test BER: {cumulative_test_ber:.6f}")

            writer.writerow([
                combination_str,
                result["best_probe_count"],
                format_probes(result["best_probes"]),
                f"{result['best_ber']:.6f}",
                f"{result['test_ber']:.6f}",
                f"{cumulative_test_ber:.6f}",
            ])
            f.flush()

            zero_sources = []
            if result["best_ber"] <= 0.0:
                zero_sources.append("best_ber")
            if result["test_ber"] <= 0.0:
                zero_sources.append("test_ber")
            zero_key = (combination_str, format_probes(result["best_probes"]))
            if zero_sources and zero_key not in zero_ber_keys:
                zero_writer.writerow([
                    combination_str,
                    result["best_probe_count"],
                    format_probes(result["best_probes"]),
                    f"{result['best_ber']:.6f}",
                    f"{result['test_ber']:.6f}",
                    ",".join(zero_sources),
                ])
                zero_f.flush()
                zero_ber_keys.add(zero_key)

    return results_file


def main() -> None:
    results_file = run_batch_experiment()
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()



