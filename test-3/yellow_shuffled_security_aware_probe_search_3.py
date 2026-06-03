#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security-aware probe and hue-mapping search for 3 legal positions.

Dataset: data/15pro/yellow_shuffled

Target:
1. Every legal device decodes its own bit with raw BER = 0.
2. Every illegal device outputs only one bit per block, and that bit is compared
   separately with each legal device's bit stream.
3. The overall min_illegal_ber is the minimum secure BER min(BER, 1-BER) across
   all illegal devices and all legal routes, and must satisfy 0.3 < min < 0.7.
4. BER is evaluated on 1000 deterministic random bit blocks per candidate.
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import os
import random
import sys
from typing import Sequence

import numpy as np


OUTPUT_RESULTS_FILENAME = "yellow_shuffled_security_aware_safe_probes_3.csv"
LEGAL_POSITION_COUNT = 3
COARSE_EVAL_BIT_BLOCK_COUNT = 200
FINAL_EVAL_BIT_BLOCK_COUNT = 1000
SEARCH_EPOCHS = 1
MAX_RANDOM_CANDIDATES_PER_COUNT = 20
TOP_K_CANDIDATES_PER_COUNT = 2
LOCAL_NEIGHBOR_SAMPLES = 8
LOCAL_ROUNDS = 2
MIN_PROBE_COUNT = 5
MAX_PROBE_COUNT = 30
DEFAULT_BASELINE_PROBE_COUNT = 8


def load_base_module():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_path = os.path.join(project_root, "test-2", "security_aware_probe_search_2.py")
    spec = importlib.util.spec_from_file_location("security_aware_probe_search_base_3", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()

base.LEGAL_POSITION_COUNT = LEGAL_POSITION_COUNT
base.LIGHT_CONDITION = "yellow_shuffled"
base.OUTPUT_RESULTS_FILENAME = OUTPUT_RESULTS_FILENAME
base.TARGET_LEGAL_BER = 0.0
base.MIN_ILLEGAL_BER = 0.3
base.MAX_ILLEGAL_BER = 0.7
base.EVAL_BIT_BLOCK_COUNT = FINAL_EVAL_BIT_BLOCK_COUNT
base.EVAL_BIT_BLOCK_SEED = 20260509
base.SEARCH_EPOCHS = SEARCH_EPOCHS
base.MAX_RANDOM_CANDIDATES_PER_COUNT = MAX_RANDOM_CANDIDATES_PER_COUNT
base.TOP_K_CANDIDATES_PER_COUNT = TOP_K_CANDIDATES_PER_COUNT
base.LOCAL_NEIGHBOR_SAMPLES = LOCAL_NEIGHBOR_SAMPLES
base.LOCAL_ROUNDS = LOCAL_ROUNDS
base.MIN_PROBE_COUNT = MIN_PROBE_COUNT
base.MAX_PROBE_COUNT = MAX_PROBE_COUNT
base.DEFAULT_BASELINE_PROBE_COUNT = DEFAULT_BASELINE_PROBE_COUNT


def generate_position_combinations(project_root: str) -> list[tuple[int, ...]]:
    return list(itertools.combinations(base.get_available_positions(project_root), LEGAL_POSITION_COUNT))


def load_existing_results(results_file: str) -> list[dict]:
    if not os.path.exists(results_file):
        return []
    with open(results_file, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_results(results_file: str, rows: Sequence[dict]) -> None:
    base.write_results(results_file, rows)


def row_from_result(legal_positions: Sequence[int], result: dict) -> dict:
    return base.row_from_result(legal_positions, result)


def evaluate_with_bit_count(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
    bit_count: int,
) -> dict:
    old_bit_count = base.EVAL_BIT_BLOCK_COUNT
    try:
        base.EVAL_BIT_BLOCK_COUNT = bit_count
        return base.evaluate_candidate(project_root, legal_positions, probes)
    finally:
        base.EVAL_BIT_BLOCK_COUNT = old_bit_count


def candidate_rank_key(candidate: dict):
    return base.candidate_rank_key(candidate)


def is_better_candidate(candidate: dict, incumbent: dict | None) -> bool:
    return base.is_better_candidate(candidate, incumbent)


def verify_candidate(project_root: str, legal_positions: Sequence[int], probes: Sequence[float]) -> dict:
    return evaluate_with_bit_count(project_root, legal_positions, probes, FINAL_EVAL_BIT_BLOCK_COUNT)


def try_local_refinement(
    project_root: str,
    legal_positions: Sequence[int],
    initial_probes: Sequence[float],
    all_probes: Sequence[float],
    rng: random.Random,
) -> tuple[np.ndarray, dict]:
    current = np.sort(np.asarray(initial_probes, dtype=float))
    current_eval = evaluate_with_bit_count(project_root, legal_positions, current, COARSE_EVAL_BIT_BLOCK_COUNT)

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
                if not base.test.is_valid_probe_set(trial, min_interval=base.min_interval_for_probe_count(len(trial))):
                    continue
                trial_eval = evaluate_with_bit_count(project_root, legal_positions, trial, COARSE_EVAL_BIT_BLOCK_COUNT)
                if is_better_candidate(trial_eval, current_eval):
                    current = trial
                    current_eval = trial_eval
                    improved = True
        if not improved:
            break

    return current, current_eval


def candidate_probe_counts(baseline_probe_count: int) -> list[int]:
    counts = sorted({
        max(MIN_PROBE_COUNT, baseline_probe_count - 3),
        max(MIN_PROBE_COUNT, baseline_probe_count - 2),
        max(MIN_PROBE_COUNT, baseline_probe_count - 1),
        baseline_probe_count,
        5,
        6,
        7,
        8,
    })
    return [count for count in counts if MIN_PROBE_COUNT <= count <= MAX_PROBE_COUNT]


def search_security_aware_probes(
    project_root: str,
    legal_positions: Sequence[int],
    baseline_probe_count: int,
    rng: random.Random,
    combination_index: int,
) -> dict:
    all_probes = base.get_all_probes(project_root, legal_positions)
    best_candidate: dict | None = None
    shortlisted_candidates: list[dict] = []

    for epoch in range(SEARCH_EPOCHS):
        print(f"  Search epoch {epoch + 1}/{SEARCH_EPOCHS}")
        for probe_count in candidate_probe_counts(baseline_probe_count):
            min_interval = base.min_interval_for_probe_count(probe_count)
            print(f"    Trying probe_count={probe_count}, min_interval={min_interval}")
            epoch_candidates: list[dict] = []
            for candidate_idx in range(MAX_RANDOM_CANDIDATES_PER_COUNT):
                probes = np.sort(np.asarray(rng.sample(all_probes, probe_count), dtype=float))
                if not base.test.is_valid_probe_set(probes, min_interval=min_interval):
                    continue

                candidate_eval = evaluate_with_bit_count(
                    project_root,
                    legal_positions,
                    probes,
                    COARSE_EVAL_BIT_BLOCK_COUNT,
                )
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
                        verify_eval = verify_candidate(project_root, legal_positions, probes)
                        verified_candidate = {
                            "probes": probes.copy(),
                            "probe_count": probe_count,
                            "min_interval": min_interval,
                            **verify_eval,
                        }
                        if verified_candidate["security_satisfied"]:
                            return verified_candidate
                        if is_better_candidate(verified_candidate, best_candidate):
                            best_candidate = verified_candidate

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
            "min_interval": base.min_interval_for_probe_count(len(refined_probes)),
            **refined_eval,
        }
        if is_better_candidate(refined_candidate, best_candidate):
            best_candidate = refined_candidate

        verify_eval = verify_candidate(project_root, legal_positions, refined_probes)
        verified_candidate = {
            "probes": refined_probes.copy(),
            "probe_count": len(refined_probes),
            "min_interval": base.min_interval_for_probe_count(len(refined_probes)),
            **verify_eval,
        }
        if verified_candidate["security_satisfied"]:
            return verified_candidate
        if is_better_candidate(verified_candidate, best_candidate):
            best_candidate = verified_candidate

    if best_candidate is None:
        raise RuntimeError("No probe candidates were evaluated")
    return best_candidate


def run_search(max_combinations: int | None = None) -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(project_root, "test-3", OUTPUT_RESULTS_FILENAME)
    existing_rows = load_existing_results(results_file)
    completed = {
        row["position_combination"]
        for row in existing_rows
        if row.get("security_satisfied") == "yes"
    }

    combinations = generate_position_combinations(project_root)
    if max_combinations is not None:
        combinations = combinations[:max_combinations]
    pending = [combination for combination in combinations if str(tuple(combination)) not in completed]
    print(f"Total combinations={len(combinations)}, safe_existing={len(completed)}, pending={len(pending)}")

    rng = random.Random(base.SELECTION_SEED)
    rows = existing_rows[:]
    for idx, legal_positions in enumerate(combinations, start=1):
        combination_key = str(tuple(legal_positions))
        if combination_key in completed:
            print(f"[{idx}/{len(combinations)}] Skip {legal_positions}: safe probes already exist.")
            continue

        print(f"[{idx}/{len(combinations)}] Searching combination {legal_positions}")
        result = search_security_aware_probes(
            project_root=project_root,
            legal_positions=legal_positions,
            baseline_probe_count=DEFAULT_BASELINE_PROBE_COUNT,
            rng=rng,
            combination_index=idx,
        )
        row = row_from_result(legal_positions, result)
        rows = [old for old in rows if old.get("position_combination") != combination_key]
        rows.append(row)
        if row["security_satisfied"] == "yes":
            completed.add(row["position_combination"])
        write_results(results_file, rows)
        print(
            f"  New result: legal_ber={row['legal_ber']}, "
            f"legal_position_bers={row['legal_position_bers']}, "
            f"min_illegal_ber={row['min_illegal_ber'] or 'N/A'}, "
            f"satisfied={result['security_satisfied']}"
        )

    print(f"Results saved to: {results_file}")
    return results_file


def main() -> None:
    max_combinations = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_search(max_combinations=max_combinations)


if __name__ == "__main__":
    main()
