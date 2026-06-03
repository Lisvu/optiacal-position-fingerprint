#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security-aware batch experiment for 4-position combinations.

Compared with batch_test.py, this script keeps the original legal BER search
logic but adds a security constraint:
1. legal test BER must be <= 0.02
2. the worst illegal single-route BER across all illegal positions and all
   legal routes must be > 0.3
3. no illegal device may decode any legal route with BER = 0
4. no illegal/legal code pair may be too highly correlated
5. among secure candidates, prefer the one whose worst illegal single-route BER
   is closest to 0.5
"""

from __future__ import annotations

import ast
import csv
import itertools
import os
import random
import sys
from typing import Iterable, Sequence

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import security_illegal_position_validation as security_eval
import test_4_simple as test


TOTAL_POSITIONS = 28
COMBINATION_SIZE = 4
SAMPLE_COUNT = 20
RESULTS_FILENAME = "batch_test_results_security_aware.csv"
TARGET_LEGAL_TEST_BER = 0.02
MIN_ILLEGAL_SINGLE_BER_THRESHOLD = 0.2
TARGET_ILLEGAL_SINGLE_BER = 0.5
MAX_ILLEGAL_ABS_CODE_CORR_THRESHOLD = 0.93
SEARCH_BITS = 10000
TEST_BITS = 10000
SECURITY_SEARCH_BITS = 3000
SECURITY_CONFIRM_BITS = 10000
MAX_SEARCH_ATTEMPTS = 10
LEGAL_SEARCH_RESTARTS = 5
LEGAL_MAX_CANDIDATES = 1800
LEGAL_CANDIDATE_BER_THRESHOLD = 0.01
MAX_SECURITY_SCREENED_CANDIDATES_PER_STRATEGY = 8
ZERO_BER_EPSILON = 1e-12
SEARCH_STRATEGIES = [
    {"label": "baseline", "min_interval": 30, "search_intensity": 1.0, "search_restarts": LEGAL_SEARCH_RESTARTS},
    {"label": "wider-gap", "min_interval": 35, "search_intensity": 1.15, "search_restarts": LEGAL_SEARCH_RESTARTS},
    {"label": "security-first", "min_interval": 40, "search_intensity": 1.35, "search_restarts": LEGAL_SEARCH_RESTARTS + 1},
]


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


def parse_position_combination(value: str) -> tuple[int, ...]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple):
        raise ValueError(f"Invalid position combination: {value}")
    return tuple(int(v) for v in parsed)


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def load_existing_results(results_file: str) -> list[dict]:
    if not os.path.exists(results_file):
        return []

    with open(results_file, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize_results_file(results_file: str) -> tuple[set[str], float, int]:
    existing_rows = load_existing_results(results_file)
    processed: set[str] = set()
    cumulative_sum = 0.0
    completed_count = 0
    fieldnames = [
        "position_combination",
        "best_probe_count",
        "best_probes",
        "best_ber",
        "test_ber",
        "min_illegal_single_ber",
        "average_illegal_single_ber",
        "worst_illegal_position",
        "worst_legal_position",
        "max_illegal_abs_code_corr",
        "max_corr_illegal_position",
        "max_corr_legal_position",
        "illegal_zero_ber_detected",
        "zero_ber_illegal_position",
        "zero_ber_legal_position",
        "security_bits",
        "security_satisfied",
        "illegal_distance_to_half",
        "attempt_count",
        "cumulative_test_ber",
    ]

    if not existing_rows:
        with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
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
            "min_illegal_single_ber": row["min_illegal_single_ber"],
            "average_illegal_single_ber": row["average_illegal_single_ber"],
            "worst_illegal_position": row["worst_illegal_position"],
            "worst_legal_position": row["worst_legal_position"],
            "max_illegal_abs_code_corr": row.get("max_illegal_abs_code_corr", ""),
            "max_corr_illegal_position": row.get("max_corr_illegal_position", ""),
            "max_corr_legal_position": row.get("max_corr_legal_position", ""),
            "illegal_zero_ber_detected": row.get("illegal_zero_ber_detected", ""),
            "zero_ber_illegal_position": row.get("zero_ber_illegal_position", ""),
            "zero_ber_legal_position": row.get("zero_ber_legal_position", ""),
            "security_bits": row["security_bits"],
            "security_satisfied": row["security_satisfied"],
            "illegal_distance_to_half": row["illegal_distance_to_half"],
            "attempt_count": row.get("attempt_count", ""),
            "cumulative_test_ber": f"{cumulative_test_ber:.6f}",
        })
        processed.add(position_combination)

    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    return processed, cumulative_sum, completed_count


def evaluate_security_for_probe_set(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
    num_bits: int,
    rng_seed: int,
) -> dict:
    legal_csv_files = test.build_csv_files_for_positions(
        project_root,
        legal_positions,
        light_condition=security_eval.LIGHT_CONDITION,
    )
    legal_models, hue_mapping = test.build_models_from_probes(
        legal_csv_files,
        np.asarray(probes, dtype=float),
        mapping_eval_bits=security_eval.MAPPING_EVAL_BITS,
        mapping_top_k=security_eval.MAPPING_TOP_K,
        rng=random.Random(rng_seed),
    )

    available_positions = security_eval.get_available_positions(project_root, security_eval.LIGHT_CONDITION)
    illegal_positions = [pos for pos in available_positions if pos not in legal_positions]
    worst_illegal_position = None
    worst_legal_position = None
    min_illegal_single_ber = float("inf")
    max_illegal_abs_code_corr = 0.0
    max_corr_illegal_position = None
    max_corr_legal_position = None
    zero_ber_illegal_position = None
    zero_ber_legal_position = None
    all_single_bers: list[float] = []

    rng = random.Random(rng_seed + 9973)
    for illegal_position in illegal_positions:
        illegal_csv_file = test.build_csv_files_for_positions(
            project_root,
            [illegal_position],
            light_condition=security_eval.LIGHT_CONDITION,
        )[0]
        probes_array = np.asarray(probes, dtype=float)
        illegal_matrix = test.load_selected_rows([illegal_csv_file], probes_array)[0]
        illegal_model = test.extract_fingerprint(probes_array, illegal_matrix, force_positive_first=True)
        code_correlations = [
            float(test.calculate_correlation(illegal_model.code, legal_model.code))
            for legal_model in legal_models
        ]
        for route_idx, corr in enumerate(code_correlations):
            abs_corr = abs(float(corr))
            if abs_corr > max_illegal_abs_code_corr:
                max_illegal_abs_code_corr = abs_corr
                max_corr_illegal_position = illegal_position
                max_corr_legal_position = legal_positions[route_idx]
        position_bers, _ = security_eval.evaluate_illegal_position_against_legal_bits(
            legal_models=legal_models,
            hue_mapping=hue_mapping,
            illegal_csv_file=illegal_csv_file,
            probes=probes,
            num_bits=num_bits,
            rng=random.Random(rng.randrange(0, 2**31)),
        )
        all_single_bers.extend(float(v) for v in position_bers)
        for route_idx, ber in enumerate(position_bers):
            if float(ber) <= ZERO_BER_EPSILON and zero_ber_illegal_position is None:
                zero_ber_illegal_position = illegal_position
                zero_ber_legal_position = legal_positions[route_idx]
        local_min = min((float(ber), idx) for idx, ber in enumerate(position_bers))
        if local_min[0] < min_illegal_single_ber:
            min_illegal_single_ber = local_min[0]
            worst_illegal_position = illegal_position
            worst_legal_position = legal_positions[local_min[1]]

    average_illegal_single_ber = (
        sum(all_single_bers) / len(all_single_bers) if all_single_bers else 0.0
    )
    return {
        "min_illegal_single_ber": float(min_illegal_single_ber if all_single_bers else 0.0),
        "average_illegal_single_ber": float(average_illegal_single_ber),
        "worst_illegal_position": worst_illegal_position,
        "worst_legal_position": worst_legal_position,
        "max_illegal_abs_code_corr": float(max_illegal_abs_code_corr),
        "max_corr_illegal_position": max_corr_illegal_position,
        "max_corr_legal_position": max_corr_legal_position,
        "illegal_zero_ber_detected": bool(zero_ber_illegal_position is not None),
        "zero_ber_illegal_position": zero_ber_illegal_position,
        "zero_ber_legal_position": zero_ber_legal_position,
        "security_bits": num_bits,
    }


def attach_security_metrics(result: dict, security_metrics: dict) -> dict:
    merged = dict(result)
    merged.update(security_metrics)
    merged["illegal_distance_to_half"] = abs(
        float(merged["min_illegal_single_ber"]) - TARGET_ILLEGAL_SINGLE_BER
    )
    merged["illegal_corr_distance_to_limit"] = (
        MAX_ILLEGAL_ABS_CODE_CORR_THRESHOLD - float(merged["max_illegal_abs_code_corr"])
    )
    merged["security_satisfied"] = (
        float(merged["test_ber"]) <= TARGET_LEGAL_TEST_BER
        and float(merged["min_illegal_single_ber"]) > MIN_ILLEGAL_SINGLE_BER_THRESHOLD
        and float(merged["max_illegal_abs_code_corr"]) < MAX_ILLEGAL_ABS_CODE_CORR_THRESHOLD
        and not bool(merged["illegal_zero_ber_detected"])
    )
    return merged


def candidate_rank_key(candidate: dict) -> tuple:
    zero_ber_penalty = 1 if candidate["illegal_zero_ber_detected"] else 0
    satisfied = 0 if candidate["security_satisfied"] else 1
    legal_gate_penalty = 0 if float(candidate["test_ber"]) <= TARGET_LEGAL_TEST_BER else 1
    corr_penalty = float(candidate["max_illegal_abs_code_corr"])
    distance_to_half = float(candidate["illegal_distance_to_half"])
    legal_test_ber = float(candidate["test_ber"])
    security_floor = -float(candidate["min_illegal_single_ber"])
    best_ber = float(candidate["best_ber"])
    return (
        zero_ber_penalty,
        satisfied,
        legal_gate_penalty,
        corr_penalty,
        distance_to_half,
        legal_test_ber,
        security_floor,
        best_ber,
    )


def evaluate_search_strategy(
    *,
    project_root: str,
    combination: Sequence[int],
    csv_files: Sequence[str],
    idx: int,
    attempt: int,
    strategy_idx: int,
    strategy: dict,
) -> dict:
    strategy_seed = 1000000 * idx + attempt * 10007 + strategy_idx * 997
    search_result = test.run_position_experiment_collect_candidates(
        csv_files=csv_files,
        search_bits=SEARCH_BITS,
        test_bits=TEST_BITS,
        min_probes=5,
        max_probes=20,
        max_candidates=LEGAL_MAX_CANDIDATES,
        seed=strategy_seed,
        search_restarts=int(strategy["search_restarts"]),
        min_interval=int(strategy["min_interval"]),
        search_intensity=float(strategy["search_intensity"]),
        candidate_ber_threshold=LEGAL_CANDIDATE_BER_THRESHOLD,
        max_collected_candidates=MAX_SECURITY_SCREENED_CANDIDATES_PER_STRATEGY,
    )
    candidate_results = search_result["candidate_results"]
    if not candidate_results:
        candidate_results = [search_result["best_result"]]

    evaluated_candidates: list[dict] = []
    for candidate_idx, result in enumerate(candidate_results, start=1):
        security_metrics = evaluate_security_for_probe_set(
            project_root=project_root,
            legal_positions=combination,
            probes=result["best_probes"],
            num_bits=SECURITY_SEARCH_BITS,
            rng_seed=2000000 * idx + attempt * 7919 + strategy_idx * 313 + candidate_idx * 41,
        )
        candidate = attach_security_metrics(result, security_metrics)
        candidate["strategy_label"] = strategy["label"]
        candidate["strategy_min_interval"] = strategy["min_interval"]
        candidate["strategy_search_intensity"] = strategy["search_intensity"]
        candidate["strategy_search_restarts"] = strategy["search_restarts"]
        candidate["strategy_candidate_index"] = candidate_idx
        evaluated_candidates.append(candidate)

    evaluated_candidates.sort(key=lambda item: (float(item["best_ber"]), float(item["test_ber"])))
    return {
        "best_result": search_result["best_result"],
        "evaluated_candidates": evaluated_candidates,
    }


def run_batch_experiment() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(project_root, "test-4", RESULTS_FILENAME)

    processed_combinations, cumulative_sum, completed_count = normalize_results_file(results_file)
    combinations = generate_unseen_random_position_combinations(
        TOTAL_POSITIONS,
        COMBINATION_SIZE,
        SAMPLE_COUNT,
        None,
        processed_combinations,
    )

    print(f"Randomly selected {len(combinations)} new position combinations")

    with open(results_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        for idx, combination in enumerate(combinations, start=1):
            combination_str = str(combination)
            if combination_str in processed_combinations:
                print(f"[{idx}/{len(combinations)}] position combination {combination} already completed, skipping")
                continue

            csv_files = test.build_csv_files_for_positions(project_root, combination, light_condition="white")
            if not all(os.path.exists(path) for path in csv_files):
                print(f"[{idx}/{len(combinations)}] position combination {combination} is missing data files, skipping")
                continue

            print(f"[{idx}/{len(combinations)}] processing security-aware combination {combination}")
            best_candidate = None
            best_zero_ber_rejected = None

            for attempt in range(1, MAX_SEARCH_ATTEMPTS + 1):
                print(
                    f"  search attempt {attempt}: target legal test BER <= {TARGET_LEGAL_TEST_BER:.3f}, "
                    f"min illegal single BER > {MIN_ILLEGAL_SINGLE_BER_THRESHOLD:.3f}, "
                    f"max |corr| < {MAX_ILLEGAL_ABS_CODE_CORR_THRESHOLD:.2f}"
                )
                attempt_best_candidate = None
                for strategy_idx, strategy in enumerate(SEARCH_STRATEGIES, start=1):
                    print(
                        f"    strategy {strategy_idx}/{len(SEARCH_STRATEGIES)} "
                        f"[{strategy['label']}]: min_interval={strategy['min_interval']}, "
                        f"search_intensity={strategy['search_intensity']:.2f}, "
                        f"search_restarts={strategy['search_restarts']}"
                    )
                    strategy_result = evaluate_search_strategy(
                        project_root=project_root,
                        combination=combination,
                        csv_files=csv_files,
                        idx=idx,
                        attempt=attempt,
                        strategy_idx=strategy_idx,
                        strategy=strategy,
                    )
                    raw_candidate_count = len(strategy_result["evaluated_candidates"])
                    print(f"      collected {raw_candidate_count} candidate probe sets for security screening.")

                    for candidate in strategy_result["evaluated_candidates"]:
                        print(
                            f"      candidate #{candidate['strategy_candidate_index']}: "
                            f"best BER = {candidate['best_ber']:.6f}, "
                            f"test BER = {candidate['test_ber']:.6f}, "
                            f"min illegal BER = {candidate['min_illegal_single_ber']:.6f}, "
                            f"max |corr| = {candidate['max_illegal_abs_code_corr']:.6f}"
                        )
                        if candidate["illegal_zero_ber_detected"]:
                            print(
                                "        rejected by hard security rule: "
                                f"illegal position {candidate['zero_ber_illegal_position']} can decode legal position "
                                f"{candidate['zero_ber_legal_position']} with BER = 0."
                            )
                        elif float(candidate["max_illegal_abs_code_corr"]) >= MAX_ILLEGAL_ABS_CODE_CORR_THRESHOLD:
                            print(
                                "        rejected by correlation guard: "
                                f"max |corr| = {candidate['max_illegal_abs_code_corr']:.6f} at "
                                f"({candidate['max_corr_illegal_position']}, {candidate['max_corr_legal_position']})."
                            )

                        if candidate["security_satisfied"] and SECURITY_CONFIRM_BITS != SECURITY_SEARCH_BITS:
                            confirmed_metrics = evaluate_security_for_probe_set(
                                project_root=project_root,
                                legal_positions=combination,
                                probes=candidate["best_probes"],
                                num_bits=SECURITY_CONFIRM_BITS,
                                rng_seed=3000000 * idx + attempt * 104729 + strategy_idx * 577 + candidate["strategy_candidate_index"] * 61,
                            )
                            confirmed_candidate = attach_security_metrics(candidate, confirmed_metrics)
                            confirmed_candidate.update({
                                "strategy_label": candidate["strategy_label"],
                                "strategy_min_interval": candidate["strategy_min_interval"],
                                "strategy_search_intensity": candidate["strategy_search_intensity"],
                                "strategy_search_restarts": candidate["strategy_search_restarts"],
                                "strategy_candidate_index": candidate["strategy_candidate_index"],
                            })
                            candidate = confirmed_candidate
                            print(
                                f"        confirmation ({SECURITY_CONFIRM_BITS} bits): "
                                f"min illegal BER = {candidate['min_illegal_single_ber']:.6f}, "
                                f"max |corr| = {candidate['max_illegal_abs_code_corr']:.6f}"
                            )

                        candidate["attempt_count"] = attempt
                        if attempt_best_candidate is None or candidate_rank_key(candidate) < candidate_rank_key(attempt_best_candidate):
                            attempt_best_candidate = candidate

                        if (
                            candidate["illegal_zero_ber_detected"]
                            or float(candidate["max_illegal_abs_code_corr"]) >= MAX_ILLEGAL_ABS_CODE_CORR_THRESHOLD
                        ):
                            if (
                                best_zero_ber_rejected is None
                                or candidate_rank_key(candidate) < candidate_rank_key(best_zero_ber_rejected)
                            ):
                                best_zero_ber_rejected = candidate
                            continue

                        if best_candidate is None or candidate_rank_key(candidate) < candidate_rank_key(best_candidate):
                            best_candidate = candidate
                            print("        updated best security-aware candidate for this combination.")

                        if candidate["security_satisfied"]:
                            print("        found a candidate that satisfies the legal/security thresholds, moving to the next combination.")
                            break

                    if best_candidate is not None and best_candidate.get("attempt_count") == attempt and best_candidate["security_satisfied"]:
                        break

                if attempt_best_candidate is not None:
                    print(
                        f"    best candidate in attempt {attempt}: strategy={attempt_best_candidate['strategy_label']}, "
                        f"legal test BER={attempt_best_candidate['test_ber']:.6f}, "
                        f"min illegal BER={attempt_best_candidate['min_illegal_single_ber']:.6f}, "
                        f"max |corr|={attempt_best_candidate['max_illegal_abs_code_corr']:.6f}"
                    )

                if (
                    best_candidate is not None
                    and best_candidate["security_satisfied"]
                ):
                    print("    found a secure candidate in this combination, stopping early.")
                    break

            if best_candidate is None:
                if best_zero_ber_rejected is not None:
                    print(
                        f"[{idx}/{len(combinations)}] no acceptable probe set found for {combination}: "
                        "every searched candidate leaked at least one legal route with BER = 0. "
                        f"Closest rejected pair = ({best_zero_ber_rejected['zero_ber_illegal_position']}, "
                        f"{best_zero_ber_rejected['zero_ber_legal_position']})."
                    )
                    continue
                raise RuntimeError(f"Failed to obtain a result for position combination {combination}")

            print(f"[{idx}/{len(combinations)}] position combination: {combination}")
            print(f"  best probe count: {best_candidate['best_probe_count']}")
            print(f"  probe combination: {best_candidate['best_probes']}")
            print(
                f"  selected strategy: {best_candidate['strategy_label']} "
                f"(min_interval={best_candidate['strategy_min_interval']}, "
                f"search_intensity={best_candidate['strategy_search_intensity']:.2f}, "
                f"search_restarts={best_candidate['strategy_search_restarts']})"
            )
            print(f"  legal best BER: {best_candidate['best_ber']:.6f}")
            print(f"  legal test BER: {best_candidate['test_ber']:.6f}")
            print(f"  min illegal single BER: {best_candidate['min_illegal_single_ber']:.6f}")
            print(f"  average illegal single BER: {best_candidate['average_illegal_single_ber']:.6f}")
            print(
                f"  max illegal |code corr|: {best_candidate['max_illegal_abs_code_corr']:.6f} "
                f"at ({best_candidate['max_corr_illegal_position']}, {best_candidate['max_corr_legal_position']})"
            )
            print(
                f"  worst illegal/legal pair: "
                f"({best_candidate['worst_illegal_position']}, {best_candidate['worst_legal_position']})"
            )
            print(f"  security satisfied: {'yes' if best_candidate['security_satisfied'] else 'no'}")

            completed_count += 1
            cumulative_sum += float(best_candidate["test_ber"])
            cumulative_test_ber = cumulative_sum / completed_count
            print(f"  cumulative test BER: {cumulative_test_ber:.6f}")

            writer.writerow([
                combination_str,
                best_candidate["best_probe_count"],
                format_probes(best_candidate["best_probes"]),
                f"{best_candidate['best_ber']:.6f}",
                f"{best_candidate['test_ber']:.6f}",
                f"{best_candidate['min_illegal_single_ber']:.6f}",
                f"{best_candidate['average_illegal_single_ber']:.6f}",
                best_candidate["worst_illegal_position"],
                best_candidate["worst_legal_position"],
                f"{best_candidate['max_illegal_abs_code_corr']:.6f}",
                best_candidate["max_corr_illegal_position"],
                best_candidate["max_corr_legal_position"],
                "yes" if best_candidate["illegal_zero_ber_detected"] else "no",
                best_candidate["zero_ber_illegal_position"],
                best_candidate["zero_ber_legal_position"],
                best_candidate["security_bits"],
                "yes" if best_candidate["security_satisfied"] else "no",
                f"{best_candidate['illegal_distance_to_half']:.6f}",
                best_candidate["attempt_count"],
                f"{cumulative_test_ber:.6f}",
            ])
            f.flush()

    return results_file


def main() -> None:
    results_file = run_batch_experiment()
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
