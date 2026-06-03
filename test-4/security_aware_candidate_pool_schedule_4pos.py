#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a large legal-reliable candidate pool, select a safe subset, and schedule it.

This script keeps the original test-4 experiments unchanged. It runs on the mid
dataset through random_probe_hopping_yellow_4pos.py's adapter.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np


LIGHT_CONDITION = "mid"
LEGAL_POSITION_COUNT = 4
SEED = 20260512
DEFAULT_COMBINATION_COUNT = 5
DEFAULT_PROBE_COUNT = 12
DEFAULT_CANDIDATE_TARGET_SIZE = 100
DEFAULT_SELECTED_POOL_SIZE = 10
DEFAULT_SCHEDULE_LENGTH = 1000
DEFAULT_SECURITY_EVAL_BITS = 500
DEFAULT_MAX_ATTEMPTS = 30000
DEFAULT_MIN_PROBE_INTERVAL = 5
DEFAULT_OVERLAP_RATIO_FOR_TIEBREAK = 0.75

DEFAULT_CANDIDATE_FILENAME = "mid_security_aware_candidate_pool_4pos.csv"
DEFAULT_SELECTED_FILENAME = "mid_security_aware_selected_pool_4pos.csv"
DEFAULT_SCHEDULE_FILENAME = "mid_security_aware_offline_schedule_4pos.csv"


def load_random_hopping_module():
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_probe_hopping_yellow_4pos.py")
    spec = importlib.util.spec_from_file_location("random_probe_hopping_mid_4pos_for_candidate_pool", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load random hopping module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rh4 = load_random_hopping_module()
base = rh4.base


@dataclass
class Candidate:
    candidate_id: int
    entry: object
    route_keys: list[tuple[int, int]]
    raw_bers: np.ndarray
    secure_bers: np.ndarray

    @property
    def min_secure_ber(self) -> float:
        return float(np.min(self.secure_bers))

    @property
    def average_secure_ber(self) -> float:
        return float(np.mean(self.secure_bers))

    @property
    def zero_leak_route_count(self) -> int:
        return int(np.sum(self.secure_bers <= 0.0))

    @property
    def worst_route(self) -> tuple[int, int]:
        return self.route_keys[int(np.argmin(self.secure_bers))]


def corrected_ber(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.minimum(values, 1.0 - values)


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def format_float_list(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.8f}" for v in values) + "]"


def format_int_list(values: Sequence[int]) -> str:
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def route_dict_text(route_keys: Sequence[tuple[int, int]], values: Sequence[float]) -> str:
    return str({f"{route[0]}->{route[1]}": f"{float(value):.8f}" for route, value in zip(route_keys, values)})


def evaluate_entry_route_raw_bers(
    project_root: str,
    legal_positions: Sequence[int],
    entry: object,
    bit_blocks_pm: Sequence[np.ndarray],
) -> dict[tuple[int, int], float]:
    route_raw_bers: dict[tuple[int, int], float] = {}
    illegal_positions = [p for p in base.get_available_positions(project_root) if p not in legal_positions]
    legal_codes = [model.code for model in entry.legal_models]

    for illegal_position in illegal_positions:
        illegal_csv = base.build_csv_files(project_root, [illegal_position])[0]
        illegal_matrix = base.test.load_selected_rows([illegal_csv], entry.probes)[0]
        illegal_model = base.test.extract_fingerprint(entry.probes, illegal_matrix, force_positive_first=True)
        probe_to_row = base.test.build_probe_to_row(entry.probes)
        errors = np.zeros(len(legal_positions), dtype=float)
        totals = np.zeros(len(legal_positions), dtype=float)

        for bits_pm in bit_blocks_pm:
            _, symbol_combinations = base.test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = base.test.map_symbol_to_hue(symbol_combinations, entry.hue_mapping)
            illegal_observation = base.test.observe_block_from_measured_matrix(hue_seq, illegal_matrix, probe_to_row)
            illegal_dec = base.test.decode_local_block(illegal_observation, illegal_model.w, illegal_model.code)
            true_bits = base.test.pm1_to_bin(bits_pm)
            for idx, true_bit in enumerate(true_bits):
                totals[idx] += 1
                if int(illegal_dec.bit_hat_bin) != int(true_bit):
                    errors[idx] += 1

        raw_bers = errors / np.maximum(totals, 1.0)
        for idx, raw_ber in enumerate(raw_bers):
            route_raw_bers[(int(illegal_position), int(legal_positions[idx]))] = float(raw_ber)
    return route_raw_bers


def max_probe_overlap_ratio(probes: Sequence[float], selected: Sequence[Candidate]) -> float:
    if not selected:
        return 0.0
    probe_set = {float(v) for v in probes}
    return max(len(probe_set & {float(v) for v in candidate.entry.probes}) / max(len(probe_set), 1) for candidate in selected)


def append_csv_rows(file_path: str, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def reset_output_files(file_paths: Sequence[str]) -> None:
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)


def collect_candidates(
    project_root: str,
    legal_positions: Sequence[int],
    probe_count: int,
    target_size: int,
    security_eval_bits: int,
    max_attempts: int,
    min_probe_interval: int,
    rng: random.Random,
) -> list[Candidate]:
    all_probes = base.get_all_probes(project_root, legal_positions)
    security_blocks = base.test.generate_random_bit_blocks(security_eval_bits, len(legal_positions), rng=rng)
    candidates: list[Candidate] = []
    seen: set[tuple[float, ...]] = set()
    attempts = 0

    while attempts < max_attempts and len(candidates) < target_size:
        attempts += 1
        probes = base.sample_valid_probe_set(all_probes, probe_count, min_probe_interval, rng)
        if probes is None:
            continue
        signature = tuple(float(v) for v in probes)
        if signature in seen:
            continue
        seen.add(signature)

        entry = base.build_probe_pool_entry(
            project_root=project_root,
            legal_positions=legal_positions,
            probes=probes,
            entry_id=len(candidates) + 1,
            rng=rng,
        )
        if not all(float(ber) == 0.0 for ber in entry.legal_position_bers):
            continue

        route_raw_bers = evaluate_entry_route_raw_bers(project_root, legal_positions, entry, security_blocks)
        route_keys = sorted(route_raw_bers)
        raw_bers = np.asarray([route_raw_bers[route] for route in route_keys], dtype=float)
        candidate = Candidate(
            candidate_id=len(candidates) + 1,
            entry=entry,
            route_keys=route_keys,
            raw_bers=raw_bers,
            secure_bers=corrected_ber(raw_bers),
        )
        candidates.append(candidate)
        print(
            f"      Candidate {len(candidates)}/{target_size}: "
            f"min_secure={candidate.min_secure_ber:.6f}, avg_secure={candidate.average_secure_ber:.6f}, "
            f"zero_routes={candidate.zero_leak_route_count}"
        )

    print(f"      Collected {len(candidates)}/{target_size} candidates after {attempts} attempts.")
    return candidates


def select_security_complementary_pool(candidates: Sequence[Candidate], selected_size: int) -> list[Candidate]:
    selected: list[Candidate] = []
    remaining = list(candidates)
    route_count = len(candidates[0].route_keys) if candidates else 0
    cumulative_raw = np.zeros(route_count, dtype=float)

    while remaining and len(selected) < selected_size:
        step = len(selected) + 1

        def score(candidate: Candidate) -> tuple[float, float, float, float, float]:
            mixed_secure = corrected_ber((cumulative_raw + candidate.raw_bers) / step)
            overlap = max_probe_overlap_ratio(candidate.entry.probes, selected)
            overlap_penalty = max(0.0, overlap - DEFAULT_OVERLAP_RATIO_FOR_TIEBREAK)
            return (
                float(np.min(mixed_secure)),
                float(np.mean(mixed_secure)),
                candidate.min_secure_ber,
                -overlap_penalty,
                -overlap,
            )

        best = max(remaining, key=score)
        selected.append(best)
        cumulative_raw += best.raw_bers
        remaining.remove(best)
        print(
            f"      Select {len(selected)}/{selected_size}: candidate={best.candidate_id}, "
            f"pool_min={float(np.min(corrected_ber(cumulative_raw / len(selected)))):.6f}, "
            f"pool_avg={float(np.mean(corrected_ber(cumulative_raw / len(selected)))):.6f}"
        )
    return selected


def optimize_usage_ratio(raw_matrix: np.ndarray) -> tuple[np.ndarray, float, str]:
    route_count, entry_count = raw_matrix.shape
    try:
        from scipy.optimize import linprog

        c = np.zeros(entry_count + 1, dtype=float)
        c[-1] = -1.0
        a_ub = []
        b_ub = []
        for route_idx in range(route_count):
            row = np.zeros(entry_count + 1, dtype=float)
            row[:entry_count] = -raw_matrix[route_idx]
            row[-1] = 1.0
            a_ub.append(row)
            b_ub.append(0.0)

            row = np.zeros(entry_count + 1, dtype=float)
            row[:entry_count] = raw_matrix[route_idx]
            row[-1] = 1.0
            a_ub.append(row)
            b_ub.append(1.0)

        result = linprog(
            c,
            A_ub=np.asarray(a_ub, dtype=float),
            b_ub=np.asarray(b_ub, dtype=float),
            A_eq=np.asarray([[*([1.0] * entry_count), 0.0]], dtype=float),
            b_eq=np.asarray([1.0], dtype=float),
            bounds=[(0.0, 1.0)] * entry_count + [(0.0, 0.5)],
            method="highs",
        )
        if result.success:
            ratio = np.maximum(result.x[:entry_count], 0.0)
            ratio = ratio / np.sum(ratio)
            objective = float(np.min(corrected_ber(raw_matrix @ ratio)))
            return ratio, objective, "linprog"
    except Exception:
        pass

    rng = np.random.default_rng(SEED)
    candidates = [np.full(entry_count, 1.0 / entry_count, dtype=float)]
    candidates.extend(np.eye(entry_count, dtype=float))
    candidates.extend(rng.dirichlet(np.ones(entry_count), size=20000))
    best_ratio = candidates[0]
    best_score = -1.0
    for ratio in candidates:
        score = float(np.min(corrected_ber(raw_matrix @ ratio)))
        if score > best_score:
            best_score = score
            best_ratio = ratio
    return np.asarray(best_ratio, dtype=float), best_score, "random_dirichlet_fallback"


def counts_from_ratio(ratio: np.ndarray, schedule_length: int) -> np.ndarray:
    raw_counts = ratio * int(schedule_length)
    counts = np.floor(raw_counts).astype(int)
    remainder = int(schedule_length) - int(np.sum(counts))
    if remainder > 0:
        for idx in np.argsort(-(raw_counts - counts))[:remainder]:
            counts[idx] += 1
    return counts


def build_risk_balanced_sequence(raw_matrix: np.ndarray, counts: np.ndarray, ids: Sequence[int]) -> list[int]:
    remaining = counts.astype(int).copy()
    cumulative_raw = np.zeros(raw_matrix.shape[0], dtype=float)
    sequence: list[int] = []
    last_idx: int | None = None

    while int(np.sum(remaining)) > 0:
        step = len(sequence) + 1
        candidates = [idx for idx, count in enumerate(remaining) if count > 0]
        non_repeat = [idx for idx in candidates if idx != last_idx]
        if non_repeat:
            candidates = non_repeat

        best_idx = max(
            candidates,
            key=lambda idx: (
                float(np.min(corrected_ber((cumulative_raw + raw_matrix[:, idx]) / step))),
                float(np.mean(corrected_ber((cumulative_raw + raw_matrix[:, idx]) / step))),
                int(remaining[idx]),
            ),
        )
        sequence.append(int(ids[best_idx]))
        cumulative_raw += raw_matrix[:, best_idx]
        remaining[best_idx] -= 1
        last_idx = best_idx
    return sequence


def candidate_fieldnames() -> list[str]:
    return [
        "position_combination",
        "candidate_id",
        "probe_count",
        "probes",
        "hue_mapping",
        "legal_position_bers",
        "min_secure_ber",
        "average_secure_ber",
        "zero_leak_route_count",
        "worst_route",
        "route_secure_bers",
        "route_raw_bers",
    ]


def selected_fieldnames() -> list[str]:
    return [
        "position_combination",
        "selection_rank",
        "candidate_id",
        "selected_pool_size",
        "probe_count",
        "probes",
        "hue_mapping",
        "legal_position_bers",
        "candidate_min_secure_ber",
        "candidate_average_secure_ber",
        "candidate_zero_leak_route_count",
        "candidate_worst_route",
        "selected_pool_min_secure_ber",
        "selected_pool_average_secure_ber",
        "route_secure_bers",
    ]


def schedule_fieldnames() -> list[str]:
    return [
        "position_combination",
        "candidate_count",
        "selected_pool_size",
        "schedule_length",
        "security_eval_bits",
        "optimizer",
        "selected_candidate_ids",
        "usage_ratio",
        "usage_counts",
        "offline_objective_min_secure_ber",
        "offline_average_secure_ber",
        "worst_route",
        "schedule_sequence",
        "route_raw_bers",
        "route_min_bers",
        "route_secure_bers",
    ]


def write_candidates(file_path: str, legal_positions: Sequence[int], candidates: Sequence[Candidate]) -> None:
    rows = []
    for candidate in candidates:
        entry = candidate.entry
        rows.append({
            "position_combination": str(tuple(legal_positions)),
            "candidate_id": candidate.candidate_id,
            "probe_count": len(entry.probes),
            "probes": format_probes(entry.probes),
            "hue_mapping": base.format_hue_mapping(entry.hue_mapping),
            "legal_position_bers": format_float_list(entry.legal_position_bers),
            "min_secure_ber": f"{candidate.min_secure_ber:.8f}",
            "average_secure_ber": f"{candidate.average_secure_ber:.8f}",
            "zero_leak_route_count": candidate.zero_leak_route_count,
            "worst_route": f"{candidate.worst_route[0]}->{candidate.worst_route[1]}",
            "route_secure_bers": route_dict_text(candidate.route_keys, candidate.secure_bers),
            "route_raw_bers": route_dict_text(candidate.route_keys, candidate.raw_bers),
        })
    append_csv_rows(file_path, candidate_fieldnames(), rows)


def write_selected(file_path: str, legal_positions: Sequence[int], selected: Sequence[Candidate]) -> None:
    rows = []
    cumulative_raw = np.zeros(len(selected[0].route_keys), dtype=float)
    for rank, candidate in enumerate(selected, start=1):
        cumulative_raw += candidate.raw_bers
        pool_secure = corrected_ber(cumulative_raw / rank)
        entry = candidate.entry
        rows.append({
            "position_combination": str(tuple(legal_positions)),
            "selection_rank": rank,
            "candidate_id": candidate.candidate_id,
            "selected_pool_size": len(selected),
            "probe_count": len(entry.probes),
            "probes": format_probes(entry.probes),
            "hue_mapping": base.format_hue_mapping(entry.hue_mapping),
            "legal_position_bers": format_float_list(entry.legal_position_bers),
            "candidate_min_secure_ber": f"{candidate.min_secure_ber:.8f}",
            "candidate_average_secure_ber": f"{candidate.average_secure_ber:.8f}",
            "candidate_zero_leak_route_count": candidate.zero_leak_route_count,
            "candidate_worst_route": f"{candidate.worst_route[0]}->{candidate.worst_route[1]}",
            "selected_pool_min_secure_ber": f"{float(np.min(pool_secure)):.8f}",
            "selected_pool_average_secure_ber": f"{float(np.mean(pool_secure)):.8f}",
            "route_secure_bers": route_dict_text(candidate.route_keys, candidate.secure_bers),
        })
    append_csv_rows(file_path, selected_fieldnames(), rows)


def write_schedule(
    file_path: str,
    legal_positions: Sequence[int],
    candidate_count: int,
    selected: Sequence[Candidate],
    schedule_length: int,
    security_eval_bits: int,
) -> None:
    route_keys = selected[0].route_keys
    raw_matrix = np.asarray([candidate.raw_bers for candidate in selected], dtype=float).T
    ratio, objective, optimizer = optimize_usage_ratio(raw_matrix)
    counts = counts_from_ratio(ratio, schedule_length)
    selected_ids = [candidate.candidate_id for candidate in selected]
    sequence = build_risk_balanced_sequence(raw_matrix, counts, selected_ids)
    actual_ratio = np.asarray([sequence.count(candidate_id) for candidate_id in selected_ids], dtype=float) / max(len(sequence), 1)
    mixed_raw = raw_matrix @ actual_ratio
    mixed_secure = corrected_ber(mixed_raw)
    worst_idx = int(np.argmin(mixed_secure))
    append_csv_rows(file_path, schedule_fieldnames(), [{
        "position_combination": str(tuple(legal_positions)),
        "candidate_count": candidate_count,
        "selected_pool_size": len(selected),
        "schedule_length": int(schedule_length),
        "security_eval_bits": int(security_eval_bits),
        "optimizer": optimizer,
        "selected_candidate_ids": format_int_list(selected_ids),
        "usage_ratio": format_float_list(actual_ratio),
        "usage_counts": format_int_list([int(value) for value in counts]),
        "offline_objective_min_secure_ber": f"{float(np.min(mixed_secure)):.8f}",
        "offline_average_secure_ber": f"{float(np.mean(mixed_secure)):.8f}",
        "worst_route": f"{route_keys[worst_idx][0]}->{route_keys[worst_idx][1]}",
        "schedule_sequence": format_int_list(sequence),
        "route_raw_bers": route_dict_text(route_keys, mixed_raw),
        "route_min_bers": route_dict_text(route_keys, mixed_secure),
        "route_secure_bers": route_dict_text(route_keys, mixed_secure),
    }])
    print(
        f"      Schedule: objective={objective:.6f}, actual_min={float(np.min(mixed_secure)):.6f}, "
        f"actual_avg={float(np.mean(mixed_secure)):.6f}"
    )


def run(args: argparse.Namespace) -> tuple[str, str, str]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "test-4")
    candidate_file = args.candidate_file or os.path.join(output_dir, DEFAULT_CANDIDATE_FILENAME)
    selected_file = args.selected_file or os.path.join(output_dir, DEFAULT_SELECTED_FILENAME)
    schedule_file = args.schedule_file or os.path.join(output_dir, DEFAULT_SCHEDULE_FILENAME)
    if args.overwrite:
        reset_output_files([candidate_file, selected_file, schedule_file])

    base.LEGAL_SEARCH_BITS = int(args.legal_search_bits)
    selection_rng = random.Random(SEED)
    combinations = base.select_position_combinations(project_root, args.combination_count, selection_rng)
    print(f"Selected {len(combinations)} random 4-position combinations from {LIGHT_CONDITION}.")

    for combo_idx, legal_positions in enumerate(combinations, start=1):
        rng = random.Random(SEED + combo_idx * 100003 + sum(int(p) * 1009 for p in legal_positions))
        print(f"[{combo_idx}/{len(combinations)}] Build candidate pool for {legal_positions}")
        candidates = collect_candidates(
            project_root=project_root,
            legal_positions=legal_positions,
            probe_count=args.probe_count,
            target_size=args.candidate_target_size,
            security_eval_bits=args.security_eval_bits,
            max_attempts=args.max_attempts,
            min_probe_interval=args.min_probe_interval,
            rng=rng,
        )
        if not candidates:
            print(f"      No legal-reliable candidates found for {legal_positions}.")
            continue
        write_candidates(candidate_file, legal_positions, candidates)

        selected_size = min(args.selected_pool_size, len(candidates))
        print(f"      Select {selected_size} security-complementary candidates")
        selected = select_security_complementary_pool(candidates, selected_size)
        write_selected(selected_file, legal_positions, selected)
        write_schedule(
            schedule_file,
            legal_positions,
            candidate_count=len(candidates),
            selected=selected,
            schedule_length=args.schedule_length,
            security_eval_bits=args.security_eval_bits,
        )

    print(f"Candidate pool saved to: {candidate_file}")
    print(f"Selected pool saved to: {selected_file}")
    print(f"Offline schedule saved to: {schedule_file}")
    return candidate_file, selected_file, schedule_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build security-aware 4-position candidate pools and schedules.")
    parser.add_argument("--combination-count", type=int, default=DEFAULT_COMBINATION_COUNT)
    parser.add_argument("--probe-count", type=int, default=DEFAULT_PROBE_COUNT)
    parser.add_argument("--candidate-target-size", type=int, default=DEFAULT_CANDIDATE_TARGET_SIZE)
    parser.add_argument("--selected-pool-size", type=int, default=DEFAULT_SELECTED_POOL_SIZE)
    parser.add_argument("--schedule-length", type=int, default=DEFAULT_SCHEDULE_LENGTH)
    parser.add_argument("--security-eval-bits", type=int, default=DEFAULT_SECURITY_EVAL_BITS)
    parser.add_argument("--legal-search-bits", type=int, default=1000)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--min-probe-interval", type=int, default=DEFAULT_MIN_PROBE_INTERVAL)
    parser.add_argument("--candidate-file", default=None)
    parser.add_argument("--selected-file", default=None)
    parser.add_argument("--schedule-file", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
