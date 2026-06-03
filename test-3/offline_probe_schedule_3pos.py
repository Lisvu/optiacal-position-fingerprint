#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build offline probe-set schedules from an existing 3-position probe pool."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np


LIGHT_CONDITION = "yellow"
DEFAULT_POOL_FILENAME = "yellow_random_probe_hopping_probe_pool_3pos.csv"
DEFAULT_OUTPUT_FILENAME = "yellow_offline_probe_schedule_3pos.csv"
DEFAULT_SCHEDULE_LENGTH = 1000
DEFAULT_EVAL_BITS = 1000
SEED = 20260510


def load_base_module():
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_probe_hopping_yellow_3pos.py")
    spec = importlib.util.spec_from_file_location("random_probe_hopping_yellow_3pos_offline_schedule", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()


@dataclass
class PoolEntry:
    position_combination: tuple[int, ...]
    pool_entry_id: int
    probes: np.ndarray
    hue_mapping: dict[tuple[int, ...], int]
    legal_models: list


def parse_probes(text: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(text), dtype=float)


def parse_hue_mapping(text: str) -> dict[tuple[int, ...], int]:
    value = ast.literal_eval(text)
    return {tuple(int(v) for v in key): int(hue) for key, hue in value.items()}


def load_pool_entries(project_root: str, pool_file: str) -> dict[tuple[int, ...], list[PoolEntry]]:
    grouped: dict[tuple[int, ...], list[PoolEntry]] = {}
    with open(pool_file, "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            legal_positions = tuple(int(v) for v in ast.literal_eval(row["position_combination"]))
            probes = parse_probes(row["probes"])
            hue_mapping = parse_hue_mapping(row["hue_mapping"])
            csv_files = base.build_csv_files(project_root, legal_positions)
            legal_models, _ = base.test.build_models_from_probes(csv_files, probes)
            grouped.setdefault(legal_positions, []).append(PoolEntry(
                position_combination=legal_positions,
                pool_entry_id=int(row["pool_entry_id"]),
                probes=probes,
                hue_mapping=hue_mapping,
                legal_models=legal_models,
            ))
    for entries in grouped.values():
        entries.sort(key=lambda entry: entry.pool_entry_id)
    return grouped


def corrected_ber(raw_ber: np.ndarray) -> np.ndarray:
    return np.minimum(raw_ber, 1.0 - raw_ber)


def evaluate_entry_route_raw_bers(
    project_root: str,
    legal_positions: Sequence[int],
    entry: PoolEntry,
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


def optimize_usage_ratio(raw_matrix: np.ndarray) -> tuple[np.ndarray, float, str]:
    route_count, entry_count = raw_matrix.shape
    try:
        from scipy.optimize import linprog

        c = np.zeros(entry_count + 1, dtype=float)
        c[-1] = -1.0
        rows = []
        bounds = []
        for route_idx in range(route_count):
            row = np.zeros(entry_count + 1, dtype=float)
            row[:entry_count] = -raw_matrix[route_idx]
            row[-1] = 1.0
            rows.append(row)
            bounds.append(0.0)

            row = np.zeros(entry_count + 1, dtype=float)
            row[:entry_count] = raw_matrix[route_idx]
            row[-1] = 1.0
            rows.append(row)
            bounds.append(1.0)

        result = linprog(
            c,
            A_ub=np.asarray(rows, dtype=float),
            b_ub=np.asarray(bounds, dtype=float),
            A_eq=np.asarray([[*([1.0] * entry_count), 0.0]], dtype=float),
            b_eq=np.asarray([1.0], dtype=float),
            bounds=[(0.0, 1.0)] * entry_count + [(0.0, 0.5)],
            method="highs",
        )
        if result.success:
            p = np.maximum(result.x[:entry_count], 0.0)
            p = p / np.sum(p)
            mixed_raw = raw_matrix @ p
            return p, float(np.min(corrected_ber(mixed_raw))), "linprog"
    except Exception:
        pass

    rng = np.random.default_rng(SEED)
    candidates = [np.full(entry_count, 1.0 / entry_count, dtype=float)]
    candidates.extend(np.eye(entry_count, dtype=float))
    candidates.extend(rng.dirichlet(np.ones(entry_count), size=20000))
    best_p = candidates[0]
    best_score = -1.0
    for p in candidates:
        score = float(np.min(corrected_ber(raw_matrix @ p)))
        if score > best_score:
            best_p = np.asarray(p, dtype=float)
            best_score = score
    return best_p, best_score, "random_dirichlet_fallback"


def counts_from_ratio(ratio: np.ndarray, schedule_length: int) -> np.ndarray:
    raw_counts = np.asarray(ratio, dtype=float) * int(schedule_length)
    counts = np.floor(raw_counts).astype(int)
    remainder = int(schedule_length) - int(np.sum(counts))
    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts


def build_risk_balanced_sequence(raw_matrix: np.ndarray, counts: np.ndarray, entry_ids: Sequence[int]) -> list[int]:
    remaining = np.asarray(counts, dtype=int).copy()
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
        sequence.append(int(entry_ids[best_idx]))
        cumulative_raw += raw_matrix[:, best_idx]
        remaining[best_idx] -= 1
        last_idx = best_idx
    return sequence


def list_text(values: Sequence[object]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def float_list_text(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.8f}" for v in values) + "]"


def run(pool_file: str, output_file: str, schedule_length: int, eval_bits: int, limit_combinations: int | None) -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rng = random.Random(SEED)
    grouped = load_pool_entries(project_root, pool_file)
    combinations = sorted(grouped)
    if limit_combinations is not None:
        combinations = combinations[:limit_combinations]

    fieldnames = [
        "position_combination",
        "schedule_length",
        "eval_bits",
        "optimizer",
        "probe_pool_size",
        "pool_entry_ids",
        "usage_ratio",
        "usage_counts",
        "offline_objective_min_secure_ber",
        "offline_average_secure_ber",
        "worst_route",
        "schedule_sequence",
        "route_secure_bers",
    ]
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for combo_idx, legal_positions in enumerate(combinations, start=1):
            entries = grouped[legal_positions]
            bit_blocks_pm = base.test.generate_random_bit_blocks(eval_bits, len(legal_positions), rng=rng)
            route_keys = None
            columns = []
            for entry in entries:
                route_bers = evaluate_entry_route_raw_bers(project_root, legal_positions, entry, bit_blocks_pm)
                if route_keys is None:
                    route_keys = sorted(route_bers)
                columns.append([route_bers[route] for route in route_keys])

            raw_matrix = np.asarray(columns, dtype=float).T
            ratio, objective, optimizer = optimize_usage_ratio(raw_matrix)
            counts = counts_from_ratio(ratio, schedule_length)
            entry_ids = [entry.pool_entry_id for entry in entries]
            sequence = build_risk_balanced_sequence(raw_matrix, counts, entry_ids)
            actual_ratio = np.asarray([sequence.count(entry_id) for entry_id in entry_ids], dtype=float) / max(len(sequence), 1)
            mixed_secure = corrected_ber(raw_matrix @ actual_ratio)
            worst_idx = int(np.argmin(mixed_secure))
            route_secure = {
                f"{route[0]}->{route[1]}": f"{mixed_secure[idx]:.8f}"
                for idx, route in enumerate(route_keys or [])
            }
            writer.writerow({
                "position_combination": str(tuple(legal_positions)),
                "schedule_length": int(schedule_length),
                "eval_bits": int(eval_bits),
                "optimizer": optimizer,
                "probe_pool_size": len(entries),
                "pool_entry_ids": list_text(entry_ids),
                "usage_ratio": float_list_text(actual_ratio),
                "usage_counts": list_text([int(v) for v in counts]),
                "offline_objective_min_secure_ber": f"{float(np.min(mixed_secure)):.8f}",
                "offline_average_secure_ber": f"{float(np.mean(mixed_secure)):.8f}",
                "worst_route": f"{route_keys[worst_idx][0]}->{route_keys[worst_idx][1]}",
                "schedule_sequence": list_text(sequence),
                "route_secure_bers": str(route_secure),
            })
            print(
                f"[{combo_idx}/{len(combinations)}] {legal_positions}: "
                f"min_secure={float(np.min(mixed_secure)):.6f}, avg_secure={float(np.mean(mixed_secure)):.6f}, "
                f"counts={counts.tolist()}"
            )
    print(f"Offline schedule saved to: {output_file}")
    return output_file


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Build offline risk-balanced probe schedules for 3-position pools.")
    parser.add_argument("--pool-file", default=os.path.join(project_root, "test-3", DEFAULT_POOL_FILENAME))
    parser.add_argument("--output-file", default=os.path.join(project_root, "test-3", DEFAULT_OUTPUT_FILENAME))
    parser.add_argument("--schedule-length", type=int, default=DEFAULT_SCHEDULE_LENGTH)
    parser.add_argument("--eval-bits", type=int, default=DEFAULT_EVAL_BITS)
    parser.add_argument("--limit-combinations", type=int, default=None)
    args = parser.parse_args()
    run(args.pool_file, args.output_file, args.schedule_length, args.eval_bits, args.limit_combinations)


if __name__ == "__main__":
    main()
