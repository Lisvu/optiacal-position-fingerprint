#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate offline probe-set schedules against random probe hopping."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import os
import random
import sys
from typing import Sequence

import numpy as np


DEFAULT_POOL_FILENAME = "yellow_random_probe_hopping_probe_pool_3pos.csv"
DEFAULT_SCHEDULE_FILENAME = "yellow_offline_probe_schedule_3pos.csv"
DEFAULT_OUTPUT_FILENAME = "yellow_offline_probe_schedule_security_eval_3pos.csv"
DEFAULT_EVAL_BITS = 1000
SEED = 20260511


def load_schedule_module():
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline_probe_schedule_3pos.py")
    spec = importlib.util.spec_from_file_location("offline_probe_schedule_3pos_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load schedule module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


schedule_mod = load_schedule_module()
base = schedule_mod.base


def corrected_ber(value: float) -> float:
    return min(float(value), 1.0 - float(value))


def load_schedules(schedule_file: str) -> dict[tuple[int, ...], list[int]]:
    schedules: dict[tuple[int, ...], list[int]] = {}
    with open(schedule_file, "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            legal_positions = tuple(int(v) for v in ast.literal_eval(row["position_combination"]))
            sequence = [int(v) for v in ast.literal_eval(row["schedule_sequence"])]
            if not sequence:
                raise ValueError(f"Empty schedule_sequence for {legal_positions}")
            schedules[legal_positions] = sequence
    return schedules


def calculate_legal_position_bers(bit_blocks_pm: Sequence[np.ndarray], selected_entries: Sequence[object]) -> list[float]:
    errors = np.zeros(len(bit_blocks_pm[0]), dtype=float)
    totals = np.zeros(len(bit_blocks_pm[0]), dtype=float)
    for bits_pm, entry in zip(bit_blocks_pm, selected_entries):
        results = base.test.simulate_blocks(entry.legal_models, [bits_pm], entry.hue_mapping)
        true_bits = results[0]["bits_bin"]
        decoded_bits = [decode.bit_hat_bin for decode in results[0]["per_position"]]
        for idx, (true_bit, decoded_bit) in enumerate(zip(true_bits, decoded_bits)):
            totals[idx] += 1
            if int(true_bit) != int(decoded_bit):
                errors[idx] += 1
    return (errors / np.maximum(totals, 1.0)).tolist()


def evaluate_selection(
    project_root: str,
    legal_positions: Sequence[int],
    selected_entries: Sequence[object],
    bit_blocks_pm: Sequence[np.ndarray],
) -> tuple[list[dict], list[float]]:
    legal_position_bers = calculate_legal_position_bers(bit_blocks_pm, selected_entries)
    illegal_positions = [p for p in base.get_available_positions(project_root) if p not in legal_positions]
    illegal_cache: dict[tuple[int, int], tuple[object, np.ndarray, dict[int, int]]] = {}
    rows = []

    for illegal_position in illegal_positions:
        errors = np.zeros(len(legal_positions), dtype=float)
        totals = np.zeros(len(legal_positions), dtype=float)
        for bits_pm, entry in zip(bit_blocks_pm, selected_entries):
            cache_key = (int(illegal_position), int(entry.pool_entry_id))
            if cache_key not in illegal_cache:
                illegal_csv = base.build_csv_files(project_root, [illegal_position])[0]
                illegal_matrix = base.test.load_selected_rows([illegal_csv], entry.probes)[0]
                illegal_model = base.test.extract_fingerprint(entry.probes, illegal_matrix, force_positive_first=True)
                probe_to_row = base.test.build_probe_to_row(entry.probes)
                illegal_cache[cache_key] = (illegal_model, illegal_matrix, probe_to_row)
            illegal_model, illegal_matrix, probe_to_row = illegal_cache[cache_key]

            legal_codes = [model.code for model in entry.legal_models]
            _, symbol_combinations = base.test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = base.test.map_symbol_to_hue(symbol_combinations, entry.hue_mapping)
            illegal_observation = base.test.observe_block_from_measured_matrix(hue_seq, illegal_matrix, probe_to_row)
            illegal_dec = base.test.decode_local_block(illegal_observation, illegal_model.w, illegal_model.code)
            true_bits = base.test.pm1_to_bin(bits_pm)
            for idx, true_bit in enumerate(true_bits):
                totals[idx] += 1
                if int(illegal_dec.bit_hat_bin) != int(true_bit):
                    errors[idx] += 1

        raw_bers = (errors / np.maximum(totals, 1.0)).tolist()
        secure_bers = [corrected_ber(ber) for ber in raw_bers]
        rows.append({
            "illegal_position": int(illegal_position),
            "raw_bers": raw_bers,
            "secure_bers": secure_bers,
            "min_secure_ber": float(min(secure_bers)),
            "average_secure_ber": float(np.mean(secure_bers)),
        })
    return rows, legal_position_bers


def select_entries_by_schedule(sequence: Sequence[int], entries_by_id: dict[int, object], bit_count: int) -> list[object]:
    return [entries_by_id[int(sequence[idx % len(sequence)])] for idx in range(bit_count)]


def select_entries_random(entries: Sequence[object], bit_count: int, rng: random.Random) -> list[object]:
    return [rng.choice(list(entries)) for _ in range(bit_count)]


def format_float_list(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def run(pool_file: str, schedule_file: str, output_file: str, eval_bits: int, limit_combinations: int | None) -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grouped = schedule_mod.load_pool_entries(project_root, pool_file)
    schedules = load_schedules(schedule_file)
    combinations = sorted(set(grouped) & set(schedules))
    if limit_combinations is not None:
        combinations = combinations[:limit_combinations]

    fieldnames = [
        "position_combination",
        "illegal_position",
        "eval_bits",
        "schedule_length",
        "scheduled_legal_position_bers",
        "random_legal_position_bers",
    ]
    for idx in range(1, 4):
        fieldnames.extend([
            f"legal_position_{idx}",
            f"scheduled_ber_vs_legal_pos_{idx}",
            f"scheduled_secure_ber_vs_legal_pos_{idx}",
            f"random_ber_vs_legal_pos_{idx}",
            f"random_secure_ber_vs_legal_pos_{idx}",
        ])
    fieldnames.extend([
        "scheduled_min_secure_ber",
        "scheduled_average_secure_ber",
        "random_min_secure_ber",
        "random_average_secure_ber",
        "min_secure_ber_improvement",
        "average_secure_ber_improvement",
    ])

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for combo_idx, legal_positions in enumerate(combinations, start=1):
            rng = random.Random(SEED + sum(int(p) * 1009 for p in legal_positions))
            entries = grouped[legal_positions]
            entries_by_id = {entry.pool_entry_id: entry for entry in entries}
            sequence = schedules[legal_positions]
            missing_ids = sorted(set(sequence) - set(entries_by_id))
            if missing_ids:
                raise ValueError(f"Schedule for {legal_positions} references missing pool_entry_id values: {missing_ids}")

            bit_blocks_pm = base.test.generate_random_bit_blocks(eval_bits, len(legal_positions), rng=rng)
            scheduled_entries = select_entries_by_schedule(sequence, entries_by_id, len(bit_blocks_pm))
            random_entries = select_entries_random(entries, len(bit_blocks_pm), rng)
            scheduled_rows, scheduled_legal_bers = evaluate_selection(project_root, legal_positions, scheduled_entries, bit_blocks_pm)
            random_rows, random_legal_bers = evaluate_selection(project_root, legal_positions, random_entries, bit_blocks_pm)
            random_by_illegal = {row["illegal_position"]: row for row in random_rows}

            for scheduled_row in scheduled_rows:
                illegal_position = scheduled_row["illegal_position"]
                random_row = random_by_illegal[illegal_position]
                row = {
                    "position_combination": str(tuple(legal_positions)),
                    "illegal_position": int(illegal_position),
                    "eval_bits": int(eval_bits),
                    "schedule_length": len(sequence),
                    "scheduled_legal_position_bers": format_float_list(scheduled_legal_bers),
                    "random_legal_position_bers": format_float_list(random_legal_bers),
                    "scheduled_min_secure_ber": f"{scheduled_row['min_secure_ber']:.6f}",
                    "scheduled_average_secure_ber": f"{scheduled_row['average_secure_ber']:.6f}",
                    "random_min_secure_ber": f"{random_row['min_secure_ber']:.6f}",
                    "random_average_secure_ber": f"{random_row['average_secure_ber']:.6f}",
                    "min_secure_ber_improvement": f"{scheduled_row['min_secure_ber'] - random_row['min_secure_ber']:.6f}",
                    "average_secure_ber_improvement": f"{scheduled_row['average_secure_ber'] - random_row['average_secure_ber']:.6f}",
                }
                for idx, legal_position in enumerate(legal_positions, start=1):
                    row[f"legal_position_{idx}"] = int(legal_position)
                    row[f"scheduled_ber_vs_legal_pos_{idx}"] = f"{scheduled_row['raw_bers'][idx - 1]:.6f}"
                    row[f"scheduled_secure_ber_vs_legal_pos_{idx}"] = f"{scheduled_row['secure_bers'][idx - 1]:.6f}"
                    row[f"random_ber_vs_legal_pos_{idx}"] = f"{random_row['raw_bers'][idx - 1]:.6f}"
                    row[f"random_secure_ber_vs_legal_pos_{idx}"] = f"{random_row['secure_bers'][idx - 1]:.6f}"
                writer.writerow(row)
            scheduled_min = min(row["min_secure_ber"] for row in scheduled_rows)
            random_min = min(row["min_secure_ber"] for row in random_rows)
            print(
                f"[{combo_idx}/{len(combinations)}] {legal_positions}: "
                f"scheduled_global_min={scheduled_min:.6f}, random_global_min={random_min:.6f}, "
                f"delta={scheduled_min - random_min:.6f}"
            )
    print(f"Schedule security evaluation saved to: {output_file}")
    return output_file


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Test offline 3-position probe schedules against random hopping.")
    parser.add_argument("--pool-file", default=os.path.join(project_root, "test-3", DEFAULT_POOL_FILENAME))
    parser.add_argument("--schedule-file", default=os.path.join(project_root, "test-3", DEFAULT_SCHEDULE_FILENAME))
    parser.add_argument("--output-file", default=os.path.join(project_root, "test-3", DEFAULT_OUTPUT_FILENAME))
    parser.add_argument("--eval-bits", type=int, default=DEFAULT_EVAL_BITS)
    parser.add_argument("--limit-combinations", type=int, default=None)
    args = parser.parse_args()
    run(args.pool_file, args.schedule_file, args.output_file, args.eval_bits, args.limit_combinations)


if __name__ == "__main__":
    main()
