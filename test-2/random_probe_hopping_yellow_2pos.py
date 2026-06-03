#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random probe hopping experiment for 2 legal positions on data/15pro/yellow."""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import types
from typing import Sequence

import numpy as np
import pandas as pd


LIGHT_CONDITION = "high"
LEGAL_POSITION_COUNT = 2
DEFAULT_COMBINATION_COUNT = 20
TARGET_POOL_SIZE = 5
OUTPUT_PROBE_POOL_FILENAME = "high_random_probe_hopping_probe_pool_2pos.csv"
OUTPUT_SECURITY_FILENAME = "high_random_probe_hopping_security_eval_2pos.csv"


def load_base_module():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(project_root, "test-3", "random_probe_hopping_yellow_3pos.py")
    spec = importlib.util.spec_from_file_location("random_probe_hopping_base_2pos", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base experiment module: {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_test2_module():
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-2-simple.py")
    spec = importlib.util.spec_from_file_location("test_2_simple_runtime_random_hopping_yellow_2pos", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test-2-simple.py: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()
test2 = load_test2_module()
base.LIGHT_CONDITION = LIGHT_CONDITION
base.LEGAL_POSITION_COUNT = LEGAL_POSITION_COUNT
base.DEFAULT_COMBINATION_COUNT = DEFAULT_COMBINATION_COUNT
base.OUTPUT_PROBE_POOL_FILENAME = OUTPUT_PROBE_POOL_FILENAME
base.OUTPUT_SECURITY_FILENAME = OUTPUT_SECURITY_FILENAME
base.TARGET_POOL_SIZE = TARGET_POOL_SIZE


def build_csv_files_for_positions(project_root: str, positions: Sequence[int], light_condition: str = LIGHT_CONDITION) -> list[str]:
    return [os.path.join(project_root, "data", "15pro", light_condition, f"{int(position)}.csv") for position in positions]


def load_selected_rows(csv_files: Sequence[str], probes: Sequence[float]) -> list[np.ndarray]:
    row_indices = [int(float(probe) / 5) - 1 for probe in probes]
    matrices = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        matrices.append(df.iloc[row_indices].values.astype(float))
    return matrices


def build_hue_mapping(models: Sequence[object], probes: Sequence[float]) -> dict[int, int]:
    z1 = np.asarray(models[0].z, dtype=float)
    z2 = np.asarray(models[1].z, dtype=float)
    features = []
    for idx, (v1, v2) in enumerate(zip(z1, z2)):
        abs_sum = abs(float(v1)) + abs(float(v2))
        features.append((idx, v1 < 0 and v2 < 0, v1 > 0 and v2 > 0, abs_sum))

    negative = [item for item in features if item[1]]
    positive = [item for item in features if item[2]]
    mixed = [item for item in features if not item[1] and not item[2]]
    neg_idx = max(negative, key=lambda item: item[3])[0] if negative else 0
    pos_idx = max(positive, key=lambda item: item[3])[0] if positive else len(probes) - 1
    zero_idx = min(mixed, key=lambda item: item[3])[0] if mixed else len(probes) // 2
    return {-2: int(probes[neg_idx]), 0: int(probes[zero_idx]), 2: int(probes[pos_idx])}


def build_models_from_probes(
    csv_files: Sequence[str],
    probes: Sequence[float],
    mapping_eval_bits: int = 500,
    mapping_top_k: int = 3,
    rng: random.Random | None = None,
):
    del mapping_eval_bits, mapping_top_k, rng
    probes_array = np.asarray(probes, dtype=float)
    matrices = load_selected_rows(csv_files, probes_array)
    models = [test2.extract_fingerprint(probes_array, matrix, force_positive_first=True) for matrix in matrices]
    return models, build_hue_mapping(models, probes_array)


def generate_random_bit_blocks(num_blocks: int, num_positions: int, rng: random.Random | None = None) -> list[np.ndarray]:
    if rng is None:
        rng = random.Random()
    return [np.asarray([rng.choice([-1, 1]) for _ in range(num_positions)], dtype=int) for _ in range(num_blocks)]


def is_valid_probe_set(probes: Sequence[float], min_interval: int) -> bool:
    sorted_probes = np.sort(np.asarray(probes, dtype=float))
    return len(sorted_probes) <= 1 or np.min(np.diff(sorted_probes)) >= min_interval


def build_symbol_sequence(bits_pm: Sequence[int], codes: Sequence[np.ndarray]):
    symbol_seq1, symbol_seq2 = test2.build_symbol_sequence(np.asarray(bits_pm, dtype=int), list(codes), list(codes))
    return symbol_seq1, (symbol_seq1, symbol_seq2)


def map_symbol_to_hue(symbol_combinations, hue_mapping: dict[int, int]) -> np.ndarray:
    symbol_seq1, symbol_seq2 = symbol_combinations
    return test2.map_symbol_to_hue(symbol_seq1, symbol_seq2, hue_mapping)


def decode_local_block(y_obs: np.ndarray, w: np.ndarray, code: np.ndarray):
    y_centered = np.asarray(y_obs, dtype=float) - np.asarray(y_obs, dtype=float).mean(axis=0)
    gamma = float(np.asarray(code, dtype=float) @ (y_centered @ np.asarray(w, dtype=float)))
    bit_hat_pm = 1 if gamma > 0 else -1
    return types.SimpleNamespace(bit_hat_pm=bit_hat_pm, bit_hat_bin=1 if bit_hat_pm > 0 else 0)


adapter = types.SimpleNamespace(
    build_csv_files_for_positions=build_csv_files_for_positions,
    load_selected_rows=load_selected_rows,
    extract_fingerprint=test2.extract_fingerprint,
    build_probe_to_row=test2.build_probe_to_row,
    pm1_to_bin=test2.pm1_to_bin,
    observe_block_from_measured_matrix=test2.observe_block_from_measured_matrix,
    simulate_blocks=test2.simulate_blocks,
    build_models_from_probes=build_models_from_probes,
    generate_random_bit_blocks=generate_random_bit_blocks,
    is_valid_probe_set=is_valid_probe_set,
    build_symbol_sequence=build_symbol_sequence,
    map_symbol_to_hue=map_symbol_to_hue,
    decode_local_block=decode_local_block,
)
base.test = adapter


def security_fieldnames() -> list[str]:
    fields = [
        "position_combination",
        "illegal_position",
        "probe_pool_size",
        "probe_count",
        "hopping_bits",
        "legal_position_bers",
    ]
    for idx in range(1, LEGAL_POSITION_COUNT + 1):
        fields.extend([
            f"legal_position_{idx}",
            f"ber_vs_legal_pos_{idx}",
            f"secure_ber_vs_legal_pos_{idx}",
        ])
    fields.extend(["min_secure_ber", "average_secure_ber"])
    return fields


def run_experiment(combination_count: int) -> tuple[str, str]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "test-2")
    probe_pool_file = os.path.join(output_dir, OUTPUT_PROBE_POOL_FILENAME)
    security_file = os.path.join(output_dir, OUTPUT_SECURITY_FILENAME)
    rng = random.Random(base.SELECTION_SEED)
    combinations = base.select_position_combinations(project_root, combination_count, rng)
    completed = base.load_completed_combinations(security_file)
    probe_pool_fields = [
        "position_combination",
        "pool_entry_id",
        "probe_pool_size",
        "probe_count",
        "probes",
        "hue_mapping",
        "legal_position_bers",
    ]

    print(f"Selected {len(combinations)} random 2-position combinations from {LIGHT_CONDITION}.")
    for idx, legal_positions in enumerate(combinations, start=1):
        combination_key = str(tuple(legal_positions))
        if combination_key in completed:
            print(f"[{idx}/{len(combinations)}] Skip {legal_positions}: already evaluated.")
            continue

        print(f"[{idx}/{len(combinations)}] Searching legal probe pool for {legal_positions}")
        probe_count, probe_pool = base.search_probe_pool(project_root, legal_positions, rng, TARGET_POOL_SIZE)
        if not probe_pool:
            print(f"  No complete legal BER=0 probe pool found for {legal_positions}.")
            continue

        pool_rows = [{
            "position_combination": combination_key,
            "pool_entry_id": entry.entry_id,
            "probe_pool_size": len(probe_pool),
            "probe_count": probe_count,
            "probes": base.format_probes(entry.probes),
            "hue_mapping": base.format_hue_mapping(entry.hue_mapping),
            "legal_position_bers": "[" + ", ".join(f"{v:.6f}" for v in entry.legal_position_bers) + "]",
        } for entry in probe_pool]
        base.append_csv_rows(probe_pool_file, probe_pool_fields, pool_rows)
        print(f"  Saved {len(pool_rows)} legal probe sets to {probe_pool_file}")

        print(f"  Evaluating random probe hopping security for {legal_positions}")
        security_rows, summary = base.evaluate_hopping_security(project_root, legal_positions, probe_pool, rng)
        base.append_csv_rows(security_file, security_fieldnames(), security_rows)
        completed.add(combination_key)
        print(
            "  Hopping legal_bers={legal}, min_illegal_secure_ber={min_ber:.6f}, "
            "worst_illegal={illegal}, worst_legal={legal_pos}".format(
                legal=[round(float(v), 6) for v in summary["legal_position_bers"]],
                min_ber=float(summary["min_illegal_secure_ber"]),
                illegal=summary["worst_illegal_position"],
                legal_pos=summary["worst_legal_position"],
            )
        )

    print(f"Probe pool results saved to: {probe_pool_file}")
    print(f"Security results saved to: {security_file}")
    return probe_pool_file, security_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2-position yellow random probe hopping experiment.")
    parser.add_argument(
        "--combination-count",
        type=int,
        default=DEFAULT_COMBINATION_COUNT,
        help="Number of random 2-position combinations to evaluate.",
    )
    args = parser.parse_args()
    run_experiment(combination_count=args.combination_count)


if __name__ == "__main__":
    main()
