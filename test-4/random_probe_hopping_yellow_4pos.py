#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random probe hopping experiment for 4 legal positions on data/15pro/mid."""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import types
from typing import Sequence


LIGHT_CONDITION = "mid"
LEGAL_POSITION_COUNT = 4
DEFAULT_COMBINATION_COUNT = 20
TARGET_POOL_SIZE = 5
OUTPUT_PROBE_POOL_FILENAME = "mid_random_probe_hopping_probe_pool_4pos.csv"
OUTPUT_SECURITY_FILENAME = "mid_random_probe_hopping_security_eval_4pos.csv"


def load_base_module():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(project_root, "test-3", "random_probe_hopping_yellow_3pos.py")
    spec = importlib.util.spec_from_file_location("random_probe_hopping_base_4pos", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base experiment module: {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()
base.LIGHT_CONDITION = LIGHT_CONDITION
base.LEGAL_POSITION_COUNT = LEGAL_POSITION_COUNT
base.DEFAULT_COMBINATION_COUNT = DEFAULT_COMBINATION_COUNT
base.OUTPUT_PROBE_POOL_FILENAME = OUTPUT_PROBE_POOL_FILENAME
base.OUTPUT_SECURITY_FILENAME = OUTPUT_SECURITY_FILENAME
base.TARGET_POOL_SIZE = TARGET_POOL_SIZE
base.test = base.load_test_module = lambda: None


def load_test4_module():
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_4_simple.py")
    spec = importlib.util.spec_from_file_location("test_4_simple_runtime_random_hopping_yellow_4pos", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test_4_simple.py: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


test4 = load_test4_module()


def promote_effective_fingerprint(model):
    if getattr(model, "eff_code", None) is not None:
        model.code = model.eff_code
    if getattr(model, "eff_w", None) is not None:
        model.w = model.eff_w
    return model


def extract_fingerprint(*args, **kwargs):
    return promote_effective_fingerprint(test4.extract_fingerprint(*args, **kwargs))


def build_models_from_probes(*args, **kwargs):
    models, hue_mapping = test4.build_models_from_probes(*args, **kwargs)
    return [promote_effective_fingerprint(model) for model in models], hue_mapping


base.test = types.SimpleNamespace(
    build_csv_files_for_positions=test4.build_csv_files_for_positions,
    load_selected_rows=test4.load_selected_rows,
    extract_fingerprint=extract_fingerprint,
    build_probe_to_row=test4.build_probe_to_row,
    pm1_to_bin=test4.pm1_to_bin,
    observe_block_from_measured_matrix=test4.observe_block_from_measured_matrix,
    simulate_blocks=test4.simulate_blocks,
    build_models_from_probes=build_models_from_probes,
    generate_random_bit_blocks=test4.generate_random_bit_blocks,
    is_valid_probe_set=test4.is_valid_probe_set,
    build_symbol_sequence=test4.build_symbol_sequence,
    map_symbol_to_hue=test4.map_symbol_to_hue,
    decode_local_block=test4.decode_local_block,
)


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


def worker_filename(filename: str, worker_count: int, worker_index: int) -> str:
    if worker_count <= 1:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}_worker{worker_index}_of_{worker_count}{ext}"


def load_completed_combinations_from_files(file_paths: Sequence[str]) -> set[str]:
    completed: set[str] = set()
    for file_path in file_paths:
        completed.update(base.load_completed_combinations(file_path))
    return completed


def run_experiment(combination_count: int, worker_count: int = 1, worker_index: int = 0) -> tuple[str, str]:
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError("worker_index must satisfy 0 <= worker_index < worker_count")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "test-4")
    global_probe_pool_file = os.path.join(output_dir, OUTPUT_PROBE_POOL_FILENAME)
    global_security_file = os.path.join(output_dir, OUTPUT_SECURITY_FILENAME)
    probe_pool_file = os.path.join(output_dir, worker_filename(OUTPUT_PROBE_POOL_FILENAME, worker_count, worker_index))
    security_file = os.path.join(output_dir, worker_filename(OUTPUT_SECURITY_FILENAME, worker_count, worker_index))
    selection_rng = random.Random(base.SELECTION_SEED)
    rng = random.Random(base.SELECTION_SEED + worker_index * 1000003)
    all_combinations = base.select_position_combinations(project_root, combination_count, selection_rng)
    combinations = [
        combination
        for combination_idx, combination in enumerate(all_combinations)
        if combination_idx % worker_count == worker_index
    ]
    completed = load_completed_combinations_from_files([
        global_probe_pool_file,
        global_security_file,
        probe_pool_file,
        security_file,
    ])

    probe_pool_fields = [
        "position_combination",
        "pool_entry_id",
        "probe_pool_size",
        "probe_count",
        "probes",
        "hue_mapping",
        "legal_position_bers",
    ]

    print(
        f"Selected {len(all_combinations)} random 4-position combinations from {LIGHT_CONDITION}; "
        f"worker {worker_index}/{worker_count} will evaluate {len(combinations)}."
    )
    print(f"Loaded {len(completed)} completed combinations from existing global/worker result files.")
    for idx, legal_positions in enumerate(combinations, start=1):
        combination_key = str(tuple(legal_positions))
        if combination_key in completed:
            print(f"[{idx}/{len(combinations)}] Skip {legal_positions}: already evaluated.")
            continue

        print(f"[{idx}/{len(combinations)}] Searching legal probe pool for {legal_positions}")
        probe_count, probe_pool = base.search_probe_pool(
            project_root=project_root,
            legal_positions=legal_positions,
            rng=rng,
            target_pool_size=TARGET_POOL_SIZE,
        )
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
        security_rows, summary = base.evaluate_hopping_security(
            project_root=project_root,
            legal_positions=legal_positions,
            probe_pool=probe_pool,
            rng=rng,
        )
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
    parser = argparse.ArgumentParser(description="Run 4-position mid random probe hopping experiment.")
    parser.add_argument(
        "--combination-count",
        type=int,
        default=DEFAULT_COMBINATION_COUNT,
        help="Number of random 4-position combinations to evaluate.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="Total number of parallel workers. Use 3 when running three processes.",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="Zero-based worker index, from 0 to worker-count - 1.",
    )
    args = parser.parse_args()
    run_experiment(
        combination_count=args.combination_count,
        worker_count=args.worker_count,
        worker_index=args.worker_index,
    )


if __name__ == "__main__":
    main()
