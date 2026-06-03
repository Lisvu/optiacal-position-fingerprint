#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual 6-position 10000-bit leakage experiment on mate40pro high_column_shuffled.

The script reuses the existing global-guarded candidate generation and local
decode logic, but fixes the legal position combination from command-line input.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Sequence

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
ORIGINAL_VECTOR_HUE_DIR = os.path.join(PROJECT_ROOT, "vector_hue")
if ORIGINAL_VECTOR_HUE_DIR not in sys.path:
    sys.path.insert(0, ORIGINAL_VECTOR_HUE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import global_guarded_core as guarded
import virtual_stream_core as core


DATASET_NAME = "mate40pro_high_column_shuffled"
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "mate40pro", "high_column_shuffled")
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "results",
    "manual_6pos_10000bits",
    DATASET_NAME,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 6 random 10000-bit streams for manually selected legal "
            "positions, then measure authorized BER and illegal leakage BER."
        )
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs=6,
        required=True,
        help="Six legal position ids, e.g. --positions 1 3 5 7 9 11",
    )
    parser.add_argument("--bits", type=int, default=10000, help="Bits per legal stream.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=20260603)
    parser.add_argument("--target-effective-k", type=int, default=8)
    parser.add_argument("--baseline-probe-subset-count", type=int, default=20)
    parser.add_argument("--targeted-probe-subset-count", type=int, default=50)
    parser.add_argument("--base-mapping-count", type=int, default=20)
    parser.add_argument("--baseline-mappings-per-subset", type=int, default=8)
    parser.add_argument("--targeted-mappings-per-subset", type=int, default=16)
    parser.add_argument("--selected-count", type=int, default=20)
    parser.add_argument("--weak-route-count", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def ensure_unique_positions(positions: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(v) for v in positions)
    if len(set(out)) != len(out):
        raise ValueError(f"Positions must be unique: {out}")
    return out


def write_rows(path: str, fields: Sequence[str], rows: Sequence[dict], overwrite: bool = True) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "w" if overwrite else "a"
    exists = os.path.exists(path)
    with open(path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        if overwrite or not exists:
            writer.writeheader()
        writer.writerows(rows)


def select_best_candidate(
    candidates: Sequence[core.Candidate],
    weak_route_count: int,
    selected_count: int,
) -> tuple[core.Candidate, list[core.Candidate], list[tuple[int, int]], core.AlignmentInfo]:
    aligned, info = core.align_candidates(candidates, [], [])
    if not aligned:
        raise RuntimeError("No aligned legal candidates were generated.")
    anchor = max(aligned, key=lambda c: (c.min_route_min_ber, c.average_route_min_ber))
    weak_routes = guarded.anchor_worst_routes(anchor, guarded.adaptive_weak_route_count(len(anchor.route_keys), weak_route_count))
    selected = core.select_candidates(aligned, selected_count=selected_count, weak_routes=weak_routes)
    best = max(selected, key=lambda c: (c.min_route_min_ber, c.average_route_min_ber, -c.authorized_max_ber))
    return best, selected, weak_routes, info


def route_rows(candidate: core.Candidate) -> list[dict]:
    rows = []
    for (illegal_position, legal_position), raw_ber, leakage_ber in zip(
        candidate.route_keys,
        candidate.route_raw_bers,
        candidate.route_min_bers,
    ):
        rows.append({
            "illegal_position": int(illegal_position),
            "legal_position": int(legal_position),
            "raw_ber": f"{float(raw_ber):.8f}",
            "leakage_ber": f"{float(leakage_ber):.8f}",
        })
    return rows


def per_legal_rows(candidate: core.Candidate, legal_positions: Sequence[int]) -> list[dict]:
    rows = []
    for stream_idx, legal_position in enumerate(legal_positions):
        values = [
            float(value)
            for route, value in zip(candidate.route_keys, candidate.route_min_bers)
            if int(route[1]) == int(legal_position)
        ]
        raw_values = [
            float(value)
            for route, value in zip(candidate.route_keys, candidate.route_raw_bers)
            if int(route[1]) == int(legal_position)
        ]
        rows.append({
            "legal_position": int(legal_position),
            "stream_index": stream_idx + 1,
            "authorized_ber": f"{float(candidate.authorized_position_bers[stream_idx]):.8f}",
            "illegal_device_count": len(values),
            "illegal_min_leakage_ber": f"{float(np.min(values)):.8f}" if values else "",
            "illegal_avg_leakage_ber": f"{float(np.mean(values)):.8f}" if values else "",
            "illegal_min_raw_ber": f"{float(np.min(raw_values)):.8f}" if raw_values else "",
            "illegal_avg_raw_ber": f"{float(np.mean(raw_values)):.8f}" if raw_values else "",
        })
    return rows


def bit_stream_rows(bit_blocks_pm: Sequence[np.ndarray], legal_positions: Sequence[int]) -> list[dict]:
    rows = []
    for bit_index, bits_pm in enumerate(bit_blocks_pm, start=1):
        bits_bin = core.base.vh.pm1_to_bin(np.asarray(bits_pm, dtype=int))[:len(legal_positions)]
        row = {"bit_index": bit_index}
        for stream_idx, (position, bit) in enumerate(zip(legal_positions, bits_bin), start=1):
            row[f"stream_{stream_idx}_position_{int(position)}"] = int(bit)
        rows.append(row)
    return rows


def run(argv: Sequence[str] | None = None) -> tuple[str, str, str, str]:
    args = parse_args(argv)
    legal_positions = ensure_unique_positions(args.positions)
    os.makedirs(args.output_dir, exist_ok=True)

    config = core.ExperimentConfig(
        real_k=6,
        output_dir=args.output_dir,
        target_effective_k=args.target_effective_k,
        sample_size=1,
        base_mapping_count=args.base_mapping_count,
        mappings_per_subset=args.targeted_mappings_per_subset,
        selected_count=args.selected_count,
        eval_bits=args.bits,
        weak_route_count=args.weak_route_count,
        dataset_dir=DATASET_DIR,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )

    dataset_dir = core.base.resolve_dataset_dir(PROJECT_ROOT, DATASET_DIR)
    position_files = core.base.vh.list_position_files(dataset_dir)
    position_files, invalid_positions = core.base.vh.filter_position_files_valid_for_probes(position_files, config.num_probes)
    available = {int(position): path for position, path in position_files}
    missing = [position for position in legal_positions if position not in available]
    if missing:
        raise ValueError(f"Positions not found or invalid for probes: {missing}. Available: {sorted(available)}")
    tuple_items = guarded.tuple_items_for_combo(position_files, legal_positions)

    rng = np.random.default_rng(args.random_seed)
    bit_blocks_pm = core.generate_effective_bit_blocks(
        real_k=config.real_k,
        virtual_count=core.virtual_count_for_k(config.real_k, config.target_effective_k),
        num_blocks=args.bits,
        rng=rng,
    )

    print(f"dataset={DATASET_NAME}")
    print(f"legal_positions={legal_positions}")
    print(f"bits_per_stream={args.bits}")
    print("Generating baseline candidates...")
    baseline_candidates, next_candidate_id = guarded.generate_candidates(
        config=config,
        position_files=position_files,
        tuple_items=tuple_items,
        legal_positions=legal_positions,
        bit_blocks_pm=bit_blocks_pm,
        subset_count=args.baseline_probe_subset_count,
        mappings_per_subset=args.baseline_mappings_per_subset,
        source="manual_baseline",
        candidate_id_start=1,
        sample_index=1,
        seed_offset=0,
    )
    baseline_candidates, _ = core.align_candidates(baseline_candidates, position_files, legal_positions)
    if not baseline_candidates:
        raise RuntimeError("No baseline candidates satisfied authorized BER threshold.")

    anchor = max(baseline_candidates, key=lambda c: (c.min_route_min_ber, c.average_route_min_ber))
    weak_routes = guarded.anchor_worst_routes(anchor, guarded.adaptive_weak_route_count(len(anchor.route_keys), args.weak_route_count))
    print(f"anchor_min_leakage_ber={anchor.min_route_min_ber:.6f}")
    print(f"weak_routes={weak_routes}")

    print("Generating targeted candidates...")
    targeted_candidates, _ = guarded.generate_candidates(
        config=config,
        position_files=position_files,
        tuple_items=tuple_items,
        legal_positions=legal_positions,
        bit_blocks_pm=bit_blocks_pm,
        subset_count=args.targeted_probe_subset_count,
        mappings_per_subset=args.targeted_mappings_per_subset,
        source="manual_targeted",
        candidate_id_start=next_candidate_id,
        sample_index=1,
        seed_offset=1000000,
    )
    all_candidates, alignment_info = core.align_candidates(
        list(baseline_candidates) + list(targeted_candidates),
        position_files,
        legal_positions,
    )
    if not all_candidates:
        raise RuntimeError("No candidates remained after route alignment.")

    selected = core.select_candidates(all_candidates, selected_count=args.selected_count, weak_routes=weak_routes)
    best = max(selected, key=lambda c: (c.min_route_min_ber, c.average_route_min_ber, -c.authorized_max_ber))

    per_legal = per_legal_rows(best, legal_positions)
    routes = route_rows(best)
    bits = bit_stream_rows(bit_blocks_pm, legal_positions)
    summary = [{
        "dataset": DATASET_NAME,
        "dataset_dir": DATASET_DIR,
        "legal_positions": str(tuple(legal_positions)),
        "bits_per_stream": args.bits,
        "real_k": config.real_k,
        "target_effective_k": config.target_effective_k,
        "virtual_stream_count": core.virtual_count_for_k(config.real_k, config.target_effective_k),
        "candidate_id": best.candidate_id,
        "candidate_source": best.source,
        "probes": core.list_text([int(v) for v in best.probes]),
        "authorized_max_ber": f"{best.authorized_max_ber:.8f}",
        "authorized_position_bers": core.float_list_text(best.authorized_position_bers),
        "illegal_device_count": len({route[0] for route in best.route_keys}),
        "route_count": len(best.route_keys),
        "illegal_min_leakage_ber": f"{best.min_route_min_ber:.8f}",
        "illegal_avg_leakage_ber": f"{best.average_route_min_ber:.8f}",
        "anchor_min_leakage_ber": f"{anchor.min_route_min_ber:.8f}",
        "common_route_count": alignment_info.common_route_count,
        "excluded_illegal_positions": core.list_text(alignment_info.excluded_illegal_positions),
        "weak_routes": core.list_text([f"{route[0]}->{route[1]}" for route in weak_routes]),
        "selected_candidate_ids": core.list_text([candidate.candidate_id for candidate in selected]),
    }]

    suffix = "_".join(str(position) for position in legal_positions)
    summary_csv = os.path.join(args.output_dir, f"manual_6pos_{suffix}_summary.csv")
    per_legal_csv = os.path.join(args.output_dir, f"manual_6pos_{suffix}_per_legal.csv")
    routes_csv = os.path.join(args.output_dir, f"manual_6pos_{suffix}_illegal_routes.csv")
    bits_csv = os.path.join(args.output_dir, f"manual_6pos_{suffix}_bits.csv")

    write_rows(summary_csv, summary[0].keys(), summary)
    write_rows(per_legal_csv, per_legal[0].keys(), per_legal)
    write_rows(routes_csv, routes[0].keys(), routes)
    write_rows(bits_csv, bits[0].keys(), bits)

    print(f"[OK] summary -> {summary_csv}")
    print(f"[OK] per legal stream -> {per_legal_csv}")
    print(f"[OK] illegal route details -> {routes_csv}")
    print(f"[OK] generated bit streams -> {bits_csv}")
    return summary_csv, per_legal_csv, routes_csv, bits_csv


def main(argv: Sequence[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()
