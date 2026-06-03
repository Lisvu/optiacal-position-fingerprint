#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual 6-position experiment using probes common to all 28 devices.

This version implements the corrected experiment idea:

1. Load mate40pro_high_column_shuffled.
2. Let the user provide 6 legal positions.
3. Find probes that are usable by all 28 positions.
4. Repair the desired probe set to nearest usable probes from that common pool.
5. Use the repaired probes as the actual transmitted probes for both legal and
   illegal receivers.
6. Generate 6 random 10000-bit streams and report authorized BER plus illegal
   leakage BER for the remaining 22 positions.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
ORIGINAL_VECTOR_HUE_DIR = os.path.join(PROJECT_ROOT, "vector_hue")
if ORIGINAL_VECTOR_HUE_DIR not in sys.path:
    sys.path.insert(0, ORIGINAL_VECTOR_HUE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import virtual_stream_core as core


DATASET_NAME = "mate40pro_high_column_shuffled"
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "mate40pro", "high_column_shuffled")
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "results",
    "manual_6pos_common_probe_10000bits",
    DATASET_NAME,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manual 6-position 10000-bit experiment with all-device common probes."
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs=6,
        required=True,
        help="Six legal position ids, e.g. --positions 1 2 3 4 5 6",
    )
    parser.add_argument("--bits", type=int, default=10000, help="Bits per legal stream.")
    parser.add_argument("--num-probes", type=int, default=15, help="Number of actual transmitted probes.")
    parser.add_argument(
        "--allow-fewer-common-probes",
        action="store_true",
        help="If fewer than --num-probes probes are common to all devices, use all common probes instead.",
    )
    parser.add_argument("--target-effective-k", type=int, default=8)
    parser.add_argument("--mapping-candidates", type=int, default=40)
    parser.add_argument("--vector-top-k", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=20260603)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def write_rows(path: str, fields: Sequence[str], rows: Sequence[dict], overwrite: bool = True) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "w" if overwrite else "a"
    exists = os.path.exists(path)
    with open(path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        if overwrite or not exists:
            writer.writeheader()
        writer.writerows(rows)


def ensure_unique_positions(positions: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(v) for v in positions)
    if len(set(out)) != len(out):
        raise ValueError(f"Positions must be unique: {out}")
    return out


def usable_probes_for_position(csv_file: str) -> set[int]:
    df = pd.read_csv(csv_file)
    values = df.values.astype(float)
    usable_rows = np.where(~np.any(values < 0, axis=1))[0]
    return {int((row_idx + 1) * 5) for row_idx in usable_rows}


def common_usable_probes(position_files: Sequence[tuple[int, str]]) -> list[int]:
    common: set[int] | None = None
    for _, csv_file in position_files:
        probes = usable_probes_for_position(csv_file)
        common = probes if common is None else common & probes
    return sorted(common or [])


def nearest_common_probe(target: int, common: Sequence[int], used: set[int]) -> int:
    choices = [probe for probe in common if int(probe) not in used]
    if not choices:
        raise ValueError("Not enough unique common probes to build requested probe set.")
    return min(choices, key=lambda probe: (abs(int(probe) - int(target)), int(probe)))


def repaired_probe_set(num_probes: int, max_row_index: int, common: Sequence[int]) -> tuple[np.ndarray, list[dict]]:
    desired = core.base.vh.generate_probes(num_probes, max_row_index, verbose=False)
    used: set[int] = set()
    repaired: list[int] = []
    substitutions: list[dict] = []
    for probe in desired:
        requested = int(probe)
        actual = nearest_common_probe(requested, common, used)
        used.add(actual)
        repaired.append(actual)
        substitutions.append({
            "requested_probe": requested,
            "actual_probe": actual,
            "distance": abs(actual - requested),
            "changed": int(actual != requested),
        })
    return np.asarray(repaired, dtype=float), substitutions


def bit_stream_rows(bit_blocks_pm: Sequence[np.ndarray], legal_positions: Sequence[int]) -> list[dict]:
    rows = []
    for bit_index, bits_pm in enumerate(bit_blocks_pm, start=1):
        bits_bin = core.base.vh.pm1_to_bin(np.asarray(bits_pm, dtype=int))[:len(legal_positions)]
        row = {"bit_index": bit_index}
        for stream_idx, (position, bit) in enumerate(zip(legal_positions, bits_bin), start=1):
            row[f"stream_{stream_idx}_position_{int(position)}"] = int(bit)
        rows.append(row)
    return rows


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
        route_values = [
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
            "illegal_device_count": len(route_values),
            "illegal_min_leakage_ber": f"{float(np.min(route_values)):.8f}",
            "illegal_avg_leakage_ber": f"{float(np.mean(route_values)):.8f}",
            "illegal_min_raw_ber": f"{float(np.min(raw_values)):.8f}",
            "illegal_avg_raw_ber": f"{float(np.mean(raw_values)):.8f}",
        })
    return rows


def per_illegal_rows(candidate: core.Candidate) -> list[dict]:
    illegal_positions = sorted({int(route[0]) for route in candidate.route_keys})
    rows = []
    for illegal_position in illegal_positions:
        values = [
            float(value)
            for route, value in zip(candidate.route_keys, candidate.route_min_bers)
            if int(route[0]) == int(illegal_position)
        ]
        raw_values = [
            float(value)
            for route, value in zip(candidate.route_keys, candidate.route_raw_bers)
            if int(route[0]) == int(illegal_position)
        ]
        rows.append({
            "illegal_position": int(illegal_position),
            "legal_stream_count": len(values),
            "min_leakage_ber": f"{float(np.min(values)):.8f}",
            "avg_leakage_ber": f"{float(np.mean(values)):.8f}",
            "min_raw_ber": f"{float(np.min(raw_values)):.8f}",
            "avg_raw_ber": f"{float(np.mean(raw_values)):.8f}",
        })
    return rows


def choose_best_candidate(
    real_models: Sequence[object],
    position_files: Sequence[tuple[int, str]],
    legal_positions: Sequence[int],
    probes: np.ndarray,
    bit_blocks_pm: Sequence[np.ndarray],
    virtual_count: int,
    mapping_candidates: int,
    vector_top_k: int,
    random_seed: int,
) -> core.Candidate:
    mappings = core.build_virtual_mapping_candidates(
        real_models=real_models,
        candidate_count=mapping_candidates,
        vector_top_k=vector_top_k,
        virtual_count=virtual_count,
        rng=random.Random(random_seed),
    )
    candidates = []
    for candidate_id, mapping in enumerate(mappings, start=1):
        candidate = core.evaluate_candidate(
            position_files=position_files,
            legal_positions=legal_positions,
            probes=probes,
            models=real_models,
            real_k=len(legal_positions),
            virtual_count=virtual_count,
            candidate_id=candidate_id,
            source="common_probe_mapping",
            hue_mapping=mapping,
            bit_blocks_pm=bit_blocks_pm,
        )
        if candidate.authorized_max_ber <= 0.0:
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError("No mapping candidate achieved zero authorized BER.")
    candidates, alignment_info = core.align_candidates(candidates, position_files, legal_positions)
    if alignment_info.excluded_illegal_positions:
        raise RuntimeError(f"Some illegal positions could not be evaluated: {alignment_info.excluded_illegal_positions}")
    return max(candidates, key=lambda c: (c.min_route_min_ber, c.average_route_min_ber))


def run(argv: Sequence[str] | None = None) -> tuple[str, str, str, str, str, str]:
    args = parse_args(argv)
    legal_positions = ensure_unique_positions(args.positions)
    os.makedirs(args.output_dir, exist_ok=True)

    position_files = core.base.vh.list_position_files(DATASET_DIR)
    file_map = {int(position): path for position, path in position_files}
    missing = [position for position in legal_positions if position not in file_map]
    if missing:
        raise ValueError(f"Legal positions not found in dataset: {missing}")
    illegal_positions = sorted(position for position, _ in position_files if int(position) not in set(legal_positions))
    if len(illegal_positions) != 22:
        raise ValueError(f"Expected 22 illegal positions, got {len(illegal_positions)}: {illegal_positions}")

    first_df = pd.read_csv(position_files[0][1])
    max_row_index = len(first_df) - 1
    common = common_usable_probes(position_files)
    actual_num_probes = args.num_probes
    if len(common) < args.num_probes:
        if not args.allow_fewer_common_probes:
            raise ValueError(
                f"Only {len(common)} probes are common to all 28 devices, need {args.num_probes}. "
                "Use --allow-fewer-common-probes to run with all available common probes, "
                "or reduce --num-probes."
            )
        actual_num_probes = len(common)
        if actual_num_probes < 2:
            raise ValueError(
                f"Only {actual_num_probes} probe is common to all 28 devices. "
                "The existing fingerprint extraction code requires at least 2 probes, "
                "so this all-device-common-probe experiment is not feasible on this dataset."
            )
        print(f"Only {len(common)} probes are common to all 28 devices; using {actual_num_probes}.")
    probes, substitutions = repaired_probe_set(actual_num_probes, max_row_index, common)

    tuple_items = [(int(position), file_map[int(position)]) for position in legal_positions]
    real_models = core.base.load_legal_models(tuple_items, probes)
    if real_models is None:
        raise RuntimeError("Failed to load legal models with repaired common probes.")

    virtual_count = core.virtual_count_for_k(len(legal_positions), args.target_effective_k)
    bit_blocks_pm = core.generate_effective_bit_blocks(
        real_k=len(legal_positions),
        virtual_count=virtual_count,
        num_blocks=args.bits,
        rng=np.random.default_rng(args.random_seed),
    )

    print(f"dataset={DATASET_NAME}")
    print(f"legal_positions={legal_positions}")
    print(f"illegal_positions={illegal_positions}")
    print(f"bits_per_stream={args.bits}")
    print(f"common_probe_count={len(common)}")
    print(f"actual_probes={[int(v) for v in probes]}")

    best = choose_best_candidate(
        real_models=real_models,
        position_files=position_files,
        legal_positions=legal_positions,
        probes=probes,
        bit_blocks_pm=bit_blocks_pm,
        virtual_count=virtual_count,
        mapping_candidates=args.mapping_candidates,
        vector_top_k=args.vector_top_k,
        random_seed=args.random_seed,
    )

    suffix = "_".join(str(position) for position in legal_positions)
    summary_csv = os.path.join(args.output_dir, f"common_probe_6pos_{suffix}_summary.csv")
    per_legal_csv = os.path.join(args.output_dir, f"common_probe_6pos_{suffix}_per_legal.csv")
    per_illegal_csv = os.path.join(args.output_dir, f"common_probe_6pos_{suffix}_per_illegal.csv")
    routes_csv = os.path.join(args.output_dir, f"common_probe_6pos_{suffix}_illegal_routes.csv")
    probes_csv = os.path.join(args.output_dir, f"common_probe_6pos_{suffix}_probes.csv")
    bits_csv = os.path.join(args.output_dir, f"common_probe_6pos_{suffix}_bits.csv")

    route_values = list(best.route_min_bers)
    summary = [{
        "dataset": DATASET_NAME,
        "dataset_dir": DATASET_DIR,
        "legal_positions": str(tuple(legal_positions)),
        "illegal_positions": str(tuple(illegal_positions)),
        "bits_per_stream": args.bits,
        "requested_num_probes": args.num_probes,
        "actual_num_probes": len(probes),
        "common_probe_count": len(common),
        "actual_probes": core.list_text([int(v) for v in probes]),
        "target_effective_k": args.target_effective_k,
        "virtual_stream_count": virtual_count,
        "mapping_candidates": args.mapping_candidates,
        "selected_candidate_id": best.candidate_id,
        "authorized_max_ber": f"{best.authorized_max_ber:.8f}",
        "authorized_position_bers": core.float_list_text(best.authorized_position_bers),
        "illegal_device_count": len(illegal_positions),
        "route_count": len(best.route_keys),
        "illegal_min_leakage_ber": f"{float(np.min(route_values)):.8f}",
        "illegal_avg_leakage_ber": f"{float(np.mean(route_values)):.8f}",
        "illegal_min_raw_ber": f"{float(np.min(best.route_raw_bers)):.8f}",
        "illegal_avg_raw_ber": f"{float(np.mean(best.route_raw_bers)):.8f}",
    }]

    probe_rows = [{
        "probe_index": idx,
        "requested_probe": row["requested_probe"],
        "actual_transmitted_probe": row["actual_probe"],
        "distance": row["distance"],
        "changed": row["changed"],
    } for idx, row in enumerate(substitutions, start=1)]

    write_rows(summary_csv, summary[0].keys(), summary)
    write_rows(per_legal_csv, per_legal_rows(best, legal_positions)[0].keys(), per_legal_rows(best, legal_positions))
    write_rows(per_illegal_csv, per_illegal_rows(best)[0].keys(), per_illegal_rows(best))
    write_rows(routes_csv, route_rows(best)[0].keys(), route_rows(best))
    write_rows(probes_csv, probe_rows[0].keys(), probe_rows)
    bits = bit_stream_rows(bit_blocks_pm, legal_positions)
    write_rows(bits_csv, bits[0].keys(), bits)

    print(f"[OK] summary -> {summary_csv}")
    print(f"[OK] per legal stream -> {per_legal_csv}")
    print(f"[OK] per illegal device -> {per_illegal_csv}")
    print(f"[OK] illegal route details -> {routes_csv}")
    print(f"[OK] actual transmitted probes -> {probes_csv}")
    print(f"[OK] generated bit streams -> {bits_csv}")
    return summary_csv, per_legal_csv, per_illegal_csv, routes_csv, probes_csv, bits_csv


def main(argv: Sequence[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()
