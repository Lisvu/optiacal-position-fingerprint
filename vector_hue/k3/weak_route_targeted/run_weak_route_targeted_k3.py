#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weak-route targeted k=3 virtual-stream experiment."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import replace
from typing import Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import virtual_stream_core as core


DEFAULT_COMBOS = [
    (7, 18, 21),
    (6, 9, 24),
    (16, 17, 22),
]


def parse_combo(value: str) -> tuple[int, ...]:
    combo = core.normalize_position_combination(value)
    if combo is None:
        raise argparse.ArgumentTypeError(f"Invalid combo: {value}")
    if len(combo) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 positions, got {value}")
    return combo


def append_weak_routes(path: str, rows: Sequence[dict]) -> None:
    exists = os.path.exists(path)
    fields = [
        "sample_index",
        "position_combination",
        "baseline_candidate_count",
        "baseline_best_min_route_min_ber",
        "weak_routes",
        "weak_route_best_values",
    ]
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def route_values_text(route_keys: Sequence[tuple[int, int]], values: Sequence[float]) -> str:
    return str({f"{route[0]}->{route[1]}": f"{float(value):.8f}" for route, value in zip(route_keys, values)})


def tuple_items_for_combo(position_files: Sequence[tuple[int, str]], combo: Sequence[int]) -> list[tuple[int, str]]:
    file_map = {int(position): path for position, path in position_files}
    missing = [int(position) for position in combo if int(position) not in file_map]
    if missing:
        raise ValueError(f"Combo has positions not available after filtering: {missing}")
    return [(int(position), file_map[int(position)]) for position in combo]


def candidate_weak_min(candidate: core.Candidate, weak_routes: Sequence[tuple[int, int]]) -> float:
    indices = core.route_indices(candidate.route_keys, weak_routes)
    if not indices:
        return candidate.min_route_min_ber
    return float(np.min(candidate.route_min_bers[indices]))


def select_targeted_candidates(
    candidates: Sequence[core.Candidate],
    selected_count: int,
    weak_routes: Sequence[tuple[int, int]],
) -> list[core.Candidate]:
    def score(candidate: core.Candidate) -> tuple[float, float, float]:
        weak_min = candidate_weak_min(candidate, weak_routes)
        blended = 0.7 * weak_min + 0.3 * candidate.min_route_min_ber
        return blended, weak_min, candidate.average_route_min_ber

    return sorted(candidates, key=score, reverse=True)[:selected_count]


def append_unique_candidate(selected: list[core.Candidate], candidate: core.Candidate) -> None:
    if all(existing.candidate_id != candidate.candidate_id for existing in selected):
        selected.append(candidate)


def select_global_guarded_candidates(
    candidates: Sequence[core.Candidate],
    selected_count: int,
    weak_routes: Sequence[tuple[int, int]],
) -> tuple[list[core.Candidate], core.Candidate]:
    anchor = max(candidates, key=lambda candidate: (candidate.min_route_min_ber, candidate.average_route_min_ber))
    selected: list[core.Candidate] = []
    append_unique_candidate(selected, anchor)

    top_global = sorted(
        candidates,
        key=lambda candidate: (candidate.min_route_min_ber, candidate.average_route_min_ber),
        reverse=True,
    )
    top_weak = sorted(
        candidates,
        key=lambda candidate: (
            candidate_weak_min(candidate, weak_routes),
            candidate.min_route_min_ber,
            candidate.average_route_min_ber,
        ),
        reverse=True,
    )
    per_route_rescuers = []
    route_count = len(candidates[0].route_keys)
    for route_idx in range(route_count):
        per_route_rescuers.append(max(
            candidates,
            key=lambda candidate: (candidate.route_min_bers[route_idx], candidate.min_route_min_ber),
        ))

    for pool in (top_global, top_weak, per_route_rescuers):
        for candidate in pool:
            append_unique_candidate(selected, candidate)
            if len(selected) >= selected_count:
                return selected, anchor
    return selected, anchor


def anchor_guarded_ratio_and_min(
    selected: Sequence[core.Candidate],
    anchor: core.Candidate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, bool]:
    raw_matrix = np.asarray([candidate.route_raw_bers for candidate in selected], dtype=float).T
    ratio, _, optimizer = core.base.optimize_usage_ratio(raw_matrix)
    mixed_raw = raw_matrix @ ratio
    mixed_min = core.base.corrected_ber(mixed_raw)
    anchor_floor = anchor.min_route_min_ber
    if float(np.min(mixed_min)) + 1e-12 >= anchor_floor:
        return ratio, mixed_raw, mixed_min, optimizer, False

    anchor_idx = next(idx for idx, candidate in enumerate(selected) if candidate.candidate_id == anchor.candidate_id)
    ratio = np.zeros(len(selected), dtype=float)
    ratio[anchor_idx] = 1.0
    return ratio, anchor.route_raw_bers, anchor.route_min_bers, "anchor_only_floor", True


def write_anchor_guarded_outputs(
    summary_csv: str,
    selected_csv: str,
    config: core.ExperimentConfig,
    sample_index: int,
    legal_positions: Sequence[int],
    selected: Sequence[core.Candidate],
    anchor: core.Candidate,
    alignment_info: core.AlignmentInfo,
) -> None:
    ratio, mixed_raw, mixed_min, optimizer, floor_applied = anchor_guarded_ratio_and_min(selected, anchor)
    worst_idx = int(np.argmin(mixed_min))
    route_keys = selected[0].route_keys
    counts = np.floor(ratio * config.schedule_length).astype(int)
    remainder = int(config.schedule_length) - int(np.sum(counts))
    if remainder > 0:
        for idx in np.argsort(-(ratio * config.schedule_length - counts))[:remainder]:
            counts[idx] += 1
    active_candidates = [candidate for candidate, weight in zip(selected, ratio) if float(weight) > 1e-12]
    if not active_candidates:
        active_candidates = [anchor]
    auth_max = max(candidate.authorized_max_ber for candidate in active_candidates)
    auth_by_position = np.max(np.asarray([candidate.authorized_position_bers for candidate in active_candidates], dtype=float), axis=0)
    effective_k = core.effective_k_for_config(config)
    virtual_count = core.virtual_count_for_k(config.real_k, config.target_effective_k)
    core.append_rows(summary_csv, core.summary_fields(), [{
        "k": config.real_k,
        "real_k": config.real_k,
        "effective_k": effective_k,
        "virtual_stream_count": virtual_count,
        "target_effective_k": config.target_effective_k,
        "sample_index": sample_index,
        "position_combination": str(tuple(legal_positions)),
        "selected_candidate_count": len(selected),
        "common_route_count": alignment_info.common_route_count,
        "excluded_illegal_positions": core.list_text(alignment_info.excluded_illegal_positions),
        "authorized_max_ber": f"{auth_max:.8f}",
        "authorized_position_bers": core.float_list_text(auth_by_position),
        "security_min_route_min_ber": f"{float(np.min(mixed_min)):.8f}",
        "security_avg_route_min_ber": f"{float(np.mean(mixed_min)):.8f}",
        "worst_route": f"{route_keys[worst_idx][0]}->{route_keys[worst_idx][1]}",
        "optimizer": optimizer,
        "usage_ratio": core.float_list_text(ratio),
        "usage_counts": core.list_text([int(v) for v in counts]),
        "selected_candidate_ids": core.list_text([candidate.candidate_id for candidate in selected]),
        "selected_probe_sets": " | ".join(core.list_text([int(v) for v in candidate.probes]) for candidate in selected),
    }])
    rows = []
    for rank, candidate in enumerate(selected, start=1):
        rows.append({
            "k": config.real_k,
            "real_k": config.real_k,
            "effective_k": effective_k,
            "virtual_stream_count": virtual_count,
            "sample_index": sample_index,
            "position_combination": str(tuple(legal_positions)),
            "selection_rank": rank,
            "candidate_id": candidate.candidate_id,
            "source": candidate.source,
            "probes": core.list_text([int(v) for v in candidate.probes]),
            "authorized_max_ber": f"{candidate.authorized_max_ber:.8f}",
            "authorized_position_bers": core.float_list_text(candidate.authorized_position_bers),
            "candidate_min_route_min_ber": f"{candidate.min_route_min_ber:.8f}",
            "candidate_avg_route_min_ber": f"{candidate.average_route_min_ber:.8f}",
            "worst_route": f"{candidate.worst_route[0]}->{candidate.worst_route[1]}",
        })
    core.append_rows(selected_csv, core.selected_fields(), rows)
    security_min = float(np.min(mixed_min))
    gain = security_min - anchor.min_route_min_ber
    print(
        f"      security_min={security_min:.6f}, anchor_min={anchor.min_route_min_ber:.6f}, "
        f"gain={gain:.6f}, auth_max={auth_max:.6f}, worst={route_keys[worst_idx][0]}->{route_keys[worst_idx][1]}, "
        f"common_routes={alignment_info.common_route_count}, optimizer={optimizer}, floor_applied={floor_applied}"
    )


def generate_candidates(
    *,
    config: core.ExperimentConfig,
    position_files: Sequence[tuple[int, str]],
    tuple_items: Sequence[tuple[int, str]],
    legal_positions: Sequence[int],
    bit_blocks_pm: Sequence[np.ndarray],
    subset_count: int,
    mappings_per_subset: int,
    source: str,
    candidate_id_start: int,
    sample_index: int,
    seed_offset: int,
) -> tuple[list[core.Candidate], int]:
    virtual_count = core.virtual_count_for_k(config.real_k, config.target_effective_k)
    probe_subsets = core.base.generate_probe_subsets(
        position_files=position_files,
        num_probes=config.num_probes,
        subset_count=subset_count,
        rng=np.random.default_rng(config.random_seed + sample_index * 7919 + seed_offset),
    )
    candidates: list[core.Candidate] = []
    candidate_id = candidate_id_start
    for subset_idx, probes in enumerate(probe_subsets, start=1):
        real_models = core.base.load_legal_models(tuple_items, probes)
        if real_models is None:
            continue
        models = core.expand_models_with_virtual(real_models, virtual_count)
        mappings = core.build_virtual_mapping_candidates(
            real_models=models,
            candidate_count=mappings_per_subset if subset_idx > 1 else config.base_mapping_count,
            vector_top_k=config.vector_top_k,
            virtual_count=virtual_count,
            rng=random.Random(config.random_seed + sample_index * 9173 + subset_idx + seed_offset),
        )
        legal_count = 0
        best_min = -1.0
        for mapping in mappings:
            candidate = core.evaluate_candidate(
                position_files=position_files,
                legal_positions=legal_positions,
                probes=probes,
                models=models,
                real_k=config.real_k,
                virtual_count=virtual_count,
                candidate_id=candidate_id,
                source=source,
                hue_mapping=mapping,
                bit_blocks_pm=bit_blocks_pm,
            )
            candidate_id += 1
            if candidate.authorized_max_ber <= config.authorized_ber_threshold:
                candidates.append(candidate)
                legal_count += 1
                best_min = max(best_min, candidate.min_route_min_ber)
        print(f"      {source} subset {subset_idx}/{len(probe_subsets)}: legal={legal_count}, best_min={best_min:.6f}")
    return candidates, candidate_id


def run_targeted_experiment(args: argparse.Namespace) -> tuple[str, str, str]:
    output_dir = os.path.dirname(os.path.abspath(__file__))
    config = core.ExperimentConfig(
        real_k=3,
        output_dir=output_dir,
        target_effective_k=args.target_effective_k,
        sample_size=len(args.combo),
        probe_subset_count=args.targeted_probe_subset_count,
        base_mapping_count=args.base_mapping_count,
        mappings_per_subset=args.targeted_mappings_per_subset,
        selected_count=args.selected_count,
        eval_bits=args.eval_bits,
        weak_route_count=args.weak_route_count,
        overwrite=args.overwrite,
    )
    summary_csv, selected_csv = core.reset_outputs(output_dir) if args.overwrite else core.output_paths(output_dir)
    weak_routes_csv = os.path.join(output_dir, "weak_routes.csv")
    if args.overwrite and os.path.exists(weak_routes_csv):
        os.remove(weak_routes_csv)

    project_root = os.path.dirname(core.BASE_DIR)
    dataset_dir = core.base.resolve_dataset_dir(project_root, config.dataset_dir)
    position_files = core.base.vh.list_position_files(dataset_dir)
    position_files, invalid_positions = core.base.vh.filter_position_files_valid_for_probes(position_files, config.num_probes)
    if invalid_positions:
        print(f"Filtered invalid baseline positions: {invalid_positions}")

    completed = core.load_completed_position_combinations(summary_csv)
    print(
        f"weak-route targeted k3, effective_k={core.effective_k_for_config(config)}, "
        f"virtual_streams={core.virtual_count_for_k(config.real_k, config.target_effective_k)}, "
        f"combos={len(args.combo)}, completed={len(completed)}, output={output_dir}"
    )

    for sample_index, combo in enumerate(args.combo, start=1):
        legal_positions = tuple(int(value) for value in combo)
        if legal_positions in completed:
            print(f"[{sample_index}/{len(args.combo)}] legal_positions={legal_positions} skipped existing")
            continue
        print(f"[{sample_index}/{len(args.combo)}] legal_positions={legal_positions}")
        tuple_items = tuple_items_for_combo(position_files, legal_positions)
        bit_blocks_pm = core.generate_effective_bit_blocks(
            real_k=config.real_k,
            virtual_count=core.virtual_count_for_k(config.real_k, config.target_effective_k),
            num_blocks=config.eval_bits,
            rng=np.random.default_rng(config.random_seed + sample_index * 1009),
        )

        baseline_candidates, next_candidate_id = generate_candidates(
            config=config,
            position_files=position_files,
            tuple_items=tuple_items,
            legal_positions=legal_positions,
            bit_blocks_pm=bit_blocks_pm,
            subset_count=args.baseline_probe_subset_count,
            mappings_per_subset=args.baseline_mappings_per_subset,
            source="baseline_weak_discovery",
            candidate_id_start=1,
            sample_index=sample_index,
            seed_offset=0,
        )
        baseline_candidates, baseline_info = core.align_candidates(baseline_candidates, position_files, legal_positions)
        if not baseline_candidates or not baseline_candidates[0].route_keys:
            print("      no baseline common routes")
            continue

        weak_routes = core.identify_weak_routes(baseline_candidates, config.weak_route_count)
        route_matrix = np.asarray([candidate.route_min_bers for candidate in baseline_candidates], dtype=float)
        route_best = np.max(route_matrix, axis=0)
        weak_indices = core.route_indices(baseline_candidates[0].route_keys, weak_routes)
        append_weak_routes(weak_routes_csv, [{
            "sample_index": sample_index,
            "position_combination": str(legal_positions),
            "baseline_candidate_count": len(baseline_candidates),
            "baseline_best_min_route_min_ber": f"{max(candidate.min_route_min_ber for candidate in baseline_candidates):.8f}",
            "weak_routes": core.list_text([f"{route[0]}->{route[1]}" for route in weak_routes]),
            "weak_route_best_values": route_values_text(weak_routes, route_best[weak_indices]),
        }])
        print(f"      weak_routes={weak_routes}")

        targeted_config = replace(config, mappings_per_subset=args.targeted_mappings_per_subset)
        targeted_candidates, _ = generate_candidates(
            config=targeted_config,
            position_files=position_files,
            tuple_items=tuple_items,
            legal_positions=legal_positions,
            bit_blocks_pm=bit_blocks_pm,
            subset_count=args.targeted_probe_subset_count,
            mappings_per_subset=args.targeted_mappings_per_subset,
            source="weak_route_targeted",
            candidate_id_start=next_candidate_id,
            sample_index=sample_index,
            seed_offset=500000,
        )

        all_candidates, alignment_info = core.align_candidates(
            list(baseline_candidates) + list(targeted_candidates),
            position_files,
            legal_positions,
        )
        if not all_candidates or not all_candidates[0].route_keys:
            print("      no targeted common routes")
            continue
        selected, anchor = select_global_guarded_candidates(
            all_candidates,
            min(config.selected_count, len(all_candidates)),
            weak_routes,
        )
        write_anchor_guarded_outputs(
            summary_csv,
            selected_csv,
            config,
            sample_index,
            legal_positions,
            selected,
            anchor,
            alignment_info,
        )
    return summary_csv, selected_csv, weak_routes_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weak-route targeted k=3 experiment.")
    parser.add_argument("--combo", type=parse_combo, action="append", default=None)
    parser.add_argument("--target-effective-k", type=int, default=8)
    parser.add_argument("--baseline-probe-subset-count", type=int, default=20)
    parser.add_argument("--targeted-probe-subset-count", type=int, default=50)
    parser.add_argument("--base-mapping-count", type=int, default=20)
    parser.add_argument("--baseline-mappings-per-subset", type=int, default=8)
    parser.add_argument("--targeted-mappings-per-subset", type=int, default=16)
    parser.add_argument("--selected-count", type=int, default=20)
    parser.add_argument("--weak-route-count", type=int, default=5)
    parser.add_argument("--eval-bits", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.combo is None:
        args.combo = DEFAULT_COMBOS
    run_targeted_experiment(args)


if __name__ == "__main__":
    main()
