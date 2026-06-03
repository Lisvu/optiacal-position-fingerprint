#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global-guarded virtual-stream experiments with anchor-preserved repair."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import virtual_stream_core as core


@dataclass
class GlobalGuardedArgs:
    real_k: int
    output_dir: str
    sample_size: int = 3
    target_effective_k: int = 8
    baseline_probe_subset_count: int = 20
    targeted_probe_subset_count: int = 50
    base_mapping_count: int = 20
    baseline_mappings_per_subset: int = 8
    targeted_mappings_per_subset: int = 16
    selected_count: int = 20
    weak_route_count: int = 5
    eval_bits: int = 1000
    dataset_dir: str = core.base.DEFAULT_DATASET_DIR
    overwrite: bool = False
    random_seed: int = 20260513


def add_global_guarded_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--target-effective-k", type=int, default=8)
    parser.add_argument("--baseline-probe-subset-count", type=int, default=20)
    parser.add_argument("--targeted-probe-subset-count", type=int, default=50)
    parser.add_argument("--base-mapping-count", type=int, default=20)
    parser.add_argument("--baseline-mappings-per-subset", type=int, default=8)
    parser.add_argument("--targeted-mappings-per-subset", type=int, default=16)
    parser.add_argument("--selected-count", type=int, default=20)
    parser.add_argument("--weak-route-count", type=int, default=5)
    parser.add_argument("--eval-bits", type=int, default=1000)
    parser.add_argument("--dataset-dir", default=core.base.DEFAULT_DATASET_DIR)
    parser.add_argument("--overwrite", action="store_true")


def args_from_namespace(real_k: int, output_dir: str, namespace: argparse.Namespace) -> GlobalGuardedArgs:
    return GlobalGuardedArgs(
        real_k=real_k,
        output_dir=output_dir,
        sample_size=namespace.sample_size,
        target_effective_k=namespace.target_effective_k,
        baseline_probe_subset_count=namespace.baseline_probe_subset_count,
        targeted_probe_subset_count=namespace.targeted_probe_subset_count,
        base_mapping_count=namespace.base_mapping_count,
        baseline_mappings_per_subset=namespace.baseline_mappings_per_subset,
        targeted_mappings_per_subset=namespace.targeted_mappings_per_subset,
        selected_count=namespace.selected_count,
        weak_route_count=namespace.weak_route_count,
        eval_bits=namespace.eval_bits,
        dataset_dir=namespace.dataset_dir,
        overwrite=namespace.overwrite,
    )


def tuple_items_for_combo(position_files: Sequence[tuple[int, str]], combo: Sequence[int]) -> list[tuple[int, str]]:
    file_map = {int(position): path for position, path in position_files}
    return [(int(position), file_map[int(position)]) for position in combo]


def route_values_text(route_keys: Sequence[tuple[int, int]], values: Sequence[float]) -> str:
    return str({f"{route[0]}->{route[1]}": f"{float(value):.8f}" for route, value in zip(route_keys, values)})


def append_weak_routes(path: str, rows: Sequence[dict]) -> None:
    exists = os.path.exists(path)
    fields = [
        "sample_index", "position_combination", "anchor_candidate_id", "anchor_source",
        "anchor_min_route_min_ber", "anchor_worst_route", "weak_routes", "weak_route_anchor_values",
    ]
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def guarded_summary_fields() -> list[str]:
    return core.summary_fields() + [
        "anchor_candidate_id", "anchor_source", "anchor_min_route_min_ber", "anchor_worst_route",
        "security_gain_over_anchor", "floor_applied", "optimizer_diagnostic", "weak_routes",
    ]


def candidate_weak_min(candidate: core.Candidate, routes: Sequence[tuple[int, int]]) -> float:
    indices = core.route_indices(candidate.route_keys, routes)
    if not indices:
        return candidate.min_route_min_ber
    return float(np.min(candidate.route_min_bers[indices]))


def anchor_worst_routes(anchor: core.Candidate, weak_route_count: int) -> list[tuple[int, int]]:
    order = np.argsort(anchor.route_min_bers)
    return [anchor.route_keys[int(idx)] for idx in order[:max(1, weak_route_count)]]


def adaptive_weak_route_count(route_count: int, requested_count: int) -> int:
    return min(int(route_count), max(int(requested_count), int(np.ceil(route_count * 0.15))))


def group_rescue_score(candidate: core.Candidate, weak_routes: Sequence[tuple[int, int]]) -> tuple[float, float, float]:
    indices = core.route_indices(candidate.route_keys, weak_routes)
    if not indices:
        weak_values = candidate.route_min_bers
    else:
        weak_values = candidate.route_min_bers[indices]
    return (
        float(np.mean(weak_values)),
        candidate.min_route_min_ber,
        candidate.average_route_min_ber,
    )


def append_unique(selected: list[core.Candidate], candidate: core.Candidate) -> None:
    if all(existing.candidate_id != candidate.candidate_id for existing in selected):
        selected.append(candidate)


def candidate_quality_floor(anchor: core.Candidate) -> float:
    return max(0.10, anchor.min_route_min_ber - 0.20)


def is_bounded_rescuer(candidate: core.Candidate, anchor: core.Candidate) -> bool:
    return candidate.candidate_id == anchor.candidate_id or candidate.min_route_min_ber >= candidate_quality_floor(anchor)


def select_global_guarded_candidates(
    candidates: Sequence[core.Candidate],
    selected_count: int,
    weak_routes: Sequence[tuple[int, int]],
) -> tuple[list[core.Candidate], core.Candidate]:
    anchor = max(candidates, key=lambda candidate: (candidate.min_route_min_ber, candidate.average_route_min_ber))
    selected: list[core.Candidate] = []
    append_unique(selected, anchor)

    bounded_candidates = [candidate for candidate in candidates if is_bounded_rescuer(candidate, anchor)]
    top_global_quota = max(1, min(5, selected_count // 4))
    group_rescue_quota = max(1, selected_count // 4)
    route_rescue_quota = max(1, selected_count // 4)
    diversity_quota = max(1, selected_count - 1 - top_global_quota - group_rescue_quota - route_rescue_quota)

    quota_pools = [
        (top_global_quota, sorted(candidates, key=lambda c: (c.min_route_min_ber, c.average_route_min_ber), reverse=True)),
        (group_rescue_quota, sorted(bounded_candidates, key=lambda c: group_rescue_score(c, weak_routes), reverse=True)),
    ]
    for quota, pool in quota_pools:
        added = 0
        for candidate in pool:
            before = len(selected)
            append_unique(selected, candidate)
            if len(selected) > before:
                added += 1
            if added >= quota or len(selected) >= selected_count:
                break

    route_indices = core.route_indices(candidates[0].route_keys, weak_routes)
    added = 0
    for route_idx in route_indices:
        candidate = max(bounded_candidates, key=lambda c: (c.route_min_bers[route_idx], c.min_route_min_ber))
        before = len(selected)
        append_unique(selected, candidate)
        if len(selected) > before:
            added += 1
        if added >= route_rescue_quota or len(selected) >= selected_count:
            break

    worst_route_counts: dict[tuple[int, int], int] = {}
    diverse_by_worst = sorted(bounded_candidates, key=lambda c: (c.average_route_min_ber, c.min_route_min_ber), reverse=True)
    added = 0
    for candidate in diverse_by_worst:
        count = worst_route_counts.get(candidate.worst_route, 0)
        if count >= 2:
            continue
        before = len(selected)
        append_unique(selected, candidate)
        if len(selected) > before:
            worst_route_counts[candidate.worst_route] = count + 1
            added += 1
        if added >= diversity_quota or len(selected) >= selected_count:
            break

    diverse = sorted(candidates, key=lambda c: (c.min_route_min_ber, c.average_route_min_ber), reverse=True)
    for candidate in diverse:
        append_unique(selected, candidate)
        if len(selected) >= selected_count:
            break
    return selected[:selected_count], anchor


def optimize_ratio_with_diagnostic(
    raw_matrix: np.ndarray,
    anchor_index: int,
    anchor_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, bool, str]:
    raw_matrix = np.asarray(raw_matrix, dtype=float)
    if raw_matrix.ndim != 2 or raw_matrix.shape[0] == 0 or raw_matrix.shape[1] == 0 or not np.all(np.isfinite(raw_matrix)):
        ratio = np.zeros(raw_matrix.shape[1] if raw_matrix.ndim == 2 else 1, dtype=float)
        ratio[min(anchor_index, len(ratio) - 1)] = 1.0
        safe_matrix = np.nan_to_num(raw_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        mixed_raw = safe_matrix @ ratio if safe_matrix.ndim == 2 else np.asarray([], dtype=float)
        mixed_min = core.base.corrected_ber(mixed_raw)
        return ratio, mixed_raw, mixed_min, "anchor_only_floor", True, f"invalid_raw_matrix shape={raw_matrix.shape}"
    try:
        from scipy.optimize import linprog

        route_count, candidate_count = raw_matrix.shape
        c = np.zeros(candidate_count + 1, dtype=float)
        c[-1] = -1.0
        a_ub = []
        b_ub = []
        for route_idx in range(route_count):
            row = np.zeros(candidate_count + 1, dtype=float)
            row[:candidate_count] = -raw_matrix[route_idx]
            row[-1] = 1.0
            a_ub.append(row)
            b_ub.append(0.0)

            row = np.zeros(candidate_count + 1, dtype=float)
            row[:candidate_count] = raw_matrix[route_idx]
            row[-1] = 1.0
            a_ub.append(row)
            b_ub.append(1.0)

        result = linprog(
            c,
            A_ub=np.asarray(a_ub, dtype=float),
            b_ub=np.asarray(b_ub, dtype=float),
            A_eq=np.asarray([[*([1.0] * candidate_count), 0.0]], dtype=float),
            b_eq=np.asarray([1.0], dtype=float),
            bounds=[(0.0, 1.0)] * candidate_count + [(0.0, 0.5)],
            method="highs",
        )
        if not result.success:
            ratio = np.zeros(raw_matrix.shape[1], dtype=float)
            ratio[anchor_index] = 1.0
            mixed_raw = raw_matrix @ ratio
            mixed_min = core.base.corrected_ber(mixed_raw)
            return ratio, mixed_raw, mixed_min, "anchor_only_floor", True, f"linprog_failed status={result.status}: {result.message}"
        ratio = np.maximum(result.x[:candidate_count], 0.0)
        ratio = ratio / np.sum(ratio)
        optimizer = "linprog"
        mixed_raw = raw_matrix @ ratio
        mixed_min = core.base.corrected_ber(mixed_raw)
        diagnostic = "ok"
    except Exception as exc:
        ratio = np.zeros(raw_matrix.shape[1], dtype=float)
        ratio[anchor_index] = 1.0
        mixed_raw = raw_matrix @ ratio
        mixed_min = core.base.corrected_ber(mixed_raw)
        return ratio, mixed_raw, mixed_min, "anchor_only_floor", True, f"optimizer_exception {type(exc).__name__}: {exc}"
    if float(np.min(mixed_min)) + 1e-12 >= anchor_floor:
        return ratio, mixed_raw, mixed_min, optimizer, False, diagnostic
    ratio = np.zeros(raw_matrix.shape[1], dtype=float)
    ratio[anchor_index] = 1.0
    mixed_raw = raw_matrix @ ratio
    mixed_min = core.base.corrected_ber(mixed_raw)
    return ratio, mixed_raw, mixed_min, "anchor_only_floor", True, f"floor_applied optimized_min={float(np.min(core.base.corrected_ber(raw_matrix @ ratio))):.8f} anchor_floor={anchor_floor:.8f}"


def anchor_guarded_ratio_and_min(
    selected: Sequence[core.Candidate],
    anchor: core.Candidate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, bool, str]:
    raw_matrix = np.asarray([candidate.route_raw_bers for candidate in selected], dtype=float).T
    anchor_idx = next(idx for idx, candidate in enumerate(selected) if candidate.candidate_id == anchor.candidate_id)
    return optimize_ratio_with_diagnostic(raw_matrix, anchor_idx, anchor.min_route_min_ber)


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
        rng=np.random.default_rng(config.random_seed + config.real_k * 100003 + sample_index * 7919 + seed_offset),
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
            rng=random.Random(config.random_seed + config.real_k * 100003 + sample_index * 9173 + subset_idx + seed_offset),
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


def write_guarded_outputs(
    summary_csv: str,
    selected_csv: str,
    config: core.ExperimentConfig,
    sample_index: int,
    legal_positions: Sequence[int],
    selected: Sequence[core.Candidate],
    anchor: core.Candidate,
    weak_routes: Sequence[tuple[int, int]],
    alignment_info: core.AlignmentInfo,
) -> None:
    ratio, _, mixed_min, optimizer, floor_applied, optimizer_diagnostic = anchor_guarded_ratio_and_min(selected, anchor)
    worst_idx = int(np.argmin(mixed_min))
    route_keys = selected[0].route_keys
    counts = np.floor(ratio * config.schedule_length).astype(int)
    remainder = int(config.schedule_length) - int(np.sum(counts))
    if remainder > 0:
        for idx in np.argsort(-(ratio * config.schedule_length - counts))[:remainder]:
            counts[idx] += 1
    active_candidates = [candidate for candidate, weight in zip(selected, ratio) if float(weight) > 1e-12] or [anchor]
    auth_max = max(candidate.authorized_max_ber for candidate in active_candidates)
    auth_by_position = np.max(np.asarray([candidate.authorized_position_bers for candidate in active_candidates], dtype=float), axis=0)
    effective_k = core.effective_k_for_config(config)
    virtual_count = core.virtual_count_for_k(config.real_k, config.target_effective_k)
    security_min = float(np.min(mixed_min))
    gain = security_min - anchor.min_route_min_ber
    core.append_rows(summary_csv, guarded_summary_fields(), [{
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
        "security_min_route_min_ber": f"{security_min:.8f}",
        "security_avg_route_min_ber": f"{float(np.mean(mixed_min)):.8f}",
        "worst_route": f"{route_keys[worst_idx][0]}->{route_keys[worst_idx][1]}",
        "optimizer": optimizer,
        "usage_ratio": core.float_list_text(ratio),
        "usage_counts": core.list_text([int(v) for v in counts]),
        "selected_candidate_ids": core.list_text([candidate.candidate_id for candidate in selected]),
        "selected_probe_sets": " | ".join(core.list_text([int(v) for v in candidate.probes]) for candidate in selected),
        "anchor_candidate_id": anchor.candidate_id,
        "anchor_source": anchor.source,
        "anchor_min_route_min_ber": f"{anchor.min_route_min_ber:.8f}",
        "anchor_worst_route": f"{anchor.worst_route[0]}->{anchor.worst_route[1]}",
        "security_gain_over_anchor": f"{gain:.8f}",
        "floor_applied": str(bool(floor_applied)),
        "optimizer_diagnostic": optimizer_diagnostic,
        "weak_routes": core.list_text([f"{route[0]}->{route[1]}" for route in weak_routes]),
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
    print(
        f"      security_min={security_min:.6f}, anchor_min={anchor.min_route_min_ber:.6f}, "
        f"gain={gain:.6f}, worst={route_keys[worst_idx][0]}->{route_keys[worst_idx][1]}, "
        f"optimizer={optimizer}, floor_applied={floor_applied}, diagnostic={optimizer_diagnostic}"
    )


def run_global_guarded_experiment(args: GlobalGuardedArgs) -> tuple[str, str, str]:
    config = core.ExperimentConfig(
        real_k=args.real_k,
        output_dir=args.output_dir,
        target_effective_k=args.target_effective_k,
        sample_size=args.sample_size,
        base_mapping_count=args.base_mapping_count,
        mappings_per_subset=args.targeted_mappings_per_subset,
        selected_count=args.selected_count,
        eval_bits=args.eval_bits,
        weak_route_count=args.weak_route_count,
        dataset_dir=args.dataset_dir,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )
    summary_csv, selected_csv = core.reset_outputs(args.output_dir) if args.overwrite else core.output_paths(args.output_dir)
    weak_routes_csv = os.path.join(args.output_dir, "weak_routes.csv")
    if args.overwrite and os.path.exists(weak_routes_csv):
        os.remove(weak_routes_csv)

    project_root = os.path.dirname(core.BASE_DIR)
    dataset_dir = core.base.resolve_dataset_dir(project_root, config.dataset_dir)
    position_files = core.base.vh.list_position_files(dataset_dir)
    position_files, invalid_positions = core.base.vh.filter_position_files_valid_for_probes(position_files, config.num_probes)
    if invalid_positions:
        print(f"Filtered invalid baseline positions: {invalid_positions}")
    sampled_sets = core.base.vh.sample_tuples(
        position_files=position_files,
        k=args.real_k,
        sample_size=args.sample_size,
        min_position_distance=0.0,
        rng=np.random.default_rng(args.random_seed + args.real_k * 100003),
    )
    completed = core.load_completed_position_combinations(summary_csv)
    print(
        f"global-guarded k={args.real_k}, effective_k={core.effective_k_for_config(config)}, "
        f"samples={len(sampled_sets)}, completed={len(completed)}, output={args.output_dir}"
    )

    for sample_index, tuple_items in enumerate(sampled_sets, start=1):
        legal_positions = tuple(int(item[0]) for item in tuple_items)
        if legal_positions in completed:
            print(f"[{sample_index}/{len(sampled_sets)}] legal_positions={legal_positions} skipped existing")
            continue
        print(f"[{sample_index}/{len(sampled_sets)}] legal_positions={legal_positions}")
        bit_blocks_pm = core.generate_effective_bit_blocks(
            real_k=args.real_k,
            virtual_count=core.virtual_count_for_k(args.real_k, args.target_effective_k),
            num_blocks=args.eval_bits,
            rng=np.random.default_rng(args.random_seed + args.real_k * 100003 + sample_index * 1009),
        )
        baseline_candidates, next_candidate_id = generate_candidates(
            config=config,
            position_files=position_files,
            tuple_items=tuple_items,
            legal_positions=legal_positions,
            bit_blocks_pm=bit_blocks_pm,
            subset_count=args.baseline_probe_subset_count,
            mappings_per_subset=args.baseline_mappings_per_subset,
            source="baseline_global_guarded",
            candidate_id_start=1,
            sample_index=sample_index,
            seed_offset=0,
        )
        baseline_candidates, _ = core.align_candidates(baseline_candidates, position_files, legal_positions)
        if not baseline_candidates or not baseline_candidates[0].route_keys:
            print("      no baseline common routes")
            continue
        baseline_anchor = max(baseline_candidates, key=lambda c: (c.min_route_min_ber, c.average_route_min_ber))
        weak_route_count = adaptive_weak_route_count(len(baseline_anchor.route_keys), args.weak_route_count)
        weak_routes = anchor_worst_routes(baseline_anchor, weak_route_count)
        weak_indices = core.route_indices(baseline_anchor.route_keys, weak_routes)
        append_weak_routes(weak_routes_csv, [{
            "sample_index": sample_index,
            "position_combination": str(legal_positions),
            "anchor_candidate_id": baseline_anchor.candidate_id,
            "anchor_source": baseline_anchor.source,
            "anchor_min_route_min_ber": f"{baseline_anchor.min_route_min_ber:.8f}",
            "anchor_worst_route": f"{baseline_anchor.worst_route[0]}->{baseline_anchor.worst_route[1]}",
            "weak_routes": core.list_text([f"{route[0]}->{route[1]}" for route in weak_routes]),
            "weak_route_anchor_values": route_values_text(weak_routes, baseline_anchor.route_min_bers[weak_indices]),
        }])
        print(f"      anchor_worst_routes={weak_routes}")

        targeted_candidates, _ = generate_candidates(
            config=config,
            position_files=position_files,
            tuple_items=tuple_items,
            legal_positions=legal_positions,
            bit_blocks_pm=bit_blocks_pm,
            subset_count=args.targeted_probe_subset_count,
            mappings_per_subset=args.targeted_mappings_per_subset,
            source="anchor_worst_targeted",
            candidate_id_start=next_candidate_id,
            sample_index=sample_index,
            seed_offset=500000,
        )
        all_candidates, alignment_info = core.align_candidates(
            list(baseline_candidates) + list(targeted_candidates), position_files, legal_positions
        )
        if not all_candidates or not all_candidates[0].route_keys:
            print("      no targeted common routes")
            continue
        anchor = max(all_candidates, key=lambda c: (c.min_route_min_ber, c.average_route_min_ber))
        weak_route_count = adaptive_weak_route_count(len(anchor.route_keys), args.weak_route_count)
        weak_routes = anchor_worst_routes(anchor, weak_route_count)
        selected, anchor = select_global_guarded_candidates(
            all_candidates,
            min(args.selected_count, len(all_candidates)),
            weak_routes,
        )
        write_guarded_outputs(summary_csv, selected_csv, config, sample_index, legal_positions, selected, anchor, weak_routes, alignment_info)
    return summary_csv, selected_csv, weak_routes_csv


def run_cli(real_k: int, output_dir: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run global-guarded vector-hue k={real_k} experiment.")
    add_global_guarded_arguments(parser)
    namespace = parser.parse_args()
    run_global_guarded_experiment(args_from_namespace(real_k, output_dir, namespace))
