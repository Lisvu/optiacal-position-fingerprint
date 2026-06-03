#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Targeted probe-subset repair for weak vector-hue routes on yellow_shuffled."""

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
import pandas as pd


DEFAULT_DATASET_DIR = "data\\15pro\\yellow_shuffled"
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 10
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_NUM_PROBES = 15
DEFAULT_BASE_MAPPING_COUNT = 20
DEFAULT_PROBE_SUBSET_COUNT = 40
DEFAULT_MAPPINGS_PER_SUBSET = 8
DEFAULT_SELECTED_COUNT = 10
DEFAULT_VECTOR_TOP_K = 4
DEFAULT_EVAL_BITS = 1000
DEFAULT_SCHEDULE_LENGTH = 1000
DEFAULT_AUTHORIZED_BER_THRESHOLD = 0.0
DEFAULT_WEAK_ROUTE_COUNT = 5
DEFAULT_RANDOM_SEED = 20260513

DEFAULT_CANDIDATE_CSV = "yellow_shuffled_targeted_probe_candidates.csv"
DEFAULT_SELECTED_CSV = "yellow_shuffled_targeted_probe_selected.csv"
DEFAULT_SCHEDULE_CSV = "yellow_shuffled_targeted_probe_schedule.csv"


def load_vector_hue_module():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_path = os.path.join(project_root, "vector_hue_random_conv_k2_to_k8_singlefile.py")
    spec = importlib.util.spec_from_file_location("vector_hue_base_for_targeted_probe_repair", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load vector-hue base module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vh = load_vector_hue_module()


@dataclass
class ProbeMappingCandidate:
    candidate_id: int
    source: str
    probes: np.ndarray
    hue_mapping: dict[tuple, tuple[int, int, int, int]]
    authorized_position_bers: list[float]
    route_keys: list[tuple[int, int]]
    route_raw_bers: np.ndarray
    route_min_bers: np.ndarray

    @property
    def authorized_max_ber(self) -> float:
        return float(max(self.authorized_position_bers)) if self.authorized_position_bers else float("nan")

    @property
    def min_route_min_ber(self) -> float:
        return float(np.min(self.route_min_bers))

    @property
    def average_route_min_ber(self) -> float:
        return float(np.mean(self.route_min_bers))

    @property
    def worst_route(self) -> tuple[int, int]:
        return self.route_keys[int(np.argmin(self.route_min_bers))]


@dataclass
class RouteAlignmentInfo:
    common_route_count: int
    excluded_illegal_positions: list[int]


def corrected_ber(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.minimum(values, 1.0 - values)


def resolve_dataset_dir(project_root: str, dataset_dir: str) -> str:
    if os.path.isabs(dataset_dir):
        if os.path.isdir(dataset_dir):
            return dataset_dir
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    path = os.path.join(project_root, dataset_dir)
    if os.path.isdir(path):
        return path
    raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")


def list_text(values: Sequence[object]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def float_list_text(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.8f}" for v in values) + "]"


def route_dict_text(route_keys: Sequence[tuple[int, int]], values: Sequence[float]) -> str:
    return str({f"{route[0]}->{route[1]}": f"{float(value):.8f}" for route, value in zip(route_keys, values)})


def output_paths_for_k(output_dir: str, k: int) -> tuple[str, str, str]:
    return (
        os.path.join(output_dir, f"yellow_shuffled_targeted_probe_k{k}_candidates.csv"),
        os.path.join(output_dir, f"yellow_shuffled_targeted_probe_k{k}_selected.csv"),
        os.path.join(output_dir, f"yellow_shuffled_targeted_probe_k{k}_schedule.csv"),
    )


def format_vector_mapping(mapping: dict[tuple, tuple[int, int, int, int]]) -> str:
    return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(mapping.items())) + "}"


def equally_spaced_probe_values(num_probes: int) -> np.ndarray:
    return np.linspace(5, 355, num_probes, dtype=int)


def available_probe_values(position_files: Sequence[tuple[int, str]]) -> list[int]:
    if not position_files:
        return []
    row_count = min(len(pd.read_csv(path)) for _, path in position_files)
    return [int((idx + 1) * 5) for idx in range(row_count)]


def generate_probe_subsets(
    position_files: Sequence[tuple[int, str]],
    num_probes: int,
    subset_count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    all_values = available_probe_values(position_files)
    baseline = tuple(int(v) for v in equally_spaced_probe_values(num_probes))
    subsets: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    if all(value in all_values for value in baseline):
        subsets.append(baseline)
        seen.add(baseline)

    attempts = 0
    max_attempts = max(1000, subset_count * 100)
    while len(subsets) < subset_count and attempts < max_attempts:
        attempts += 1
        picked = tuple(sorted(int(v) for v in rng.choice(all_values, size=num_probes, replace=False)))
        if picked in seen:
            continue
        seen.add(picked)
        subsets.append(picked)
    return [np.asarray(values, dtype=float) for values in subsets]


def mapping_signature(mapping: dict[tuple, tuple[int, int, int, int]]) -> tuple[tuple[tuple, tuple[int, int, int, int]], ...]:
    return tuple(sorted(mapping.items()))


def build_vector_mapping_candidates(
    models: Sequence[object],
    candidate_count: int,
    vector_top_k: int,
    rng: random.Random,
) -> list[dict[tuple, tuple[int, int, int, int]]]:
    candidate_dict = vh._build_vector_candidate_dict(list(models), top_k=vector_top_k)
    mappings: list[dict[tuple, tuple[int, int, int, int]]] = []
    seen: set[tuple[tuple[tuple, tuple[int, int, int, int]], ...]] = set()

    top1 = {key: values[0] for key, values in candidate_dict.items()}
    mappings.append(top1)
    seen.add(mapping_signature(top1))

    keys = sorted(candidate_dict)
    attempts = 0
    max_attempts = max(1000, candidate_count * 100)
    while len(mappings) < candidate_count and attempts < max_attempts:
        attempts += 1
        mapping = {key: rng.choice(candidate_dict[key]) for key in keys}
        signature = mapping_signature(mapping)
        if signature in seen:
            continue
        seen.add(signature)
        mappings.append(mapping)
    return mappings


def authorized_position_bers(results: Sequence[dict], num_positions: int) -> list[float]:
    errors = np.zeros(num_positions, dtype=float)
    totals = np.zeros(num_positions, dtype=float)
    for block in results:
        true_bits = block["bits_bin"]
        decoded_bits = [decode.bit_hat_bin for decode in block["per_position"]]
        for idx, (true_bit, decoded_bit) in enumerate(zip(true_bits, decoded_bits)):
            totals[idx] += 1
            if int(true_bit) != int(decoded_bit):
                errors[idx] += 1
    return (errors / np.maximum(totals, 1.0)).tolist()


def evaluate_candidate(
    position_files: Sequence[tuple[int, str]],
    legal_positions: Sequence[int],
    probes: np.ndarray,
    models: Sequence[object],
    candidate_id: int,
    source: str,
    hue_mapping: dict[tuple, tuple[int, int, int, int]],
    bit_blocks_pm: Sequence[np.ndarray],
) -> ProbeMappingCandidate:
    results = vh.simulate_blocks_vector(list(models), list(bit_blocks_pm), hue_mapping)
    legal_bers = authorized_position_bers(results, len(legal_positions))
    true_stream_bits = vh.extract_truth_bits(results, num_positions=len(legal_positions))
    legal_set = set(int(v) for v in legal_positions)
    file_map = {int(position): path for position, path in position_files}
    route_keys: list[tuple[int, int]] = []
    route_raw_bers: list[float] = []

    for illegal_position in sorted(file_map):
        if illegal_position in legal_set:
            continue
        try:
            illegal_model = vh.load_single_position_model(file_map[illegal_position], probes=probes)
            probe_to_row = vh.build_probe_to_row(illegal_model.probes)
            decoded_bits = []
            for block in results:
                y_obs = vh.observe_block_from_vector_mapping(block["hue_seq_vector"], illegal_model.Y, probe_to_row)
                dec = vh.decode_local_block(y_obs, illegal_model.w, illegal_model.code)
                decoded_bits.append(int(dec.bit_hat_bin))
            decoded = np.asarray(decoded_bits, dtype=int)
            for stream_idx, legal_position in enumerate(legal_positions):
                truth = np.asarray(true_stream_bits[stream_idx], dtype=int)
                route_keys.append((int(illegal_position), int(legal_position)))
                route_raw_bers.append(float(np.mean(decoded != truth)))
        except Exception:
            continue

    raw = np.asarray(route_raw_bers, dtype=float)
    return ProbeMappingCandidate(
        candidate_id=candidate_id,
        source=source,
        probes=np.asarray(probes, dtype=float),
        hue_mapping=hue_mapping,
        authorized_position_bers=legal_bers,
        route_keys=route_keys,
        route_raw_bers=raw,
        route_min_bers=corrected_ber(raw),
    )


def route_indices(route_keys: Sequence[tuple[int, int]], routes: Sequence[tuple[int, int]]) -> list[int]:
    index = {route: idx for idx, route in enumerate(route_keys)}
    return [index[route] for route in routes if route in index]


def identify_weak_routes(candidates: Sequence[ProbeMappingCandidate], weak_route_count: int) -> list[tuple[int, int]]:
    route_keys = candidates[0].route_keys
    route_matrix = np.asarray([candidate.route_min_bers for candidate in candidates], dtype=float)
    route_best = np.max(route_matrix, axis=0)
    order = np.argsort(route_best)
    return [route_keys[int(idx)] for idx in order[:max(1, weak_route_count)]]


def candidate_score(candidate: ProbeMappingCandidate, weak_routes: Sequence[tuple[int, int]]) -> tuple[float, float, float]:
    indices = route_indices(candidate.route_keys, weak_routes)
    weak_min = float(np.min(candidate.route_min_bers[indices])) if indices else candidate.min_route_min_ber
    return (weak_min, candidate.min_route_min_ber, candidate.average_route_min_ber)


def select_candidates(
    candidates: Sequence[ProbeMappingCandidate],
    selected_count: int,
    weak_routes: Sequence[tuple[int, int]],
) -> list[ProbeMappingCandidate]:
    return sorted(candidates, key=lambda candidate: candidate_score(candidate, weak_routes), reverse=True)[:selected_count]


def align_candidates_to_common_routes(
    candidates: Sequence[ProbeMappingCandidate],
    position_files: Sequence[tuple[int, str]],
    legal_positions: Sequence[int],
) -> tuple[list[ProbeMappingCandidate], RouteAlignmentInfo]:
    if not candidates:
        return [], RouteAlignmentInfo(common_route_count=0, excluded_illegal_positions=[])
    common = set(candidates[0].route_keys)
    for candidate in candidates[1:]:
        common &= set(candidate.route_keys)
    common_keys = [route for route in candidates[0].route_keys if route in common]
    legal_set = set(int(v) for v in legal_positions)
    expected_illegals = {int(position) for position, _ in position_files if int(position) not in legal_set}
    common_illegals = {int(route[0]) for route in common_keys}
    excluded_illegals = sorted(expected_illegals - common_illegals)
    info = RouteAlignmentInfo(
        common_route_count=len(common_keys),
        excluded_illegal_positions=excluded_illegals,
    )
    aligned: list[ProbeMappingCandidate] = []
    for candidate in candidates:
        index = {route: idx for idx, route in enumerate(candidate.route_keys)}
        indices = [index[route] for route in common_keys]
        aligned.append(ProbeMappingCandidate(
            candidate_id=candidate.candidate_id,
            source=candidate.source,
            probes=candidate.probes,
            hue_mapping=candidate.hue_mapping,
            authorized_position_bers=candidate.authorized_position_bers,
            route_keys=list(common_keys),
            route_raw_bers=candidate.route_raw_bers[indices],
            route_min_bers=candidate.route_min_bers[indices],
        ))
    return aligned, info


def optimize_usage_ratio(raw_matrix: np.ndarray) -> tuple[np.ndarray, float, str]:
    route_count, candidate_count = raw_matrix.shape
    try:
        from scipy.optimize import linprog

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
        if result.success:
            ratio = np.maximum(result.x[:candidate_count], 0.0)
            ratio = ratio / np.sum(ratio)
            objective = float(np.min(corrected_ber(raw_matrix @ ratio)))
            return ratio, objective, "linprog"
    except Exception:
        pass

    best_idx = int(np.argmax(np.min(corrected_ber(raw_matrix), axis=0)))
    ratio = np.zeros(candidate_count, dtype=float)
    ratio[best_idx] = 1.0
    objective = float(np.min(corrected_ber(raw_matrix @ ratio)))
    return ratio, objective, "best_single_fallback"


def append_rows(file_path: str, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    exists = os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def reset_files(paths: Sequence[str]) -> None:
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def candidate_fields() -> list[str]:
    return [
        "k", "position_combination", "candidate_id", "source", "probes", "common_route_count",
        "excluded_illegal_positions", "authorized_position_bers",
        "authorized_max_ber", "min_route_min_ber", "average_route_min_ber", "worst_route",
        "route_raw_bers", "route_min_bers", "vector_hue_mapping",
    ]


def selected_fields() -> list[str]:
    return [
        "k", "position_combination", "selection_rank", "candidate_id", "source", "probes",
        "common_route_count", "excluded_illegal_positions",
        "candidate_min_route_min_ber", "candidate_average_route_min_ber", "worst_route",
        "route_min_bers",
    ]


def schedule_fields() -> list[str]:
    return [
        "k", "position_combination", "common_route_count", "excluded_illegal_positions",
        "selected_candidate_ids", "selected_sources", "usage_ratio", "usage_counts",
        "optimizer", "offline_objective_min_route_min_ber", "offline_average_route_min_ber", "worst_route",
        "weak_routes", "route_raw_bers", "route_min_bers",
    ]


def write_outputs(
    candidate_csv: str,
    selected_csv: str,
    schedule_csv: str,
    k: int,
    legal_positions: Sequence[int],
    candidates: Sequence[ProbeMappingCandidate],
    selected: Sequence[ProbeMappingCandidate],
    schedule_length: int,
    weak_routes: Sequence[tuple[int, int]],
    alignment_info: RouteAlignmentInfo,
) -> None:
    candidate_rows = []
    for candidate in candidates:
        candidate_rows.append({
            "k": int(k),
            "position_combination": str(tuple(legal_positions)),
            "candidate_id": candidate.candidate_id,
            "source": candidate.source,
            "probes": list_text([int(v) for v in candidate.probes]),
            "common_route_count": int(alignment_info.common_route_count),
            "excluded_illegal_positions": list_text(alignment_info.excluded_illegal_positions),
            "authorized_position_bers": float_list_text(candidate.authorized_position_bers),
            "authorized_max_ber": f"{candidate.authorized_max_ber:.8f}",
            "min_route_min_ber": f"{candidate.min_route_min_ber:.8f}",
            "average_route_min_ber": f"{candidate.average_route_min_ber:.8f}",
            "worst_route": f"{candidate.worst_route[0]}->{candidate.worst_route[1]}",
            "route_raw_bers": route_dict_text(candidate.route_keys, candidate.route_raw_bers),
            "route_min_bers": route_dict_text(candidate.route_keys, candidate.route_min_bers),
            "vector_hue_mapping": format_vector_mapping(candidate.hue_mapping),
        })
    append_rows(candidate_csv, candidate_fields(), candidate_rows)

    selected_rows = []
    for rank, candidate in enumerate(selected, start=1):
        selected_rows.append({
            "k": int(k),
            "position_combination": str(tuple(legal_positions)),
            "selection_rank": rank,
            "candidate_id": candidate.candidate_id,
            "source": candidate.source,
            "probes": list_text([int(v) for v in candidate.probes]),
            "common_route_count": int(alignment_info.common_route_count),
            "excluded_illegal_positions": list_text(alignment_info.excluded_illegal_positions),
            "candidate_min_route_min_ber": f"{candidate.min_route_min_ber:.8f}",
            "candidate_average_route_min_ber": f"{candidate.average_route_min_ber:.8f}",
            "worst_route": f"{candidate.worst_route[0]}->{candidate.worst_route[1]}",
            "route_min_bers": route_dict_text(candidate.route_keys, candidate.route_min_bers),
        })
    append_rows(selected_csv, selected_fields(), selected_rows)

    raw_matrix = np.asarray([candidate.route_raw_bers for candidate in selected], dtype=float).T
    ratio, objective, optimizer = optimize_usage_ratio(raw_matrix)
    mixed_raw = raw_matrix @ ratio
    mixed_min = corrected_ber(mixed_raw)
    worst_idx = int(np.argmin(mixed_min))
    route_keys = selected[0].route_keys
    counts = np.floor(ratio * schedule_length).astype(int)
    remainder = int(schedule_length) - int(np.sum(counts))
    if remainder > 0:
        for idx in np.argsort(-(ratio * schedule_length - counts))[:remainder]:
            counts[idx] += 1
    append_rows(schedule_csv, schedule_fields(), [{
        "k": int(k),
        "position_combination": str(tuple(legal_positions)),
        "common_route_count": int(alignment_info.common_route_count),
        "excluded_illegal_positions": list_text(alignment_info.excluded_illegal_positions),
        "selected_candidate_ids": list_text([candidate.candidate_id for candidate in selected]),
        "selected_sources": list_text([candidate.source for candidate in selected]),
        "usage_ratio": float_list_text(ratio),
        "usage_counts": list_text([int(v) for v in counts]),
        "optimizer": optimizer,
        "offline_objective_min_route_min_ber": f"{float(np.min(mixed_min)):.8f}",
        "offline_average_route_min_ber": f"{float(np.mean(mixed_min)):.8f}",
        "worst_route": f"{route_keys[worst_idx][0]}->{route_keys[worst_idx][1]}",
        "weak_routes": list_text([f"{route[0]}->{route[1]}" for route in weak_routes]),
        "route_raw_bers": route_dict_text(route_keys, mixed_raw),
        "route_min_bers": route_dict_text(route_keys, mixed_min),
    }])
    print(
        f"      Schedule objective={objective:.6f}, actual_min={float(np.min(mixed_min)):.6f}, "
        f"actual_avg={float(np.mean(mixed_min)):.6f}, worst={route_keys[worst_idx][0]}->{route_keys[worst_idx][1]}"
    )


def load_legal_models(tuple_items: Sequence[tuple[int, str]], probes: np.ndarray) -> list[object] | None:
    models = []
    try:
        for _, path in tuple_items:
            models.append(vh.load_single_position_model(path, probes=probes))
    except Exception:
        return None
    return models


def run_for_k(
    args: argparse.Namespace,
    dataset_dir: str,
    output_dir: str,
    position_files: Sequence[tuple[int, str]],
    k: int,
) -> tuple[str, str, str]:
    candidate_csv, selected_csv, schedule_csv = output_paths_for_k(output_dir, k)
    if args.overwrite:
        reset_files([candidate_csv, selected_csv, schedule_csv])
    sampled_sets = vh.sample_tuples(
        position_files=position_files,
        k=k,
        sample_size=args.sample_size,
        min_position_distance=0.0,
        rng=np.random.default_rng(args.random_seed + k * 100003),
    )
    print(f"Targeted probe-subset repair: k={k}, sampled_sets={len(sampled_sets)}")

    for combo_idx, tuple_items in enumerate(sampled_sets, start=1):
        legal_positions = tuple(int(item[0]) for item in tuple_items)
        print(f"[{combo_idx}/{len(sampled_sets)}] legal_positions={legal_positions}")
        bit_blocks_pm = vh.generate_random_bit_blocks(
            num_positions=k,
            num_blocks=args.eval_bits,
            rng=np.random.default_rng(args.random_seed + k * 100003 + combo_idx * 1009),
        )
        probe_subsets = generate_probe_subsets(
            position_files=position_files,
            num_probes=args.num_probes,
            subset_count=args.probe_subset_count,
            rng=np.random.default_rng(args.random_seed + k * 100003 + combo_idx * 7919),
        )

        candidates: list[ProbeMappingCandidate] = []
        candidate_id = 1
        for subset_idx, probes in enumerate(probe_subsets, start=1):
            models = load_legal_models(tuple_items, probes)
            if models is None:
                print(f"      Probe subset {subset_idx}/{len(probe_subsets)}: skipped invalid legal probes")
                continue
            mappings = build_vector_mapping_candidates(
                models=models,
                candidate_count=args.mappings_per_subset if subset_idx > 1 else args.base_mapping_count,
                vector_top_k=args.vector_top_k,
                rng=random.Random(args.random_seed + k * 100003 + combo_idx * 9173 + subset_idx),
            )
            legal_count = 0
            best_min = -1.0
            for mapping in mappings:
                candidate = evaluate_candidate(
                    position_files=position_files,
                    legal_positions=legal_positions,
                    probes=probes,
                    models=models,
                    candidate_id=candidate_id,
                    source="baseline" if subset_idx == 1 else "probe_repair",
                    hue_mapping=mapping,
                    bit_blocks_pm=bit_blocks_pm,
                )
                candidate_id += 1
                if candidate.authorized_max_ber <= args.authorized_ber_threshold:
                    candidates.append(candidate)
                    legal_count += 1
                    best_min = max(best_min, candidate.min_route_min_ber)
            print(f"      Probe subset {subset_idx}/{len(probe_subsets)}: legal={legal_count}, best_min={best_min:.6f}")

        if not candidates:
            print("      No legal candidates found.")
            continue

        candidates, alignment_info = align_candidates_to_common_routes(candidates, position_files, legal_positions)
        if not candidates or not candidates[0].route_keys:
            print("      No common evaluable routes across candidates.")
            continue
        print(
            f"      Common routes={alignment_info.common_route_count}, "
            f"excluded_illegal_positions={alignment_info.excluded_illegal_positions}"
        )

        weak_routes = identify_weak_routes(candidates, args.weak_route_count)
        print("      Weak routes: " + ", ".join(f"{route[0]}->{route[1]}" for route in weak_routes))
        selected = select_candidates(candidates, min(args.selected_count, len(candidates)), weak_routes)
        write_outputs(
            candidate_csv,
            selected_csv,
            schedule_csv,
            k,
            legal_positions,
            candidates,
            selected,
            args.schedule_length,
            weak_routes,
            alignment_info,
        )

    print(f"Candidates saved to: {candidate_csv}")
    print(f"Selected saved to: {selected_csv}")
    print(f"Schedule saved to: {schedule_csv}")
    return candidate_csv, selected_csv, schedule_csv


def run(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = resolve_dataset_dir(project_root, args.dataset_dir)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    position_files = vh.list_position_files(dataset_dir)
    position_files, invalid_positions = vh.filter_position_files_valid_for_probes(position_files, args.num_probes)
    if invalid_positions:
        print(f"Filtered invalid positions for baseline num_probes={args.num_probes}: {invalid_positions}")
    if args.k is not None:
        k_values = [int(args.k)]
    else:
        k_values = list(range(int(args.k_min), int(args.k_max) + 1))
    print(f"Dataset: {dataset_dir}")
    print(f"K values: {k_values}; sample_size_per_k={args.sample_size}")
    outputs = []
    for k in k_values:
        outputs.append(run_for_k(args, dataset_dir, output_dir, position_files, k))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted probe-subset repair for yellow_shuffled vector hue mappings.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--k", type=int, default=None, help="Run one k only. By default runs --k-min..--k-max.")
    parser.add_argument("--k-min", type=int, default=DEFAULT_K_MIN)
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--num-probes", type=int, default=DEFAULT_NUM_PROBES)
    parser.add_argument("--base-mapping-count", type=int, default=DEFAULT_BASE_MAPPING_COUNT)
    parser.add_argument("--probe-subset-count", type=int, default=DEFAULT_PROBE_SUBSET_COUNT)
    parser.add_argument("--mappings-per-subset", type=int, default=DEFAULT_MAPPINGS_PER_SUBSET)
    parser.add_argument("--selected-count", type=int, default=DEFAULT_SELECTED_COUNT)
    parser.add_argument("--vector-top-k", type=int, default=DEFAULT_VECTOR_TOP_K)
    parser.add_argument("--eval-bits", type=int, default=DEFAULT_EVAL_BITS)
    parser.add_argument("--schedule-length", type=int, default=DEFAULT_SCHEDULE_LENGTH)
    parser.add_argument("--authorized-ber-threshold", type=float, default=DEFAULT_AUTHORIZED_BER_THRESHOLD)
    parser.add_argument("--weak-route-count", type=int, default=DEFAULT_WEAK_ROUTE_COUNT)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
