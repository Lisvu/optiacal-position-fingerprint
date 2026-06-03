#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline scheduling of multiple vector-hue mappings on fixed probes.

Scheme A:
1. Use one fixed, equally spaced probe set.
2. Generate multiple vector-hue mappings for the same legal-position group.
3. Keep mappings that satisfy the legal BER constraint.
4. Select a security-complementary subset of mappings.
5. Optimize an offline usage ratio and emit a cyclic mapping schedule.

The online phase only needs: mapping_id = schedule_sequence[t % schedule_length].
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np


DEFAULT_DATASET_DIR = "data\\15pro\\yellow_shuffled"
DEFAULT_K = 8
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_NUM_PROBES = 15
DEFAULT_MAPPING_COUNT = 30
DEFAULT_SELECTED_MAPPING_COUNT = 10
DEFAULT_VECTOR_TOP_K = 4
DEFAULT_REPAIR_MAPPING_COUNT = 80
DEFAULT_REPAIR_KEEP_COUNT = 20
DEFAULT_REPAIR_VECTOR_TOP_K = 8
DEFAULT_WEAK_ROUTE_COUNT = 5
DEFAULT_EVAL_BITS = 1000
DEFAULT_SCHEDULE_LENGTH = 1000
DEFAULT_AUTHORIZED_BER_THRESHOLD = 0.0
SEED = 20260513

DEFAULT_MAPPING_CANDIDATE_CSV = "yellow_shuffled_vector_mapping_candidates.csv"
DEFAULT_SELECTED_MAPPING_CSV = "yellow_shuffled_vector_mapping_selected.csv"
DEFAULT_SCHEDULE_CSV = "yellow_shuffled_vector_mapping_schedule.csv"


def load_vector_hue_module():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_path = os.path.join(project_root, "vector_hue_random_conv_k2_to_k8_singlefile.py")
    spec = importlib.util.spec_from_file_location("vector_hue_base_for_offline_mapping_schedule", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load vector-hue base module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vh = load_vector_hue_module()


@dataclass
class MappingCandidate:
    mapping_id: int
    source: str
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


def corrected_ber(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.minimum(values, 1.0 - values)


def list_text(values: Sequence[object]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def float_list_text(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.8f}" for v in values) + "]"


def route_dict_text(route_keys: Sequence[tuple[int, int]], values: Sequence[float]) -> str:
    return str({f"{route[0]}->{route[1]}": f"{float(value):.8f}" for route, value in zip(route_keys, values)})


def format_vector_mapping(mapping: dict[tuple, tuple[int, int, int, int]]) -> str:
    return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(mapping.items())) + "}"


def mapping_signature(mapping: dict[tuple, tuple[int, int, int, int]]) -> tuple[tuple[tuple, tuple[int, int, int, int]], ...]:
    return tuple(sorted(mapping.items()))


def resolve_dataset_dir(project_root: str, dataset_dir: str) -> str:
    if os.path.isabs(dataset_dir):
        if os.path.isdir(dataset_dir):
            return dataset_dir
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    path = os.path.join(project_root, dataset_dir)
    if os.path.isdir(path):
        return path
    # Accept the user's common spelling without the trailing 'd'.
    fallback = os.path.join(project_root, "data", "15pro", "yellow_shuffled")
    if dataset_dir.replace("/", "\\").endswith("yellow_shuffle") and os.path.isdir(fallback):
        return fallback
    raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")


def sample_legal_position_sets(
    position_files: list[tuple[int, str]],
    k: int,
    sample_size: int,
    rng: np.random.Generator,
) -> list[tuple[tuple[int, str], ...]]:
    return vh.sample_tuples(
        position_files=position_files,
        k=k,
        sample_size=sample_size,
        min_position_distance=0.0,
        rng=rng,
    )


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
    signature = tuple(sorted(top1.items()))
    mappings.append(top1)
    seen.add(signature)

    attempts = 0
    max_attempts = max(1000, candidate_count * 200)
    keys = sorted(candidate_dict)
    while len(mappings) < candidate_count and attempts < max_attempts:
        attempts += 1
        mapping = {}
        for key in keys:
            values = candidate_dict[key]
            if attempts % 3 == 0:
                # Bias some candidates toward high-ranked vectors.
                weights = np.asarray([1.0 / (idx + 1) for idx in range(len(values))], dtype=float)
                weights = weights / np.sum(weights)
                pick_idx = int(np.random.default_rng(SEED + attempts + hash(key) % 100000).choice(len(values), p=weights))
                mapping[key] = values[pick_idx]
            else:
                mapping[key] = rng.choice(values)
        signature = tuple(sorted(mapping.items()))
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


def evaluate_mapping_candidate(
    dataset_dir: str,
    position_files: Sequence[tuple[int, str]],
    legal_positions: Sequence[int],
    probes: np.ndarray,
    models: Sequence[object],
    mapping_id: int,
    source: str,
    hue_mapping: dict[tuple, tuple[int, int, int, int]],
    bit_blocks_pm: Sequence[np.ndarray],
) -> MappingCandidate:
    results = vh.simulate_blocks_vector(list(models), list(bit_blocks_pm), hue_mapping)
    legal_bers = authorized_position_bers(results, len(legal_positions))
    true_stream_bits = vh.extract_truth_bits(results, num_positions=len(legal_positions))
    route_keys: list[tuple[int, int]] = []
    route_raw_bers: list[float] = []
    legal_set = set(int(v) for v in legal_positions)
    file_map = {int(position): path for position, path in position_files}

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
    return MappingCandidate(
        mapping_id=mapping_id,
        source=source,
        hue_mapping=hue_mapping,
        authorized_position_bers=legal_bers,
        route_keys=route_keys,
        route_raw_bers=raw,
        route_min_bers=corrected_ber(raw),
    )


def identify_weak_routes(candidates: Sequence[MappingCandidate], weak_route_count: int) -> list[tuple[int, int]]:
    if not candidates:
        return []
    route_keys = candidates[0].route_keys
    route_matrix = np.asarray([candidate.route_min_bers for candidate in candidates], dtype=float)
    route_best = np.max(route_matrix, axis=0)
    order = np.argsort(route_best)
    return [route_keys[int(idx)] for idx in order[:max(1, weak_route_count)]]


def route_indices(route_keys: Sequence[tuple[int, int]], routes: Sequence[tuple[int, int]]) -> list[int]:
    index = {route: idx for idx, route in enumerate(route_keys)}
    return [index[route] for route in routes if route in index]


def repair_score(candidate: MappingCandidate, weak_routes: Sequence[tuple[int, int]]) -> tuple[float, float, float]:
    indices = route_indices(candidate.route_keys, weak_routes)
    if not indices:
        weak_min = candidate.min_route_min_ber
    else:
        weak_min = float(np.min(candidate.route_min_bers[indices]))
    return (
        weak_min,
        candidate.min_route_min_ber,
        candidate.average_route_min_ber,
    )


def select_security_complementary_mappings(
    candidates: Sequence[MappingCandidate],
    selected_count: int,
) -> list[MappingCandidate]:
    selected: list[MappingCandidate] = []
    remaining = list(candidates)
    if not remaining:
        return selected
    cumulative_raw = np.zeros(len(remaining[0].route_keys), dtype=float)

    while remaining and len(selected) < selected_count:
        step = len(selected) + 1

        def score(candidate: MappingCandidate) -> tuple[float, float, float]:
            mixed_min = corrected_ber((cumulative_raw + candidate.route_raw_bers) / step)
            return (
                float(np.min(mixed_min)),
                float(np.mean(mixed_min)),
                candidate.average_route_min_ber,
            )

        best = max(remaining, key=score)
        selected.append(best)
        cumulative_raw += best.route_raw_bers
        remaining.remove(best)
        pool_min = corrected_ber(cumulative_raw / len(selected))
        print(
            f"      Select {len(selected)}/{selected_count}: mapping={best.mapping_id}, "
            f"pool_min={float(np.min(pool_min)):.6f}, pool_avg={float(np.mean(pool_min)):.6f}"
        )
    return selected


def optimize_usage_ratio(raw_matrix: np.ndarray) -> tuple[np.ndarray, float, str]:
    route_count, mapping_count = raw_matrix.shape
    try:
        from scipy.optimize import linprog

        c = np.zeros(mapping_count + 1, dtype=float)
        c[-1] = -1.0
        a_ub = []
        b_ub = []
        for route_idx in range(route_count):
            row = np.zeros(mapping_count + 1, dtype=float)
            row[:mapping_count] = -raw_matrix[route_idx]
            row[-1] = 1.0
            a_ub.append(row)
            b_ub.append(0.0)

            row = np.zeros(mapping_count + 1, dtype=float)
            row[:mapping_count] = raw_matrix[route_idx]
            row[-1] = 1.0
            a_ub.append(row)
            b_ub.append(1.0)

        result = linprog(
            c,
            A_ub=np.asarray(a_ub, dtype=float),
            b_ub=np.asarray(b_ub, dtype=float),
            A_eq=np.asarray([[*([1.0] * mapping_count), 0.0]], dtype=float),
            b_eq=np.asarray([1.0], dtype=float),
            bounds=[(0.0, 1.0)] * mapping_count + [(0.0, 0.5)],
            method="highs",
        )
        if result.success:
            ratio = np.maximum(result.x[:mapping_count], 0.0)
            ratio = ratio / np.sum(ratio)
            objective = float(np.min(corrected_ber(raw_matrix @ ratio)))
            return ratio, objective, "linprog"
    except Exception:
        pass

    rng = np.random.default_rng(SEED)
    ratios = [np.full(mapping_count, 1.0 / mapping_count, dtype=float)]
    ratios.extend(np.eye(mapping_count, dtype=float))
    ratios.extend(rng.dirichlet(np.ones(mapping_count), size=30000))
    best_ratio = ratios[0]
    best_score = -1.0
    for ratio in ratios:
        score = float(np.min(corrected_ber(raw_matrix @ ratio)))
        if score > best_score:
            best_score = score
            best_ratio = ratio
    return np.asarray(best_ratio, dtype=float), best_score, "random_dirichlet_fallback"


def counts_from_ratio(ratio: np.ndarray, schedule_length: int) -> np.ndarray:
    raw_counts = np.asarray(ratio, dtype=float) * int(schedule_length)
    counts = np.floor(raw_counts).astype(int)
    remainder = int(schedule_length) - int(np.sum(counts))
    if remainder > 0:
        for idx in np.argsort(-(raw_counts - counts))[:remainder]:
            counts[idx] += 1
    return counts


def build_risk_balanced_sequence(raw_matrix: np.ndarray, counts: np.ndarray, mapping_ids: Sequence[int]) -> list[int]:
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
        sequence.append(int(mapping_ids[best_idx]))
        cumulative_raw += raw_matrix[:, best_idx]
        remaining[best_idx] -= 1
        last_idx = best_idx
    return sequence


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
        "position_combination", "mapping_id", "candidate_source", "k", "num_probes", "probes", "authorized_position_bers",
        "authorized_max_ber", "min_route_min_ber", "average_route_min_ber", "worst_route",
        "route_raw_bers", "route_min_bers", "vector_hue_mapping",
    ]


def selected_fields() -> list[str]:
    return [
        "position_combination", "selection_rank", "mapping_id", "candidate_source", "selected_mapping_count",
        "authorized_position_bers", "authorized_max_ber", "candidate_min_route_min_ber",
        "candidate_average_route_min_ber", "selected_pool_min_route_min_ber",
        "selected_pool_average_route_min_ber", "worst_route", "route_min_bers",
    ]


def schedule_fields() -> list[str]:
    return [
        "position_combination", "k", "num_probes", "eval_bits", "mapping_candidate_count",
        "selected_mapping_count", "schedule_length", "optimizer", "selected_mapping_ids",
        "usage_ratio", "usage_counts", "offline_objective_min_route_min_ber",
        "offline_average_route_min_ber", "worst_route", "schedule_sequence",
        "route_raw_bers", "route_min_bers",
    ]


def write_candidates(path: str, combo: Sequence[int], probes: np.ndarray, candidates: Sequence[MappingCandidate]) -> None:
    rows = []
    for candidate in candidates:
        rows.append({
            "position_combination": str(tuple(combo)),
            "mapping_id": candidate.mapping_id,
            "candidate_source": candidate.source,
            "k": len(combo),
            "num_probes": len(probes),
            "probes": list_text([int(v) for v in probes]),
            "authorized_position_bers": float_list_text(candidate.authorized_position_bers),
            "authorized_max_ber": f"{candidate.authorized_max_ber:.8f}",
            "min_route_min_ber": f"{candidate.min_route_min_ber:.8f}",
            "average_route_min_ber": f"{candidate.average_route_min_ber:.8f}",
            "worst_route": f"{candidate.worst_route[0]}->{candidate.worst_route[1]}",
            "route_raw_bers": route_dict_text(candidate.route_keys, candidate.route_raw_bers),
            "route_min_bers": route_dict_text(candidate.route_keys, candidate.route_min_bers),
            "vector_hue_mapping": format_vector_mapping(candidate.hue_mapping),
        })
    append_rows(path, candidate_fields(), rows)


def write_selected(path: str, combo: Sequence[int], selected: Sequence[MappingCandidate]) -> None:
    rows = []
    cumulative_raw = np.zeros(len(selected[0].route_keys), dtype=float)
    for rank, candidate in enumerate(selected, start=1):
        cumulative_raw += candidate.route_raw_bers
        pool_min = corrected_ber(cumulative_raw / rank)
        rows.append({
            "position_combination": str(tuple(combo)),
            "selection_rank": rank,
            "mapping_id": candidate.mapping_id,
            "candidate_source": candidate.source,
            "selected_mapping_count": len(selected),
            "authorized_position_bers": float_list_text(candidate.authorized_position_bers),
            "authorized_max_ber": f"{candidate.authorized_max_ber:.8f}",
            "candidate_min_route_min_ber": f"{candidate.min_route_min_ber:.8f}",
            "candidate_average_route_min_ber": f"{candidate.average_route_min_ber:.8f}",
            "selected_pool_min_route_min_ber": f"{float(np.min(pool_min)):.8f}",
            "selected_pool_average_route_min_ber": f"{float(np.mean(pool_min)):.8f}",
            "worst_route": f"{candidate.worst_route[0]}->{candidate.worst_route[1]}",
            "route_min_bers": route_dict_text(candidate.route_keys, candidate.route_min_bers),
        })
    append_rows(path, selected_fields(), rows)


def write_schedule(
    path: str,
    combo: Sequence[int],
    probes: np.ndarray,
    eval_bits: int,
    mapping_candidate_count: int,
    selected: Sequence[MappingCandidate],
    schedule_length: int,
) -> None:
    raw_matrix = np.asarray([candidate.route_raw_bers for candidate in selected], dtype=float).T
    ratio, objective, optimizer = optimize_usage_ratio(raw_matrix)
    counts = counts_from_ratio(ratio, schedule_length)
    selected_ids = [candidate.mapping_id for candidate in selected]
    sequence = build_risk_balanced_sequence(raw_matrix, counts, selected_ids)
    actual_ratio = np.asarray([sequence.count(mapping_id) for mapping_id in selected_ids], dtype=float) / max(len(sequence), 1)
    mixed_raw = raw_matrix @ actual_ratio
    mixed_min = corrected_ber(mixed_raw)
    worst_idx = int(np.argmin(mixed_min))
    route_keys = selected[0].route_keys
    append_rows(path, schedule_fields(), [{
        "position_combination": str(tuple(combo)),
        "k": len(combo),
        "num_probes": len(probes),
        "eval_bits": int(eval_bits),
        "mapping_candidate_count": int(mapping_candidate_count),
        "selected_mapping_count": len(selected),
        "schedule_length": int(schedule_length),
        "optimizer": optimizer,
        "selected_mapping_ids": list_text(selected_ids),
        "usage_ratio": float_list_text(actual_ratio),
        "usage_counts": list_text([int(v) for v in counts]),
        "offline_objective_min_route_min_ber": f"{float(np.min(mixed_min)):.8f}",
        "offline_average_route_min_ber": f"{float(np.mean(mixed_min)):.8f}",
        "worst_route": f"{route_keys[worst_idx][0]}->{route_keys[worst_idx][1]}",
        "schedule_sequence": list_text(sequence),
        "route_raw_bers": route_dict_text(route_keys, mixed_raw),
        "route_min_bers": route_dict_text(route_keys, mixed_min),
    }])
    print(
        f"      Schedule objective={objective:.6f}, actual_min={float(np.min(mixed_min)):.6f}, "
        f"actual_avg={float(np.mean(mixed_min)):.6f}, worst={route_keys[worst_idx][0]}->{route_keys[worst_idx][1]}"
    )


def run(args: argparse.Namespace) -> tuple[str, str, str]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = resolve_dataset_dir(project_root, args.dataset_dir)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_csv = args.candidate_csv or os.path.join(output_dir, DEFAULT_MAPPING_CANDIDATE_CSV)
    selected_csv = args.selected_csv or os.path.join(output_dir, DEFAULT_SELECTED_MAPPING_CSV)
    schedule_csv = args.schedule_csv or os.path.join(output_dir, DEFAULT_SCHEDULE_CSV)
    if args.overwrite:
        reset_files([candidate_csv, selected_csv, schedule_csv])

    position_files = vh.list_position_files(dataset_dir)
    position_files, invalid_positions = vh.filter_position_files_valid_for_probes(position_files, args.num_probes)
    if invalid_positions:
        print(f"Filtered invalid positions for num_probes={args.num_probes}: {invalid_positions}")
    sampled_sets = sample_legal_position_sets(
        position_files=position_files,
        k=args.k,
        sample_size=args.sample_size,
        rng=np.random.default_rng(args.random_seed),
    )
    print(f"Dataset: {dataset_dir}")
    print(f"Scheme A vector-hue mapping schedule: k={args.k}, sampled_sets={len(sampled_sets)}")

    for combo_idx, tuple_items in enumerate(sampled_sets, start=1):
        legal_positions = tuple(int(item[0]) for item in tuple_items)
        print(f"[{combo_idx}/{len(sampled_sets)}] legal_positions={legal_positions}")
        models = vh._load_models_for_target(
            tuple_items=tuple_items,
            num_probes=args.num_probes,
            target_subset_size=args.num_probes,
        )
        probes = np.asarray(models[0].probes, dtype=float)
        bit_blocks_pm = vh.generate_random_bit_blocks(
            num_positions=args.k,
            num_blocks=args.eval_bits,
            rng=np.random.default_rng(args.random_seed + combo_idx * 1009),
        )
        mappings = build_vector_mapping_candidates(
            models=models,
            candidate_count=args.mapping_count,
            vector_top_k=args.vector_top_k,
            rng=random.Random(args.random_seed + combo_idx * 9173),
        )
        candidates: list[MappingCandidate] = []
        seen_signatures: set[tuple[tuple[tuple, tuple[int, int, int, int]], ...]] = set()
        for mapping_idx, mapping in enumerate(mappings, start=1):
            seen_signatures.add(mapping_signature(mapping))
            candidate = evaluate_mapping_candidate(
                dataset_dir=dataset_dir,
                position_files=position_files,
                legal_positions=legal_positions,
                probes=probes,
                models=models,
                mapping_id=mapping_idx,
                source="base",
                hue_mapping=mapping,
                bit_blocks_pm=bit_blocks_pm,
            )
            print(
                f"      Mapping {mapping_idx}/{len(mappings)}: auth_max={candidate.authorized_max_ber:.6f}, "
                f"route_min={candidate.min_route_min_ber:.6f}, route_avg={candidate.average_route_min_ber:.6f}"
            )
            if candidate.authorized_max_ber <= args.authorized_ber_threshold:
                candidates.append(candidate)

        if not candidates:
            print(f"      No mappings satisfy authorized BER <= {args.authorized_ber_threshold}.")
            continue
        if args.repair_mapping_count > 0:
            weak_routes = identify_weak_routes(candidates, args.weak_route_count)
            print(
                "      Weak routes for repair: "
                + ", ".join(f"{route[0]}->{route[1]}" for route in weak_routes)
            )
            repair_mappings = build_vector_mapping_candidates(
                models=models,
                candidate_count=args.repair_mapping_count,
                vector_top_k=args.repair_vector_top_k,
                rng=random.Random(args.random_seed + combo_idx * 19001),
            )
            repair_candidates: list[MappingCandidate] = []
            next_mapping_id = max(candidate.mapping_id for candidate in candidates) + 1
            for repair_mapping in repair_mappings:
                signature = mapping_signature(repair_mapping)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                candidate = evaluate_mapping_candidate(
                    dataset_dir=dataset_dir,
                    position_files=position_files,
                    legal_positions=legal_positions,
                    probes=probes,
                    models=models,
                    mapping_id=next_mapping_id,
                    source="repair",
                    hue_mapping=repair_mapping,
                    bit_blocks_pm=bit_blocks_pm,
                )
                next_mapping_id += 1
                if candidate.authorized_max_ber <= args.authorized_ber_threshold:
                    repair_candidates.append(candidate)
            repair_candidates.sort(key=lambda candidate: repair_score(candidate, weak_routes), reverse=True)
            kept_repair_candidates = repair_candidates[:args.repair_keep_count]
            if kept_repair_candidates:
                best_repair = kept_repair_candidates[0]
                weak_indices = route_indices(best_repair.route_keys, weak_routes)
                best_weak_min = float(np.min(best_repair.route_min_bers[weak_indices])) if weak_indices else best_repair.min_route_min_ber
                print(
                    f"      Keep {len(kept_repair_candidates)}/{len(repair_candidates)} legal repair mappings; "
                    f"best_repair_weak_min={best_weak_min:.6f}, "
                    f"best_repair_global_min={best_repair.min_route_min_ber:.6f}"
                )
                candidates.extend(kept_repair_candidates)
            else:
                print("      No legal repair mappings kept.")

        write_candidates(candidate_csv, legal_positions, probes, candidates)
        selected_count = min(args.selected_mapping_count, len(candidates))
        print(f"      Select {selected_count} security-complementary mappings from {len(candidates)} legal mappings")
        selected = select_security_complementary_mappings(candidates, selected_count)
        write_selected(selected_csv, legal_positions, selected)
        write_schedule(
            schedule_csv,
            legal_positions,
            probes,
            eval_bits=args.eval_bits,
            mapping_candidate_count=len(candidates),
            selected=selected,
            schedule_length=args.schedule_length,
        )

    print(f"Mapping candidates saved to: {candidate_csv}")
    print(f"Selected mappings saved to: {selected_csv}")
    print(f"Mapping schedule saved to: {schedule_csv}")
    return candidate_csv, selected_csv, schedule_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline schedule multiple vector-hue mappings on fixed probes.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--num-probes", type=int, default=DEFAULT_NUM_PROBES)
    parser.add_argument("--mapping-count", type=int, default=DEFAULT_MAPPING_COUNT)
    parser.add_argument("--selected-mapping-count", type=int, default=DEFAULT_SELECTED_MAPPING_COUNT)
    parser.add_argument("--vector-top-k", type=int, default=DEFAULT_VECTOR_TOP_K)
    parser.add_argument("--repair-mapping-count", type=int, default=DEFAULT_REPAIR_MAPPING_COUNT)
    parser.add_argument("--repair-keep-count", type=int, default=DEFAULT_REPAIR_KEEP_COUNT)
    parser.add_argument("--repair-vector-top-k", type=int, default=DEFAULT_REPAIR_VECTOR_TOP_K)
    parser.add_argument("--weak-route-count", type=int, default=DEFAULT_WEAK_ROUTE_COUNT)
    parser.add_argument("--eval-bits", type=int, default=DEFAULT_EVAL_BITS)
    parser.add_argument("--schedule-length", type=int, default=DEFAULT_SCHEDULE_LENGTH)
    parser.add_argument("--authorized-ber-threshold", type=float, default=DEFAULT_AUTHORIZED_BER_THRESHOLD)
    parser.add_argument("--random-seed", type=int, default=SEED)
    parser.add_argument("--candidate-csv", default=None)
    parser.add_argument("--selected-csv", default=None)
    parser.add_argument("--schedule-csv", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
