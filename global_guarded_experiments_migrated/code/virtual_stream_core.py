#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact per-k vector-hue experiments with key-controlled virtual streams."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import targeted_probe_subset_repair_yellow_shuffled as base


@dataclass
class ExperimentConfig:
    real_k: int
    output_dir: str
    target_effective_k: int = 8
    sample_size: int = 100
    num_probes: int = 15
    probe_subset_count: int = 200
    base_mapping_count: int = 20
    mappings_per_subset: int = 8
    selected_count: int = 10
    vector_top_k: int = 4
    eval_bits: int = 1000
    schedule_length: int = 1000
    authorized_ber_threshold: float = 0.0
    weak_route_count: int = 5
    random_seed: int = 20260513
    dataset_dir: str = base.DEFAULT_DATASET_DIR
    overwrite: bool = False


@dataclass
class Candidate:
    candidate_id: int
    source: str
    probes: np.ndarray
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
class AlignmentInfo:
    common_route_count: int
    excluded_illegal_positions: list[int]


def virtual_count_for_k(real_k: int, target_effective_k: int) -> int:
    return max(0, int(target_effective_k) - int(real_k))


def effective_k_for_config(config: ExperimentConfig) -> int:
    return config.real_k + virtual_count_for_k(config.real_k, config.target_effective_k)


def list_text(values: Sequence[object]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def float_list_text(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.8f}" for v in values) + "]"


def append_rows(path: str, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def reset_outputs(output_dir: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, "results_summary.csv")
    selected_csv = os.path.join(output_dir, "results_selected.csv")
    for path in (summary_csv, selected_csv):
        if os.path.exists(path):
            os.remove(path)
    return summary_csv, selected_csv


def output_paths(output_dir: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    return (
        os.path.join(output_dir, "results_summary.csv"),
        os.path.join(output_dir, "results_selected.csv"),
    )


def normalize_position_combination(value: str) -> tuple[int, ...] | None:
    try:
        cleaned = value.strip().strip("()").replace(" ", "")
        if not cleaned:
            return None
        return tuple(int(part) for part in cleaned.split(",") if part != "")
    except Exception:
        return None


def load_completed_position_combinations(summary_csv: str) -> set[tuple[int, ...]]:
    if not os.path.exists(summary_csv):
        return set()
    completed: set[tuple[int, ...]] = set()
    with open(summary_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            combo = normalize_position_combination(row.get("position_combination", ""))
            if combo is not None:
                completed.add(combo)
    return completed


def expand_models_with_virtual(real_models: Sequence[object], virtual_count: int) -> list[object]:
    return list(real_models)


def generate_effective_bit_blocks(
    real_k: int,
    virtual_count: int,
    num_blocks: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    real_blocks = base.vh.generate_random_bit_blocks(real_k, num_blocks, rng)
    if virtual_count <= 0:
        return real_blocks
    expanded = []
    for real_bits in real_blocks:
        virtual_bits_bin = rng.integers(0, 2, size=virtual_count)
        virtual_bits_pm = np.where(virtual_bits_bin > 0, 1, -1).astype(int)
        expanded.append(np.concatenate([np.asarray(real_bits, dtype=int), virtual_bits_pm]))
    return expanded


def full_codes_for_virtual(real_models: Sequence[object], virtual_count: int) -> list[np.ndarray]:
    real_codes = [np.asarray(model.code, dtype=int) for model in real_models]
    codes = list(real_codes)
    for idx in range(virtual_count):
        codes.append(real_codes[idx % len(real_codes)])
    return codes


def build_virtual_mapping_candidates(
    real_models: Sequence[object],
    candidate_count: int,
    vector_top_k: int,
    virtual_count: int,
    rng: random.Random,
) -> list[dict[tuple, tuple[int, int, int, int]]]:
    real_candidate_dict = base.vh._build_vector_candidate_dict(list(real_models), top_k=vector_top_k)
    real_k = len(real_models)
    effective_k = real_k + virtual_count
    full_keys = [tuple(bits) for bits in product([1, -1], repeat=effective_k)]
    mappings: list[dict[tuple, tuple[int, int, int, int]]] = []
    seen: set[tuple[tuple[tuple, tuple[int, int, int, int]], ...]] = set()

    top1 = {key: real_candidate_dict[key[:real_k]][0] for key in full_keys}
    mappings.append(top1)
    seen.add(base.mapping_signature(top1))

    attempts = 0
    max_attempts = max(1000, candidate_count * 100)
    while len(mappings) < candidate_count and attempts < max_attempts:
        attempts += 1
        mapping = {}
        for key in full_keys:
            values = real_candidate_dict[key[:real_k]]
            mapping[key] = rng.choice(values)
        signature = base.mapping_signature(mapping)
        if signature in seen:
            continue
        seen.add(signature)
        mappings.append(mapping)
    return mappings


def simulate_blocks_vector_with_virtual(
    real_models: Sequence[object],
    bit_blocks_pm: Sequence[np.ndarray],
    hue_mapping: dict[tuple, tuple[int, int, int, int]],
    virtual_count: int,
) -> list[dict]:
    real_models = list(real_models)
    codes = full_codes_for_virtual(real_models, virtual_count)
    probe_to_rows = [base.vh.build_probe_to_row(model.probes) for model in real_models]
    results = []
    for bits_pm in bit_blocks_pm:
        bits_pm = np.asarray(bits_pm, dtype=int)
        symbol_combinations = base.vh._build_symbol_combinations(bits_pm, codes)
        hue_vec_seq = np.asarray([hue_mapping[key] for key in symbol_combinations], dtype=int)
        block_info = {
            "bits_pm": bits_pm,
            "bits_bin": base.vh.pm1_to_bin(bits_pm),
            "hue_seq_vector": hue_vec_seq,
            "per_position": [],
        }
        for rx_idx, model in enumerate(real_models):
            y_obs = base.vh.observe_block_from_vector_mapping(hue_vec_seq, model.Y, probe_to_rows[rx_idx])
            dec = base.vh.decode_local_block(y_obs, model.w, model.code)
            block_info["per_position"].append(dec)
        results.append(block_info)
    return results


def authorized_position_bers_real(results: Sequence[dict], real_k: int) -> list[float]:
    errors = np.zeros(real_k, dtype=float)
    totals = np.zeros(real_k, dtype=float)
    for block in results:
        true_bits = np.asarray(block["bits_bin"], dtype=int)[:real_k]
        decoded_bits = [decode.bit_hat_bin for decode in block["per_position"][:real_k]]
        for idx, (true_bit, decoded_bit) in enumerate(zip(true_bits, decoded_bits)):
            totals[idx] += 1
            if int(true_bit) != int(decoded_bit):
                errors[idx] += 1
    return (errors / np.maximum(totals, 1.0)).tolist()


def truth_bits_real(results: Sequence[dict], real_k: int) -> list[list[int]]:
    streams = [[] for _ in range(real_k)]
    for block in results:
        bits = np.asarray(block["bits_bin"], dtype=int)[:real_k]
        for idx, bit in enumerate(bits):
            streams[idx].append(int(bit))
    return streams


def evaluate_candidate(
    position_files: Sequence[tuple[int, str]],
    legal_positions: Sequence[int],
    probes: np.ndarray,
    models: Sequence[object],
    real_k: int,
    virtual_count: int,
    candidate_id: int,
    source: str,
    hue_mapping: dict[tuple, tuple[int, int, int, int]],
    bit_blocks_pm: Sequence[np.ndarray],
) -> Candidate:
    results = simulate_blocks_vector_with_virtual(list(models), list(bit_blocks_pm), hue_mapping, virtual_count)
    legal_bers = authorized_position_bers_real(results, real_k)
    true_stream_bits = truth_bits_real(results, real_k)
    legal_set = set(int(v) for v in legal_positions)
    file_map = {int(position): path for position, path in position_files}
    route_keys: list[tuple[int, int]] = []
    route_raw_bers: list[float] = []

    for illegal_position in sorted(file_map):
        if illegal_position in legal_set:
            continue
        try:
            illegal_model = base.vh.load_single_position_model(file_map[illegal_position], probes=probes)
            probe_to_row = base.vh.build_probe_to_row(illegal_model.probes)
            decoded_bits = []
            for block in results:
                y_obs = base.vh.observe_block_from_vector_mapping(block["hue_seq_vector"], illegal_model.Y, probe_to_row)
                dec = base.vh.decode_local_block(y_obs, illegal_model.w, illegal_model.code)
                decoded_bits.append(int(dec.bit_hat_bin))
            decoded = np.asarray(decoded_bits, dtype=int)
            for stream_idx, legal_position in enumerate(legal_positions):
                truth = np.asarray(true_stream_bits[stream_idx], dtype=int)
                route_keys.append((int(illegal_position), int(legal_position)))
                route_raw_bers.append(float(np.mean(decoded != truth)))
        except Exception:
            continue

    raw = np.asarray(route_raw_bers, dtype=float)
    return Candidate(
        candidate_id=candidate_id,
        source=source,
        probes=np.asarray(probes, dtype=float),
        authorized_position_bers=legal_bers,
        route_keys=route_keys,
        route_raw_bers=raw,
        route_min_bers=base.corrected_ber(raw),
    )


def align_candidates(
    candidates: Sequence[Candidate],
    position_files: Sequence[tuple[int, str]],
    legal_positions: Sequence[int],
) -> tuple[list[Candidate], AlignmentInfo]:
    if not candidates:
        return [], AlignmentInfo(0, [])
    common = set(candidates[0].route_keys)
    for candidate in candidates[1:]:
        common &= set(candidate.route_keys)
    common_keys = [route for route in candidates[0].route_keys if route in common]
    legal_set = set(int(v) for v in legal_positions)
    expected_illegals = {int(position) for position, _ in position_files if int(position) not in legal_set}
    common_illegals = {int(route[0]) for route in common_keys}
    info = AlignmentInfo(len(common_keys), sorted(expected_illegals - common_illegals))
    aligned = []
    for candidate in candidates:
        index = {route: idx for idx, route in enumerate(candidate.route_keys)}
        indices = [index[route] for route in common_keys]
        aligned.append(Candidate(
            candidate.candidate_id,
            candidate.source,
            candidate.probes,
            candidate.authorized_position_bers,
            list(common_keys),
            candidate.route_raw_bers[indices],
            candidate.route_min_bers[indices],
        ))
    return aligned, info


def identify_weak_routes(candidates: Sequence[Candidate], weak_route_count: int) -> list[tuple[int, int]]:
    route_keys = candidates[0].route_keys
    route_matrix = np.asarray([candidate.route_min_bers for candidate in candidates], dtype=float)
    route_best = np.max(route_matrix, axis=0)
    order = np.argsort(route_best)
    return [route_keys[int(idx)] for idx in order[:max(1, weak_route_count)]]


def route_indices(route_keys: Sequence[tuple[int, int]], routes: Sequence[tuple[int, int]]) -> list[int]:
    index = {route: idx for idx, route in enumerate(route_keys)}
    return [index[route] for route in routes if route in index]


def select_candidates(candidates: Sequence[Candidate], selected_count: int, weak_routes: Sequence[tuple[int, int]]) -> list[Candidate]:
    def score(candidate: Candidate) -> tuple[float, float, float]:
        indices = route_indices(candidate.route_keys, weak_routes)
        weak_min = float(np.min(candidate.route_min_bers[indices])) if indices else candidate.min_route_min_ber
        return weak_min, candidate.min_route_min_ber, candidate.average_route_min_ber
    return sorted(candidates, key=score, reverse=True)[:selected_count]


def summary_fields() -> list[str]:
    return [
        "k", "real_k", "effective_k", "virtual_stream_count", "target_effective_k",
        "sample_index", "position_combination", "selected_candidate_count", "common_route_count",
        "excluded_illegal_positions", "authorized_max_ber", "authorized_position_bers",
        "security_min_route_min_ber", "security_avg_route_min_ber", "worst_route",
        "optimizer", "usage_ratio", "usage_counts", "selected_candidate_ids", "selected_probe_sets",
    ]


def selected_fields() -> list[str]:
    return [
        "k", "real_k", "effective_k", "virtual_stream_count", "sample_index",
        "position_combination", "selection_rank", "candidate_id", "source", "probes",
        "authorized_max_ber", "authorized_position_bers", "candidate_min_route_min_ber",
        "candidate_avg_route_min_ber", "worst_route",
    ]


def write_compact_outputs(
    summary_csv: str,
    selected_csv: str,
    config: ExperimentConfig,
    sample_index: int,
    legal_positions: Sequence[int],
    selected: Sequence[Candidate],
    alignment_info: AlignmentInfo,
) -> None:
    raw_matrix = np.asarray([candidate.route_raw_bers for candidate in selected], dtype=float).T
    ratio, _, optimizer = base.optimize_usage_ratio(raw_matrix)
    mixed_raw = raw_matrix @ ratio
    mixed_min = base.corrected_ber(mixed_raw)
    worst_idx = int(np.argmin(mixed_min))
    route_keys = selected[0].route_keys
    counts = np.floor(ratio * config.schedule_length).astype(int)
    remainder = int(config.schedule_length) - int(np.sum(counts))
    if remainder > 0:
        for idx in np.argsort(-(ratio * config.schedule_length - counts))[:remainder]:
            counts[idx] += 1
    auth_max = max(candidate.authorized_max_ber for candidate in selected)
    auth_by_position = np.max(np.asarray([candidate.authorized_position_bers for candidate in selected], dtype=float), axis=0)
    effective_k = effective_k_for_config(config)
    virtual_count = virtual_count_for_k(config.real_k, config.target_effective_k)
    append_rows(summary_csv, summary_fields(), [{
        "k": config.real_k,
        "real_k": config.real_k,
        "effective_k": effective_k,
        "virtual_stream_count": virtual_count,
        "target_effective_k": config.target_effective_k,
        "sample_index": sample_index,
        "position_combination": str(tuple(legal_positions)),
        "selected_candidate_count": len(selected),
        "common_route_count": alignment_info.common_route_count,
        "excluded_illegal_positions": list_text(alignment_info.excluded_illegal_positions),
        "authorized_max_ber": f"{auth_max:.8f}",
        "authorized_position_bers": float_list_text(auth_by_position),
        "security_min_route_min_ber": f"{float(np.min(mixed_min)):.8f}",
        "security_avg_route_min_ber": f"{float(np.mean(mixed_min)):.8f}",
        "worst_route": f"{route_keys[worst_idx][0]}->{route_keys[worst_idx][1]}",
        "optimizer": optimizer,
        "usage_ratio": float_list_text(ratio),
        "usage_counts": list_text([int(v) for v in counts]),
        "selected_candidate_ids": list_text([candidate.candidate_id for candidate in selected]),
        "selected_probe_sets": " | ".join(list_text([int(v) for v in candidate.probes]) for candidate in selected),
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
            "probes": list_text([int(v) for v in candidate.probes]),
            "authorized_max_ber": f"{candidate.authorized_max_ber:.8f}",
            "authorized_position_bers": float_list_text(candidate.authorized_position_bers),
            "candidate_min_route_min_ber": f"{candidate.min_route_min_ber:.8f}",
            "candidate_avg_route_min_ber": f"{candidate.average_route_min_ber:.8f}",
            "worst_route": f"{candidate.worst_route[0]}->{candidate.worst_route[1]}",
        })
    append_rows(selected_csv, selected_fields(), rows)
    print(
        f"      security_min={float(np.min(mixed_min)):.6f}, auth_max={auth_max:.6f}, "
        f"worst={route_keys[worst_idx][0]}->{route_keys[worst_idx][1]}, common_routes={alignment_info.common_route_count}"
    )


def run_k_experiment(config: ExperimentConfig) -> tuple[str, str]:
    project_root = os.path.dirname(BASE_DIR)
    dataset_dir = base.resolve_dataset_dir(project_root, config.dataset_dir)
    summary_csv, selected_csv = reset_outputs(config.output_dir) if config.overwrite else output_paths(config.output_dir)
    completed_combos = load_completed_position_combinations(summary_csv)
    position_files = base.vh.list_position_files(dataset_dir)
    position_files, invalid_positions = base.vh.filter_position_files_valid_for_probes(position_files, config.num_probes)
    if invalid_positions:
        print(f"Filtered invalid baseline positions: {invalid_positions}")
    sampled_sets = base.vh.sample_tuples(
        position_files=position_files,
        k=config.real_k,
        sample_size=config.sample_size,
        min_position_distance=0.0,
        rng=np.random.default_rng(config.random_seed + config.real_k * 100003),
    )
    virtual_count = virtual_count_for_k(config.real_k, config.target_effective_k)
    effective_k = effective_k_for_config(config)
    print(
        f"k={config.real_k}, effective_k={effective_k}, virtual_streams={virtual_count}, "
        f"samples={len(sampled_sets)}, completed={len(completed_combos)}, output={config.output_dir}"
    )
    for sample_index, tuple_items in enumerate(sampled_sets, start=1):
        legal_positions = tuple(int(item[0]) for item in tuple_items)
        if legal_positions in completed_combos:
            print(f"[{sample_index}/{len(sampled_sets)}] legal_positions={legal_positions} skipped existing")
            continue
        print(f"[{sample_index}/{len(sampled_sets)}] legal_positions={legal_positions}")
        bit_blocks_pm = generate_effective_bit_blocks(
            real_k=config.real_k,
            virtual_count=virtual_count,
            num_blocks=config.eval_bits,
            rng=np.random.default_rng(config.random_seed + config.real_k * 100003 + sample_index * 1009),
        )
        probe_subsets = base.generate_probe_subsets(
            position_files=position_files,
            num_probes=config.num_probes,
            subset_count=config.probe_subset_count,
            rng=np.random.default_rng(config.random_seed + config.real_k * 100003 + sample_index * 7919),
        )
        candidates: list[Candidate] = []
        candidate_id = 1
        for subset_idx, probes in enumerate(probe_subsets, start=1):
            real_models = base.load_legal_models(tuple_items, probes)
            if real_models is None:
                continue
            models = expand_models_with_virtual(real_models, virtual_count)
            mappings = build_virtual_mapping_candidates(
                real_models=models,
                candidate_count=config.mappings_per_subset if subset_idx > 1 else config.base_mapping_count,
                vector_top_k=config.vector_top_k,
                virtual_count=virtual_count,
                rng=random.Random(config.random_seed + config.real_k * 100003 + sample_index * 9173 + subset_idx),
            )
            legal_count = 0
            best_min = -1.0
            for mapping in mappings:
                candidate = evaluate_candidate(
                    position_files=position_files,
                    legal_positions=legal_positions,
                    probes=probes,
                    models=models,
                    real_k=config.real_k,
                    virtual_count=virtual_count,
                    candidate_id=candidate_id,
                    source="baseline" if subset_idx == 1 else "probe_repair",
                    hue_mapping=mapping,
                    bit_blocks_pm=bit_blocks_pm,
                )
                candidate_id += 1
                if candidate.authorized_max_ber <= config.authorized_ber_threshold:
                    candidates.append(candidate)
                    legal_count += 1
                    best_min = max(best_min, candidate.min_route_min_ber)
            print(f"      subset {subset_idx}/{len(probe_subsets)}: legal={legal_count}, best_min={best_min:.6f}")
        if not candidates:
            print("      no legal candidates")
            continue
        candidates, alignment_info = align_candidates(candidates, position_files, legal_positions)
        if not candidates or not candidates[0].route_keys:
            print("      no common routes")
            continue
        weak_routes = identify_weak_routes(candidates, config.weak_route_count)
        selected = select_candidates(candidates, min(config.selected_count, len(candidates)), weak_routes)
        write_compact_outputs(summary_csv, selected_csv, config, sample_index, legal_positions, selected, alignment_info)
    return summary_csv, selected_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one compact virtual-stream vector-hue k experiment.")
    parser.add_argument("--real-k", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-effective-k", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--probe-subset-count", type=int, default=200)
    parser.add_argument("--base-mapping-count", type=int, default=20)
    parser.add_argument("--mappings-per-subset", type=int, default=8)
    parser.add_argument("--selected-count", type=int, default=10)
    parser.add_argument("--eval-bits", type=int, default=1000)
    parser.add_argument("--schedule-length", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(
        real_k=args.real_k,
        output_dir=args.output_dir,
        target_effective_k=args.target_effective_k,
        sample_size=args.sample_size,
        probe_subset_count=args.probe_subset_count,
        base_mapping_count=args.base_mapping_count,
        mappings_per_subset=args.mappings_per_subset,
        selected_count=args.selected_count,
        eval_bits=args.eval_bits,
        schedule_length=args.schedule_length,
        overwrite=True if args.overwrite else False,
    )
    run_k_experiment(config)


if __name__ == "__main__":
    main()
