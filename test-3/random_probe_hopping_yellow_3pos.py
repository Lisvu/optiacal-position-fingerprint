#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random probe hopping experiment for 3 legal positions on data/15pro/yellow.

For each sampled legal-position combination:
1. Search about K legal-reliable probe sets with the same probe count M.
2. Each probe set has its own hue_mapping and must decode legal devices with BER=0.
3. Save the legal-reliable probe pool to a CSV file.
4. Evaluate random probe hopping: each bit block randomly selects one probe set from
   the pool, then uses that set's hue_mapping for transmission and decoding.
5. Save per-illegal-position BER against every legal position to another CSV file.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import os
import random
import sys
import types
from typing import Sequence

import numpy as np
import pandas as pd


LIGHT_CONDITION = "yellow"
LEGAL_POSITION_COUNT = 3
DEFAULT_COMBINATION_COUNT = 20
TARGET_POOL_SIZE = 5
PROBE_COUNT_CANDIDATES = [12, 14, 16, 18, 20]
MAX_CANDIDATES_PER_PROBE_COUNT = 5000
MAX_STAGNANT_CANDIDATES_PER_PROBE_COUNT = 1200
LEGAL_SEARCH_BITS = 1000
HOPPING_EVAL_BITS = 1000
POOL_LEAK_EVAL_BITS = 1000
MAPPING_EVAL_BITS = 500
MAPPING_TOP_K = 3
MIN_PROBE_INTERVAL = 5
MAX_EVALUATED_PROBE_OVERLAP_RATIO = 0.75
LEAK_SECURE_BER_THRESHOLD = 0.0
MAX_ZERO_LEAKS_PER_ROUTE = 2
MIN_POOL_ROUTE_AVERAGE_SECURE_BER = 0.2
SELECTION_SEED = 20260509
OUTPUT_PROBE_POOL_FILENAME = "yellow_random_probe_hopping_probe_pool_3pos.csv"
OUTPUT_SECURITY_FILENAME = "yellow_random_probe_hopping_security_eval_3pos.csv"


class ProbePoolEntry:
    def __init__(
        self,
        entry_id: int,
        probes: np.ndarray,
        legal_models: list,
        hue_mapping: dict[tuple[int, ...], int],
        legal_position_bers: list[float],
    ) -> None:
        self.entry_id = entry_id
        self.probes = probes
        self.legal_models = legal_models
        self.hue_mapping = hue_mapping
        self.legal_position_bers = legal_position_bers
        self.route_secure_bers: dict[tuple[int, int], float] | None = None


def load_test_module() -> types.ModuleType:
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_3_simple.py")
    with open(module_path, "r", encoding="utf-8-sig") as f:
        source = f.read().lstrip("\ufeff")
    module = types.ModuleType("test_3_simple_runtime_random_hopping_yellow_3pos")
    module.__file__ = module_path
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(source, module_path, "exec"), module.__dict__)
    return module


test = load_test_module()


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def format_hue_mapping(hue_mapping: dict[tuple[int, ...], int]) -> str:
    return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(hue_mapping.items())) + "}"


def parse_positions(text: str) -> tuple[int, ...]:
    value = ast.literal_eval(text)
    if not isinstance(value, tuple) or len(value) != LEGAL_POSITION_COUNT:
        raise ValueError(f"Invalid 3-position combination: {text}")
    return tuple(int(v) for v in value)


def get_available_positions(project_root: str) -> list[int]:
    data_dir = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION)
    positions = []
    for entry in os.listdir(data_dir):
        stem, ext = os.path.splitext(entry)
        if ext == ".csv" and stem.isdigit():
            positions.append(int(stem))
    return sorted(positions)


def build_csv_files(project_root: str, positions: Sequence[int]) -> list[str]:
    return test.build_csv_files_for_positions(project_root, positions, light_condition=LIGHT_CONDITION)


def get_all_probes(project_root: str, legal_positions: Sequence[int]) -> list[float]:
    first_csv = build_csv_files(project_root, legal_positions)[0]
    first_df = pd.read_csv(first_csv)
    return (5 + np.arange(len(first_df)) * 5).astype(float).tolist()


def min_interval_for_probe_count(probe_count: int) -> int:
    return MIN_PROBE_INTERVAL


def sample_valid_probe_set(
    all_probes: Sequence[float],
    probe_count: int,
    min_interval: int,
    rng: random.Random,
) -> np.ndarray | None:
    probes = np.sort(np.asarray(all_probes, dtype=float))
    if len(probes) < probe_count:
        return None
    step = float(np.median(np.diff(probes))) if len(probes) > 1 else 5.0
    min_index_gap = int(np.ceil(float(min_interval) / step))
    adjusted_count = len(probes) - (probe_count - 1) * min_index_gap
    if adjusted_count <= 0:
        return None

    offsets = sorted(rng.randrange(adjusted_count) for _ in range(probe_count))
    indices = [offset + idx * min_index_gap for idx, offset in enumerate(offsets)]
    selected = probes[indices]
    if not test.is_valid_probe_set(selected, min_interval=min_interval):
        return None
    return np.sort(np.asarray(selected, dtype=float))


def is_diverse_probe_set(
    probes: Sequence[float],
    evaluated_probe_sets: Sequence[set[float]],
) -> bool:
    if not evaluated_probe_sets:
        return True
    probe_set = {float(probe) for probe in probes}
    max_allowed_overlap = int(np.floor(len(probe_set) * MAX_EVALUATED_PROBE_OVERLAP_RATIO))
    return all(len(probe_set & old_probe_set) <= max_allowed_overlap for old_probe_set in evaluated_probe_sets)


def corrected_ber(ber: float) -> float:
    return min(float(ber), 1.0 - float(ber))


def calculate_position_bers(results: Sequence[dict]) -> list[float]:
    position_errors = np.zeros(len(results[0]["per_position"]), dtype=float)
    position_total = np.zeros(len(results[0]["per_position"]), dtype=float)
    for result in results:
        true_bits = result["bits_bin"]
        decoded_bits = [decode.bit_hat_bin for decode in result["per_position"]]
        for idx, (true_bit, decoded_bit) in enumerate(zip(true_bits, decoded_bits)):
            position_total[idx] += 1
            if int(true_bit) != int(decoded_bit):
                position_errors[idx] += 1
    return (position_errors / np.maximum(position_total, 1.0)).tolist()


def legal_position_bers_for_entry(entry: ProbePoolEntry, bit_blocks_pm: Sequence[np.ndarray]) -> list[float]:
    results = test.simulate_blocks(entry.legal_models, list(bit_blocks_pm), entry.hue_mapping)
    return [float(v) for v in calculate_position_bers(results)]


def evaluate_entry_route_secure_bers(
    project_root: str,
    legal_positions: Sequence[int],
    entry: ProbePoolEntry,
    bit_blocks_pm: Sequence[np.ndarray],
) -> dict[tuple[int, int], float]:
    route_secure_bers: dict[tuple[int, int], float] = {}
    all_positions = get_available_positions(project_root)
    illegal_positions = [position for position in all_positions if position not in legal_positions]
    legal_codes = [model.code for model in entry.legal_models]

    for illegal_position in illegal_positions:
        illegal_csv = build_csv_files(project_root, [illegal_position])[0]
        illegal_matrix = test.load_selected_rows([illegal_csv], entry.probes)[0]
        illegal_model = test.extract_fingerprint(entry.probes, illegal_matrix, force_positive_first=True)
        probe_to_row = test.build_probe_to_row(entry.probes)
        position_errors = np.zeros(len(legal_positions), dtype=float)
        position_total = np.zeros(len(legal_positions), dtype=float)

        for bits_pm in bit_blocks_pm:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, entry.hue_mapping)
            illegal_observation = test.observe_block_from_measured_matrix(hue_seq, illegal_matrix, probe_to_row)
            illegal_dec = test.decode_local_block(illegal_observation, illegal_model.w, illegal_model.code)
            true_bits = test.pm1_to_bin(bits_pm)
            for idx, true_bit in enumerate(true_bits):
                position_total[idx] += 1
                if int(illegal_dec.bit_hat_bin) != int(true_bit):
                    position_errors[idx] += 1

        raw_bers = (position_errors / np.maximum(position_total, 1.0)).tolist()
        secure_bers = [corrected_ber(ber) for ber in raw_bers]
        for idx, secure_ber in enumerate(secure_bers):
            route_secure_bers[(int(illegal_position), int(legal_positions[idx]))] = float(secure_ber)
    return route_secure_bers


def evaluate_entry_leak_routes(
    project_root: str,
    legal_positions: Sequence[int],
    entry: ProbePoolEntry,
    bit_blocks_pm: Sequence[np.ndarray],
) -> set[tuple[int, int]]:
    route_secure_bers = evaluate_entry_route_secure_bers(project_root, legal_positions, entry, bit_blocks_pm)
    return {
        route
        for route, secure_ber in route_secure_bers.items()
        if float(secure_ber) <= LEAK_SECURE_BER_THRESHOLD
    }


def remove_one_pool_constraint_violating_entry(
    project_root: str,
    legal_positions: Sequence[int],
    entries: list[ProbePoolEntry],
    leak_eval_blocks: Sequence[np.ndarray],
) -> bool:
    if not entries:
        return False

    secure_bers_by_entry = {}
    for entry in entries:
        if entry.route_secure_bers is None:
            entry.route_secure_bers = evaluate_entry_route_secure_bers(project_root, legal_positions, entry, leak_eval_blocks)
        secure_bers_by_entry[entry.entry_id] = entry.route_secure_bers
    all_routes = set().union(*secure_bers_by_entry.values()) if secure_bers_by_entry else set()
    violating_routes: list[tuple[tuple[int, int], int, float, float]] = []
    for route in sorted(all_routes):
        values = [secure_bers_by_entry[entry.entry_id][route] for entry in entries]
        zero_count = sum(1 for value in values if float(value) <= LEAK_SECURE_BER_THRESHOLD)
        average_secure_ber = float(np.mean(values))
        min_secure_ber = float(min(values))
        if (
            zero_count > MAX_ZERO_LEAKS_PER_ROUTE
            or average_secure_ber < MIN_POOL_ROUTE_AVERAGE_SECURE_BER
        ):
            violating_routes.append((route, zero_count, average_secure_ber, min_secure_ber))

    if not violating_routes:
        return False

    violating_route_set = {route for route, _, _, _ in violating_routes}
    remove_entry = min(
        entries,
        key=lambda entry: (
            sum(secure_bers_by_entry[entry.entry_id][route] for route in violating_route_set),
            -sum(
                1
                for route in violating_route_set
                if secure_bers_by_entry[entry.entry_id][route] <= LEAK_SECURE_BER_THRESHOLD
            ),
        ),
    )
    route_text = ", ".join(
        f"illegal {route[0]}->legal {route[1]} zeros={zero_count} avg={average_secure_ber:.3f}"
        f" min={min_secure_ber:.3f}"
        for route, zero_count, average_secure_ber, min_secure_ber in violating_routes[:5]
    )
    if len(violating_routes) > 5:
        route_text += f", ... total={len(violating_routes)}"
    print(
        f"      Pool constraint violation ({route_text}); "
        f"remove set {remove_entry.entry_id} and continue searching."
    )
    entries.remove(remove_entry)
    for new_id, entry in enumerate(entries, start=1):
        entry.entry_id = new_id
    return True


def would_exceed_zero_leak_limit(
    entries: Sequence[ProbePoolEntry],
    candidate_route_secure_bers: dict[tuple[int, int], float],
) -> tuple[bool, tuple[int, int] | None, int]:
    route_zero_counts: dict[tuple[int, int], int] = {}
    for entry in entries:
        if entry.route_secure_bers is None:
            continue
        for route, secure_ber in entry.route_secure_bers.items():
            if float(secure_ber) <= LEAK_SECURE_BER_THRESHOLD:
                route_zero_counts[route] = route_zero_counts.get(route, 0) + 1

    for route, secure_ber in candidate_route_secure_bers.items():
        zero_count = route_zero_counts.get(route, 0)
        if float(secure_ber) <= LEAK_SECURE_BER_THRESHOLD:
            zero_count += 1
        if zero_count > MAX_ZERO_LEAKS_PER_ROUTE:
            return True, route, zero_count
    return False, None, 0


def build_probe_pool_entry(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
    entry_id: int,
    rng: random.Random,
) -> ProbePoolEntry:
    probes_array = np.sort(np.asarray(probes, dtype=float))
    legal_csv_files = build_csv_files(project_root, legal_positions)
    legal_models, hue_mapping = test.build_models_from_probes(
        legal_csv_files,
        probes_array,
        mapping_eval_bits=MAPPING_EVAL_BITS,
        mapping_top_k=MAPPING_TOP_K,
        rng=rng,
    )
    legal_blocks = test.generate_random_bit_blocks(LEGAL_SEARCH_BITS, len(legal_positions), rng=rng)
    results = test.simulate_blocks(legal_models, legal_blocks, hue_mapping)
    legal_position_bers = [float(v) for v in calculate_position_bers(results)]
    return ProbePoolEntry(
        entry_id=entry_id,
        probes=probes_array,
        legal_models=legal_models,
        hue_mapping=hue_mapping,
        legal_position_bers=legal_position_bers,
    )


def search_probe_pool(
    project_root: str,
    legal_positions: Sequence[int],
    rng: random.Random,
    target_pool_size: int,
) -> tuple[int, list[ProbePoolEntry]]:
    all_probes = get_all_probes(project_root, legal_positions)
    leak_eval_rng = random.Random(SELECTION_SEED + sum(int(position) * 1009 for position in legal_positions))
    leak_eval_blocks = test.generate_random_bit_blocks(POOL_LEAK_EVAL_BITS, len(legal_positions), rng=leak_eval_rng)
    for probe_count in PROBE_COUNT_CANDIDATES:
        min_interval = min_interval_for_probe_count(probe_count)
        print(f"    Search legal probe pool: M={probe_count}, min_interval={min_interval}")
        entries: list[ProbePoolEntry] = []
        seen: set[tuple[float, ...]] = set()
        evaluated_probe_sets: list[set[float]] = []
        attempts = 0
        evaluated = 0
        stagnant_candidates = 0
        while attempts < MAX_CANDIDATES_PER_PROBE_COUNT and len(entries) < target_pool_size:
            if stagnant_candidates >= MAX_STAGNANT_CANDIDATES_PER_PROBE_COUNT and entries:
                print(
                    f"      Stop M={probe_count}: no accepted set in the last "
                    f"{stagnant_candidates} evaluated candidates."
                )
                break
            attempts += 1
            probes = sample_valid_probe_set(all_probes, probe_count, min_interval, rng)
            if probes is None:
                continue
            signature = tuple(float(v) for v in probes)
            if signature in seen:
                continue
            if not is_diverse_probe_set(probes, evaluated_probe_sets):
                continue
            seen.add(signature)
            evaluated_probe_sets.append({float(probe) for probe in probes})
            evaluated += 1
            stagnant_candidates += 1

            entry = build_probe_pool_entry(
                project_root=project_root,
                legal_positions=legal_positions,
                probes=probes,
                entry_id=len(entries) + 1,
                rng=rng,
            )
            if all(ber == 0.0 for ber in entry.legal_position_bers):
                entry.route_secure_bers = evaluate_entry_route_secure_bers(
                    project_root,
                    legal_positions,
                    entry,
                    leak_eval_blocks,
                )
                exceeds_zero_limit, route, zero_count = would_exceed_zero_leak_limit(entries, entry.route_secure_bers)
                if exceeds_zero_limit:
                    print(
                        f"      Reject legal BER=0 set: adding it makes "
                        f"illegal {route[0]}->legal {route[1]} zeros={zero_count}>{MAX_ZERO_LEAKS_PER_ROUTE}."
                    )
                    continue
                entries.append(entry)
                stagnant_candidates = 0
                print(f"      Found legal BER=0 set {len(entries)}/{target_pool_size}: {format_probes(entry.probes)}")
                if len(entries) >= target_pool_size:
                    if remove_one_pool_constraint_violating_entry(project_root, legal_positions, entries, leak_eval_blocks):
                        stagnant_candidates = 0
                        continue
                    return probe_count, entries

        print(
            f"      Only found {len(entries)}/{target_pool_size} legal BER=0 sets "
            f"for M={probe_count}; evaluated_valid_candidates={evaluated}"
        )
    return 0, []


def evaluate_hopping_security(
    project_root: str,
    legal_positions: Sequence[int],
    probe_pool: Sequence[ProbePoolEntry],
    rng: random.Random,
) -> tuple[list[dict], dict]:
    all_positions = get_available_positions(project_root)
    illegal_positions = [position for position in all_positions if position not in legal_positions]
    bit_blocks_pm = test.generate_random_bit_blocks(HOPPING_EVAL_BITS, len(legal_positions), rng=rng)
    selected_entries = [rng.choice(list(probe_pool)) for _ in bit_blocks_pm]

    legal_errors = np.zeros(len(legal_positions), dtype=float)
    legal_total = np.zeros(len(legal_positions), dtype=float)
    for bits_pm, entry in zip(bit_blocks_pm, selected_entries):
        results = test.simulate_blocks(entry.legal_models, [bits_pm], entry.hue_mapping)
        true_bits = results[0]["bits_bin"]
        decoded_bits = [decode.bit_hat_bin for decode in results[0]["per_position"]]
        for idx, (true_bit, decoded_bit) in enumerate(zip(true_bits, decoded_bits)):
            legal_total[idx] += 1
            if int(true_bit) != int(decoded_bit):
                legal_errors[idx] += 1
    legal_position_bers = (legal_errors / np.maximum(legal_total, 1.0)).tolist()

    illegal_rows: list[dict] = []
    global_min_secure_ber = float("inf")
    worst_illegal_position: int | None = None
    worst_legal_position: int | None = None

    illegal_cache: dict[tuple[int, int], tuple[object, np.ndarray, dict[int, int]]] = {}
    for illegal_position in illegal_positions:
        position_errors = np.zeros(len(legal_positions), dtype=float)
        position_total = np.zeros(len(legal_positions), dtype=float)
        for bits_pm, entry in zip(bit_blocks_pm, selected_entries):
            cache_key = (illegal_position, entry.entry_id)
            if cache_key not in illegal_cache:
                illegal_csv = build_csv_files(project_root, [illegal_position])[0]
                illegal_matrix = test.load_selected_rows([illegal_csv], entry.probes)[0]
                illegal_model = test.extract_fingerprint(entry.probes, illegal_matrix, force_positive_first=True)
                probe_to_row = test.build_probe_to_row(entry.probes)
                illegal_cache[cache_key] = (illegal_model, illegal_matrix, probe_to_row)
            illegal_model, illegal_matrix, probe_to_row = illegal_cache[cache_key]

            legal_codes = [model.code for model in entry.legal_models]
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, entry.hue_mapping)
            illegal_observation = test.observe_block_from_measured_matrix(hue_seq, illegal_matrix, probe_to_row)
            illegal_dec = test.decode_local_block(illegal_observation, illegal_model.w, illegal_model.code)
            true_bits = test.pm1_to_bin(bits_pm)
            for idx, true_bit in enumerate(true_bits):
                position_total[idx] += 1
                if int(illegal_dec.bit_hat_bin) != int(true_bit):
                    position_errors[idx] += 1

        raw_bers = (position_errors / np.maximum(position_total, 1.0)).tolist()
        secure_bers = [corrected_ber(ber) for ber in raw_bers]
        average_secure_ber = float(np.mean(secure_bers))
        local_min_idx = int(np.argmin(secure_bers))
        local_min_secure_ber = float(secure_bers[local_min_idx])
        if local_min_secure_ber < global_min_secure_ber:
            global_min_secure_ber = local_min_secure_ber
            worst_illegal_position = illegal_position
            worst_legal_position = int(legal_positions[local_min_idx])

        row = {
            "position_combination": str(tuple(legal_positions)),
            "illegal_position": illegal_position,
            "probe_pool_size": len(probe_pool),
            "probe_count": len(probe_pool[0].probes),
            "hopping_bits": HOPPING_EVAL_BITS,
            "legal_position_bers": "[" + ", ".join(f"{v:.6f}" for v in legal_position_bers) + "]",
            "min_secure_ber": f"{local_min_secure_ber:.6f}",
            "average_secure_ber": f"{average_secure_ber:.6f}",
        }
        for idx, legal_position in enumerate(legal_positions, start=1):
            row[f"legal_position_{idx}"] = int(legal_position)
            row[f"ber_vs_legal_pos_{idx}"] = f"{raw_bers[idx - 1]:.6f}"
            row[f"secure_ber_vs_legal_pos_{idx}"] = f"{secure_bers[idx - 1]:.6f}"
        illegal_rows.append(row)

    summary = {
        "legal_position_bers": legal_position_bers,
        "min_illegal_secure_ber": 0.0 if global_min_secure_ber == float("inf") else global_min_secure_ber,
        "worst_illegal_position": worst_illegal_position,
        "worst_legal_position": worst_legal_position,
    }
    return illegal_rows, summary


def select_position_combinations(project_root: str, count: int, rng: random.Random) -> list[tuple[int, ...]]:
    combinations = list(itertools.combinations(get_available_positions(project_root), LEGAL_POSITION_COUNT))
    if count >= len(combinations):
        return [tuple(int(v) for v in combination) for combination in combinations]
    return sorted(tuple(int(v) for v in combination) for combination in rng.sample(combinations, count))


def append_csv_rows(file_path: str, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    file_exists = os.path.exists(file_path)
    if file_exists:
        with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        if list(existing_fieldnames) != list(fieldnames):
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({field: row.get(field, "") for field in fieldnames})

    with open(file_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def load_completed_combinations(file_path: str) -> set[str]:
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
        return {row["position_combination"] for row in csv.DictReader(f)}


def run_experiment(combination_count: int) -> tuple[str, str]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "test-3")
    probe_pool_file = os.path.join(output_dir, OUTPUT_PROBE_POOL_FILENAME)
    security_file = os.path.join(output_dir, OUTPUT_SECURITY_FILENAME)
    rng = random.Random(SELECTION_SEED)
    combinations = select_position_combinations(project_root, combination_count, rng)
    completed = load_completed_combinations(security_file)

    probe_pool_fields = [
        "position_combination",
        "pool_entry_id",
        "probe_pool_size",
        "probe_count",
        "probes",
        "hue_mapping",
        "legal_position_bers",
    ]
    security_fields = [
        "position_combination",
        "illegal_position",
        "probe_pool_size",
        "probe_count",
        "hopping_bits",
        "legal_position_bers",
        "legal_position_1",
        "ber_vs_legal_pos_1",
        "secure_ber_vs_legal_pos_1",
        "legal_position_2",
        "ber_vs_legal_pos_2",
        "secure_ber_vs_legal_pos_2",
        "legal_position_3",
        "ber_vs_legal_pos_3",
        "secure_ber_vs_legal_pos_3",
        "min_secure_ber",
        "average_secure_ber",
    ]

    print(f"Selected {len(combinations)} random 3-position combinations from {LIGHT_CONDITION}.")
    for idx, legal_positions in enumerate(combinations, start=1):
        combination_key = str(tuple(legal_positions))
        if combination_key in completed:
            print(f"[{idx}/{len(combinations)}] Skip {legal_positions}: already evaluated.")
            continue

        print(f"[{idx}/{len(combinations)}] Searching legal probe pool for {legal_positions}")
        probe_count, probe_pool = search_probe_pool(
            project_root=project_root,
            legal_positions=legal_positions,
            rng=rng,
            target_pool_size=TARGET_POOL_SIZE,
        )
        if not probe_pool:
            print(f"  No complete legal BER=0 probe pool found for {legal_positions}.")
            continue

        pool_rows = []
        for entry in probe_pool:
            pool_rows.append({
                "position_combination": combination_key,
                "pool_entry_id": entry.entry_id,
                "probe_pool_size": len(probe_pool),
                "probe_count": probe_count,
                "probes": format_probes(entry.probes),
                "hue_mapping": format_hue_mapping(entry.hue_mapping),
                "legal_position_bers": "[" + ", ".join(f"{v:.6f}" for v in entry.legal_position_bers) + "]",
            })
        append_csv_rows(probe_pool_file, probe_pool_fields, pool_rows)
        print(f"  Saved {len(pool_rows)} legal probe sets to {probe_pool_file}")

        print(f"  Evaluating random probe hopping security for {legal_positions}")
        security_rows, summary = evaluate_hopping_security(
            project_root=project_root,
            legal_positions=legal_positions,
            probe_pool=probe_pool,
            rng=rng,
        )
        append_csv_rows(security_file, security_fields, security_rows)
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
    parser = argparse.ArgumentParser(description="Run 3-position yellow random probe hopping experiment.")
    parser.add_argument(
        "--combination-count",
        type=int,
        default=DEFAULT_COMBINATION_COUNT,
        help="Number of random 3-position combinations to evaluate.",
    )
    args = parser.parse_args()
    run_experiment(combination_count=args.combination_count)


if __name__ == "__main__":
    main()
