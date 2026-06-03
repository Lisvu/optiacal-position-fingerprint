#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-component batch search: find probe sets that satisfy BOTH legal BER and security.

For each 4-position combination:
  1. Run staged_beam_probe_selection (multi-component system)
  2. Exact-eval legal BER on all 16 blocks
  3. If legal_ber <= 0.005:
       - Exact-eval all illegal positions (16 blocks each)
       - If min_illegal_ber >= 0.2:
           - FEC confirm (10000 bits)
           - Save if both conditions pass

Output: multi_comp_security_results_4.csv
"""

from __future__ import annotations

import csv
import itertools
import os
import random
import sys
from typing import Sequence

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

LIGHT_CONDITION = "white"
OUTPUT_RESULTS_FILENAME = "multi_comp_security_results_4.csv"

TARGET_LEGAL_BER = 0.005
MIN_ILLEGAL_BER = 0.2
MAX_ILLEGAL_BER = 0.8

# Beam search params (adapted for multi-component)
NUM_PROBES_RANGE = (6, 14)
MIN_INTERVAL = 30
COARSE_BITS = 800
MAPPING_EVAL_BITS = 300
MAPPING_TOP_K = 3

# FEC confirmation
CONFIRM_LEGAL_BITS = 10000
CONFIRM_ILLEGAL_BITS = 10000

SEED = 20260509


def get_available_positions(project_root: str) -> list[int]:
    data_dir = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION)
    positions = []
    for entry in os.listdir(data_dir):
        stem, ext = os.path.splitext(entry)
        if ext.lower() == ".csv" and stem.isdigit():
            positions.append(int(stem))
    return sorted(positions)


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def format_hue_mapping(hue_mapping: dict) -> str:
    if not hue_mapping:
        return ""
    return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(hue_mapping.items())) + "}"


def build_models(project_root: str, positions: Sequence[int], probes: np.ndarray):
    csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition=LIGHT_CONDITION)
    matrices = test.load_selected_rows(csv_files, probes)
    models = [test.extract_fingerprint(probes, mat, force_positive_first=True) for mat in matrices]
    models = test.align_model_directions(models)
    hue_mapping = test.build_hue_mapping(
        models, probes,
        mapping_eval_bits=MAPPING_EVAL_BITS,
        top_k_per_combination=MAPPING_TOP_K,
    )
    return models, hue_mapping


def exact_eval_legal(models: list, hue_mapping: dict) -> float:
    blocks = test.generate_all_bit_blocks(len(models))
    return test.evaluate_blocks_ber(models, blocks, hue_mapping)


def exact_eval_illegal(
    project_root: str,
    legal_positions: Sequence[int],
    probes: np.ndarray,
    legal_models: list,
    hue_mapping: dict,
) -> dict:
    legal_codes = [m.eff_code if m.eff_code is not None else m.code for m in legal_models]
    probe_to_row = test.build_probe_to_row(probes)
    blocks = test.generate_all_bit_blocks(len(legal_positions))
    all_positions = get_available_positions(project_root)

    global_min = float("inf")
    raw_at_min = 0.0
    worst_pos = None
    worst_idx = None
    all_secure = []

    for pos in all_positions:
        if pos in legal_positions:
            continue
        csv_file = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION, f"{pos}.csv")
        mat = test.load_csv_matrix(csv_file)
        row_indices = [int((p / 5) - 1) for p in probes]
        illegal_mat = mat[row_indices].astype(float, copy=False)
        illegal_model = test.extract_fingerprint(probes, illegal_mat, force_positive_first=True)

        corr = test.calculate_correlation(legal_models[0].eff_code if legal_models[0].eff_code is not None else legal_models[0].code,
                                          illegal_model.eff_code if illegal_model.eff_code is not None else illegal_model.code)
        if corr < 0:
            if illegal_model.W is not None and illegal_model.W.ndim > 1:
                illegal_model.W = -illegal_model.W
                illegal_model.Z = -illegal_model.Z
                illegal_model.multi_code = -illegal_model.multi_code
                illegal_model.eff_w = -illegal_model.eff_w
                illegal_model.eff_code = -illegal_model.eff_code
            else:
                illegal_model.w = -illegal_model.w
                illegal_model.z = -illegal_model.z
                illegal_model.code = -illegal_model.code

        errors = np.zeros(len(legal_positions), dtype=float)
        totals = np.zeros(len(legal_positions), dtype=float)
        for bits_pm in blocks:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            obs = test.observe_block_from_measured_matrix(hue_seq, illegal_model.Y, probe_to_row)
            if illegal_model.eff_w is not None and illegal_model.eff_code is not None:
                dec = test.decode_local_block(obs, illegal_model.eff_w, illegal_model.eff_code)
            else:
                dec = test.decode_local_block(obs, illegal_model.w, illegal_model.code)
            true_bits = test.pm1_to_bin(bits_pm)
            for li in range(len(legal_positions)):
                totals[li] += 1
                if dec.bit_hat_bin != true_bits[li]:
                    errors[li] += 1

        raw_bers = errors / np.maximum(totals, 1.0)
        secure_bers = np.minimum(raw_bers, 1.0 - raw_bers)
        all_secure.extend(secure_bers.tolist())
        local_min = float(np.min(secure_bers))
        local_idx = int(np.argmin(secure_bers))
        if local_min < global_min:
            global_min = local_min
            raw_at_min = float(raw_bers[local_idx])
            worst_pos = pos
            worst_idx = local_idx

    return {
        "min_illegal_ber": float(global_min if all_secure else 0.0),
        "raw_ber_at_min": float(raw_at_min),
        "average_illegal_ber": float(np.mean(all_secure)) if all_secure else 0.0,
        "worst_illegal_position": worst_pos,
        "worst_legal_index": worst_idx,
    }


def fec_confirm(
    project_root: str,
    legal_positions: Sequence[int],
    probes: np.ndarray,
    legal_models: list,
    hue_mapping: dict,
    rng: random.Random,
) -> tuple[float, float]:
    legal_ber = test.evaluate_blocks_ber_with_convolutional_fec(
        legal_models,
        test.generate_random_information_bits(CONFIRM_LEGAL_BITS, len(legal_positions), rng=rng),
        hue_mapping,
    )
    if legal_ber > TARGET_LEGAL_BER:
        return legal_ber, 0.0

    # 为了速度，只检查 worst illegal position
    sec = exact_eval_illegal(project_root, legal_positions, probes, legal_models, hue_mapping)
    worst_pos = sec["worst_illegal_position"]
    if worst_pos is None:
        return legal_ber, 1.0

    # FEC eval on worst illegal position
    legal_codes = [m.eff_code if m.eff_code is not None else m.code for m in legal_models]
    probe_to_row = test.build_probe_to_row(probes)
    info_bits = test.generate_random_information_bits(CONFIRM_ILLEGAL_BITS, len(legal_positions), rng=rng)
    bit_blocks_pm = test.build_convolutional_bit_blocks(info_bits)

    csv_file = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION, f"{worst_pos}.csv")
    mat = test.load_csv_matrix(csv_file)
    row_indices = [int((p / 5) - 1) for p in probes]
    illegal_mat = mat[row_indices].astype(float, copy=False)
    illegal_model = test.extract_fingerprint(probes, illegal_mat, force_positive_first=True)

    corr = test.calculate_correlation(legal_models[0].eff_code if legal_models[0].eff_code is not None else legal_models[0].code,
                                      illegal_model.eff_code if illegal_model.eff_code is not None else illegal_model.code)
    if corr < 0:
        if illegal_model.W is not None and illegal_model.W.ndim > 1:
            illegal_model.W = -illegal_model.W
            illegal_model.Z = -illegal_model.Z
            illegal_model.multi_code = -illegal_model.multi_code
            illegal_model.eff_w = -illegal_model.eff_w
            illegal_model.eff_code = -illegal_model.eff_code
        else:
            illegal_model.w = -illegal_model.w
            illegal_model.z = -illegal_model.z
            illegal_model.code = -illegal_model.code

    received = [[] for _ in legal_positions]
    for bits_pm in bit_blocks_pm:
        _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
        hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
        obs = test.observe_block_from_measured_matrix(hue_seq, illegal_model.Y, probe_to_row)
        if illegal_model.eff_w is not None and illegal_model.eff_code is not None:
            dec = test.decode_local_block(obs, illegal_model.eff_w, illegal_model.eff_code)
        else:
            dec = test.decode_local_block(obs, illegal_model.w, illegal_model.code)
        for li in range(len(legal_positions)):
            received[li].append(dec.bit_hat_bin)

    global_min = float("inf")
    for li in range(len(legal_positions)):
        decoded = test.viterbi_decode_hard(received[li])
        ref = info_bits[:, li]
        cl = min(len(decoded), len(ref))
        if cl > 0:
            raw_ber = float(np.mean(decoded[:cl] != ref[:cl]))
            secure_ber = min(raw_ber, 1.0 - raw_ber)
            if secure_ber < global_min:
                global_min = secure_ber

    return legal_ber, float(global_min)


def search_combination(project_root: str, positions: tuple[int, ...], rng: random.Random) -> dict:
    print(f"\n[Search] {positions}")
    all_probes = (5 + np.arange(73) * 5).astype(float).tolist()

    # Phase 1: staged beam search for best probe set (try multiple counts)
    print("  Phase 1: beam search for probe set...")
    best_probes = None
    best_ber = float("inf")
    best_count = 0
    
    for num_probes in [8, 9, 10, 11, 12]:
        print(f"    Trying num_probes={num_probes}...")
        try:
            probes, ber = test.staged_beam_probe_selection(
                csv_files=test.build_csv_files_for_positions(project_root, positions, light_condition=LIGHT_CONDITION),
                num_probes=num_probes,
                num_bits=3000,
                min_interval=MIN_INTERVAL,
                coarse_bits=400,
                mapping_eval_bits=200,
                mapping_top_k=3,
                neighborhood_samples=12,
                local_rounds=3,
                beam_width=12,
                initial_sample_size=30,
                expansion_sample_size=16,
                finalist_count=8,
                sa_iterations=32,
                repeat_eval=2,
                candidate_pool_size=40,
                base_seed=rng.randint(0, 2**31),
                rng=rng,
            )
            print(f"      -> coarse_ber={ber:.6f}, probes={format_probes(probes)}")
            if ber < best_ber:
                best_ber = ber
                best_probes = probes
                best_count = num_probes
        except Exception as e:
            print(f"      -> error: {e}")
            continue
    
    if best_probes is None:
        print("  No probe set found!")
        return {
            "position_combination": str(positions),
            "probe_count": 0,
            "probes": "",
            "hue_mapping": "",
            "legal_ber": "1.0",
            "min_illegal_ber": "",
            "security_satisfied": "no",
        }
    
    print(f"  Best probe set: count={best_count}, coarse_ber={best_ber:.6f}")
    print(f"  Probes: {format_probes(best_probes)}")

    # Phase 2: exact eval
    print("  Phase 2: exact evaluation...")
    models, hue_mapping = build_models(project_root, positions, best_probes)
    exact_legal = exact_eval_legal(models, hue_mapping)
    print(f"  Exact legal BER: {exact_legal:.6f}")

    if exact_legal > TARGET_LEGAL_BER:
        print(f"  FAIL: exact legal_ber > {TARGET_LEGAL_BER}")
        return {
            "position_combination": str(positions),
            "probe_count": len(best_probes),
            "probes": format_probes(best_probes),
            "hue_mapping": format_hue_mapping(hue_mapping),
            "legal_ber": f"{exact_legal:.6f}",
            "min_illegal_ber": "",
            "security_satisfied": "no",
        }

    # Phase 3: security exact eval
    print("  Phase 3: security evaluation...")
    sec = exact_eval_illegal(project_root, positions, best_probes, models, hue_mapping)
    print(f"  min_illegal_ber={sec['min_illegal_ber']:.6f}, worst_pos={sec['worst_illegal_position']}")

    if sec["min_illegal_ber"] < MIN_ILLEGAL_BER:
        print(f"  FAIL: min_illegal_ber < {MIN_ILLEGAL_BER}")
        return {
            "position_combination": str(positions),
            "probe_count": len(best_probes),
            "probes": format_probes(best_probes),
            "hue_mapping": format_hue_mapping(hue_mapping),
            "legal_ber": f"{exact_legal:.6f}",
            "min_illegal_ber": f"{sec['min_illegal_ber']:.6f}",
            "security_satisfied": "no",
        }

    # Phase 4: FEC confirmation
    print("  Phase 4: FEC confirmation...")
    fec_legal, fec_illegal = fec_confirm(project_root, positions, best_probes, models, hue_mapping, rng)
    print(f"  FEC: legal={fec_legal:.6f}, illegal={fec_illegal:.6f}")

    satisfied = fec_legal <= TARGET_LEGAL_BER and fec_illegal >= MIN_ILLEGAL_BER
    print(f"  {'*** SECURITY SATISFIED ***' if satisfied else 'FAIL on FEC confirmation'}")

    return {
        "position_combination": str(positions),
        "probe_count": len(best_probes),
        "probes": format_probes(best_probes),
        "hue_mapping": format_hue_mapping(hue_mapping),
        "legal_ber": f"{fec_legal:.6f}",
        "min_illegal_ber": f"{fec_illegal:.6f}",
        "security_satisfied": "yes" if satisfied else "no",
    }


def write_results(results_file: str, rows: list[dict]) -> None:
    fieldnames = [
        "position_combination", "probe_count", "probes", "hue_mapping",
        "legal_ber", "min_illegal_ber", "security_satisfied",
    ]
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(project_root, "test-4", OUTPUT_RESULTS_FILENAME)

    positions = get_available_positions(project_root)
    print(f"Total positions: {len(positions)}")
    print(f"Multi-component config: N_COMP={test.N_COMPONENTS}, ALPHA={test.ALPHA}")

    # 先测前5个组合
    combinations = list(itertools.combinations(positions, 4))[:5]
    print(f"Testing {len(combinations)} combinations")

    rng = random.Random(SEED)
    rows = []

    for idx, pos in enumerate(combinations, start=1):
        print(f"\n[{idx}/{len(combinations)}] Processing {pos}")
        result = search_combination(project_root, pos, rng)
        rows.append(result)
        write_results(results_file, rows)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
