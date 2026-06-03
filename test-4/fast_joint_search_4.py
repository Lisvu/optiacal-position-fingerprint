#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast joint probe + hue mapping search for 4 positions (multi-component system).

针对多主成分系统重新搜索 probe sets，不再依赖旧 zero-BER configs。
流程：
  1. 对每个 4-position 组合，随机采样 probe sets（6~13 probes）
  2. 用 exact evaluation（16 blocks）快速筛选 legal BER <= 0.02 的候选
  3. 对 top 候选评估 illegal security（exact eval，所有非法位置）
  4. 通过初筛的用 FEC 10000 bits 做最终确认
  5. 目标：legal_ber <= 0.005，min_illegal_ber >= 0.2
"""

from __future__ import annotations

import csv
import os
import random
import sys
from typing import Sequence

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LIGHT_CONDITION = "white"
OUTPUT_RESULTS_FILENAME = "fast_joint_search_results_4.csv"

TARGET_LEGAL_BER = 0.005
MIN_ILLEGAL_BER = 0.2
MAX_ILLEGAL_BER = 0.8

# Probe search
PROBE_COUNTS = [7, 8, 9, 10, 11, 12, 13]
RANDOM_SAMPLES_PER_COUNT = 60
PROBE_MIN_INTERVAL = 30

# Hue mapping
MAPPING_EVAL_BITS = 200   # 4 positions -> exact 16 blocks automatically
MAPPING_TOP_K = 3

# Security screening (fast exact eval)
SECURITY_EXACT_EVAL = True

# Final confirmation (FEC)
CONFIRM_LEGAL_BITS = 10000
CONFIRM_ILLEGAL_BITS = 10000

SEED = 20260508


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_available_positions(project_root: str) -> list[int]:
    data_dir = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION)
    positions = []
    for entry in os.listdir(data_dir):
        stem, ext = os.path.splitext(entry)
        if ext.lower() == ".csv" and stem.isdigit():
            positions.append(int(stem))
    return sorted(positions)


def get_all_probes(project_root: str, legal_positions: Sequence[int]) -> list[float]:
    csv_file = test.build_csv_files_for_positions(project_root, legal_positions, light_condition=LIGHT_CONDITION)[0]
    first_df = test.load_csv_matrix(csv_file)
    return (5 + np.arange(len(first_df)) * 5).astype(float).tolist()


def min_interval_for_probe_count(probe_count: int) -> int:
    if probe_count <= 12:
        return 30
    if probe_count <= 16:
        return 20
    return 15


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def format_hue_mapping(hue_mapping: dict) -> str:
    if not hue_mapping:
        return ""
    return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(hue_mapping.items())) + "}"


# ---------------------------------------------------------------------------
# Fast illegal evaluation (exact, all illegal positions)
# ---------------------------------------------------------------------------
def evaluate_illegal_exact(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
    legal_models: list,
    hue_mapping: dict,
) -> dict:
    probes_array = np.asarray(probes, dtype=float)
    legal_codes = [m.eff_code if m.eff_code is not None else m.code for m in legal_models]
    probe_to_row = test.build_probe_to_row(probes_array)
    bit_blocks_pm = test.generate_all_bit_blocks(len(legal_positions))

    data_dir = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION)
    all_positions = get_available_positions(project_root)

    global_min = float("inf")
    raw_at_min = 0.0
    worst_pos = None
    worst_idx = None
    all_secure = []

    for pos in all_positions:
        if pos in legal_positions:
            continue
        csv_file = os.path.join(data_dir, f"{pos}.csv")
        mat = test.load_csv_matrix(csv_file)
        row_indices = [int((p / 5) - 1) for p in probes_array]
        illegal_mat = mat[row_indices].astype(float, copy=False)
        illegal_model = test.extract_fingerprint(probes_array, illegal_mat, force_positive_first=True)

        # align direction with first legal model
        corr = test.calculate_correlation(legal_models[0].code, illegal_model.code)
        if corr < 0:
            if illegal_model.W is not None and illegal_model.W.ndim > 1:
                illegal_model.W = -illegal_model.W
                illegal_model.Z = -illegal_model.Z
                illegal_model.multi_code = -illegal_model.multi_code
                k = illegal_model.W.shape[1]
                alpha = test.ALPHA[:k] if hasattr(test, 'ALPHA') else np.ones(k)
                alpha = np.asarray(alpha, dtype=float)
                illegal_model.eff_code = np.sign(illegal_model.Z @ alpha).astype(int)
            else:
                illegal_model.w = -illegal_model.w
                illegal_model.z = -illegal_model.z
                illegal_model.code = -illegal_model.code

        errors = np.zeros(len(legal_models), dtype=float)
        totals = np.zeros(len(legal_models), dtype=float)

        for bits_pm in bit_blocks_pm:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            obs = test.observe_block_from_measured_matrix(hue_seq, illegal_model.Y, probe_to_row)
            if illegal_model.eff_w is not None and illegal_model.eff_code is not None:
                dec = test.decode_local_block(obs, illegal_model.eff_w, illegal_model.eff_code)
            else:
                dec = test.decode_local_block(obs, illegal_model.w, illegal_model.code)
            true_bits = test.pm1_to_bin(bits_pm)
            for li in range(len(legal_models)):
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


# ---------------------------------------------------------------------------
# FEC confirmation
# ---------------------------------------------------------------------------
def confirm_with_fec(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
    legal_models: list,
    hue_mapping: dict,
    rng: random.Random,
) -> tuple[float, float]:
    # Legal FEC
    legal_ber = test.evaluate_blocks_ber_with_convolutional_fec(
        legal_models,
        test.generate_random_information_bits(CONFIRM_LEGAL_BITS, len(legal_positions), rng=rng),
        hue_mapping,
    )
    if legal_ber > TARGET_LEGAL_BER:
        return legal_ber, 0.0

    # Illegal FEC (expensive - sample a subset of illegal positions for speed)
    probes_array = np.asarray(probes, dtype=float)
    legal_codes = [m.eff_code if m.eff_code is not None else m.code for m in legal_models]
    probe_to_row = test.build_probe_to_row(probes_array)
    info_bits = test.generate_random_information_bits(CONFIRM_ILLEGAL_BITS, len(legal_positions), rng=rng)
    bit_blocks_pm = test.build_convolutional_bit_blocks(info_bits)

    data_dir = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION)
    all_positions = get_available_positions(project_root)
    illegal_positions = [p for p in all_positions if p not in legal_positions]

    # 为了速度，随机选最多 6 个非法位置做 FEC 确认
    if len(illegal_positions) > 6:
        illegal_positions = rng.sample(illegal_positions, 6)

    global_min = float("inf")

    for pos in illegal_positions:
        csv_file = os.path.join(data_dir, f"{pos}.csv")
        mat = test.load_csv_matrix(csv_file)
        row_indices = [int((p / 5) - 1) for p in probes_array]
        illegal_mat = mat[row_indices].astype(float, copy=False)
        illegal_model = test.extract_fingerprint(probes_array, illegal_mat, force_positive_first=True)

        corr = test.calculate_correlation(legal_models[0].code, illegal_model.code)
        if corr < 0:
            if illegal_model.W is not None and illegal_model.W.ndim > 1:
                illegal_model.W = -illegal_model.W
                illegal_model.Z = -illegal_model.Z
                illegal_model.multi_code = -illegal_model.multi_code
                k = illegal_model.W.shape[1]
                alpha = test.ALPHA[:k] if hasattr(test, 'ALPHA') else np.ones(k)
                alpha = np.asarray(alpha, dtype=float)
                illegal_model.eff_code = np.sign(illegal_model.Z @ alpha).astype(int)
            else:
                illegal_model.w = -illegal_model.w
                illegal_model.z = -illegal_model.z
                illegal_model.code = -illegal_model.code

        received = [[] for _ in legal_models]
        for bits_pm in bit_blocks_pm:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            obs = test.observe_block_from_measured_matrix(hue_seq, illegal_model.Y, probe_to_row)
            if illegal_model.eff_w is not None and illegal_model.eff_code is not None:
                dec = test.decode_local_block(obs, illegal_model.eff_w, illegal_model.eff_code)
            else:
                dec = test.decode_local_block(obs, illegal_model.w, illegal_model.code)
            for li in range(len(legal_models)):
                received[li].append(dec.bit_hat_bin)

        for li in range(len(legal_models)):
            decoded = test.viterbi_decode_hard(received[li])
            ref = info_bits[:, li]
            cl = min(len(decoded), len(ref))
            if cl > 0:
                raw_ber = float(np.mean(decoded[:cl] != ref[:cl]))
                secure_ber = min(raw_ber, 1.0 - raw_ber)
                if secure_ber < global_min:
                    global_min = secure_ber

    return legal_ber, float(global_min)


# ---------------------------------------------------------------------------
# Core: probe set search
# ---------------------------------------------------------------------------
def search_probe_sets(
    project_root: str,
    legal_positions: Sequence[int],
    all_probes: list[float],
    rng: random.Random,
) -> list[tuple[float, np.ndarray, int, list, dict]]:
    """返回 [(legal_ber, probes, count, models, hue_mapping), ...]，按 legal_ber 排序"""
    candidates = []
    seen = set()

    for count in PROBE_COUNTS:
        min_interval = min_interval_for_probe_count(count)
        print(f"    Trying probe_count={count}, interval>={min_interval}")
        found_for_count = 0

        for _ in range(RANDOM_SAMPLES_PER_COUNT):
            probes = np.sort(np.asarray(rng.sample(all_probes, count), dtype=float))
            key = tuple(probes.tolist())
            if key in seen or not test.is_valid_probe_set(probes, min_interval):
                continue
            seen.add(key)

            try:
                csv_files = test.build_csv_files_for_positions(project_root, legal_positions, light_condition=LIGHT_CONDITION)
                models, hue_mapping = test.build_models_from_probes(
                    csv_files, probes,
                    mapping_eval_bits=MAPPING_EVAL_BITS,
                    mapping_top_k=MAPPING_TOP_K,
                    rng=rng,
                )
                bit_blocks = test.generate_all_bit_blocks(len(legal_positions))
                legal_ber = test.evaluate_blocks_ber(models, bit_blocks, hue_mapping)

                if legal_ber <= 0.02:  # 宽松初筛
                    candidates.append((legal_ber, probes.copy(), count, models, hue_mapping))
                    found_for_count += 1

            except Exception as e:
                continue

        print(f"      Found {found_for_count} candidates with legal_ber <= 0.02")

    candidates.sort(key=lambda x: x[0])
    return candidates[:20]  # 保留 top 20


# ---------------------------------------------------------------------------
# Main search for one combination
# ---------------------------------------------------------------------------
def search_for_combination(project_root: str, legal_positions: tuple[int, ...], rng: random.Random) -> dict:
    print(f"\n[Search] Combination {legal_positions}")
    all_probes = get_all_probes(project_root, legal_positions)
    print(f"  Available probes: {len(all_probes)} (5~{int(max(all_probes))})")

    # Phase 1: search probes
    probe_candidates = search_probe_sets(project_root, legal_positions, all_probes, rng)
    if not probe_candidates:
        print("  No probe candidates found!")
        return {
            "position_combination": str(legal_positions),
            "probe_count": 0,
            "probes": "",
            "hue_mapping": "",
            "legal_ber": 1.0,
            "min_illegal_ber": "",
            "security_satisfied": "no",
        }

    print(f"  Top probe candidate: legal_ber={probe_candidates[0][0]:.6f}")

    # Phase 2: evaluate security for top candidates
    best_result = None
    for idx, (legal_ber, probes, count, models, hue_mapping) in enumerate(probe_candidates[:10]):
        print(f"  [{idx+1}] probe_count={count}, legal_ber(exact)={legal_ber:.6f}")

        if legal_ber > TARGET_LEGAL_BER:
            print(f"      Skip: legal_ber > {TARGET_LEGAL_BER}")
            continue

        sec = evaluate_illegal_exact(project_root, legal_positions, probes, models, hue_mapping)
        min_illegal = sec["min_illegal_ber"]
        print(f"      min_illegal_ber={min_illegal:.6f}, worst_pos={sec['worst_illegal_position']}")

        if min_illegal >= MIN_ILLEGAL_BER:
            print(f"      -> Passing exact screen, running FEC confirmation...")
            confirm_legal, confirm_illegal = confirm_with_fec(
                project_root, legal_positions, probes, models, hue_mapping, rng
            )
            print(f"      FEC confirm: legal={confirm_legal:.6f}, illegal={confirm_illegal:.6f}")

            if confirm_legal <= TARGET_LEGAL_BER and confirm_illegal >= MIN_ILLEGAL_BER:
                print(f"      *** SECURITY SATISFIED ***")
                return {
                    "position_combination": str(legal_positions),
                    "probe_count": count,
                    "probes": format_probes(probes),
                    "hue_mapping": format_hue_mapping(hue_mapping),
                    "legal_ber": f"{confirm_legal:.6f}",
                    "min_illegal_ber": f"{confirm_illegal:.6f}",
                    "average_illegal_ber": f"{sec['average_illegal_ber']:.6f}",
                    "worst_illegal_position": sec["worst_illegal_position"] or "",
                    "security_satisfied": "yes",
                }

        # 记录 best non-satisfied
        if best_result is None or min_illegal > float(best_result.get("min_illegal_ber", 0) or 0):
            best_result = {
                "position_combination": str(legal_positions),
                "probe_count": count,
                "probes": format_probes(probes),
                "hue_mapping": format_hue_mapping(hue_mapping),
                "legal_ber": f"{legal_ber:.6f}",
                "min_illegal_ber": f"{min_illegal:.6f}",
                "average_illegal_ber": f"{sec['average_illegal_ber']:.6f}",
                "worst_illegal_position": sec["worst_illegal_position"] or "",
                "security_satisfied": "no",
            }

    return best_result or {
        "position_combination": str(legal_positions),
        "probe_count": 0,
        "probes": "",
        "hue_mapping": "",
        "legal_ber": 1.0,
        "min_illegal_ber": "",
        "security_satisfied": "no",
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def write_results(results_file: str, rows: list[dict]) -> None:
    fieldnames = [
        "position_combination", "probe_count", "probes", "hue_mapping",
        "legal_ber", "min_illegal_ber", "average_illegal_ber",
        "worst_illegal_position", "security_satisfied",
    ]
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(project_root, "test-4", OUTPUT_RESULTS_FILENAME)

    all_positions = get_available_positions(project_root)
    print(f"Total positions: {len(all_positions)}")

    # 只测前几个组合快速验证（可以改成全部）
    import itertools
    combinations = list(itertools.combinations(all_positions, 4))
    print(f"Total 4-position combinations: {len(combinations)}")

    # 如需限制数量，取消下面注释
    combinations = combinations[:5]

    rng = random.Random(SEED)
    rows = []

    for idx, legal_positions in enumerate(combinations, start=1):
        print(f"\n[{idx}/{len(combinations)}] Processing {legal_positions}")
        result = search_for_combination(project_root, legal_positions, rng)
        rows.append(result)
        write_results(results_file, rows)
        print(f"  Result: legal_ber={result['legal_ber']}, min_illegal_ber={result.get('min_illegal_ber', 'N/A')}, satisfied={result['security_satisfied']}")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
