#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双探针联合解码安全实验 (Dual-Probe Joint Decoding)

核心思想：
  1. 为每个合法位置选两组互补探针 P_a 和 P_b
  2. 发送端用 P_a 编码（和原来一样）
  3. 接收端用 P_a 和 P_b 分别独立解码
  4. 只有两次解码结果一致时才接受
  5. 非法位置即使知道 P_a，不知道 P_b 就无法验证，BER 接近 0.5

安全增益来源：
  - 非法位置要同时猜对两组探针才能正确解码
  - 猜对一组探针的概率 ≈ 1/C(73,8)，猜对两组的概率 ≈ [1/C(73,8)]^2
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

# ============================================================================
# Config
# ============================================================================
LIGHT_CONDITION = "white"
OUTPUT_RESULTS_FILENAME = "dual_probe_security_results_4.csv"

# 安全目标
TARGET_LEGAL_BER = 0.02       # 放宽到 0.02，因为双重验证会有损失
MIN_ILLEGAL_BER = 0.2         # 非法位置 secure BER >= 0.2

# 探针配置
PROBE_COUNT_A = 8             # 第一组探针数量
PROBE_COUNT_B = 8             # 第二组探针数量（互补）
MIN_INTERVAL = 30             # 最小探针间隔

# 评估
FEC_CONFIRM_BITS = 5000

SEED = 20260510


# ============================================================================
# Helpers
# ============================================================================
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


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def format_hue_mapping(hue_mapping: dict) -> str:
    if not hue_mapping:
        return ""
    return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(hue_mapping.items())) + "}"


# ============================================================================
# Dual-Probe Model Builder
# ============================================================================
def build_dual_probe_models(
    project_root: str,
    legal_positions: Sequence[int],
    probes_a: np.ndarray,
    probes_b: np.ndarray,
) -> tuple[list, list, dict]:
    """
    为每个合法位置构建两组探针的指纹模型。
    
    返回: (models_a, models_b, hue_mapping)
    - models_a: 用 P_a 构建的模型列表
    - models_b: 用 P_b 构建的模型列表
    - hue_mapping: 基于 P_a 的 hue mapping（发送端只用 P_a）
    """
    csv_files = test.build_csv_files_for_positions(project_root, legal_positions, light_condition=LIGHT_CONDITION)
    
    # Build models with P_a
    matrices_a = test.load_selected_rows(csv_files, probes_a)
    models_a = [test.extract_fingerprint(probes_a, mat, force_positive_first=True) for mat in matrices_a]
    models_a = test.align_model_directions(models_a)
    
    # Build models with P_b
    matrices_b = test.load_selected_rows(csv_files, probes_b)
    models_b = [test.extract_fingerprint(probes_b, mat, force_positive_first=True) for mat in matrices_b]
    models_b = test.align_model_directions(models_b)
    
    # Hue mapping based on P_a (sender uses P_a only)
    hue_mapping = test.build_hue_mapping(
        models_a, probes_a,
        mapping_eval_bits=500, top_k_per_combination=3,
        use_amplitude_aware=True,
    )
    
    return models_a, models_b, hue_mapping


# ============================================================================
# Dual-Probe Decoder
# ============================================================================
def decode_dual_probe(
    models_a: list,
    models_b: list,
    hue_seq: np.ndarray,
    probe_to_row_a: dict,
    probe_to_row_b: dict,
) -> list[int]:
    """
    双探针联合解码：
    1. 用 P_a 的 model 解码 → bit_a
    2. 用 P_b 的 model 解码 → bit_b
    3. 只有 bit_a == bit_b 时才返回结果，否则返回随机（模拟检测到不一致）
    
    实际上在实验中，我们分别计算两组的 BER，看非法位置是否无法同时满足两组。
    """
    bits_a = []
    bits_b = []
    
    for model_a, model_b in zip(models_a, models_b):
        # Decode with P_a
        Y_obs_a = test.observe_block_from_measured_matrix(hue_seq, model_a.Y, probe_to_row_a)
        if model_a.eff_w is not None and model_a.eff_code is not None:
            dec_a = test.decode_local_block(Y_obs_a, model_a.eff_w, model_a.eff_code)
        else:
            dec_a = test.decode_local_block(Y_obs_a, model_a.w, model_a.code)
        bits_a.append(dec_a.bit_hat_bin)
        
        # Decode with P_b
        Y_obs_b = test.observe_block_from_measured_matrix(hue_seq, model_b.Y, probe_to_row_b)
        if model_b.eff_w is not None and model_b.eff_code is not None:
            dec_b = test.decode_local_block(Y_obs_b, model_b.eff_w, model_b.eff_code)
        else:
            dec_b = test.decode_local_block(Y_obs_b, model_b.w, model_b.code)
        bits_b.append(dec_b.bit_hat_bin)
    
    # Joint decision: only accept if both agree
    # If disagree, return the opposite of ground truth (simulating detection)
    # But in evaluation, we treat disagreement as error
    return bits_a, bits_b


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_dual_probe_legal(
    models_a: list,
    models_b: list,
    hue_mapping: dict,
    probe_to_row_a: dict,
    probe_to_row_b: dict,
) -> tuple[float, float, float]:
    """
    评估合法位置的双探针解码性能。
    
    返回: (ber_a, ber_b, ber_joint)
    - ber_a: 只用 P_a 解码的 BER
    - ber_b: 只用 P_b 解码的 BER
    - ber_joint: 联合解码（两结果必须一致）的 BER
    """
    blocks = test.generate_all_bit_blocks(len(models_a))
    
    errors_a = np.zeros(len(models_a))
    errors_b = np.zeros(len(models_a))
    errors_joint = np.zeros(len(models_a))
    totals = np.zeros(len(models_a))
    
    codes_a = [m.eff_code if m.eff_code is not None else m.code for m in models_a]
    
    for bits_pm in blocks:
        _, symbol_combinations = test.build_symbol_sequence(bits_pm, codes_a)
        hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
        bits_tx = test.pm1_to_bin(bits_pm)
        
        bits_a, bits_b = decode_dual_probe(models_a, models_b, hue_seq, probe_to_row_a, probe_to_row_b)
        
        for li in range(len(models_a)):
            totals[li] += 1
            if bits_a[li] != bits_tx[li]:
                errors_a[li] += 1
            if bits_b[li] != bits_tx[li]:
                errors_b[li] += 1
            # Joint: error if either one is wrong OR they disagree
            if bits_a[li] != bits_tx[li] or bits_b[li] != bits_tx[li] or bits_a[li] != bits_b[li]:
                errors_joint[li] += 1
    
    raw_a = errors_a / totals
    raw_b = errors_b / totals
    raw_joint = errors_joint / totals
    
    secure_a = np.minimum(raw_a, 1 - raw_a)
    secure_b = np.minimum(raw_b, 1 - raw_b)
    secure_joint = np.minimum(raw_joint, 1 - raw_joint)
    
    return (
        float(np.mean(secure_a)),
        float(np.mean(secure_b)),
        float(np.mean(secure_joint)),
    )


def evaluate_dual_probe_illegal(
    project_root: str,
    legal_positions: Sequence[int],
    probes_a: np.ndarray,
    probes_b: np.ndarray,
    models_a: list,
    models_b: list,
    hue_mapping: dict,
    probe_to_row_a: dict,
    probe_to_row_b: dict,
) -> dict:
    """评估非法位置的安全性。"""
    legal_codes_a = [m.eff_code if m.eff_code is not None else m.code for m in models_a]
    blocks = test.generate_all_bit_blocks(len(legal_positions))
    all_positions = get_available_positions(project_root)
    
    global_min_a = float("inf")
    global_min_b = float("inf")
    global_min_joint = float("inf")
    worst_pos = None
    
    for pos in all_positions:
        if pos in legal_positions:
            continue
        
        csv_file = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION, f"{pos}.csv")
        mat = test.load_csv_matrix(csv_file)
        
        # Build illegal model with P_a
        row_indices_a = [int((p / 5) - 1) for p in probes_a]
        illegal_mat_a = mat[row_indices_a].astype(float, copy=False)
        illegal_model_a = test.extract_fingerprint(probes_a, illegal_mat_a, force_positive_first=True)
        
        # Build illegal model with P_b
        row_indices_b = [int((p / 5) - 1) for p in probes_b]
        illegal_mat_b = mat[row_indices_b].astype(float, copy=False)
        illegal_model_b = test.extract_fingerprint(probes_b, illegal_mat_b, force_positive_first=True)
        
        # Align directions
        corr_a = test.calculate_correlation(models_a[0].eff_code if models_a[0].eff_code is not None else models_a[0].code,
                                            illegal_model_a.eff_code if illegal_model_a.eff_code is not None else illegal_model_a.code)
        if corr_a < 0:
            if illegal_model_a.W is not None and illegal_model_a.W.ndim > 1:
                illegal_model_a.W = -illegal_model_a.W
                illegal_model_a.Z = -illegal_model_a.Z
                illegal_model_a.multi_code = -illegal_model_a.multi_code
                illegal_model_a.eff_w = -illegal_model_a.eff_w
                illegal_model_a.eff_code = -illegal_model_a.eff_code
            else:
                illegal_model_a.w = -illegal_model_a.w
                illegal_model_a.z = -illegal_model_a.z
                illegal_model_a.code = -illegal_model_a.code
        
        corr_b = test.calculate_correlation(models_b[0].eff_code if models_b[0].eff_code is not None else models_b[0].code,
                                            illegal_model_b.eff_code if illegal_model_b.eff_code is not None else illegal_model_b.code)
        if corr_b < 0:
            if illegal_model_b.W is not None and illegal_model_b.W.ndim > 1:
                illegal_model_b.W = -illegal_model_b.W
                illegal_model_b.Z = -illegal_model_b.Z
                illegal_model_b.multi_code = -illegal_model_b.multi_code
                illegal_model_b.eff_w = -illegal_model_b.eff_w
                illegal_model_b.eff_code = -illegal_model_b.eff_code
            else:
                illegal_model_b.w = -illegal_model_b.w
                illegal_model_b.z = -illegal_model_b.z
                illegal_model_b.code = -illegal_model_b.code
        
        errors_a = np.zeros(len(legal_positions))
        errors_b = np.zeros(len(legal_positions))
        errors_joint = np.zeros(len(legal_positions))
        totals = np.zeros(len(legal_positions))
        
        for bits_pm in blocks:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes_a)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            bits_tx = test.pm1_to_bin(bits_pm)
            
            bits_a = []
            bits_b = []
            for li in range(len(legal_positions)):
                # P_a decode
                Y_obs_a = test.observe_block_from_measured_matrix(hue_seq, illegal_model_a.Y, probe_to_row_a)
                if illegal_model_a.eff_w is not None and illegal_model_a.eff_code is not None:
                    dec_a = test.decode_local_block(Y_obs_a, illegal_model_a.eff_w, illegal_model_a.eff_code)
                else:
                    dec_a = test.decode_local_block(Y_obs_a, illegal_model_a.w, illegal_model_a.code)
                bits_a.append(dec_a.bit_hat_bin)
                
                # P_b decode
                Y_obs_b = test.observe_block_from_measured_matrix(hue_seq, illegal_model_b.Y, probe_to_row_b)
                if illegal_model_b.eff_w is not None and illegal_model_b.eff_code is not None:
                    dec_b = test.decode_local_block(Y_obs_b, illegal_model_b.eff_w, illegal_model_b.eff_code)
                else:
                    dec_b = test.decode_local_block(Y_obs_b, illegal_model_b.w, illegal_model_b.code)
                bits_b.append(dec_b.bit_hat_bin)
            
            for li in range(len(legal_positions)):
                totals[li] += 1
                if bits_a[li] != bits_tx[li]:
                    errors_a[li] += 1
                if bits_b[li] != bits_tx[li]:
                    errors_b[li] += 1
                if bits_a[li] != bits_tx[li] or bits_b[li] != bits_tx[li] or bits_a[li] != bits_b[li]:
                    errors_joint[li] += 1
        
        raw_a = errors_a / totals
        raw_b = errors_b / totals
        raw_joint = errors_joint / totals
        
        secure_a = np.minimum(raw_a, 1 - raw_a)
        secure_b = np.minimum(raw_b, 1 - raw_b)
        secure_joint = np.minimum(raw_joint, 1 - raw_joint)
        
        if secure_joint.min() < global_min_joint:
            global_min_joint = secure_joint.min()
            global_min_a = secure_a.min()
            global_min_b = secure_b.min()
            worst_pos = pos
    
    return {
        "min_illegal_ber_a": float(global_min_a),
        "min_illegal_ber_b": float(global_min_b),
        "min_illegal_ber_joint": float(global_min_joint),
        "worst_illegal_position": worst_pos,
    }


# ============================================================================
# Search
# ============================================================================
def random_search_dual_probes(
    project_root: str,
    legal_positions: Sequence[int],
    all_probes: list,
    rng: random.Random,
    max_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, float, float, float, list, list, dict, dict, dict]:
    """
    随机搜索两组互补探针。
    
    返回最佳组合的 (probes_a, probes_b, ber_joint, illegal_joint, ...)
    """
    best_probes_a = None
    best_probes_b = None
    best_ber_joint = float("inf")
    best_illegal_joint = 0.0
    best_models_a = None
    best_models_b = None
    best_mapping = None
    best_probe_to_row_a = None
    best_probe_to_row_b = None
    
    for i in range(max_samples):
        # Sample P_a
        count_a = rng.choice([7, 8, 9, 10])
        probes_a = np.sort(np.asarray(rng.sample(all_probes, count_a), dtype=float))
        if not test.is_valid_probe_set(probes_a, MIN_INTERVAL):
            continue
        
        # Sample P_b (complementary: mostly different probes)
        remaining = [p for p in all_probes if p not in probes_a]
        if len(remaining) < 7:
            continue
        count_b = rng.choice([7, 8, 9, 10])
        probes_b = np.sort(np.asarray(rng.sample(remaining, count_b), dtype=float))
        if not test.is_valid_probe_set(probes_b, MIN_INTERVAL):
            continue
        
        try:
            models_a, models_b, hue_mapping = build_dual_probe_models(
                project_root, legal_positions, probes_a, probes_b
            )
            probe_to_row_a = test.build_probe_to_row(probes_a)
            probe_to_row_b = test.build_probe_to_row(probes_b)
            
            ber_a, ber_b, ber_joint = evaluate_dual_probe_legal(
                models_a, models_b, hue_mapping, probe_to_row_a, probe_to_row_b
            )
            
            if ber_joint < best_ber_joint:
                best_ber_joint = ber_joint
                best_probes_a = probes_a.copy()
                best_probes_b = probes_b.copy()
                best_models_a = models_a
                best_models_b = models_b
                best_mapping = hue_mapping
                best_probe_to_row_a = probe_to_row_a
                best_probe_to_row_b = probe_to_row_b
                
                if i % 20 == 0:
                    print(f"    sample {i}: new best joint_ber={ber_joint:.4f} (a={ber_a:.4f}, b={ber_b:.4f})")
            
            if ber_joint <= TARGET_LEGAL_BER:
                print(f"    sample {i}: reached target! joint_ber={ber_joint:.4f}")
                # Evaluate illegal
                sec = evaluate_dual_probe_illegal(
                    project_root, legal_positions, probes_a, probes_b,
                    models_a, models_b, hue_mapping, probe_to_row_a, probe_to_row_b
                )
                if sec["min_illegal_ber_joint"] >= MIN_ILLEGAL_BER:
                    print(f"    *** SECURITY SATISFIED! illegal_joint={sec['min_illegal_ber_joint']:.4f} ***")
                    return (probes_a, probes_b, ber_joint,
                            sec["min_illegal_ber_joint"], sec["min_illegal_ber_a"],
                            models_a, models_b, hue_mapping, probe_to_row_a, probe_to_row_b)
        except Exception as e:
            continue
    
    # Return best found even if not satisfied
    if best_probes_a is not None:
        sec = evaluate_dual_probe_illegal(
            project_root, legal_positions, best_probes_a, best_probes_b,
            best_models_a, best_models_b, best_mapping,
            best_probe_to_row_a, best_probe_to_row_b
        )
        return (best_probes_a, best_probes_b, best_ber_joint,
                sec["min_illegal_ber_joint"], sec["min_illegal_ber_a"],
                best_models_a, best_models_b, best_mapping,
                best_probe_to_row_a, best_probe_to_row_b)
    
    return None


# ============================================================================
# Main
# ============================================================================
def search_combination(project_root: str, positions: tuple[int, ...], rng: random.Random) -> dict:
    print(f"\n[Search] {positions}")
    all_probes = (5 + np.arange(73) * 5).astype(float).tolist()
    
    result = random_search_dual_probes(project_root, positions, all_probes, rng, max_samples=300)
    
    if result is None:
        return {
            "position_combination": str(positions),
            "probe_count_a": 0, "probes_a": "",
            "probe_count_b": 0, "probes_b": "",
            "legal_ber_a": "", "legal_ber_b": "", "legal_ber_joint": "1.0",
            "min_illegal_ber_a": "", "min_illegal_ber_b": "", "min_illegal_ber_joint": "",
            "security_satisfied": "no",
        }
    
    probes_a, probes_b, ber_joint, illegal_joint, illegal_a, models_a, models_b, hue_mapping, _, _ = result
    
    # 重新评估以获取 ber_a 和 ber_b
    probe_to_row_a = test.build_probe_to_row(probes_a)
    probe_to_row_b = test.build_probe_to_row(probes_b)
    ber_a, ber_b, _ = evaluate_dual_probe_legal(models_a, models_b, hue_mapping, probe_to_row_a, probe_to_row_b)
    
    print(f"  Best result:")
    print(f"    P_a: count={len(probes_a)}, probes={format_probes(probes_a)}")
    print(f"    P_b: count={len(probes_b)}, probes={format_probes(probes_b)}")
    print(f"    P_a only legal BER: {ber_a:.4f}")
    print(f"    P_b only legal BER: {ber_b:.4f}")
    print(f"    Joint legal BER: {ber_joint:.4f}")
    print(f"    Illegal joint BER: {illegal_joint:.4f}")
    print(f"    Illegal P_a only: {illegal_a:.4f}")
    
    satisfied = ber_joint <= TARGET_LEGAL_BER and illegal_joint >= MIN_ILLEGAL_BER
    
    return {
        "position_combination": str(positions),
        "probe_count_a": len(probes_a), "probes_a": format_probes(probes_a),
        "probe_count_b": len(probes_b), "probes_b": format_probes(probes_b),
        "legal_ber_a": f"{ber_a:.6f}",
        "legal_ber_b": f"{ber_b:.6f}",
        "legal_ber_joint": f"{ber_joint:.6f}",
        "min_illegal_ber_a": f"{illegal_a:.6f}",
        "min_illegal_ber_b": "",
        "min_illegal_ber_joint": f"{illegal_joint:.6f}",
        "security_satisfied": "yes" if satisfied else "no",
    }


def write_results(results_file: str, rows: list[dict]) -> None:
    fieldnames = [
        "position_combination", "probe_count_a", "probes_a", "probe_count_b", "probes_b",
        "legal_ber_joint", "min_illegal_ber_joint", "security_satisfied",
        "legal_ber_a", "legal_ber_b", "min_illegal_ber_a", "min_illegal_ber_b",
    ]
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(project_root, "test-4", OUTPUT_RESULTS_FILENAME)
    
    positions = get_available_positions(project_root)
    print(f"Total positions: {len(positions)}")
    print(f"Dual-Probe Joint Decoding Experiment")
    print(f"Config: N_COMP={test.N_COMPONENTS}, USE_AMPLITUDE_AWARE={test.USE_AMPLITUDE_AWARE}")
    print(f"Targets: joint_legal<={TARGET_LEGAL_BER}, illegal_joint>={MIN_ILLEGAL_BER}")
    
    # Test a few combinations
    combinations = list(itertools.combinations(positions, 4))[:5]
    print(f"Testing {len(combinations)} combinations")
    
    rng = random.Random(SEED)
    rows = []
    
    for idx, pos in enumerate(combinations, start=1):
        print(f"\n[{idx}/{len(combinations)}] Processing {pos}")
        result = search_combination(project_root, pos, rng)
        rows.append(result)
        write_results(results_file, rows)
        print(f"  Summary: legal_joint={result.get('legal_ber_joint', 'N/A')}, "
              f"illegal_joint={result.get('min_illegal_ber_joint', 'N/A')}, "
              f"satisfied={result['security_satisfied']}")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
