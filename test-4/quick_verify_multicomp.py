#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证：用新的多主成分系统重新评估旧 zero-BER probe sets
"""
import os
import sys
import csv
import ast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test
import numpy as np

LIGHT_CONDITION = "white"
SOURCE_FILE = "batch_test_zero_ber_results.csv"

def parse_tuple(text):
    return tuple(ast.literal_eval(text))

def parse_list(text):
    return ast.literal_eval(text)

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_path = os.path.join(project_root, "test-4", SOURCE_FILE)
    
    configs = []
    with open(source_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["test_ber"]) != 0.0:
                continue
            configs.append({
                "position_combination": parse_tuple(row["position_combination"]),
                "probes": parse_list(row["best_probes"]),
            })
    
    print(f"Loaded {len(configs)} zero-BER configs")
    print(f"Using N_COMPONENTS={test.N_COMPONENTS}, ALPHA={test.ALPHA}, BETA={test.BETA}")
    print("-" * 80)
    
    for idx, config in enumerate(configs[:5], 1):
        positions = config["position_combination"]
        probes = np.asarray(config["probes"], dtype=float)
        
        csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition=LIGHT_CONDITION)
        matrices = test.load_selected_rows(csv_files, probes)
        models = [test.extract_fingerprint(probes, mat, force_positive_first=True) for mat in matrices]
        models = test.align_model_directions(models)
        
        # 用默认 hue mapping 测试 legal BER
        hue_mapping = test.build_hue_mapping(models, probes, mapping_eval_bits=500, top_k_per_combination=3)
        
        # FEC 评估
        rng = test.random.Random(42)
        info_bits = test.generate_random_information_bits(2000, len(positions), rng=rng)
        legal_ber = test.evaluate_blocks_ber_with_convolutional_fec(models, info_bits, hue_mapping)
        
        print(f"[{idx}] {positions}")
        print(f"    Probes: {config['probes']}")
        print(f"    Legal BER (new multi-comp system): {legal_ber:.6f}")
        
        # 显示每个位置的多主成分结构
        for mi, model in enumerate(models):
            if model.Z is not None and model.Z.ndim > 1:
                s2_ratio = np.linalg.norm(model.Z[:, 1]) / np.linalg.norm(model.Z[:, 0]) if model.Z.shape[1] > 1 else 0
                s3_ratio = np.linalg.norm(model.Z[:, 2]) / np.linalg.norm(model.Z[:, 0]) if model.Z.shape[1] > 2 else 0
                print(f"    Pos {positions[mi]}: PC2 energy={s2_ratio:.3f}, PC3 energy={s3_ratio:.3f}")
        print()

if __name__ == "__main__":
    main()
