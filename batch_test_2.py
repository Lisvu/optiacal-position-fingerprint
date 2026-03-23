#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
批量测量test-2-simple的实验效果

遍历所有C(28,2)位置组合，每次发送10000位随机比特，
计算误码率和累计误码率，并将结果输出到CSV文件。
"""

import os
import itertools
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# 导入test-2-simple.py中的相关函数
import importlib.util
import sys

# 加载test-2-simple.py模块
spec = importlib.util.spec_from_file_location("test_2", "test-2-simple.py")
test = importlib.util.module_from_spec(spec)
sys.modules["test_2"] = test
spec.loader.exec_module(test)

Array = np.ndarray

# 设备列表
devices = ["15pro", "mate40pro", "p40"]

# 光线情况列表
light_conditions = ["high", "low", "mid", "white", "yellow"]

# 位置数量
num_positions = 28

# 发送比特数
num_bits = 10000

# 探针数量
num_probes = 15

def generate_bit_blocks(num_positions: int, num_bits: int) -> List[Array]:
    """
    生成随机的比特块
    
    Parameters
    ----------
    num_positions : 位置数量
    num_bits : 发送比特数
    
    Returns
    -------
    List[Array] : 比特块列表
    """
    bit_blocks = []
    for _ in range(num_bits):
        # 生成随机比特，1表示+1，0表示-1
        bits = np.random.randint(0, 2, num_positions)
        # 转换为+1/-1形式
        bits_pm = np.where(bits > 0, 1, -1)
        bit_blocks.append(bits_pm)
    return bit_blocks

def calculate_ber(results: List[dict]) -> float:
    """
    计算误码率
    
    Parameters
    ----------
    results : 解码结果列表
    
    Returns
    -------
    float : 误码率
    """
    total_bits = 0
    error_bits = 0
    
    for res in results:
        bits_tx = test.pm1_to_bin(res["bits_pm"])
        for p, dec in enumerate(res["per_position"]):
            total_bits += 1
            if dec.bit_hat_bin != bits_tx[p]:
                error_bits += 1
    
    if total_bits == 0:
        return 0.0
    
    return error_bits / total_bits

def run_experiment(device: str, light: str, positions: Tuple[int, int]) -> float:
    """
    运行单个实验，计算误码率
    
    Parameters
    ----------
    device : 设备名称
    light : 光线情况
    positions : 两个位置的索引
    
    Returns
    -------
    float : 误码率
    """
    # 构建文件路径
    base_path = f"data\\{device}\\{light}"
    csv_files = [f"{base_path}\\{pos}.csv" for pos in positions]
    
    # 检查文件是否存在
    for file_path in csv_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
    
    try:
        print(f"  Step 1: Loading data...")
        # 获取数据
        models, _, hue_mapping = test.get_data_from_csv(*csv_files, num_probes)
        
        print(f"  Step 2: Generating bit blocks...")
        # 生成比特块
        bit_blocks = generate_bit_blocks(2, num_bits)
        
        print(f"  Step 3: Running simulation...")
        # 运行仿真
        results = test.simulate_blocks(models, bit_blocks, hue_mapping)
        
        print(f"  Step 4: Calculating BER...")
        # 计算误码率
        ber = calculate_ber(results)
        
        print(f"  Experiment completed successfully")
        return ber
    except Exception as e:
        import traceback
        print(f"Error running experiment for {device}/{light}/{positions}:")
        print(traceback.format_exc())
        return None

def main() -> None:
    """
    主函数，遍历所有设备、光线情况和位置组合，计算误码率
    """
    # 生成所有可能的两个位置的组合
    position_combinations = list(itertools.combinations(range(1, num_positions + 1), 2))
    print(f"Total position combinations: {len(position_combinations)}")
    
    # 保存结果到CSV文件
    results_file = "batch_test_2_results.csv"
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["device", "light", "positions", "ber", "cumulative_ber"])
    
    # 累计误码率统计
    cumulative_errors = 0
    cumulative_bits = 0
    
    # 遍历所有设备
    for device in devices:
        # 遍历所有光线情况
        for light in light_conditions:
            print(f"\n=== Running experiments for {device}/{light} ===")
            
            # 遍历所有位置组合
            for i, positions in enumerate(position_combinations):
                print(f"\nExperiment {i+1}/{len(position_combinations)}: {positions}")
                
                # 运行实验
                ber = run_experiment(device, light, positions)
                
                if ber is not None:
                    # 更新累计误码率
                    experiment_errors = int(ber * num_bits * 2)  # 2 positions * num_bits
                    experiment_bits = num_bits * 2
                    cumulative_errors += experiment_errors
                    cumulative_bits += experiment_bits
                    cumulative_ber = cumulative_errors / cumulative_bits if cumulative_bits > 0 else 0.0
                    
                    # 保存结果
                    with open(results_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([device, light, str(positions), ber, cumulative_ber])
                    
                    print(f"  BER: {ber:.6f}, Cumulative BER: {cumulative_ber:.6f}")
                else:
                    print(f"  Experiment failed, skipping...")
    
    print(f"\n=== All experiments completed ===")
    print(f"Total cumulative BER: {cumulative_ber:.6f}")

if __name__ == "__main__":
    import csv
    main()