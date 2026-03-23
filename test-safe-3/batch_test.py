#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
批量测量test-3的实验效果

遍历三种设备（15pro、mate40pro、p40）、五种光线情况（high、low、mid、white、yellow），
以及每种设备光线的28个位置的所有三个位置的组合情况（C(3,28)），计算总误码率，
并将结果输出为表格。
"""

import os
import itertools
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 导入test-3.py中的相关函数
import importlib.util
import sys

# 加载test-3-simple.py模块
spec = importlib.util.spec_from_file_location("test_3", "test-3-simple.py")
test = importlib.util.module_from_spec(spec)
sys.modules["test_3"] = test
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
    计算误码率，只统计未被丢弃的比特
    
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
        bits_tx = res["bits_bin"]
        for p, dec in enumerate(res["per_position"]):
            if not dec.is_discarded:
                total_bits += 1
                if dec.bit_hat_bin != bits_tx[p]:
                    error_bits += 1
    
    if total_bits == 0:
        return 0.0
    
    return error_bits / total_bits


def calculate_distance(pos1, pos2):
    """
    计算两个位置之间的距离
    
    Parameters
    ----------
    pos1 : 第一个位置编号
    pos2 : 第二个位置编号
    
    Returns
    -------
    float : 两个位置之间的距离
    """
    # 计算行和列
    row1 = (pos1 - 1) // 7 + 1
    col1 = (pos1 - 1) % 7 + 1
    row2 = (pos2 - 1) // 7 + 1
    col2 = (pos2 - 1) % 7 + 1
    
    # 计算欧氏距离
    distance = ((row1 - row2) ** 2 + (col1 - col2) ** 2) ** 0.5
    return distance

def generate_valid_position_combinations():
    """
    生成有效的位置组合，确保两两之间距离至少为2m
    
    Returns
    -------
    List[Tuple[int, int, int]] : 有效的位置组合列表
    """
    valid_combinations = []
    
    # 生成所有可能的三个位置的组合
    all_combinations = list(itertools.combinations(range(1, num_positions + 1), 3))
    
    # 筛选出两两之间距离至少为2m的组合
    for combo in all_combinations:
        pos1, pos2, pos3 = combo
        
        # 计算两两之间的距离
        dist12 = calculate_distance(pos1, pos2)
        dist13 = calculate_distance(pos1, pos3)
        dist23 = calculate_distance(pos2, pos3)
        
        # 检查是否都至少为2m
        if dist12 >= 2 and dist13 >= 2 and dist23 >= 2:
            valid_combinations.append(combo)
    
    print(f"Total possible combinations: {len(all_combinations)}")
    print(f"Valid combinations (distance >= 2m): {len(valid_combinations)}")
    
    return valid_combinations

def run_experiment(device: str, light: str, positions: Tuple[int, int, int]) -> float:
    """
    运行单个实验，计算误码率
    
    Parameters
    ----------
    device : 设备名称
    light : 光线情况
    positions : 三个位置的索引
    
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
        print(f"  Step 1: Finding optimal probe count...")
        # 自动找出最佳探针数（范围5~15，减少计算量）
        best_probe_result = test.find_optimal_probe_count(*csv_files, min_probes=5, max_probes=15)
        
        # 处理返回结果
        if isinstance(best_probe_result, tuple) and len(best_probe_result) == 2:
            best_probe_count, best_probes = best_probe_result
        else:
            # 兼容旧版本
            best_probe_count = best_probe_result
            best_probes = None
        
        # 清除当前行并打印最佳探针数
        print(" " * 80, end="\r")
        print(f"  Best probe count: {best_probe_count}")
        
        print(f"  Step 2: Loading data...")
        # 获取数据
        models, _, hue_mapping, _, _ = test.get_data_from_csv(*csv_files, best_probe_count, best_probes)
        
        print(f"  Step 3: Generating bit blocks...")
        # 生成比特块（减少数量，提高速度）
        bit_blocks = generate_bit_blocks(3, min(1000, num_bits))
        
        print(f"  Step 4: Finding optimal threshold...")
        # 自动学习最优阈值
        optimal_threshold = test.find_optimal_threshold(models, bit_blocks, hue_mapping)
        
        print(f"  Step 5: Running simulation...")
        # 运行仿真，使用最优阈值
        results = test.simulate_blocks(models, bit_blocks, hue_mapping, threshold=optimal_threshold)
        
        print(f"  Step 6: Calculating BER...")
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
    # 生成有效的位置组合，确保两两之间距离至少为2m
    position_combinations = generate_valid_position_combinations()
    print(f"Generated {len(position_combinations)} valid position combinations")
    
    # 检查是否存在已有的结果文件
    results_file = "batch_test_results_new.csv"
    existing_results = {}
    total_ber = 0.0
    total_experiments_completed = 0
    
    if os.path.exists(results_file):
        df_existing = pd.read_csv(results_file)
        for _, row in df_existing.iterrows():
            key = (row["device"], row["light"], tuple([row["position1"], row["position2"], row["position3"]]))
            existing_results[key] = row["ber"]
            # 累计已完成实验的误码率
            total_ber += row["ber"]
            total_experiments_completed += 1
    
    # 计算总实验次数
    total_experiments = len(devices) * len(light_conditions) * len(position_combinations)
    current_experiment = total_experiments_completed
    
    # 遍历所有设备、光线情况和位置组合
    for device in devices:
        for light in light_conditions:
            for positions in position_combinations:
                # 检查是否已经计算过
                key = (device, light, positions)
                if key in existing_results:
                    print(f"Skipping {device}/{light}/{positions} (already calculated)")
                    continue
                
                # 更新进度
                current_experiment += 1
                print(f"\nRunning experiment {current_experiment}/{total_experiments}: {device}/{light}/{positions}")
                
                # 运行实验
                ber = run_experiment(device, light, positions)
                
                if ber is not None:
                    # 更新累计误码率
                    total_ber += ber
                    total_experiments_completed += 1
                    # 计算累计平均误码率
                    cumulative_ber = total_ber / total_experiments_completed if total_experiments_completed > 0 else 0.0
                    
                    # 立即保存结果
                    result = {
                        "device": device,
                        "light": light,
                        "position1": positions[0],
                        "position2": positions[1],
                        "position3": positions[2],
                        "ber": ber,
                        "cumulative_ber": cumulative_ber
                    }
                    
                    # 创建DataFrame并写入文件
                    df = pd.DataFrame([result])
                    if os.path.exists(results_file):
                        # 检查文件是否有cumulative_ber列
                        df_existing = pd.read_csv(results_file)
                        if "cumulative_ber" not in df_existing.columns:
                            # 如果没有，添加列名
                            with open(results_file, 'r') as f:
                                header = f.readline().strip() + ",cumulative_ber\n"
                            with open(results_file, 'w') as f:
                                f.write(header)
                                f.writelines(f.readlines()[1:])
                        df.to_csv(results_file, mode='a', header=False, index=False)
                    else:
                        df.to_csv(results_file, index=False)
                    
                    print(f"  BER: {ber:.6f}")
                    print(f"  Cumulative BER: {cumulative_ber:.6f}")
                    print(f"  Result saved to {results_file}")
    
    print("\nAll experiments completed.")
    
    # 生成可视化表格
    if os.path.exists(results_file):
        print("\n========== 实验结果可视化 ==========")
        
        # 读取结果
        df = pd.read_csv(results_file)
        
        # 计算平均误码率
        avg_df = df.groupby(['device', 'light'])['ber'].mean().unstack()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 创建热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(avg_df, annot=True, cmap='coolwarm', fmt='.4f', linewidths=.5)
        plt.title('不同设备和光线条件下的平均误码率')
        plt.xlabel('设备')
        plt.ylabel('光线情况')
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('batch_test_results.png')
        print("Visualization saved to batch_test_results.png")
        
        # 显示图表
        plt.show()
    else:
        print("No results to visualize.")


if __name__ == "__main__":
    main()