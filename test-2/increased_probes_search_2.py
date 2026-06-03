#!/usr/bin/env python3
"""
增加探针数量搜索2位置系统（yellow_shuffled数据集）

核心思路：
- 2位置系统有4种symbol combination，对应4个探针
- 预筛选：检查是否存在4个探针，使得每个非法位置攻击每个合法位置时，
  在bit=-1的2个探针或bit=+1的2个探针中至少有一个不匹配
- 对通过预筛选的探针集合，做完整hue mapping评估

目标：legal BER = 0, min illegal BER > 0.3
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yellow_shuffled_probe_search_2 import (
    build_legal_models,
    find_security_aware_hue_mapping,
    LIGHT_CONDITION,
)
import numpy as np
import pandas as pd
import itertools

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
all_probes = np.arange(5, 361, 5, dtype=float)


def get_codes_for_probes(pos, probes):
    """获取某位置在给定探针集合上的code"""
    csv_file = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION, f"{pos}.csv")
    df = pd.read_csv(csv_file)
    row_indices = [int((p/5)-1) for p in probes]
    mat = df.values[row_indices].astype(float)
    x = probes.astype(float)
    A = np.column_stack([x, np.ones_like(x)])
    coeffs = np.linalg.lstsq(A, mat, rcond=None)[0]
    trend = A @ coeffs
    residual = mat - trend
    _, _, Vt = np.linalg.svd(residual, full_matrices=False)
    w = Vt[0].copy()
    z = residual @ w
    if z[0] < 0:
        w = -w
        z = -z
    code = np.where(z >= 0, 1, -1)
    return code


def fast_prescreen(probes, pos_a, pos_b):
    """
    快速预筛选：检查是否存在4个探针，使得每个非法位置攻击每个合法位置时
    在bit=-1的2个探针或bit=+1的2个探针中至少有一个不匹配。
    
    返回：(is_safe, best_4tuple, failure_count)
    """
    code_a = get_codes_for_probes(pos_a, probes)
    code_b = get_codes_for_probes(pos_b, probes)
    
    n = len(probes)
    all_illegal = [p for p in range(1, 29) if p not in (pos_a, pos_b)]
    
    best_tuple = None
    best_failure = float('inf')
    
    # 遍历所有4元组
    for quad in itertools.combinations(range(n), 4):
        i1, i2, i3, i4 = quad
        
        # 我们需要分配4个symbol combination到4个探针
        # 为了最大化安全性，尝试所有4! = 24种分配方式
        # symbol: (-1,-1), (-1,+1), (+1,-1), (+1,+1)
        # 其中 (-1,-1)和(-1,+1)对应bit_a=-1
        # (+1,-1)和(+1,+1)对应bit_a=+1
        # (-1,-1)和(+1,-1)对应bit_b=-1
        # (-1,+1)和(+1,+1)对应bit_b=+1
        
        # 对每个分配，计算最坏非法位置的错误数
        best_assign_failure = float('inf')
        best_assign = None
        
        for perm in itertools.permutations(quad):
            p_m11, p_m1p1, p_p1m1, p_p1p1 = perm  # probe indices for symbols
            
            # 检查所有非法位置
            total_failures = 0
            for pos_il in all_illegal:
                code_il = get_codes_for_probes(pos_il, probes)
                
                # 攻击 pos_a (bit_a)
                # bit_a = -1: probes p_m11, p_m1p1
                match_m1_a = (code_il[p_m11] == code_a[p_m11]) and (code_il[p_m1p1] == code_a[p_m1p1])
                # bit_a = +1: probes p_p1m1, p_p1p1
                match_p1_a = (code_il[p_p1m1] == code_a[p_p1m1]) and (code_il[p_p1p1] == code_a[p_p1p1])
                
                if match_m1_a or match_p1_a:
                    # 非法位置可以正确解码 pos_a 的至少一个bit
                    total_failures += 1
                
                # 攻击 pos_b (bit_b)
                # bit_b = -1: probes p_m11, p_p1m1
                match_m1_b = (code_il[p_m11] == code_b[p_m11]) and (code_il[p_p1m1] == code_b[p_p1m1])
                # bit_b = +1: probes p_m1p1, p_p1p1
                match_p1_b = (code_il[p_m1p1] == code_b[p_m1p1]) and (code_il[p_p1p1] == code_b[p_p1p1])
                
                if match_m1_b or match_p1_b:
                    total_failures += 1
            
            if total_failures < best_assign_failure:
                best_assign_failure = total_failures
                best_assign = perm
        
        if best_assign_failure < best_failure:
            best_failure = best_assign_failure
            best_tuple = (quad, best_assign)
        
        if best_failure == 0:
            return True, best_tuple, 0
    
    return False, best_tuple, best_failure


def search_with_more_probes(pos_a, pos_b, probe_counts=[12, 14, 16], max_trials_per_count=100):
    """
    增加探针数量搜索2位置系统
    """
    print(f"Searching for positions ({pos_a}, {pos_b})")
    print(f"Dataset: {LIGHT_CONDITION}")
    print(f"Probe counts to try: {probe_counts}")
    print(f"Max trials per count: {max_trials_per_count}")
    print()
    
    rng = np.random.RandomState(42)
    best_result = None
    best_min_illegal = -1
    
    for probe_count in probe_counts:
        print(f"\nProbe count = {probe_count}")
        passed_prescreen = 0
        
        for trial in range(max_trials_per_count):
            probes = np.sort(rng.choice(all_probes, size=probe_count, replace=False))
            
            # 快速预筛选
            is_safe, best_tuple, failure_count = fast_prescreen(probes, pos_a, pos_b)
            
            if not is_safe:
                continue
            
            passed_prescreen += 1
            print(f"  Trial {trial+1}: PRESCREEN PASSED (failures={failure_count})")
            
            # 做完整评估
            try:
                result = find_security_aware_hue_mapping(project_root, (pos_a, pos_b), probes)
                
                if result['legal_ber'] > 0.001:
                    print(f"    Full eval: legal BER too high ({result['legal_ber']:.4f})")
                    continue
                
                min_il = result['min_illegal_ber']
                print(f"    Full eval: min_illegal={min_il:.4f}, avg={result['average_illegal_ber']:.4f}")
                
                if min_il > best_min_illegal:
                    best_min_illegal = min_il
                    best_result = result
                    best_result['probes'] = probes.copy()
                    print(f"    *** NEW BEST: min_illegal={min_il:.4f} ***")
                    
                    if min_il > 0.3:
                        print(f"    *** TARGET ACHIEVED! ***")
                        return best_result
                        
            except Exception as e:
                print(f"    Full eval ERROR: {e}")
                continue
        
        print(f"  Passed prescreen: {passed_prescreen}/{max_trials_per_count}")
    
    return best_result


# 测试所有7个预筛选的安全组合
safe_combos = [(5, 13), (5, 16), (5, 23), (6, 14), (13, 16), (13, 23), (16, 23)]

print(f"Testing {len(safe_combos)} pre-screened safe combinations")
print(f"With increased probe counts (12, 14, 16)\n")
print("="*70)

best_overall = None
best_overall_min = -1

for pos_a, pos_b in safe_combos:
    result = search_with_more_probes(pos_a, pos_b, probe_counts=[12, 14, 16], max_trials_per_count=50)
    
    if result and result['min_illegal_ber'] > best_overall_min:
        best_overall_min = result['min_illegal_ber']
        best_overall = result
        best_overall['positions'] = (pos_a, pos_b)
    
    print()

print("="*70)
print("FINAL RESULT:")
if best_overall:
    print(f"  Best combination: {best_overall['positions']}")
    print(f"  Best probes: {list(best_overall['probes'])}")
    print(f"  Best min illegal BER: {best_overall['min_illegal_ber']:.4f}")
    print(f"  Legal BER: {best_overall['legal_ber']:.4f}")
    print(f"  Avg illegal BER: {best_overall['average_illegal_ber']:.4f}")
    print(f"  Security satisfied: {best_overall['security_satisfied']}")
else:
    print("  No satisfactory result found.")
print("="*70)
print("\nDone.")
