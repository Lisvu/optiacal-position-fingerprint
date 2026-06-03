# Global Guarded 实验与结果说明

本文档整理 `global_guarded_experiments_migrated` 下的实验代码、对应数据集、结果文件和当前已完成的位置组合数量。

## 目录结构

- `code/`: 实验入口脚本和核心实现。
- `results/`: 已生成的实验结果。
- `results/dataset_compare/<dataset>/k*/`: 指定数据集上的对比实验结果。
- `results/k*/global_guarded/`: 默认数据集 `data\15pro\yellow_shuffled` 上的结果。

## 实验入口与数据集

| 实验代码 | 数据集 | k 范围 | 结果位置 |
|---|---|---:|---|
| `code/run_global_guarded_dataset_compare.py` | `data\mate40pro\high`, `data\15pro\mid` | `k2`, `k3`, `k7` | `results/dataset_compare/mate40pro_high/`, `results/dataset_compare/15pro_mid/` |
| `code/run_mate40pro_high_column_shuffled.py` | `data\mate40pro\high_column_shuffled` | `k2`, `k3`, `k4`, `k7`, `k8` | `results/dataset_compare/mate40pro_high_column_shuffled/` |
| `code/run_mate40pro_high_column_shuffled_batch.py` | `data\mate40pro\high_column_shuffled` | `k2` 到 `k20` | `results/dataset_compare/mate40pro_high_column_shuffled/` |
| `code/run_p40_low_column_shuffled_batch.py` | `data\p40\low_column_shuffled` | `k2` 到 `k20` | `results/dataset_compare/p40_low_column_shuffled/` |
| `code/run_global_guarded_20samples.py` | 默认 `data\15pro\yellow_shuffled` | `k2` 到 `k10` | `results/k*/global_guarded/results_summary_20samples.csv` |
| `code/k*/global_guarded/run_k*_global_guarded.py` | 默认 `data\15pro\yellow_shuffled` | 单个 k wrapper: `k2`, `k3`, `k4`, `k7`, `k8` | `results/k*/global_guarded/` |

## 数据集完成情况汇总

说明：

- `位置组合数`: `results_summary*.csv` 中完成的不同 `position_combination` 数量。
- `max authorized_max_ber`: 授权位置中最差 BER 的最大值，越低越好。
- `avg security_min_route_min_ber`: 每组实验中最弱非法路径 BER 的平均值，越接近 `0.5` 越安全。
- `avg security_gain_over_anchor`: 相比 anchor 的平均安全增益。

| 数据集 | 已有 k | 位置组合数 | max authorized_max_ber | avg security_min_route_min_ber | avg security_gain_over_anchor |
|---|---|---:|---:|---:|---:|
| `15pro_mid` | `k2` | 2 | 0.000000 | 0.082846 | 0.057346 |
| `15pro_yellow_shuffled (default)` | `k3`, `k4`, `k5`, `k6`, `k7`, `k8`, `k9`, `k10` | 151 | 0.000000 | 0.448440 | 0.117902 |
| `mate40pro_high` | `k2`, `k3` | 5 | 0.000000 | 0.138850 | 0.098850 |
| `mate40pro_high_column_shuffled` | `k2`, `k3`, `k4`, `k5`, `k6`, `k7`, `k8`, `k9`, `k10`, `k11` | 43 | 0.000000 | 0.438194 | 0.065648 |
| `p40_low_column_shuffled` | `k2`, `k3`, `k4`, `k5`, `k6`, `k7`, `k8`, `k9`, `k10`, `k11`, `k12`, `k14` | 50 | 0.000000 | 0.402370 | 0.105670 |

整体看，当前结果里授权端 `authorized_max_ber` 全部为 `0`，说明已保存的实验都满足授权位置无误码。安全性方面，`15pro_yellow_shuffled`、`mate40pro_high_column_shuffled` 和 `p40_low_column_shuffled` 的非法路径最小 BER 平均值较高，接近随机猜测区间；`mate40pro_high` 和 `15pro_mid` 当前样本较少，且平均最弱非法路径 BER 较低，需要更多样本或进一步修复。

## 每个结果目录的效果

| 数据集 | k | summary 文件 | 位置组合数 | max authorized_max_ber | avg security_min_route_min_ber | min security_min_route_min_ber | avg security_avg_route_min_ber | avg security_gain_over_anchor |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `15pro_mid` | `k2` | `results_summary.csv` | 2 | 0.000000 | 0.082846 | 0.007694 | 0.308022 | 0.057346 |
| `15pro_yellow_shuffled (default)` | `k3` | `results_summary.csv` | 3 | 0.000000 | 0.444428 | 0.434116 | 0.470759 | 0.098428 |
| `15pro_yellow_shuffled (default)` | `k4` | `results_summary.csv` | 3 | 0.000000 | 0.444023 | 0.440979 | 0.469073 | 0.107357 |
| `15pro_yellow_shuffled (default)` | `k4` | `results_summary_20samples.csv` | 20 | 0.000000 | 0.447536 | 0.428314 | 0.471321 | 0.099636 |
| `15pro_yellow_shuffled (default)` | `k5` | `results_summary_20samples.csv` | 20 | 0.000000 | 0.453108 | 0.442692 | 0.474388 | 0.099658 |
| `15pro_yellow_shuffled (default)` | `k6` | `results_summary_20samples.csv` | 20 | 0.000000 | 0.448574 | 0.411449 | 0.474270 | 0.110674 |
| `15pro_yellow_shuffled (default)` | `k7` | `results_summary.csv` | 3 | 0.000000 | 0.455898 | 0.453007 | 0.476057 | 0.117231 |
| `15pro_yellow_shuffled (default)` | `k7` | `results_summary_20samples.csv` | 20 | 0.000000 | 0.454386 | 0.447277 | 0.476495 | 0.111536 |
| `15pro_yellow_shuffled (default)` | `k8` | `results_summary.csv` | 3 | 0.000000 | 0.455336 | 0.447461 | 0.477429 | 0.117336 |
| `15pro_yellow_shuffled (default)` | `k8` | `results_summary_20samples.csv` | 20 | 0.000000 | 0.452348 | 0.438506 | 0.476526 | 0.119898 |
| `15pro_yellow_shuffled (default)` | `k9` | `results_summary_20samples.csv` | 20 | 0.000000 | 0.443297 | 0.403318 | 0.470595 | 0.148997 |
| `15pro_yellow_shuffled (default)` | `k10` | `results_summary_20samples.csv` | 19 | 0.000000 | 0.433912 | 0.387326 | 0.468466 | 0.166175 |
| `mate40pro_high` | `k2` | `results_summary.csv` | 3 | 0.000000 | 0.120572 | 0.050000 | 0.277814 | 0.040572 |
| `mate40pro_high` | `k3` | `results_summary.csv` | 2 | 0.000000 | 0.157128 | 0.128783 | 0.325880 | 0.157128 |
| `mate40pro_high_column_shuffled` | `k2` | `results_summary.csv` | 5 | 0.000000 | 0.439766 | 0.388907 | 0.461184 | 0.084166 |
| `mate40pro_high_column_shuffled` | `k3` | `results_summary.csv` | 5 | 0.000000 | 0.468539 | 0.459688 | 0.478415 | 0.066539 |
| `mate40pro_high_column_shuffled` | `k4` | `results_summary.csv` | 5 | 0.000000 | 0.450865 | 0.430977 | 0.470618 | 0.099465 |
| `mate40pro_high_column_shuffled` | `k5` | `results_summary.csv` | 5 | 0.000000 | 0.460937 | 0.434339 | 0.475916 | 0.074137 |
| `mate40pro_high_column_shuffled` | `k6` | `results_summary.csv` | 5 | 0.000000 | 0.469257 | 0.455875 | 0.480561 | 0.063457 |
| `mate40pro_high_column_shuffled` | `k7` | `results_summary.csv` | 5 | 0.000000 | 0.452877 | 0.372871 | 0.476838 | 0.050677 |
| `mate40pro_high_column_shuffled` | `k8` | `results_summary.csv` | 4 | 0.000000 | 0.472691 | 0.469556 | 0.481762 | 0.052191 |
| `mate40pro_high_column_shuffled` | `k9` | `results_summary.csv` | 4 | 0.000000 | 0.415731 | 0.356908 | 0.463959 | 0.099231 |
| `mate40pro_high_column_shuffled` | `k10` | `results_summary.csv` | 2 | 0.000000 | 0.349111 | 0.214000 | 0.467875 | 0.030111 |
| `mate40pro_high_column_shuffled` | `k11` | `results_summary.csv` | 3 | 0.000000 | 0.402167 | 0.367000 | 0.452803 | 0.036500 |
| `p40_low_column_shuffled` | `k2` | `results_summary.csv` | 5 | 0.000000 | 0.425300 | 0.388209 | 0.457315 | 0.112900 |
| `p40_low_column_shuffled` | `k3` | `results_summary.csv` | 5 | 0.000000 | 0.451896 | 0.440154 | 0.472332 | 0.105296 |
| `p40_low_column_shuffled` | `k4` | `results_summary.csv` | 5 | 0.000000 | 0.438460 | 0.402263 | 0.470552 | 0.118660 |
| `p40_low_column_shuffled` | `k5` | `results_summary.csv` | 5 | 0.000000 | 0.447585 | 0.437800 | 0.473374 | 0.087185 |
| `p40_low_column_shuffled` | `k6` | `results_summary.csv` | 5 | 0.000000 | 0.452723 | 0.444633 | 0.476444 | 0.090923 |
| `p40_low_column_shuffled` | `k7` | `results_summary.csv` | 5 | 0.000000 | 0.426939 | 0.337026 | 0.472424 | 0.112739 |
| `p40_low_column_shuffled` | `k8` | `results_summary.csv` | 5 | 0.000000 | 0.435986 | 0.404542 | 0.471576 | 0.111386 |
| `p40_low_column_shuffled` | `k9` | `results_summary.csv` | 5 | 0.000000 | 0.403285 | 0.358476 | 0.466223 | 0.143885 |
| `p40_low_column_shuffled` | `k10` | `results_summary.csv` | 5 | 0.000000 | 0.405827 | 0.372204 | 0.465075 | 0.173627 |
| `p40_low_column_shuffled` | `k11` | `results_summary.csv` | 2 | 0.000000 | 0.434591 | 0.417463 | 0.470353 | 0.146591 |
| `p40_low_column_shuffled` | `k12` | `results_summary.csv` | 2 | 0.000000 | 0.320848 | 0.234000 | 0.460260 | 0.064848 |
| `p40_low_column_shuffled` | `k14` | `results_summary.csv` | 1 | 0.000000 | 0.185000 | 0.185000 | 0.452905 | 0.000000 |

## 结果文件保存内容

每个完整实验目录通常包含三类 CSV。

### `results_summary.csv`

保存每个位置组合的一行最终实验结果，是最主要的汇总文件。

主要字段：

- `k` / `real_k`: 真实授权位置数量。
- `effective_k`: 加入虚拟流后的等效 k。
- `virtual_stream_count`: 虚拟流数量。
- `sample_index`: 第几个采样组合。
- `position_combination`: 本次实验选择的授权位置组合。
- `selected_candidate_count`: 最终参与调度/混合的候选方案数量。
- `common_route_count`: 候选之间共同覆盖的非法攻击路径数量。
- `excluded_illegal_positions`: 因数据或探针不满足条件被排除的非法位置。
- `authorized_max_ber`: 授权位置中最差的 BER，越低越好。
- `authorized_position_bers`: 每个授权位置各自的 BER。
- `security_min_route_min_ber`: 最弱非法路径的 BER，越接近 `0.5` 越安全。
- `security_avg_route_min_ber`: 所有共同非法路径的平均 BER。
- `worst_route`: 当前最弱的非法路径，格式为 `illegal_position->legal_position`。
- `optimizer`: 用于组合候选方案的优化器，例如 `linprog`。
- `usage_ratio`: 每个 selected candidate 在最终调度中的使用比例。
- `usage_counts`: 把 `usage_ratio` 离散化成调度长度后的使用次数。
- `selected_candidate_ids`: 被最终选择的候选方案 ID。
- `selected_probe_sets`: 每个候选方案使用的 probe 集合。
- `anchor_candidate_id` / `anchor_source`: anchor 候选方案信息。
- `anchor_min_route_min_ber`: anchor 自身的最弱非法路径 BER。
- `anchor_worst_route`: anchor 自身最弱非法路径。
- `security_gain_over_anchor`: 最终方案相对 anchor 的安全增益。
- `floor_applied`: 优化时是否应用了 floor 约束。
- `optimizer_diagnostic`: 优化器诊断信息。
- `weak_routes`: anchor 阶段识别出的弱路径列表。

### `results_summary_20samples.csv`

字段含义与 `results_summary.csv` 相同。区别是它来自 `run_global_guarded_20samples.py`，通常表示同一个 k 下运行了 20 个随机位置组合样本，并把最终 summary 复制到对应 `k*/global_guarded/` 目录。

### `results_selected.csv`

保存每个被选中候选方案的详细信息。一个位置组合通常会对应多行，因为最终调度可能混合多个 candidate。

主要字段：

- `sample_index` / `position_combination`: 对应哪一次位置组合实验。
- `candidate_id`: 候选方案 ID。
- `candidate_source`: 候选来源，例如 baseline 或 targeted repair。
- `candidate_weight`: 该候选在最终混合调度中的权重。
- `candidate_usage_count`: 该候选在离散调度中的使用次数。
- `probes`: 候选方案使用的 probe 子集。
- `authorized_max_ber` / `authorized_position_bers`: 该候选自身的授权端 BER。
- `candidate_min_route_min_ber`: 该候选自身最弱非法路径 BER。
- `candidate_avg_route_min_ber`: 该候选自身非法路径平均 BER。
- `worst_route`: 该候选自身最弱非法路径。

这个文件适合用来追踪最终 summary 中的结果到底由哪些候选 probe/mapping 组合贡献。

### `weak_routes.csv`

保存 anchor 阶段识别出的弱非法路径，后续 targeted repair 会重点修复这些路径。

主要字段：

- `sample_index` / `position_combination`: 对应哪一次位置组合实验。
- `anchor_candidate_id`: 用来识别弱路径的 anchor candidate。
- `anchor_source`: anchor 来源。
- `anchor_min_route_min_ber`: anchor 的最弱非法路径 BER。
- `anchor_worst_route`: anchor 的最弱路径。
- `weak_routes`: 被挑出来重点修复的弱路径列表。
- `weak_route_anchor_values`: 这些弱路径在 anchor 上对应的 BER 数值。

这个文件适合用来解释 targeted repair 为什么选择某些非法路径作为优化目标。

## 读数建议

- 先看 `results_summary*.csv`：判断每个位置组合的最终通信与安全效果。
- 再看 `results_selected.csv`：分析最终方案由哪些候选 probe 集合组成。
- 最后看 `weak_routes.csv`：理解安全修复针对的是哪些非法路径。
- 授权端指标看 `authorized_max_ber`，当前保存的结果均为 `0`。
- 安全端指标看 `security_min_route_min_ber`，越接近 `0.5` 越接近随机猜测，泄露越少。
