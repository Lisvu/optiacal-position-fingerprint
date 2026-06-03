# Global Guarded 实验方案与当前结果总结

更新时间：2026-05-13

## 目标

本轮实验目标是优化 `vector_hue` 中不同真实 k 值的安全性，使最终的 usage-rate 混合结果在全局最差 route 上尽量高于最佳单候选 anchor，同时保留 anchor 下界保护。

核心指标是 `security_min_route_min_ber`：对某个样本，先按 LP 得到多个候选方案的 usage ratio，再计算所有 route 的混合 minBER，最后取其中最差 route 的 minBER。该值越高越好，越接近 0.5 表示最弱 route 也越接近随机猜测。

## 最新实验方案

### 方法名称

`global_guarded`

### 适用 k 范围

当前计划对 `k=2` 到 `k=10` 分别随机运行 20 组实验。

### 核心思想

- 先生成 baseline candidates。
- 在 baseline candidates 中选择全局最强单候选作为初始 anchor。
- 根据 anchor 的最弱 routes 识别当前瓶颈 routes。
- 继续生成 targeted candidates，用于补强 anchor 的弱 route。
- 在全部 candidates 中重新选择最终 anchor。
- 使用 hybrid selected pool：保留 anchor，同时加入全局高分候选、弱 route group rescuer、bounded per-route rescuer 和多样性候选。
- 用 `scipy.optimize.linprog` 显式求解 usage ratio，使混合后的全局最差 route minBER 最大化。
- 若 LP 不可用或混合结果低于 anchor floor，则回退到 anchor-only floor，保证 `security_min_route_min_ber >= anchor_min_route_min_ber`。

### 关键保护机制

- `anchor_min_route_min_ber` 是最佳单候选 anchor 的全局最差 route minBER。
- `security_gain_over_anchor = security_min_route_min_ber - anchor_min_route_min_ber`。
- 正常目标是 `security_gain_over_anchor > 0`。
- 保底要求是 `security_gain_over_anchor >= 0`。
- `optimizer_diagnostic=ok` 表示 LP 正常求解。
- `floor_applied=False` 表示没有触发 anchor-only fallback。

## 主要代码文件

- `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\global_guarded_core.py`
- `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\run_global_guarded_20samples.py`
- `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\test_global_guarded_core.py`
- `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\test_run_global_guarded_20samples.py`

## 单个 k 的运行方式

推荐使用专门脚本分别运行每个 k。

示例：运行 `k=7`。

```powershell
& "E:\LuminaLink\Position_fingerprint_experiment\.venv\Scripts\python.exe" "E:\LuminaLink\Position_fingerprint_experiment\vector_hue\run_global_guarded_20samples.py" --k 7
```

运行其他 k 时只需要修改 `--k` 后面的数字，例如 `--k 2` 到 `--k 10`。

### 输出文件

每个 k 的 20 组 summary 结果会写入对应目录下的独立文件：

```text
E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k{K}\global_guarded\results_summary_20samples.csv
```

例如 `k=7` 的输出是：

```text
E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k7\global_guarded\results_summary_20samples.csv
```

### 脚本行为

- 实际实验先写入临时目录：`%TEMP%\opencode\global_guarded_20samples\k{K}`。
- 运行完成后复制临时目录中的 `results_summary.csv` 到对应 k 目录下的 `results_summary_20samples.csv`。
- 这样可以避免覆盖已有的 `results_summary.csv`。

## 当前已验证的 3 样本结果

以下结果来自当前实际存在的 `results_summary.csv` 文件。当前存在结果的 k 为 `k=3,4,7,8`。每个 k 有 3 个样本。

| k | 样本数 | 平均 security_min | 最低 security_min | 最高 security_min | 平均 anchor_min | 平均 gain | 最低 gain | 最高 gain | linprog | floor_applied | diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| k3 | 3 | 0.444428 | 0.434116 | 0.458000 | 0.346000 | 0.098428 | 0.075167 | 0.111116 | 3/3 | 0 | ok: 3 |
| k4 | 3 | 0.444023 | 0.440979 | 0.446692 | 0.336667 | 0.107357 | 0.083979 | 0.146692 | 3/3 | 0 | ok: 3 |
| k7 | 3 | 0.455898 | 0.453007 | 0.459182 | 0.338667 | 0.117231 | 0.103182 | 0.128007 | 3/3 | 0 | ok: 3 |
| k8 | 3 | 0.455336 | 0.447461 | 0.460371 | 0.338000 | 0.117336 | 0.088176 | 0.135461 | 3/3 | 0 | ok: 3 |

### 当前结果判断

- 当前 3 样本结果中，`k3/k4/k7/k8` 全部使用 `linprog` 成功求解。
- 所有样本的 `optimizer_diagnostic` 都是 `ok`。
- 所有样本都没有触发 floor fallback。
- 所有样本的 `security_gain_over_anchor` 都大于 0。
- 当前 3 样本平均表现排序为：`k7 > k8 > k3 > k4`。

## 当前文件状态

截至本次记录，实际扫描到的 summary 文件为：

```text
vector_hue\k3\global_guarded\results_summary.csv
vector_hue\k4\global_guarded\results_summary.csv
vector_hue\k7\global_guarded\results_summary.csv
vector_hue\k8\global_guarded\results_summary.csv
```

尚未扫描到任何 `results_summary_20samples.csv`。这表示 k=2 到 k=10 的 20 组正式结果还没有完整生成到目标文件中。

另外，之前尝试运行 20 样本全量实验时被中断，`k2\global_guarded\results_summary.csv` 已不存在。因此 k2 需要重新运行。

## 20 组实验完成后的检查命令

运行完各 k 后，可用以下命令汇总 `results_summary_20samples.csv`：

```powershell
& "E:\LuminaLink\Position_fingerprint_experiment\.venv\Scripts\python.exe" -c "import csv, glob, os, statistics as st; base=r'E:\LuminaLink\Position_fingerprint_experiment\vector_hue'; files=sorted(glob.glob(base+r'\k*\global_guarded\results_summary_20samples.csv'), key=lambda p:int(os.path.basename(os.path.dirname(os.path.dirname(p)))[1:])); print('k,n,sec_avg,sec_min,sec_max,gain_avg,gain_min,gain_max,linprog,diag');
for path in files:
    rows=list(csv.DictReader(open(path,encoding='utf-8-sig',newline='')))
    k=os.path.basename(os.path.dirname(os.path.dirname(path)))
    secs=[float(r['security_min_route_min_ber']) for r in rows]
    gains=[float(r.get('security_gain_over_anchor',0)) for r in rows]
    lin=sum(1 for r in rows if r.get('optimizer')=='linprog')
    diag={}
    for r in rows:
        d=r.get('optimizer_diagnostic','')
        diag[d]=diag.get(d,0)+1
    print(k,len(rows),f'{st.mean(secs):.6f}',f'{min(secs):.6f}',f'{max(secs):.6f}',f'{st.mean(gains):.6f}',f'{min(gains):.6f}',f'{max(gains):.6f}',lin,diag)"
```

预期每个 k 应满足：

- `n=20`
- `linprog=20`
- `diag={'ok': 20}`
- `security_gain_over_anchor` 平均值大于 0
- 最低 `security_gain_over_anchor` 不应小于 0

## 注意事项

- 必须使用项目 `.venv` 中的 Python 运行实验。
- `.venv` 中必须安装 `scipy`，否则 LP 会失败并回退到 anchor-only floor。
- 若看到 `optimizer_diagnostic` 包含 `ModuleNotFoundError: No module named 'scipy'`，说明当前运行环境没有安装 scipy。
- 若运行被中断，临时目录可能有部分文件，但最终目标 `results_summary_20samples.csv` 只有在单个 k 完整跑完后才会复制生成。
