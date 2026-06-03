# Virtual Stream K Folders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-k experiment folders that run independently and save compact accuracy/security results, with virtual random streams for small k.

**Architecture:** Create one shared core module in `vector_hue/virtual_stream_core.py` and thin `k2/run_k2.py` through `k10/run_k10.py` wrappers. The core reuses the existing vector-hue helpers, extends mapping dimension with key-controlled virtual random streams when `k < target_effective_k`, and writes compact `results_summary.csv` and `results_selected.csv` only.

**Tech Stack:** Python 3, NumPy, pandas, SciPy linprog through the existing scheduler helper, existing `targeted_probe_subset_repair_yellow_shuffled.py` utilities.

---

### Task 1: Add failing import/output checks

**Files:**
- Test command only

- [ ] Run `python -c "import importlib.util, sys; p=r'E:\LuminaLink\Position_fingerprint_experiment\vector_hue\virtual_stream_core.py'; spec=importlib.util.spec_from_file_location('virtual_stream_core', p); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); assert hasattr(mod, 'run_k_experiment'); assert mod.virtual_count_for_k(2,8)==6; assert mod.virtual_count_for_k(8,8)==0"`
- Expected before implementation: fails because `virtual_stream_core.py` does not exist.

### Task 2: Implement shared virtual-stream core

**Files:**
- Create: `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\virtual_stream_core.py`

- [ ] Implement `virtual_count_for_k`, virtual bit block expansion, real-stream-only authorized BER, real-route-only illegal BER, compact CSV output, and `run_k_experiment(config)`.
- [ ] Verify compile with `python -m py_compile "E:\LuminaLink\Position_fingerprint_experiment\vector_hue\virtual_stream_core.py"`.

### Task 3: Create per-k wrappers

**Files:**
- Create directories and files `vector_hue\k2\run_k2.py` through `vector_hue\k10\run_k10.py`.

- [ ] Each wrapper imports `virtual_stream_core`, sets `real_k`, `target_effective_k=8`, `sample_size=20`, and writes into its own folder.
- [ ] Verify all wrappers compile with `python -m py_compile`.

### Task 4: Smoke test one wrapper

**Files:**
- Output: `vector_hue\k2\results_summary.csv`
- Output: `vector_hue\k2\results_selected.csv`

- [ ] Run a tiny smoke by invoking core with `sample_size=1`, `probe_subset_count=2`, `base_mapping_count=2`, `mappings_per_subset=1`, `eval_bits=50`.
- [ ] Verify compact CSV headers include probe choice, authorized BER, security min/avg, worst route, common route count, excluded illegal positions, and virtual stream metadata.

### Self-Review

Spec coverage: The plan covers independent per-k folders, virtual stream effective-k behavior, compact results, probe selection, accuracy, and security fields.

Placeholder scan: No placeholders remain.

Type consistency: Core names `run_k_experiment` and `virtual_count_for_k` are used consistently.
