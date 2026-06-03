# Global Guarded Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a batch experiment for k=2,3,4,7,8 that preserves the best single candidate while using global guarded weak-route repair to search for usage-rate improvements.

**Architecture:** Add a shared `vector_hue/global_guarded_core.py` module that reuses `virtual_stream_core` primitives, implements anchor-worst route targeting, quota-based candidate selection, anchor-floor optimization, and per-k output directories. Add separate per-k entry files under `k2`, `k3`, `k4`, `k7`, and `k8`, plus a lightweight test script for selection and fallback invariants.

**Tech Stack:** Python 3, NumPy, SciPy linprog through existing optimizer, CSV outputs.

---

### Task 1: Global Guarded Core And Per-K Scripts

**Files:**
- Create: `vector_hue/global_guarded_core.py`
- Create: `vector_hue/k2/global_guarded/run_k2_global_guarded.py`
- Create: `vector_hue/k3/global_guarded/run_k3_global_guarded.py`
- Create: `vector_hue/k4/global_guarded/run_k4_global_guarded.py`
- Create: `vector_hue/k7/global_guarded/run_k7_global_guarded.py`
- Create: `vector_hue/k8/global_guarded/run_k8_global_guarded.py`

- [ ] **Step 1: Define shared CLI and output layout**

Implement shared arguments for `--sample-size`, subset/mapping counts, eval bits, selected count, weak route count, and `--overwrite`. Each per-k script writes outputs to its own `vector_hue/k{K}/global_guarded` directory.

- [ ] **Step 2: Implement candidate generation**

Reuse `core.evaluate_candidate`, `core.build_virtual_mapping_candidates`, and `core.base.generate_probe_subsets`. Keep search evaluation at `search_eval_bits` for speed.

- [ ] **Step 3: Implement anchor and route selection**

Find anchor by global `min_route_min_ber`; derive weak routes from the anchor's worst routes, not baseline `route_best`.

- [ ] **Step 4: Implement quota selected pool**

Build selected pool from anchor, top-global, top anchor-worst, per-route rescuers, and diversity candidates.

- [ ] **Step 5: Implement guarded output**

Optimize usage rate over selected candidates, then fallback to anchor-only if optimized minBER falls below anchor.

### Task 2: Regression Checks

**Files:**
- Create: `vector_hue/test_global_guarded_batch.py`

- [ ] **Step 1: Verify anchor is first and preserved**

Use fake candidates to check the selected pool includes the global best candidate.

- [ ] **Step 2: Verify fallback floor**

Patch the optimizer to return a bad ratio and assert final minBER is not lower than anchor.

### Task 3: Run Batch Experiment

**Files:**
- Generated outputs under `vector_hue/k2/global_guarded`, `vector_hue/k3/global_guarded`, `vector_hue/k4/global_guarded`, `vector_hue/k7/global_guarded`, `vector_hue/k8/global_guarded`

- [ ] **Step 1: Run regression checks**

Run: `python vector_hue/test_global_guarded_batch.py`
Expected: prints `global guarded batch checks passed`.

- [ ] **Step 2: Start per-k runs**

Run each per-k file: `python vector_hue/k2/global_guarded/run_k2_global_guarded.py --overwrite` and equivalent for k3, k4, k7, k8.
Expected: each k prints progress and writes `results_summary.csv`, `results_selected.csv`, and `weak_routes.csv`.

### Self-Review

Spec coverage: Covers global anchor preservation, anchor-worst route targeting, quota selection, optimizer fallback, k batch execution, and result files.

Placeholder scan: No placeholders remain.

Type consistency: File paths and helper names match the planned implementation.
