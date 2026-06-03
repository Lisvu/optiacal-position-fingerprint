# Global Guarded 20 Samples K2-K10 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run global-guarded experiments for k=2 through k=10 with 20 random samples each and save one independent summary file per k.

**Architecture:** Do not modify the optimizer or existing result files. Run each k into an isolated temporary output directory, then copy only the generated summary CSV to `vector_hue/k{K}/global_guarded/results_summary_20samples.csv`.

**Tech Stack:** Windows PowerShell, `.venv` Python, existing `vector_hue/global_guarded_core.py` runner, CSV summary outputs.

---

## File Structure

- Create/update: `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k2\global_guarded\results_summary_20samples.csv`
- Create/update: `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k3\global_guarded\results_summary_20samples.csv`
- Create/update: `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k4\global_guarded\results_summary_20samples.csv`
- Create/update: `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k5\global_guarded\results_summary_20samples.csv`
- Create/update: `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k6\global_guarded\results_summary_20samples.csv`
- Create/update: `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k7\global_guarded\results_summary_20samples.csv`
- Create/update: `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k8\global_guarded\results_summary_20samples.csv`
- Create/update: `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k9\global_guarded\results_summary_20samples.csv`
- Create/update: `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k10\global_guarded\results_summary_20samples.csv`

## Task 1: Run Isolated 20-Sample Experiments

**Files:**
- Temporary output: `C:\Users\ASUS\AppData\Local\Temp\opencode\global_guarded_20samples\k{K}`
- Final output: `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k{K}\global_guarded\results_summary_20samples.csv`

- [ ] **Step 1: Create output directories**

Run PowerShell checks and create missing `global_guarded` directories under k=5, k=6, k=9, and k=10 before copying results.

- [ ] **Step 2: Run k=2 through k=10 in temp outputs**

Use `.venv\Scripts\python.exe` and call `global_guarded_core.run_global_guarded_experiment()` with `sample_size=20`, `overwrite=True`, and the temp output directory for each k.

- [ ] **Step 3: Copy summaries to final files**

Copy each temp `results_summary.csv` to the matching `results_summary_20samples.csv` in that k's `global_guarded` directory.

## Task 2: Verify Results

**Files:**
- Read: `E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k*\global_guarded\results_summary_20samples.csv`

- [ ] **Step 1: Count rows**

Verify every k from 2 through 10 has exactly 20 rows.

- [ ] **Step 2: Check optimizer health**

Verify every row has `optimizer=linprog`, `optimizer_diagnostic=ok`, and `floor_applied=False`, or report exceptions.

- [ ] **Step 3: Summarize effect**

For every k, report average/min/max `security_min_route_min_ber`, average/min/max `security_gain_over_anchor`, and count of successful LP rows.

## Self-Review

- The plan does not change algorithm code.
- The plan avoids overwriting existing `results_summary.csv` files.
- The plan creates one independent summary file per k, as requested.
- The plan includes verification criteria for row count and optimizer health.
