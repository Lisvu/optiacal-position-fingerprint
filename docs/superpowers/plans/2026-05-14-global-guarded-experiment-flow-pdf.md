# Global Guarded Experiment Flow PDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete Chinese LaTeX/PDF explanation of `vector_hue/global_guarded_core.py`, including real-data examples from current experiment outputs and concrete vector hue / hue mapping / legal and illegal decoding calculations.

**Architecture:** Keep the deliverable as one self-contained `ctexart` document under `docs/`. Use the existing code and CSV outputs as the source of truth, then add a small reproducible helper script only if needed to compute intermediate numeric examples from the same functions used by the experiment.

**Tech Stack:** Python 3, NumPy, existing `vector_hue` modules, CSV result files, LaTeX `ctexart`, `latexmk` or `xelatex`.

---

### Task 1: Extract Real Experiment Facts

**Files:**
- Read: `E:/LuminaLink/Position_fingerprint_experiment/vector_hue/global_guarded_core.py`
- Read: `E:/LuminaLink/Position_fingerprint_experiment/vector_hue/virtual_stream_core.py`
- Read: `E:/LuminaLink/Position_fingerprint_experiment/vector_hue/targeted_probe_subset_repair_yellow_shuffled.py`
- Read: `E:/LuminaLink/Position_fingerprint_experiment/vector_hue/k3/global_guarded/results_summary.csv`
- Read: `E:/LuminaLink/Position_fingerprint_experiment/vector_hue/k3/global_guarded/results_selected.csv`
- Read: `E:/LuminaLink/Position_fingerprint_experiment/vector_hue/k3/global_guarded/weak_routes.csv`

- [ ] **Step 1: Identify the code-level flow**

Record these functions and their roles: `run_global_guarded_experiment`, `generate_candidates`, `build_virtual_mapping_candidates`, `simulate_blocks_vector_with_virtual`, `evaluate_candidate`, `select_global_guarded_candidates`, `anchor_guarded_ratio_and_min`, `write_guarded_outputs`.

- [ ] **Step 2: Select one concrete sample**

Use `k=3`, `sample_index=1`, `position_combination=(2, 5, 26)` from `k3/global_guarded/results_summary.csv` as the running example.

- [ ] **Step 3: Extract summary numbers**

Use these real values in the document: `real_k=3`, `effective_k=8`, `virtual_stream_count=5`, `authorized_max_ber=0.00000000`, `authorized_position_bers=[0,0,0]`, `security_min_route_min_ber=0.43411580`, `security_avg_route_min_ber=0.46757386`, `worst_route=25->26`, `optimizer=linprog`, `selected_candidate_count=20`, and the usage counts from the CSV.

### Task 2: Compute Concrete Numeric Walkthrough

**Files:**
- Optional temporary script: `C:/Users/ASUS/AppData/Local/Temp/opencode/global_guarded_walkthrough.py`
- Output data to embed manually in LaTeX.

- [ ] **Step 1: Load one selected candidate**

Use candidate `267` from `results_selected.csv` for the clearest example. Its probe set is `[25, 75, 100, 115, 180, 215, 225, 250, 270, 295, 315, 325, 335, 340, 355]`.

- [ ] **Step 2: Build legal models and mapping**

Use existing functions to load legal position models `(2,5,26)`, build vector hue mapping candidates, and select the matching candidate mapping for `candidate_id=267` if reproducible from the deterministic seed path. If exact mapping reconstruction is difficult, use candidate outputs and explicitly state the candidate-level metrics are CSV-real while the single-block decoding example is recomputed from the same code path.

- [ ] **Step 3: Compute and capture examples**

Capture at least one bit block, its effective 8-stream `bits_pm`, the per-chip `hue_seq_vector`, the first few local projections/correlation values for legal positions, and one illegal route failure for route `25->26` or another route available from the candidate.

### Task 3: Write LaTeX Document

**Files:**
- Create: `E:/LuminaLink/Position_fingerprint_experiment/docs/global_guarded_experiment_flow.tex`

- [ ] **Step 1: Create the document skeleton**

Use `ctexart`, `amsmath`, `booktabs`, `longtable`, `geometry`, `hyperref`, and `xcolor`. Title: `Global Guarded Vector Hue 位置指纹通信实验流程说明`.

- [ ] **Step 2: Write the conceptual flow**

Explain the full pipeline in Chinese: data input, fingerprint extraction, virtual streams, vector hue mapping, legal decoding, illegal decoding, candidate generation, weak-route targeting, global guarded selection, linear-program usage ratio, and result CSV interpretation.

- [ ] **Step 3: Add real-data walkthrough**

Embed the selected `(2,5,26)` sample with true CSV numbers and a readable example from sending bits through vector hue to legal decode success and illegal decode failure.

- [ ] **Step 4: Make it beginner-readable**

Define every term when first used: Probe, chip, vector hue, hue mapping, legal device, illegal device, BER, corrected BER, weak route, anchor, usage ratio.

### Task 4: Compile and Verify PDF

**Files:**
- Input: `E:/LuminaLink/Position_fingerprint_experiment/docs/global_guarded_experiment_flow.tex`
- Output: `E:/LuminaLink/Position_fingerprint_experiment/docs/global_guarded_experiment_flow.pdf`

- [ ] **Step 1: Compile**

Run `xelatex -interaction=nonstopmode global_guarded_experiment_flow.tex` from the `docs` directory. Repeat if needed for references.

- [ ] **Step 2: Verify artifacts**

Confirm that the PDF exists and that the log has no fatal errors.

- [ ] **Step 3: Final response**

Report the `.tex` and `.pdf` paths and summarize the real-data example used.

---

## Self-Review

Spec coverage: The plan covers code flow, real-data extraction, numeric walkthrough, LaTeX writing, compilation, and verification. Placeholder scan: no TBD/TODO placeholders remain. Type consistency: file names and function names match the explored code.
