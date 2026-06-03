# Global Guarded Weak Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure weak-route targeted k=3 experiments optimize global minBER while preserving the best discovered single candidate as a hard floor.

**Architecture:** Replace blended weak-route-only selection with a global-guarded selector that always includes the best single candidate anchor, adds global/weak/per-route support candidates, and applies an anchor-only fallback after usage-ratio optimization if the mixed schedule is worse than the anchor. Keep changes localized to the k3 targeted script and add a lightweight regression script.

**Tech Stack:** Python 3, NumPy, existing `virtual_stream_core` candidate and optimizer utilities.

---

### Task 1: Add Anchor-Preserved Selection

**Files:**
- Modify: `vector_hue/k3/weak_route_targeted/run_weak_route_targeted_k3.py`

- [ ] **Step 1: Add a selector that always includes the global best candidate**

Add `append_unique_candidate()` and `select_global_guarded_candidates()` near the existing selection helpers. The selector must return `(selected, anchor)` and include candidates from global ranking, weak-route ranking, and per-route rescuers.

- [ ] **Step 2: Replace the old selector call**

Change the experiment loop to call `select_global_guarded_candidates(...)` instead of `select_targeted_candidates(...)`.

### Task 2: Add Anchor Floor Output

**Files:**
- Modify: `vector_hue/k3/weak_route_targeted/run_weak_route_targeted_k3.py`

- [ ] **Step 1: Add a writer wrapper**

Add `write_anchor_guarded_outputs()` that mirrors `core.write_compact_outputs()` but checks optimized `security_min` against `anchor.min_route_min_ber` and falls back to anchor-only when needed.

- [ ] **Step 2: Use the wrapper in the experiment loop**

Pass `selected` and `anchor` to the new writer. The printed output should include `anchor_min`, `security_min`, `gain`, and `optimizer`.

### Task 3: Add Regression Check

**Files:**
- Create: `vector_hue/k3/weak_route_targeted/test_anchor_guarded_selection.py`

- [ ] **Step 1: Write a script-level regression test**

Create simple fake candidates and verify that anchor selection includes the global best candidate and that anchor fallback produces a final minBER not below the anchor.

- [ ] **Step 2: Run the regression script**

Run: `python vector_hue/k3/weak_route_targeted/test_anchor_guarded_selection.py`
Expected: exits successfully and prints `anchor guarded selection checks passed`.

### Self-Review

Spec coverage: The plan covers global anchor preservation, weak-route candidate retention, per-route coverage, and final minBER floor enforcement.

Placeholder scan: No placeholders remain.

Type consistency: Helper names and paths match the existing script structure.
