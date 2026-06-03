# Weak-Route Targeted K3 Experiment

This folder isolates the k=3 weak-route targeted candidate-generation experiment.

The runner uses a two-pass flow for each fixed legal-position combination:

1. Generate baseline candidates and identify the weakest illegal routes.
2. Generate additional candidates with a weak-route-targeted source label.
3. Select candidates using a score weighted toward the baseline weak routes.
4. Reuse the existing scheduler/linprog output format.

Default target combinations are the current weak k3 cases:

- `(7, 18, 21)`
- `(6, 9, 24)`
- `(16, 17, 22)`

## Run

```powershell
python "E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k3\weak_route_targeted\run_weak_route_targeted_k3.py" --overwrite
```

For the effective-k=9 variant:

```powershell
python "E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k3\weak_route_targeted\run_weak_route_targeted_k3.py" --target-effective-k 9 --overwrite
```

For a faster smoke test:

```powershell
python "E:\LuminaLink\Position_fingerprint_experiment\vector_hue\k3\weak_route_targeted\run_weak_route_targeted_k3.py" --combo "(7,18,21)" --baseline-probe-subset-count 2 --targeted-probe-subset-count 2 --base-mapping-count 2 --baseline-mappings-per-subset 1 --targeted-mappings-per-subset 1 --selected-count 4 --eval-bits 50 --overwrite
```

## Outputs

- `results_summary.csv`: compact scheduled result for each combination.
- `results_selected.csv`: selected candidate details.
- `weak_routes.csv`: baseline weak routes used to guide targeted selection.

Success indicators:

- `authorized_max_ber` remains `0.00000000`.
- `security_min_route_min_ber` improves over the original k3 result for the same combination.
- `optimizer` becomes `linprog`, or `usage_ratio` contains multiple non-zero entries.
