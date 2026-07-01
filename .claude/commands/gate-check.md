Print the Gate 0 or Gate 1 evaluation decision for the pctrans project, using PLAN.md's Gate
Decision Architecture thresholds. Read-only: does not modify, commit, or push anything.

## Arguments

`$ARGUMENTS` is optional: `0` (Gate 0, Day 7 sanity check) or `1` (Gate 1, Day 10 kNN evaluation).
If omitted, auto-detect which gate applies:
- If `models/best_model.pt` and `reports/eval_summary.json` both exist → Gate 1.
- Else if an MLflow run or training log exists → Gate 0.
- Else print `No training artifacts found yet — nothing to gate-check. Run /day 7 first.` and stop.

## Gate 0 (Day 7): Data + Architecture Sanity

Read PLAN.md's "Gate 0 (Day 7)" section for the exact rule. Check the most recent 1-epoch sanity
run (MLflow run or printed training output):
- Loss is a positive finite scalar (no NaN/Inf) → required for PASS.
- Val kNN accuracy > 0 (even ~33% random baseline counts) → required for PASS.

Print:
```
[Gate 0: Does 1 training epoch complete without NaN and with kNN > 0?]
Loss:      <value or "NaN/Inf detected">
Val kNN:   <value>
DECISION:  PASS — proceed to Week 2 / FAIL — debug before Day 8
```

If FAIL, print PLAN.md's three debug hints verbatim (NaN loss → check scaler for Inf values on
zero-variance columns; shape errors → check sampler `__len__`/`__getitem__`; kNN = 0 → check
`LINEAGE_TO_IDX` consistency across datasets).

## Gate 1 (Day 10): kNN Retrieval Accuracy

Read `reports/eval_summary.json` (written by `pctrans-evaluate`). Extract overall kNN@5, per-lineage
kNN@5 (LUAD/BRCA/SKCM), silhouette score, and TFS. Apply PLAN.md's Gate Decision Architecture
4-tier threshold table:

```
≥ 70%    → DEPLOY PATH (Days 11-14)
60-70%   → SOFT FAIL: debug 1 fix (lower lr to 1e-4, increase warmup to 10 epochs, re-check tau clamp), re-run 10 epochs, re-evaluate Day 11
50-60%   → HARD FAIL: batch construction issue (re-check sampler for within-lineage negatives, consider L2 reg), re-evaluate Day 11
< 50%    → ARCHITECTURE FAILURE: PIVOT (Harmony baseline, or reframe as a documented failure analysis) — do NOT proceed to Streamlit deployment
```

Print the exact report format from PLAN.md Day 10 task 6:
```
══════════════════════════════════════════════
           GATE 1 EVALUATION REPORT
══════════════════════════════════════════════
Overall kNN@5 Accuracy:  XX.X%   (threshold: 70%)
Per-lineage kNN@5:
  LUAD:  XX.X%
  BRCA:  XX.X%
  SKCM:  XX.X%
Silhouette Score:  +X.XX   (> 0 = good alignment)
TFS (composite):   X.XX    (> 0.6 = deploy)
Random baseline:   33.3%
PCA+kNN baseline:  ~55%
Harmony baseline:  ~63%
══════════════════════════════════════════════
DECISION: [DEPLOY / DEBUG]
══════════════════════════════════════════════
```

## Notes

- This command only reads existing artifacts and prints the decision — it never trains, evaluates,
  commits, or pushes. `/day 7` and `/day 10` already run the underlying training/evaluation steps
  and commit the outcome; use `/gate-check` to re-print the decision later without re-running anything.
- If `reports/eval_summary.json` is missing when Gate 1 is requested, say so and suggest running
  `uv run pctrans-evaluate --model models/best_model.pt --data-dir data/processed/` first.
