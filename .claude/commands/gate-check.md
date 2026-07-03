Print the Gate 0, Gate 1, or Gate 2 evaluation decision for the pctrans project, using PLAN.md's
Gate Decision Architecture thresholds (Gate 0/1) or PLAN-phase2.md's Phase 2 Gate section (Gate 2).
Read-only: does not modify, commit, or push anything.

## Arguments

`$ARGUMENTS` is optional: `0` (Gate 0, Day 7 sanity check), `1` (Gate 1, Day 10 kNN evaluation), or
`2` (Gate 2, Day 24 validation report). If omitted, auto-detect which gate(s) apply:
- If `reports/multiseed_results.json`, `reports/baselines.json`, `reports/permutation_test.json`,
  `reports/confounder_purity.json`, `reports/eval_summary_15.json`, and `reports/braf_casestudy.json`
  all exist → print Gate 0, 1, **and** 2.
- Else if `models/best_model.pt` and `reports/eval_summary.json` both exist → Gate 1 (and Gate 0).
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

## Gate 2 (Day 24): Validation Report — the honesty checkpoint

Read PLAN-phase2.md's "Phase 2 Gate (Gate 2)" section for the exact five criteria + bonus. Gate 2
is a narrative, not a single pass/fail number: report every criterion even when the day's own
tasks would let you stop early.

Read these artefacts:
- `reports/multiseed_results.json` → `aggregate.knn5.{mean,sd,ci_low,ci_high}` (G2-1)
- `reports/baselines.json` → `harmony_knn`, `combat_knn`, `scanorama_knn`, `supervised_ceiling`,
  `contrastive_knn5`, `beats_best_baseline_by`, `matches_supervised_ceiling` (G2-2)
- `reports/permutation_test.json` → `eval_only.p_value`, `retrain.p_value`, `decision` (G2-3)
- `reports/confounder_purity.json` → `domain_axis_purity_correlation.r`,
  `purity_stratified_knn.{high_purity,low_purity}.overall_accuracy`,
  `silhouette_before_residualisation`, `silhouette_after_residualisation` (G2-4)
- `reports/eval_summary_15.json` → `overall_knn_accuracy`, `knn_wilson_ci`, `confusion_matrix` /
  `confusion_labels` (G2-5)
- `reports/braf_casestudy.json` → `placement.{p_value,effect_size,effect_size_ci_low,
  effect_size_ci_high}`, `response_link.{rho,p_value,ci_low,ci_high}` (G2-BONUS)

Apply PLAN-phase2.md's thresholds:
```
G2-1  PASS if multiseed kNN@5 95% CI lower bound ≥ 0.90
G2-2  PASS if contrastive kNN@5 beats best of {Harmony, ComBat, Scanorama} by ≥ 5 pts
      AND does not merely match the supervised ceiling (matches_supervised_ceiling == false)
G2-3  PASS if empirical p < 0.01 (both eval-only and retrain variants)
G2-4  PASS if kNN@5 holds in both purity strata AND |r| is moderate, not dominant (embedding is
      not primarily a purity axis) AND silhouette survives residualisation (stays positive)
G2-5  REPORT (not pass/fail): 15-lineage kNN@5 < 100% is the expected, healthy outcome; check that
      the top off-diagonal confusion mass sits on the named biologically-confusable pairs
      (LUAD↔LUSC, GBM↔LGG, COAD↔READ), not scattered randomly
G2-BONUS  REPORT (not pass/fail): state the placement-test and response-link results plainly,
      including if either is weak or null — do not upgrade a p≈0.05 or a null rho to a clean win
```

Print:
```
══════════════════════════════════════════════════════════
                 GATE 2 — VALIDATION REPORT
══════════════════════════════════════════════════════════
G2-1  Multi-seed kNN@5 (10 seeds): mean X.XXX ± X.XXX, 95% CI [lo, hi]        [PASS/FAIL]
G2-2  Beats best real baseline (<name>, XX.X%) by +XX.X pts; supervised ceiling XX.X% (not matched) [PASS/FAIL]
G2-3  Label-shuffle permutation: real XX.X% vs null mean X.X% (max X.X%), p = 0.00XX             [PASS/FAIL]
G2-4  Purity strata kNN@5: high XX.X% (n=XXX) / low XX.X% (n=XXX); domain-purity r = ±0.XXX;
      silhouette +0.XXX -> +0.XXX after residualisation                                          [PASS/FAIL]
G2-5  15-lineage kNN@5: XX.X% (Wilson XX.X-XX.X%, n=XXX); confusions: <named pairs and their share>
G2-BONUS  Vemurafenib case study: Part A p=0.0XX, effect X.XXX (CI [lo,hi]); Part B rho=0.XXX
      (CI [lo,hi]), p=0.XX  <state directionally positive / weak / null plainly>
──────────────────────────────────────────────────────────
DECISION:  PORTFOLIO-READY  /  NEEDS-WORK  (name the specific failing rung, if any)
══════════════════════════════════════════════════════════
```

If any of G2-1 through G2-4 fails, name that criterion explicitly in the decision line rather than
rolling everything into a bare NEEDS-WORK. G2-5 and G2-BONUS never block the decision on their own
(they are report-only) but a G2-5 confusion pattern that looks random, or a G2-BONUS result that
was silently reframed as positive in other docs, should be flagged as a documentation-honesty issue.

## Notes

- This command only reads existing artifacts and prints the decision — it never trains, evaluates,
  commits, or pushes. `/day 7`, `/day 10`, and `/day 24` already run the underlying
  training/evaluation steps and commit the outcome; use `/gate-check` to re-print the decision later
  without re-running anything.
- If `reports/eval_summary.json` is missing when Gate 1 is requested, say so and suggest running
  `uv run pctrans-evaluate --model models/best_model.pt --data-dir data/processed/` first.
- If any Gate 2 artefact is missing, say so by name and suggest which `/day N` (15/17/19/20/21/23)
  produces it, rather than printing a partial Gate 2 report.
