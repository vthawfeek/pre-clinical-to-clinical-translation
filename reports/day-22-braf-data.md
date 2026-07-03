# Day 22: Vemurafenib + BRAF Mutation Data Assembly

**Date:** 2026-07-03
**Commit:** `day 22: vemurafenib + BRAF mutation data assembly (DepMap/GDSC/MC3)`

## What Was Built

- `pctrans/casestudy/braf_vemurafenib.py` — data loaders for the Rung-4 case study:
  - `PrismClient.download_dose_response` — downloads the DepMap PRISM Repurposing 20Q2
    secondary-screen dose-response table (Figshare, ~290 MB) and validates it against a
    minimum-size check, same pattern as `CCLEClient`/`TCGAClient`.
  - `CBioPortalClient` — thin wrapper around three cBioPortal REST endpoints
    (`mutations/fetch`, `sample-lists/.../sample-ids`, `studies/.../clinical-data`), each caching
    one JSON response under `data/raw/cbioportal/` and skipping re-fetch unless `force=True`.
  - `is_braf_v600` / `classify_braf_status` — flags a real BRAF V600 substitution (V600E/K/D/...),
    explicitly rejecting both a `mutationType == "Silent"` record and a `proteinChange == "V600V"`
    (reference residue recurring — synonymous at the hotspot codon, not a substitution).
  - `load_vemurafenib_sensitivity` — PRISM AUC per CCLE line, averaged across replicate
    screens/doses, plus a z-scored column.
  - `load_ccle_depmap_id_map` — cBioPortal `ccle_name` sampleId -> DepMap `ACH-` id, from the
    `DEPMAPID` clinical attribute (needed because cBioPortal's CCLE study keys samples by name,
    not by DepMap id).
  - `assemble_braf_table` — joins all of the above onto the SKCM slice of the already-embedded
    3-lineage cell lines and TCGA patients, dropping any sample without a resolvable BRAF call.
  - `coverage_summary` — N per domain, BRAF mutant/WT split, and vemurafenib join coverage.
- `pctrans/scripts/casestudy.py` — new `pctrans-casestudy` CLI: downloads the three raw sources,
  calls `assemble_braf_table`, writes `data/processed/braf_vemurafenib.parquet` +
  `reports/braf_coverage.json`.
- `pyproject.toml` — registers `pctrans-casestudy`.
- `tests/test_casestudy.py` — 8 tests: BRAF-status parsing (`test_braf_status_parsing`,
  `test_is_braf_v600_rejects_silent_and_non_v600`), vemurafenib replicate aggregation, DepMap-id
  mapping, and full `assemble_braf_table`/`coverage_summary` round-trips on tiny synthetic
  fixtures (no network in tests — the real download path is exercised once, live, by the CLI run
  below).

## What Was Learned

- **The classic CCLE compound panel never tested vemurafenib.** Barretina et al. (2012) profiled
  PLX4720, the Plexxikon tool-compound precursor of vemurafenib (PLX4032) — same chemical series,
  same BRAF-V600E target, but a different molecule. Confirmed by inspecting both current GDSC1/2
  releases (`cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/`) and cBioPortal's `ccle_broad_2019`
  drug-treatment profiles directly: "PLX-4720" appears in all of them, "vemurafenib" in none. The
  actual clinical compound only shows up in DepMap's PRISM Repurposing screen (Corsello et al.
  2020), which specifically targets approved/clinical-stage drugs — the right source for this case
  study, not a compromise.
- **cBioPortal is a far cheaper way to ask "is this sample BRAF-mutant?" than downloading the raw
  mutation calls the plan named.** DepMap's `OmicsSomaticMutations.csv` is every gene x every one
  of ~2,600 lines (multi-GB); the public MC3 pan-cancer MAF is ~28 GB unpacked. cBioPortal's
  `mutations/fetch` endpoint returns just the BRAF calls for one study in a few hundred KB, backed
  by the same underlying MAF-derived pipeline. Used for both CCLE (`ccle_broad_2019`) and TCGA SKCM
  (`skcm_tcga_pan_can_atlas_2018`) — one consistent source instead of three.
- **cBioPortal's CCLE study (2019 vintage) doesn't cover every line in DepMap 24Q4.** Of our 110
  SKCM cell lines, 61 have a `DEPMAPID` mapping into `ccle_broad_2019`; the other 49 are
  ACH-0019xx/ACH-002xxx ids — DepMap additions from after 2019 that the cBioPortal snapshot
  predates. This caps CCLE-side coverage below 110 regardless of the vemurafenib join.
- **PRISM's IC50 column is mostly unusable; AUC is not.** Across the 843 vemurafenib dose-response
  rows, IC50 is missing 86% of the time (many lines never drop below 50% viability in the tested
  dose range, so the curve fit has no IC50 to report). AUC is always fit. Some lines were screened
  in two PRISM batches (`MTS010`/`HTS002`) with different plated dose ranges — those got averaged
  into one AUC per line rather than picking one batch arbitrarily.

## Key Decisions

1. **Used cBioPortal instead of raw DepMap `OmicsSomaticMutations.csv` / MC3 MAF for BRAF status.**
   Both raw sources are two to three orders of magnitude larger than a single-gene lookup needs.
   cBioPortal's public REST API returns the same underlying calls at the scale the task actually
   requires, and unifies the CCLE and TCGA lookups behind one client class instead of three
   different raw-file formats.
2. **Used DepMap PRISM Repurposing (not GDSC, not the classic CCLE panel) for vemurafenib
   sensitivity.** It is the only public cell-line screen that tested the literal clinical compound
   by name, confirmed by direct inspection of the alternatives (see "What Was Learned" above)
   rather than assumed from the plan's "PRISM Repurposing (or GDSC)" phrasing.
3. **Reused existing embeddings rather than computing new ones.** Cell lines: the full 259-line
   `ccle_embeddings.npz` (Day 12), so BRAF/vemurafenib coverage isn't capped by the small held-out
   test split. Patients: the held-out `embeddings_test.npz` TCGA test slice (Day 11) — 71 SKCM
   patients, 65 with a resolvable BRAF call and a near-even 32/33 mutant/WT split, already enough
   for Day 23's planned Mann-Whitney comparison without adding a new full-cohort TCGA embedding
   pipeline this early.
4. **`is_braf_v600` checks two independent guards, not one.** A hotspot hit could slip through as a
   false mutant either via an explicit `mutationType == "Silent"` annotation or via a literal
   `V600V` protein change (no upstream mutationType at all) — the real cBioPortal data hit neither
   case (no silent BRAF calls in either study's `mutations/fetch` response), which is exactly why
   the plan asks for a dedicated unit test rather than relying on real data to exercise it.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
........................................................................ [ 58%]
...................................................                      [100%]
123 passed, 6 deselected in 27.96s

$ uv run pytest tests/test_casestudy.py -q
......                                                                   [100%]
6 passed in 0.56s
```

Real run (`pctrans-casestudy`, live network — PRISM Figshare + cBioPortal):

```
Downloading PRISM Repurposing 20Q2 secondary-screen dose response...
prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv: 100%|##########| 290M/290M [00:16<00:00, 17.1MB/s]
Downloading cBioPortal BRAF calls (CCLE + TCGA-SKCM)...
Wrote data\processed\braf_vemurafenib.parquet  (126 rows)

====================================================
     DAY 22 -- BRAF / VEMURAFENIB DATA COVERAGE
====================================================
SKCM cell lines (BRAF status resolved): 61
  BRAF split: {'mutant': 47, 'WT': 14}
  ...with a vemurafenib readout: 41
  BRAF split (vemurafenib subset): {'mutant': 33, 'WT': 8}
SKCM patients (BRAF status resolved): 65
  BRAF split: {'WT': 33, 'mutant': 32}
====================================================
Wrote reports\braf_coverage.json
```

## Numbers

| Quantity | Value |
|---|---|
| SKCM cell lines in the 3-lineage catalogue (all 259) | 110 |
| ...with a BRAF call via cBioPortal `ccle_broad_2019` | 61 (47 mutant / 14 WT) |
| ...of those, with a PRISM vemurafenib AUC | 41 (33 mutant / 8 WT) |
| SKCM patients in the held-out test split | 71 |
| ...with a BRAF call via cBioPortal `skcm_tcga_pan_can_atlas_2018` | 65 (32 mutant / 33 WT) |
| Assembled table rows (`braf_vemurafenib.parquet`) | 126 (61 cell lines + 65 patients) |
| PRISM dose-response file size | 290 MB (843 vemurafenib rows, 2 broad_id batches) |
| Embedding dim | 64 |

## Next Up

- Day 23: Part A — is a BRAF-mutant SKCM cell line embedded closer to the BRAF-mutant SKCM patient
  centroid than a WT line (Mann-Whitney on distance-to-centroid)? Part B — does that same proximity
  correlate with vemurafenib AUC (Spearman, n=41, 33 mutant/8 WT)? Report the finding honestly
  either way — this is explicitly framed in the plan as a positive control, not a foregone
  conclusion.
- Day 23 also adds `test_centroid_distance_math` / `test_response_correlation_runs_and_bounds` and
  a `03_evaluation.ipynb` Section 6 with the figures.
- Day 24: Gate 2 evaluation folds in this case study's outcome alongside Days 15/17/20/21.
- Day 25: Blog Post 3 draft.
