# Day 1: Project Scaffold + Environment

**Date:** 2026-07-01
**Commit:** `day 1: scaffold pctrans package, pyproject.toml, CI workflow, stub modules`

## What Was Built

- `PLAN.md` — copied verbatim from the plan-mode source (`~/.claude/plans/use-the-below-plan-concurrent-kahan.md`) into the repo root, as the plan's own "How to Use This Plan" section requires.
- `pyproject.toml` — uv/hatchling build, all runtime + dev dependencies, 5 CLI entry points (`pctrans-download`, `pctrans-preprocess`, `pctrans-train`, `pctrans-evaluate`, `pctrans-query`), ruff config, pytest `slow`/`integration` markers.
- `pctrans/` package with `__init__.py` (`version = "0.1.0"`) and stub modules (empty classes/functions raising `NotImplementedError`) for every file in the Project Structure: `data/{ccle_client,tcga_client,preprocessor,dataset,sampler}.py`, `models/{encoders,dual_tower,losses}.py`, `training/{trainer,callbacks}.py`, `evaluation/{knn,silhouette,tfs,viz}.py`, `inference/api.py`, `scripts/{download,preprocess,train,evaluate,query}.py` (Typer `app` skeletons wired to the CLI entry points).
- `configs/model.yaml`, `configs/training.yaml`, `configs/data.yaml` — initial hyperparameter values from the plan's Architecture Specifications.
- `CLAUDE.md` — Quick Reference extracted from PLAN.md, plus a "How to trigger a day's work" section and a day-by-day status tracker.
- `tests/conftest.py` — `tiny_ccle` (10x50 synthetic expression + lineage column), `tiny_tcga` (20x50), `tiny_model` (config dict, embed_dim=8; full model fixtures land once `encoders.py` is implemented Day 6).
- `tests/test_data.py` — smoke test asserting `tiny_ccle` shape.
- `.github/workflows/ci.yml` — runs `ruff check` + `pytest` on every push.
- `.gitignore` — excludes `.venv/`, `data/raw/`, `data/processed/`, `mlruns/`, `models/*.pt`.
- `.claude/commands/day.md`, `.claude/commands/blog-draft.md`, `.claude/commands/gate-check.md` — the `/day N`, `/blog-draft N`, `/gate-check` slash commands, adapted from the working pattern in the sibling `mtdna-foundation-model` project.

## What Was Learned

- `/day 1` couldn't run because it's a project-scoped custom slash command that lives in `.claude/commands/`, and this project had never been scaffolded — nothing existed to create that command file until Day 1's own tasks ran it once, manually.
- `umap-learn`'s transitive dependency chain (`pynndescent` -> `numba` -> `llvmlite`) resolved to a 2021-era `numba==0.53.1` / `llvmlite==0.36.0` under uv's default resolver, and both explicitly refuse to build on Python 3.11 (`requires_python <3.10` self-check in old `llvmlite` setup.py). Neither package declares a PEP 621 `requires-python` in a way uv can see without building, so uv picked the oldest version satisfying `numba>=0.51.2` instead of a modern one. Fixed by adding explicit floor constraints `numba>=0.59` and `llvmlite>=0.42` to `pyproject.toml`.
- `harmonypy` jumped from a pure-Python `0.2.0` straight to a C++/CMake rewrite at `2.0.0`, which needs MSVC build tools (`nmake`, `CMAKE_CXX_COMPILER`) not present on this machine. Pinned `harmonypy>=0.0.9,<2.0` to stay on the pure-Python implementation, sufficient for the Day 10 Harmony baseline comparison.

## Key Decisions

- CLAUDE.md's day-status lines omit an embedded commit hash (`COMPLETE (commit <hash>)`) since the hash can't be known before the commit that would contain it exists; the `/day` command instead notes the hash can be added retroactively via `git log` if desired. The sibling project's own CLAUDE.md is inconsistent about this for the same reason.
- `/blog-draft` and `/gate-check` were designed from scratch (no sibling-project equivalent for gate-check) rather than copied verbatim, since pctrans's PLAN.md already fully scripts blog/LinkedIn/X copy in its Content Calendar section and defines explicit numeric gate thresholds in its Gate Decision Architecture section — the commands are built to consume those sections directly rather than generating content or thresholds independently.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
.                                                                        [100%]
1 passed in 0.19s

$ uv run python -c "import pctrans; print(pctrans.__version__)"
0.1.0

$ uv run pctrans-download --help / pctrans-preprocess --help / pctrans-train --help / pctrans-evaluate --help / pctrans-query --help
(all 5 entry points resolve and print usage)
```

## Numbers

- 20 stub source files created under `pctrans/` across 6 subpackages.
- `uv sync --all-extras` resolves 180 packages, including torch 2.12.1, streamlit 1.58.0, umap-learn 0.5.12.
- 1/1 tests passing, 0 ruff errors.

## Next Up

- Day 2: implement `CCLEClient` (DepMap download), `filter_lineages`, `pctrans-download ccle`.
- Populate `data/raw/ccle/` and verify lineage counts (~97-112 LUAD, ~125-145 BRCA, ~62-78 SKCM).
- Add CCLE-specific tests to `tests/test_data.py`.
