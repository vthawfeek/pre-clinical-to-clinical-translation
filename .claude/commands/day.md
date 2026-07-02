Execute Day $ARGUMENTS of the Pre-Clinical to Clinical Translation (pctrans) project.

## Context

Working directory: c:\Users\vthawfeek.Shajitha\Documents\Projects\pre-clinical-to-clinical-translation
Plan files: `PLAN.md` holds Phase 1 (Days 1–14); `PLAN-phase2.md` holds Phase 2 (Days 15–25). Both live in the repo root and are tracked in git.
GitHub repo: https://github.com/vthawfeek/pre-clinical-to-clinical-translation

## Instructions

1. Pick the plan file by day number: **Days 1–14 → `PLAN.md`; Days 15–25 → `PLAN-phase2.md`.** Read that file and find the section titled "### Day $ARGUMENTS:". Execute every task listed there, in order. Do not skip tasks. If a task produces an error, fix the error before moving on. Do not invent tasks that are not in the plan.

2. If CLAUDE.md's "Current status" section already marks Day $ARGUMENTS as COMPLETE, skip the implementation tasks but still make sure the daily report exists, then continue from step 4 (commit and push).

3. For each source file created or modified, follow these rules:
   - All Python code must pass `uv run ruff check pctrans/ tests/`
   - All tests must pass `uv run pytest tests/ -q -m "not slow and not integration"`
   - Fix any failures before proceeding to the next task

4. After all tasks are complete, write a daily report at:
   `reports/day-$ARGUMENTS-<short-topic>.md`

   Use this exact structure (PLAN.md's Daily Report Template):
   ```
   # Day $ARGUMENTS: [Short Title]

   **Date:** YYYY-MM-DD
   **Commit:** `day $ARGUMENTS: <description>`

   ## What Was Built

   [Bullet list of files created/modified with what they contain]

   ## What Was Learned

   [2-4 bullet points: surprises, unexpected behaviour, scientific observations]

   ## Key Decisions

   [1-3 decisions made today with brief justification. Only include non-obvious decisions.]

   ## Verification

   [Paste actual output of: ruff check, pytest, and any key print statements]

   ## Numbers (if applicable)

   [Any metrics, shapes, counts, or timings from today's work]

   ## Next Up

   [Day $ARGUMENTS+1 tasks — 3-5 bullets]
   ```

5. Run the end-of-day quality gate:
   ```
   uv run ruff check pctrans/ tests/
   uv run pytest tests/ -q -m "not slow and not integration"
   ```
   Both must pass before committing. Fix any failures first.

6. Update CLAUDE.md: find the line for Day $ARGUMENTS in "Current status" and set it to
   `- Day $ARGUMENTS: COMPLETE — <short description>`. Don't try to embed the commit hash here —
   it isn't known until after the commit in step 7; add it retroactively on a later day's pass if
   convenient by checking `git log --oneline`.

7. Stage and commit:
   - Stage: all new and modified files under `pctrans/, tests/, configs/, docs/, notebooks/, reports/, app/, .claude/, .github/, PLAN.md, CLAUDE.md, pyproject.toml`
   - Do NOT stage: `data/, models/*.pt, mlruns/, .venv/, *.pyc, __pycache__/`
   - Commit message format: `day $ARGUMENTS: <short description of what was built>`
   - Add co-author: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`

8. Push to GitHub:
   ```
   git push origin main
   ```
   If the push fails because there is no remote, remind the user to run:
   ```
   gh repo create vthawfeek/pre-clinical-to-clinical-translation --public --source=. --remote=origin
   git push -u origin main
   ```

9. If Day $ARGUMENTS is a blog-draft milestone day — `7` or `12` (Phase 1 Content Calendar) or `25`
   (Phase 2 Blog Post 3) — generate content drafts by following all the steps in
   `.claude/commands/blog-draft.md` for Day $ARGUMENTS.

   Those steps will read the report just written, draft a blog post + LinkedIn post + X thread from
   the active plan's Content Calendar, write them to `reports/blog-0X-<slug>.md`,
   `reports/linkedin-0X.txt`, `reports/x-thread-0X.txt`, commit and push them, and print a preview.

10. If Day $ARGUMENTS is a gate day — `7` (Gate 0), `10` (Gate 1), or `24` (Gate 2, defined in
    `PLAN-phase2.md`) — after finishing the day's own tasks, also run `.claude/commands/gate-check.md`
    to double check the printed decision matches what got committed. Note: Day 24 itself extends
    `/gate-check` with the Gate 2 report, so run it after that extension is in place.

## Important

- Follow the plan precisely. The active plan file (`PLAN.md` for Days 1–14, `PLAN-phase2.md` for
  Days 15–25) has specific file names, function signatures, and architecture decisions — use them
  exactly as written.
- Phase 2 keeps the Phase 1 deliverables reproducible: save new artefacts under distinct prefixes
  (e.g. `*_15.parquet`, `best_model_15.pt`, `gene_list_trainhvg.txt`) rather than overwriting the
  3-lineage Phase 1 outputs, exactly as `PLAN-phase2.md` specifies.
- Do not skip tasks that are in the plan, and do not skip the quality gate.
- Gate days (Day 7 = Gate 0, Day 10 = Gate 1 in `PLAN.md`; Day 24 = Gate 2 in `PLAN-phase2.md`) have
  explicit pass/fail thresholds in their plan's "Gate Decision Architecture" / "Phase 2 Gate"
  section — follow the debug protocol there if a gate does not pass, rather than silently proceeding.
