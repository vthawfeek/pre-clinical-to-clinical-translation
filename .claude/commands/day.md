Execute Day $ARGUMENTS of the Pre-Clinical to Clinical Translation (pctrans) project.

## Context

Working directory: c:\Users\vthawfeek.Shajitha\Documents\Projects\pre-clinical-to-clinical-translation
Plan file: PLAN.md (in the repo root — this is the primary copy tracked in git)
GitHub repo: https://github.com/vthawfeek/pre-clinical-to-clinical-translation

## Instructions

1. Read PLAN.md and find the section titled "### Day $ARGUMENTS:". Execute every task listed there, in order. Do not skip tasks. If a task produces an error, fix the error before moving on. Do not invent tasks that are not in the plan.

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

9. If Day $ARGUMENTS is `7` or `12` (the blog-draft milestone days per PLAN.md's Content Calendar),
   generate content drafts by following all the steps in `.claude/commands/blog-draft.md` for Day $ARGUMENTS.

   Those steps will read the report just written, draft a blog post + LinkedIn post + X thread from
   PLAN.md's Content Calendar, write them to `reports/blog-0X-<slug>.md`, `reports/linkedin-0X.txt`,
   `reports/x-thread-0X.txt`, commit and push them, and print a preview.

10. If Day $ARGUMENTS is `10` (Gate 1), after finishing the day's own evaluation tasks, also run
    `.claude/commands/gate-check.md` to double check the printed decision matches what got committed.

## Important

- Follow the plan precisely. PLAN.md has specific file names, function signatures, and architecture
  decisions — use them exactly as written.
- Do not skip tasks that are in the plan, and do not skip the quality gate.
- Gate days (Day 7 = Gate 0, Day 10 = Gate 1) have explicit pass/fail thresholds in PLAN.md's
  "Gate Decision Architecture" section — follow the debug protocol there if a gate does not pass,
  rather than silently proceeding.
