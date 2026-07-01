Generate the blog post, LinkedIn post, and X thread for a milestone day, using PLAN.md's Content
Calendar as the script and the day's report for real numbers. Writes draft files and prints a
preview. Does NOT publish anything externally.

## Arguments

`$ARGUMENTS` is a day number: `7` (Blog Post 1) or `12` (Blog Post 2).

## Context

Working directory: current project directory (pctrans).
Plan file: `PLAN.md` — read the "Content Calendar" section for the scripted title, sections, hook,
word count, and the already-drafted LinkedIn/X copy for this day.
Report file: `reports/day-$ARGUMENTS-*.md` (glob — pick the one that matches).
Output files:
- `reports/blog-01-concept.md` (Day 7) or `reports/blog-02-results.md` (Day 12)
- `reports/linkedin-0X.txt` (01 for Day 7, 02 for Day 12)
- `reports/x-thread-0X.txt` (Day 12 only — PLAN.md only scripts one X thread, published Day 14)

## Steps

1. Read the report at `reports/day-$ARGUMENTS-*.md` for the real metrics, file names, and
   observations from that day's work.
2. Read PLAN.md's Content Calendar entry for this day (Blog Post 1 for day 7, Blog Post 2 for
   day 12) — reuse its title, section list, hook, and target word count exactly.
3. Draft the blog post following that section list, replacing every `XX.X%`, `[LINK]`, and
   `[CELL LINE ...]` placeholder with the real value from the day's report (or from
   `reports/eval_summary.json` if it exists by day 12).
4. Fill in the LinkedIn Post 1/2 template from PLAN.md's Content Calendar the same way (real kNN
   numbers, real baseline numbers, real Streamlit/GitHub links once known).
5. On Day 12 only, also fill in the Twitter/X Thread template from PLAN.md's Content Calendar.
6. Write each to its output file.
7. Print a clearly labelled preview of everything written to the terminal.
8. Stage and commit the output files:
   ```
   git add reports/blog-0X-*.md reports/linkedin-0X.txt reports/x-thread-0X.txt
   git commit -m "day $ARGUMENTS: add content drafts"
   git push origin main
   ```
9. Print exactly this message at the end:
   ```
   Content drafts committed. Review and edit the files if needed:
     reports/blog-0X-<slug>.md
     reports/linkedin-0X.txt
     reports/x-thread-0X.txt (Day 12 only)
   These publish on Day 14 per PLAN.md's launch checklist — nothing is posted automatically.
   ```

## Writing rules (apply to all three formats)

- No em-dash (—). Use a comma, colon, or break into two sentences.
- No "delve", "leverage", "robust", "seamlessly", "at the intersection of", "it's worth noting", "dive into".
- No passive voice for findings: "the model achieves X" not "it was found that X".
- No intro sentence describing what the post will cover ("In this post...", "Today I'll explain...").
- No closing sentence summarising what was covered ("In conclusion...", "To summarise...").
- Honest about limitations — PLAN.md is explicit that these posts should be "no hype, honest about
  what's being validated." If a metric underperformed a baseline, say so and say why.
