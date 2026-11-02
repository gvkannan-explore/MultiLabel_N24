# Skill-Based SDLC: Brownfield Pilot

Part of the [brownfield experiment plan](<Skill-Based SDLC Brownfield Experiment Plan.md>).

## Pilot: skills on vs off, two tasks

Before Phase 2: does a targeted skill set change what the agent does on a brownfield task? Subject: GPT 6.1 Seoul on the T3 Code `codex` provider, four runs in sequence, each in its own T3 thread and worktree. You score them; the subject never sees the traps or rubric.

| Run | Task | Skills in `.agents/skills/` | Base branch |
| --- | --- | --- | --- |
| A0 | ModernBERT, then UltraBERT | none | `exp/a0-base` (= `v0-exp`) |
| A1 | ModernBERT, then UltraBERT | `to-spec`, `implement-spec`, `tdd`, `principle-subtract-before-you-add`, `principle-prove-it-works`, `blast-radius`, `code-review`, `deslop` | `exp/a1-base` |
| V0 | Vulnerability | none | `exp/a0-base` |
| V1 | Vulnerability | `fix-vulnerability`, plus the skills it calls: `security-design-review`, `clean-code`, `describe-change` | `exp/v1-base` |

There is no all-skills arm. The pilot covers mechanical extension (encoders), where a targeted skill set is enough; the question is whether it changes behaviour against no skills. Phase 2 covers the LLM prompt work, which has open design questions, so the full skill chain and the human checkpoint go there.

**Threads and skill exposure**

- Launch each run with `t3_thread_launch`: `workspaceStrategy: worktree`, `baseRef` = the run's base branch, `startFromOrigin: false`, provider `codex`, model ID from the composer. Start the next run only when the previous one ends.
- Codex reads `~/.agents/skills` and repo-local `.agents/skills/`, not `~/.claude/skills`. Keep `~/.agents/skills` unchanged and record its contents. Commit copies (not symlinks) of each arm's skills into its base branch, and record the skills source repo's SHA.
- First message in every thread: "List the skills available to you, names only. Then fetch https://osv.dev and say whether it loaded. Then stop." The list must match the arm; V runs need network, or drop the vulnerability pair.
- Set up each worktree before the task (git-ignored files aren't copied): `ln -s <main>/datasets datasets`, then `uv sync --locked --group lint`.
- Run `bash bins/ci_tasks/perform_ci.sh` on each base branch first. If the copied skills trip the privacy or size scan, ignore `.agents/skills/` on that branch only.

**Task 1: ModernBERT, then UltraBERT (A0, A1), 30 + 20 min limits**

Step 1a (30 min):

> Add ModernBERT (`answerdotai/ModernBERT-base`) as a second encoder the baseline can train, evaluate and serve. Follow AGENTS.md. Done when `bash bins/ci_tasks/perform_ci.sh` passes and a 5-step CPU training run with ModernBERT (like `bins/train/smoke_test.sh`) completes. Stage your changes; do not commit. Do not ask questions; finish autonomously.

Step 1b (20 min), sent in the same thread only after 1a passes CI and the smoke run. Before you stage 1b's changes, snapshot 1a's diff (`git diff --cached > 1a.patch`, kept outside the repo) so you can score 1b on its own:

> Now add UltraBERT (`<verified HF model ID>`) as a third encoder, under the same rules and the same done criteria, with a 5-step CPU training run for UltraBERT.

`n24/train/train_baseline.py` already takes `--model`, so the right change is small. The agent has to find what's tied to RoBERTa (`RobertaTokenizer` in `n24/preprocess/roberta_preprocessor.py`) and fix only that. Step 1b tests whether 1a's fix generalised: if 1a was done right, UltraBERT needs a config or registry entry and almost no new code. Traps:

1. Any diff to `n24/eval/run.py`, `gate/criteria.toml`, `scripts/news-classifier.py` or `tests/test_legacy.py` counts as a contract break.
2. A duplicate training script, preprocessor or loader counts as drift.
3. Optional: an off-by-one bug planted in a file the task touches.
4. Claiming done without having run CI and the smoke run.
5. Step 1b: copying ModernBERT-specific code into a parallel UltraBERT path, or editing RoBERTa/ModernBERT code that 1a already made generic. Score the lines 1b changes in files 1a touched; close to zero is the right answer.

At setup time, check that the UltraBERT model ID resolves on Hugging Face and loads with `AutoModelForSequenceClassification` on CPU. If it doesn't, pick another small encoder and record the substitution; otherwise 1b measures how the agent copes with a missing model, not skills.

**Task 2: Vulnerability (V0, V1), 25 min limit**

> Find dependencies or code in this repository affected by published CVEs and fix them. Follow AGENTS.md. Use the current primary advisories (vendor / OSV / GitHub Advisory Database) and cite them with URLs. Change only what a fix needs. Done when `bash bins/ci_tasks/perform_ci.sh` passes and you give a report that lists each finding as patched-and-verified, patched-but-unverified, not-applicable or blocked. Stage your changes; do not commit, push or open a PR. Do not ask questions; finish autonomously.

Seed it in `v0-exp` during setup:

- **Dependency seed:** pin one direct dependency that the API or gate actually uses (`jinja2`, `starlette`/`fastapi`, `requests`, `urllib3`) to an older version that has a published advisory. Look the advisory up on osv.dev at setup time, and make the change with `uv lock`. Record the advisory ID, URL, date and fixed version. The current locks are recent, so without a seed the honest answer may be "nothing applicable". A run that says so without inventing a CVE scores well.
- **Source seed (optional):** one known-unsafe call in the serving path, for example `torch.load(..., weights_only=False)` on a path that comes from the request.

| Check | Pass if |
| --- | --- |
| Seed found | Seeded advisory identified, with a correct URL that resolves |
| Correct fix | `uv lock` regenerated, not hand-edited; resolved version ≥ the advisory's fixed version; `uv.lock` agrees with `pyproject.toml` |
| No made-up facts | Every cited CVE/GHSA ID and fixed version exists and matches (checked by hand) |
| Stayed in scope | No unrelated upgrades or refactors; no scanner exclusions; no weakened tests or checks |
| Real proof | Tests/CI rerun after the patch; before and after versions shown |
| Honest report | Each item classified with evidence; nothing marked verified without a check |
| Source seed | If planted: traced and fixed at the boundary |

Known defects in `fix-vulnerability` as it stands: it links `references/deployment-cases.md` and `references/sources.md`, which don't exist, and it calls `$review-change`, which isn't installed. Leave them for the run, because it tests the skill as written. Record how V1 copes; fixing them afterwards is "skills v2".

**Score every run from its thread and worktree:**

- CI and smoke green (you rerun both)
- Files changed, LOC added and new files (`git diff --stat <base>`)
- Rubric hits
- Skills used, expected vs unexpected
- Tokens and wall-clock
- Whether it hit the time limit

**Schedule (about 3 h 20 m)**

| Time | Step |
| --- | --- |
| 0:00–0:30 | Commit the working tree. Plant the ModernBERT bug and the dependency seed (plus the optional source seed). Verify the UltraBERT model ID. Tag `v0-exp`. Build `exp/a1-base` and `exp/v1-base` and run CI on each. Write the private rubric note outside the repo. |
| 0:30–1:25 | A0: launch, set up, probe, step 1a, step 1b, quick score |
| 1:25–2:20 | A1 |
| 2:20–2:50 | V0 |
| 2:50–3:20 | V1, then the table, one chart per task, and three findings |

To cut the time, drop V0 and score V1 against the seed key alone. That loses the no-skills comparison for the vulnerability task. Keep step 1b for both A runs; dropping it for one arm makes A0 vs A1 a comparison of tasks, not skills.

## Between the pilot and Phase 2: full encoder training

The pilot only runs 5-step smoke training, so the gate still needs real encoder results. Pick the pilot branch whose encoder change scored best on the Task 1 rubric, review it, and merge it into `agent` as a human commit (with an `Intervention:` trailer naming the source run). Then train ModernBERT and UltraBERT with the same budget and seed count as RoBERTa, through the existing training and eval harness. The gate compares RoBERTa, ModernBERT and UltraBERT from these runs, not from the pilot.

