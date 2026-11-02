# A1 run files

Run from this worktree with the locked environment and `datasets/prepared` in place:

```bash
uv sync --locked --group lint
```

## Five-step CPU smoke only

This executes a real model training run. It does not certify the full pilot acceptance criteria. Choose a new output directory each time; the scripts never remove an existing run.

```bash
bash bins/train/encoder_smoke.sh answerdotai/ModernBERT-base experiment_runs/a1-modernbert-manual-01
```

Fixed defaults: five steps, batch size 2, max length 128, seed 42, CPU. `OMP_NUM_THREADS` defaults to 4. Public model downloads can occur if the model is absent from the Hugging Face cache. No Modal or GPU jobs are dispatched.

## Pilot runs with CI gates

```bash
bash bins/experiments/run_a1.sh 1a experiment_runs/a1-modernbert-pilot-01
bash bins/experiments/run_a1.sh 1b experiment_runs/a1-ultrabert-pilot-01 experiment_runs/a1-modernbert-pilot-01
```

The driver stops on failed CI before training and reruns CI after the smoke. Before step 1b, it checks the saved ModernBERT CPU run and weights hash, requires UltraBERT deployment support, and saves the staged step-1a diff. It writes CI and smoke logs into a fresh ignored `.audit/a1-*` folder. It never stages, commits, or pushes changes.

These scripts execute the current branch. They do not launch a fresh agent, implement pending encoder support, or reproduce a blinded skills experiment.

## Current status

The existing `experiment_runs/a1-modernbert-smoke` run already completed five CPU steps. Its checkpoint passed evaluation, freezing, and the local prediction API. There is no need to retrain it to inspect those results; see `.audit/acceptance-summary.json` and `.audit/modernbert-artifact-check.json`.

Full CI remains blocked by 32 existing history-author privacy findings. The pilot driver therefore stops at CI. UltraBERT-150M loaded and ran on CPU during setup, but its deployment support and step-1b acceptance remain pending. Do not label a smoke-only run as a completed A1 experiment.

See [A1_SKILLS_USED.md](A1_SKILLS_USED.md) for the skill tally and protocol deviations.

## Agent task prompts

For a new controlled trial, send the contents of `a1-prompts/00-probe.txt`, then `a1-prompts/01-modernbert.txt`. Send `a1-prompts/02-ultrabert.txt` in the same subject thread only after step 1a passes CI and its CPU smoke. The pilot limits agent work to 30 minutes for 1a and 20 minutes for 1b; the shell scripts do not launch or time an agent.

A fresh trial needs its own clean skills-on base and subject thread. This modified branch already contains the partial A1 implementation and is not a fresh trial. Keep rubric and scoring notes outside the subject's checkout. Snapshot the staged 1a diff outside the repository before staging 1b; the driver also keeps an ignored audit copy.

For remote GPU runs, use [A1_MODAL_RUNBOOK.md](A1_MODAL_RUNBOOK.md). The local commands above still force CPU.
