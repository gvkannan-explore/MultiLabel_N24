# A1 smoke experiments on Modal

These run files use the existing `n24.train.modal_train` GPU function. Its configured GPU preference is L4, then A10G, with a two-hour function timeout. Each invocation requests exactly five training steps, batch size 2, max length 128, seed 42, and the trainer's default learning rate. Dev evaluation and checkpoint/report writing still run afterward.

No jobs were launched while creating these files. Running the commands below starts billable Modal work. These are GPU smoke experiments; they do not replace the original pilot's five-step CPU acceptance check or perform full training.

## One-time setup

Use the A1 worktree and the Modal account/environment that owns the volumes:

```bash
uv sync --locked --group lint
uv run --locked modal setup
```

If `n24-data` does not exist, create it. Upload the prepared dataset once if `/data/prepared` is absent; do not overwrite an existing dataset just to rerun a smoke:

```bash
uv run --locked modal volume create n24-data
uv run --locked modal volume put n24-data datasets/prepared /data/prepared
```

The existing trainer mounts this at `/news24-data/data/prepared` and writes runs to the `n24-runs` volume. The image uses locked dependencies, caches the default RoBERTa model, and downloads the requested encoder when needed.

## Standalone GPU smoke runs

Choose a fresh name for each run:

```bash
bash bins/train/encoder_smoke_modal.sh answerdotai/ModernBERT-base a1-modernbert-modal-01
bash bins/train/encoder_smoke_modal.sh dmis-lab/UltraBERT-150M a1-ultrabert-modal-01
```

These commands train and download artifacts only. They do not require CI to pass, implement missing deployment support, or certify that an encoder is ready to serve. ModernBERT is currently implemented; UltraBERT's freeze support remains pending. If preserving the pilot sequence, do not run UltraBERT before step 1a meets its gate.

After each successful run, the script downloads the complete remote run into `runs_modal/<name>/` and prints its `metrics.json` locally. This includes the selected GPU, model, five-step history, hashes, and dev metrics. The checkpoint and PDF are downloaded too. Existing local runs are refused; the trainer also refuses existing remote output directories. Failed submissions stop before downloading. No changes are staged, committed, or pushed by these scripts.

If a remote run completed but download failed, retrieve it without retraining:

```bash
mkdir -p runs_modal
uv run --locked modal volume get n24-runs a1-modernbert-modal-01 runs_modal/
```

The download refuses existing files unless explicitly forced. Do not use a new run submission just to retry a download.

## GPU variant with pilot CI gates

```bash
bash bins/experiments/run_a1_modal.sh 1a a1-modernbert-modal-gated-01
bash bins/experiments/run_a1_modal.sh 1b a1-ultrabert-modal-gated-01 runs_modal/a1-modernbert-modal-gated-01
```

The driver runs local CI before any paid submission and after downloading the run. For step 1b, it requires a matching five-step GPU ModernBERT artifact, verifies its weights hash, requires UltraBERT freeze support, and snapshots the staged diff. Keep the step-1a snapshot before staging any step-1b changes. Logs remain in ignored `.audit/a1-*` directories.

CI currently fails on the documented history privacy findings, so the gated driver stops before launching Modal. Use the standalone commands only when you intentionally want a smoke run despite those unresolved acceptance checks.

Offline tests verify CLI arguments, failure propagation, downloaded-metrics printing, name validation, existing-run protection, and CI gating. The remote image, model training, account authentication, GPU allocation, and volume contents were not tested live.

References: [Modal volume CLI](https://modal.com/docs/cli/latest/volume), [GPU configuration](https://modal.com/docs/guide/gpu).
