# Evaluation

Three tools score models without judging them: the eval harness for trained
runs, zero-shot scoring for LLMs, and the external probes. The
[deployment gate](../README.md) judges only eval harness results.

## Eval harness

`n24/eval/run.py` scores one trained run on the test split and writes its
results file. It measures and does not judge.

```bash
uv run python -m n24.eval.run --run-dir runs_modal/roberta-baseline
```

`--data-dir` defaults to `datasets/prepared` and `--results-dir` to `results/nlp`.

- It runs on CPU with `torch.use_deterministic_algorithms(True)` and tokenizes
  with the run's own `max_length` from `metrics.json`, so the input matches
  training.
- It predicts the test split twice, loading the model fresh each time. The
  number of sections that differ between the passes and the largest logit
  difference are the repeatability numbers.
- It recomputes the weights and prepared dataset hashes and records them next
  to the values in `metrics.json`. These are reproducibility evidence.
- It refuses to overwrite an existing results file.

The results file `results/nlp/<run>.json` holds the eval version, the test split
hash, the run name and directory, the git commit, both pairs of hashes, test
accuracy, test macro F1, per-section test F1, the repeatability numbers, the
predicted section for every test article, and the torch and transformers
versions.

`EVAL_VERSION` in `n24/eval/run.py` is the eval version. Bump it whenever the
harness or the test split changes, then rerun the harness for every model,
the baseline model included. Results are only compared within one eval version
and test split ([ADR 0005](adr/0005-eval-comparability.md)).

## Zero-shot LLMs

`n24/eval/zero_shot.py` scores an instruction-tuned LLM on the test split with
no training, using vLLM on a Modal GPU. Run it from the repository root:

```bash
uv run modal run -m n24.eval.zero_shot::main --model Qwen/Qwen3-0.6B
uv run modal run -m n24.eval.zero_shot::main --model HuggingFaceTB/SmolLM2-360M-Instruct
uv run modal run -m n24.eval.zero_shot::main --model Qwen/Qwen3.5-9B --gpu L40S
uv run modal run -m n24.eval.zero_shot::main --model Qwen/Qwen3.5-27B-FP8 --gpu L40S
```

Without `--gpu` it runs on the first free L4 or A10G (24 GB), which fits
models up to about 4B parameters in BF16. Qwen3.5-9B (about 20 GB) and
Qwen3.5-27B-FP8 (about 30 GB) need an L40S (48 GB); the BF16 Qwen3.5-27B
(about 54 GB) needs `--gpu H100`. `main`, `probes`, and `probe` all take
`--gpu`.

- The prompt lists the sections and gives the abstract. vLLM decodes
  greedily and may answer only with a section name.
- It predicts the test split twice, each pass in a fresh container, and
  records how many sections differ between the passes.
- It pins the model to its current Hugging Face commit and records that
  commit, the prompt's sha256, and the git commit.
- It writes `results/llm/zs-<model name>.json` with the same eval version, test
  split hash, and metrics as the eval harness, and refuses to overwrite it.
- vLLM runs text only (`language_model_only`), so it never loads the vision
  encoder of a model such as Qwen3.5.
- Model weights are cached in the `n24-hf-cache` Modal volume, so only the
  first container for a model downloads them.

vLLM pins torch and transformers below the root `uv.lock`, so the Modal image
installs its own project, `zero-shot/`. After you change
`zero-shot/pyproject.toml`, run `uv lock` in that folder. The deployment
gate does not judge zero-shot results.

## External probes

`n24/eval/probes.py` scores a model on the external probes in the git-ignored
`datasets/external_probes/examples.jsonl`. It keeps only the `external` rows,
prints one row per probe with a correct count, and writes
`results/probes/<run>.json`. The probes are a small diagnostic, not part of the
test split or the deployment gate.

```bash
uv run python -m n24.eval.probes --run-dir runs_modal/baseline-roberta-epoch3-Oct312026
uv run modal run -m n24.eval.zero_shot::probes --model Qwen/Qwen3-0.6B
```

The LLM gets the same prompt and constrained decoding as its test-split run.
