"""Score a trained run on the test split and write its results file.

Predicts the test split twice, each pass with a fresh model load, on CPU with
deterministic algorithms, and records how far the passes differ
(repeatability). Recomputes the weights and dataset hashes next to the ones in
the run's metrics.json (reproducibility). It only measures; the deployment
gate in src/gate/report.py judges the results file.

    uv run python -m src.eval.run --run-dir runs_modal/roberta-baseline
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import DatasetDict, load_from_disk
from pydantic import BaseModel
from rich import print
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.train.train_baseline import dataset_sha256, f1_per_section, sha256_file

# Bump when this harness or the test split changes, then rerun every model.
EVAL_VERSION = 1
BATCH_SIZE = 32


class RunMetrics(BaseModel):
    """The fields of a run's metrics.json that the harness reads."""

    git_commit: str | None
    weights_sha256: str
    dataset_sha256: str
    args: dict[str, str]


class Hashes(BaseModel):
    recorded: str
    recomputed: str


class Repeatability(BaseModel):
    label_mismatches: int
    max_abs_logit_diff: float


class Results(BaseModel):
    eval_version: int
    run: str
    run_dir: str
    test_split_sha256: str
    git_commit: str | None
    weights_sha256: Hashes
    dataset_sha256: Hashes
    test_accuracy: float
    test_f1_macro: float
    test_f1_per_section: dict[str, float]
    repeatability: Repeatability
    predictions: list[int]
    torch_version: str
    transformers_version: str


def predict(model_dir: Path, abstracts: list[str], max_length: int) -> np.ndarray:
    """Logits for each abstract, from a fresh load of the model."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(abstracts), BATCH_SIZE):
            inputs = tokenizer(
                abstracts[start : start + BATCH_SIZE],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            batches.append(model(**inputs).logits.numpy())
    return np.concatenate(batches)


def evaluate(run_dir: Path, data_dir: Path, results_dir: Path) -> Results:
    """Score run_dir/best_model on the test split and write
    results_dir/<run>.json. Refuses to overwrite an existing results file."""

    torch.use_deterministic_algorithms(True)
    metrics = RunMetrics.model_validate_json((run_dir / "metrics.json").read_text())
    prepared = load_from_disk(str(data_dir))
    if not isinstance(prepared, DatasetDict):
        raise ValueError(f"{data_dir} is not a prepared dataset with splits")
    test = prepared["test"]
    sections = test.features["label"].names
    labels = np.array(test["label"])

    # Tokenize with the run's own max length, so the input matches training.
    model_dir = run_dir / "best_model"
    max_length = int(metrics.args["max_length"])
    first = predict(model_dir, test["abstract"], max_length)
    second = predict(model_dir, test["abstract"], max_length)
    preds = first.argmax(axis=-1)
    per_section = f1_per_section(labels, preds, len(sections))

    results = Results(
        eval_version=EVAL_VERSION,
        run=run_dir.name,
        run_dir=str(run_dir),
        test_split_sha256=dataset_sha256(data_dir / "test"),
        git_commit=metrics.git_commit,
        weights_sha256=Hashes(
            recorded=metrics.weights_sha256,
            recomputed=sha256_file(model_dir / "model.safetensors"),
        ),
        dataset_sha256=Hashes(
            recorded=metrics.dataset_sha256, recomputed=dataset_sha256(data_dir)
        ),
        test_accuracy=float(np.mean(preds == labels)),
        test_f1_macro=float(per_section.mean()),
        test_f1_per_section={
            section: float(score)
            for section, score in zip(sections, per_section, strict=True)
        },
        repeatability=Repeatability(
            label_mismatches=int(np.sum(preds != second.argmax(axis=-1))),
            max_abs_logit_diff=float(np.max(np.abs(first - second))),
        ),
        predictions=preds.tolist(),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_dir.name}.json"
    with open(results_path, "x") as stream:
        stream.write(results.model_dump_json(indent=2) + "\n")
    print(
        f"{results.run}: test macro F1 {results.test_f1_macro:.4f}, "
        f"accuracy {results.test_accuracy:.4f} -> {results_path}"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/prepared"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    evaluate(args.run_dir, args.data_dir, args.results_dir)


if __name__ == "__main__":
    main()
