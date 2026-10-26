"""Fine-tune RoBERTa on the prepared dataset's abstracts.

Keeps the epoch with the best macro F1 on the dev split, then writes
best_model/ and metrics.json to the output directory. The test split is not
used. src/train/modal_train.py runs it on a Modal GPU.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import cast

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[2]


def f1_per_section(labels: np.ndarray, preds: np.ndarray, n: int) -> np.ndarray:
    """F1 for each section: 2 * true positives / (predicted + actual). A section
    with no predictions and no examples scores 0, as with scikit-learn's
    zero_division=0."""
    true_positives = np.bincount(labels[labels == preds], minlength=n)
    counts = np.bincount(preds, minlength=n) + np.bincount(labels, minlength=n)
    return np.divide(2 * true_positives, counts, out=np.zeros(n), where=counts > 0)


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = eval_pred.label_ids
    if isinstance(labels, tuple):
        labels = labels[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(np.mean(preds == labels)),
        "f1_macro": float(f1_per_section(labels, preds, logits.shape[-1]).mean()),
    }


def git_commit() -> str | None:
    """The commit the code ran from, with a -dirty suffix for uncommitted changes."""
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=40"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_sha256(data_dir: Path) -> str:
    """Hash the prepared dataset's files. state.json is skipped because the
    datasets library writes a random fingerprint into it on every save."""
    digest = hashlib.sha256()
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.name != "state.json":
            digest.update(path.relative_to(data_dir).as_posix().encode())
            digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def train(args: argparse.Namespace, commit: str | None) -> dict:
    """Train and write the run. `commit` is recorded in metrics.json; the
    caller passes it because a Modal container has no git checkout."""
    output_dir = Path(args.output_dir)
    # Refuse to overwrite an earlier run.
    output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)

    prepared = load_from_disk(str(args.data_dir))
    if not isinstance(prepared, DatasetDict):
        raise ValueError(f"{args.data_dir} is not a prepared dataset with splits")
    sections = prepared["train"].features["label"].names
    id2label = dict(enumerate(sections))
    label2id = {section: label for label, section in id2label.items()}

    # The model input is the abstract. Padding happens per batch in the collator.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized = DatasetDict({split: prepared[split] for split in ("train", "dev")}).map(
        lambda batch: tokenizer(
            batch["abstract"], truncation=True, max_length=args.max_length
        ),
        batched=True,
        remove_columns=["headline", "abstract", "body"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(sections),
        id2label=id2label,
        label2id=label2id,
    )

    # Smoke runs set max_steps, so they evaluate and save at that step instead
    # of at the end of each epoch.
    smoke = args.max_steps > 0
    strategy = "steps" if smoke else "epoch"
    checkpoint_dir = output_dir / "checkpoints"
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            learning_rate=args.lr,
            lr_scheduler_type="linear",
            warmup_steps=0.06,  # a float below 1 is a ratio of total steps
            weight_decay=0.01,
            eval_strategy=strategy,
            save_strategy=strategy,
            eval_steps=args.max_steps if smoke else None,
            save_steps=args.max_steps if smoke else 500,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            save_total_limit=1,
            save_only_model=True,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            logging_steps=50,
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    print(f"Training {args.model} on {trainer.args.device} -> {output_dir}")
    trainer.train()

    # The best epoch is loaded. Score it on the dev split once more for the
    # per-section F1.
    dev = trainer.predict(cast(TorchDataset, tokenized["dev"]), metric_key_prefix="dev")
    logits = (
        dev.predictions[0] if isinstance(dev.predictions, tuple) else dev.predictions
    )
    labels = dev.label_ids[0] if isinstance(dev.label_ids, tuple) else dev.label_ids
    if labels is None:
        raise ValueError("Expected labels in the dev split")
    per_section = f1_per_section(labels, np.argmax(logits, axis=-1), len(sections))

    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    shutil.rmtree(checkpoint_dir)

    metrics = {
        **(dev.metrics or {}),
        "dev_f1_per_section": {
            section: float(score)
            for section, score in zip(sections, per_section, strict=True)
        },
        "args": {key: str(value) for key, value in vars(args).items()},
        "git_commit": commit,
        "dataset_sha256": dataset_sha256(Path(args.data_dir)),
        "weights_sha256": sha256_file(best_dir / "model.safetensors"),
        # One entry per logging step and per evaluation: the training curves.
        "log_history": trainer.state.log_history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in metrics.items() if k.startswith("dev_")}, indent=2)
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/prepared"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="FacebookAI/roberta-base")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after this many steps for a smoke run; -1 trains all epochs",
    )
    return parser.parse_args(argv)


def main() -> None:
    train(parse_args(), git_commit())


if __name__ == "__main__":
    main()
