"""Tests for the baseline model's data preparation and trainer.

They run offline on tiny synthetic split files with three sections.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_data = load_script("prepare_data")
train_baseline = load_script("train_baseline")

SECTIONS = ["Travel", "Books", "Sports"]


def record(section: str, n: int) -> dict:
    return {
        "section": section,
        "headline": f"  {section} headline\n{n} ",
        "abstract": f"Abstract about {section},\tnumber {n}.  ",
        "article": f"Body of {section}\n\nstory {n}.",
        "image": "ignored",
    }


def write_splits(directory: Path, splits: dict[str, list[dict]]) -> None:
    for split, records in splits.items():
        path = directory / prepare_data.SPLIT_FILES[split]
        path.write_text(json.dumps(records))


def fixture_splits(per_section: int = 4) -> dict[str, list[dict]]:
    return {
        split: [record(s, n) for n in range(per_section) for s in SECTIONS]
        for split in prepare_data.SPLIT_FILES
    }


class PrepareDataTests(unittest.TestCase):
    def prepare(self, splits):
        with tempfile.TemporaryDirectory() as tmp:
            write_splits(Path(tmp), splits)
            return prepare_data.prepare(Path(tmp), num_sections=len(SECTIONS))

    def test_writes_sorted_sections_and_matching_labels(self):
        prepared = self.prepare(fixture_splits())
        self.assertEqual(set(prepared), {"train", "dev", "test"})
        train = prepared["train"]
        self.assertEqual(train.column_names, ["headline", "abstract", "body", "label"])
        names = train.features["label"].names
        self.assertEqual(names, sorted(SECTIONS))
        for row, source in zip(train, fixture_splits()["train"], strict=True):
            self.assertEqual(names[row["label"]], source["section"])

    def test_collapses_whitespace_and_keeps_numbers(self):
        row = self.prepare(fixture_splits())["dev"][0]
        self.assertEqual(row["headline"], "Travel headline 0")
        self.assertEqual(row["abstract"], "Abstract about Travel, number 0.")
        self.assertEqual(row["body"], "Body of Travel story 0.")

    def test_rejects_section_missing_from_train(self):
        splits = fixture_splits()
        splits["dev"].append(record("Opinion", 0))
        with self.assertRaisesRegex(ValueError, "train does not have"):
            self.prepare(splits)

    def test_rejects_empty_abstract(self):
        splits = fixture_splits()
        splits["test"][2]["abstract"] = " \n "
        with self.assertRaisesRegex(ValueError, "test record 2 has an empty abstract"):
            self.prepare(splits)

    def test_rejects_wrong_section_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_splits(Path(tmp), fixture_splits())
            with self.assertRaisesRegex(ValueError, "Expected 24 sections"):
                prepare_data.prepare(Path(tmp))


def save_tiny_roberta(directory: Path, texts: list[str]) -> None:
    """Save a 2-layer RoBERTa and a tokenizer trained on the fixture text, so
    the trainer runs without downloading roberta-base."""
    from tokenizers.implementations import ByteLevelBPETokenizer
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

    bpe = ByteLevelBPETokenizer()
    # Special tokens first, so their ids match roberta-base (<pad> is 1).
    bpe.train_from_iterator(
        iter(texts),
        vocab_size=300,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    directory.mkdir()
    bpe.save_model(str(directory))
    tokenizer = RobertaTokenizerFast(
        vocab_file=str(directory / "vocab.json"),
        merges_file=str(directory / "merges.txt"),
    )
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        # RoBERTa positions start after the padding id, so allow 128 + 2.
        max_position_embeddings=130,
    )
    # An encoder with no classification head, like roberta-base.
    RobertaModel(config).save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))


class TrainBaselineTests(unittest.TestCase):
    def test_metrics_match_scikit_learn(self):
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score

        rng = np.random.default_rng(0)
        labels = rng.integers(0, 5, 200)
        # Section 4 is never predicted, so its F1 is 0 rather than undefined.
        logits = rng.normal(size=(200, 5))
        logits[:, 4] = -10
        preds = logits.argmax(axis=-1)
        metrics = train_baseline.compute_metrics(
            train_baseline.EvalPrediction(predictions=logits, label_ids=labels)
        )
        # scikit-learn accepts average=None; its inferred annotation says str.
        expected = f1_score(labels, preds, average=None, labels=range(5))  # pyright: ignore[reportArgumentType]
        np.testing.assert_allclose(
            train_baseline.f1_per_section(labels, preds, 5), expected
        )
        self.assertAlmostEqual(metrics["f1_macro"], float(np.mean(expected)))
        self.assertAlmostEqual(metrics["accuracy"], accuracy_score(labels, preds))

    def test_smoke_run_writes_best_model_and_metrics(self):
        from transformers import AutoModelForSequenceClassification

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "news").mkdir()
            write_splits(root / "news", fixture_splits())
            prepared = prepare_data.prepare(root / "news", num_sections=3)
            prepared.save_to_disk(str(root / "prepared"))
            save_tiny_roberta(
                root / "model", [r["abstract"] for r in fixture_splits()["train"]]
            )
            argv = [
                "--data-dir", str(root / "prepared"),
                "--output-dir", str(root / "run"),
                "--model", str(root / "model"),
                "--batch-size", "4",
                "--max-steps", "2",
            ]  # fmt: skip

            metrics = train_baseline.train(train_baseline.parse_args(argv))

            saved = json.loads((root / "run/metrics.json").read_text())
            self.assertEqual(saved, metrics)
            for key in ("dev_accuracy", "dev_f1_macro"):
                self.assertGreaterEqual(saved[key], 0.0)
                self.assertLessEqual(saved[key], 1.0)
            self.assertEqual(set(saved["dev_f1_per_section"]), set(SECTIONS))
            self.assertRegex(saved["weights_sha256"], "^[0-9a-f]{64}$")
            self.assertRegex(saved["dataset_sha256"], "^[0-9a-f]{64}$")
            self.assertEqual(saved["args"]["max_steps"], "2")
            self.assertFalse((root / "run/checkpoints").exists())

            model = AutoModelForSequenceClassification.from_pretrained(
                str(root / "run/best_model")
            )
            self.assertEqual(model.config.id2label, dict(enumerate(sorted(SECTIONS))))

            with self.assertRaises(FileExistsError):
                train_baseline.train(train_baseline.parse_args(argv))


if __name__ == "__main__":
    unittest.main()
