"""Tests for zero-shot scoring.

A fake predict function stands in for vLLM on Modal, so they run offline on
CPU against the tiny synthetic splits.
"""

import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

from sklearn.metrics import accuracy_score, f1_score
from test_baseline_model import SECTIONS, fixture_splits, write_splits

from n24.data import prepare_data
from n24.eval import run, zero_shot
from n24.train import train_baseline


class ZeroShotTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "news").mkdir()
        write_splits(self.root / "news", fixture_splits())
        self.data_dir = self.root / "prepared"
        prepare_data.prepare(
            self.root / "news", num_sections=len(SECTIONS)
        ).save_to_disk(str(self.data_dir))
        test = run.load_test_split(self.data_dir)
        self.sections: list[str] = test.features["label"].names
        self.abstracts = cast(list[str], test["abstract"])
        self.labels = cast(list[int], test["label"])

    def evaluate(self, *passes: list[int]) -> zero_shot.ZeroShotResults:
        """Score with a fake predict that returns the given passes in order."""
        answers = iter(passes)
        self.calls: list[tuple[list[zero_shot.Chat], list[str]]] = []

        def predict(chats: list[zero_shot.Chat], sections: list[str]) -> list[int]:
            self.calls.append((chats, sections))
            return next(answers)

        return zero_shot.evaluate(
            model="org/Tiny-LM",
            revision="0123abc",
            data_dir=self.data_dir,
            results_dir=self.root / "results",
            predict=predict,
            commit="abc123",
        )

    def test_prompt_lists_every_section_and_the_abstract(self):
        abstract = self.abstracts[0]
        messages = zero_shot.build_messages(abstract, self.sections)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        for section in self.sections:
            self.assertIn(f"- {section}\n", messages[1]["content"])
        self.assertTrue(messages[1]["content"].endswith(abstract))

    def test_predict_runs_twice_on_every_test_abstract(self):
        labels = self.labels
        self.evaluate(labels, labels)
        self.assertEqual(len(self.calls), 2)
        chats, sections = self.calls[0]
        self.assertEqual(sections, self.sections)
        self.assertEqual(
            chats,
            [zero_shot.build_messages(a, sections) for a in self.abstracts],
        )

    def test_results_record_ruler_and_identity(self):
        labels = self.labels
        self.evaluate(labels, labels)
        results = zero_shot.ZeroShotResults.model_validate_json(
            (self.root / "results/zs-Tiny-LM.json").read_text()
        )
        self.assertEqual(results.eval_version, run.EVAL_VERSION)
        self.assertEqual(
            results.test_split_sha256,
            train_baseline.dataset_sha256(self.data_dir / "test"),
        )
        self.assertEqual(results.run, "zs-Tiny-LM")
        self.assertEqual(results.model, "org/Tiny-LM")
        self.assertEqual(results.model_revision, "0123abc")
        self.assertEqual(results.prompt_sha256, zero_shot.prompt_sha256())
        self.assertEqual(results.git_commit, "abc123")

    def test_metrics_match_scikit_learn(self):
        labels = self.labels
        # Answer the first section for every article.
        preds = [0] * len(labels)
        results = self.evaluate(preds, preds)
        self.assertEqual(results.predictions, preds)
        self.assertAlmostEqual(results.test_accuracy, accuracy_score(labels, preds))
        self.assertAlmostEqual(
            results.test_f1_macro,
            f1_score(labels, preds, average="macro", labels=range(len(SECTIONS))),
        )
        self.assertEqual(set(results.test_f1_per_section), set(SECTIONS))

    def test_counts_sections_that_differ_between_passes(self):
        labels = self.labels
        second = [(label + 1) % len(SECTIONS) for label in labels[:2]] + labels[2:]
        results = self.evaluate(labels, second)
        self.assertEqual(results.label_mismatches, 2)
        self.assertEqual(results.predictions, labels)

    def test_refuses_to_overwrite_results(self):
        labels = self.labels
        self.evaluate(labels, labels)
        with self.assertRaises(FileExistsError):
            self.evaluate(labels, labels)


class GpuTests(unittest.TestCase):
    def test_without_gpu_the_function_keeps_its_default_gpus(self):
        function = zero_shot.predict_remote
        self.assertIs(zero_shot.on_gpu(function, None), function)

    def test_gpu_overrides_the_default(self):
        function = zero_shot.predict_remote
        with mock.patch.object(function, "with_options") as with_options:
            zero_shot.on_gpu(function, "L40S")
        with_options.assert_called_once_with(gpu="L40S")


if __name__ == "__main__":
    unittest.main()
