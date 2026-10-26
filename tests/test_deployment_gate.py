"""Tests for the eval harness and the deployment gate.

They train a tiny RoBERTa for two steps on synthetic splits, offline, then
score it with the harness and judge the results file with the gate.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from test_baseline_model import (
    SECTIONS,
    fixture_splits,
    save_tiny_roberta,
    write_splits,
)

from src.data import prepare_data
from src.eval import run
from src.gate import report
from src.train import train_baseline


def write_criteria(path: Path, **overrides: object) -> Path:
    values: dict[str, object] = {
        "eval_version": run.EVAL_VERSION,
        "test_split_sha256": "",
        "min_test_f1_macro": 0.0,
        "min_test_accuracy": 0.0,
        "min_section_f1": 0.0,
        "logits_atol": 1e-5,
    } | overrides
    path.write_text(
        f"eval_version = {values['eval_version']}\n"
        f'test_split_sha256 = "{values["test_split_sha256"]}"\n'
        "[accuracy]\n"
        f"min_test_f1_macro = {values['min_test_f1_macro']}\n"
        f"min_test_accuracy = {values['min_test_accuracy']}\n"
        f"min_section_f1 = {values['min_section_f1']}\n"
        "[repeatability]\n"
        f"logits_atol = {values['logits_atol']}\n"
    )
    return path


class DeploymentGateTests(unittest.TestCase):
    _tmp: tempfile.TemporaryDirectory[str]
    root: Path
    data_dir: Path
    run_dir: Path
    results_path: Path
    split_sha256: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        (cls.root / "news").mkdir()
        write_splits(cls.root / "news", fixture_splits())
        prepared = prepare_data.prepare(cls.root / "news", num_sections=len(SECTIONS))
        cls.data_dir = cls.root / "prepared"
        prepared.save_to_disk(str(cls.data_dir))
        save_tiny_roberta(
            cls.root / "model", [r["abstract"] for r in fixture_splits()["train"]]
        )
        cls.run_dir = cls.root / "run"
        argv = [
            "--data-dir", str(cls.data_dir),
            "--output-dir", str(cls.run_dir),
            "--model", str(cls.root / "model"),
            "--batch-size", "4",
            "--max-steps", "2",
        ]  # fmt: skip
        train_baseline.train(train_baseline.parse_args(argv), "abc123")
        run.evaluate(cls.run_dir, cls.data_dir, cls.root / "results")
        cls.results_path = cls.root / "results/run.json"
        cls.split_sha256 = train_baseline.dataset_sha256(cls.data_dir / "test")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def evaluate_edited_run(self, name: str, **edits: str) -> Path:
        """Copy the run, overwrite fields in its metrics.json, and score it."""
        run_dir = self.root / name
        shutil.copytree(self.run_dir, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        (run_dir / "metrics.json").write_text(json.dumps(metrics | edits))
        run.evaluate(run_dir, self.data_dir, self.root / "results")
        return self.root / f"results/{name}.json"

    def gate(self, results_path: Path, **overrides: object) -> tuple[int, Path]:
        tmp = Path(tempfile.mkdtemp(dir=self.root))
        criteria = write_criteria(
            tmp / "criteria.toml",
            **({"test_split_sha256": self.split_sha256} | overrides),
        )
        reports = tmp / "reports"
        code = report.main(
            [
                str(results_path),
                "--criteria",
                str(criteria),
                "--reports-dir",
                str(reports),
            ]
        )
        return code, reports / f"{results_path.stem}.html"

    def failed_checks(self, results_path: Path) -> set[str]:
        results = run.Results.model_validate_json(results_path.read_text())
        criteria = report.load_criteria(
            write_criteria(
                Path(tempfile.mkdtemp(dir=self.root)) / "criteria.toml",
                test_split_sha256=self.split_sha256,
            )
        )
        return {
            check.name
            for check in report.checks(results, criteria, results_path)
            if not check.passed
        }

    def test_results_file_records_version_split_hash_and_identity(self):
        results = json.loads(self.results_path.read_text())
        self.assertEqual(results["eval_version"], run.EVAL_VERSION)
        self.assertEqual(results["test_split_sha256"], self.split_sha256)
        self.assertEqual(results["git_commit"], "abc123")
        metrics = json.loads((self.run_dir / "metrics.json").read_text())
        self.assertEqual(
            results["weights_sha256"]["recorded"], metrics["weights_sha256"]
        )
        self.assertEqual(
            results["weights_sha256"]["recomputed"], metrics["weights_sha256"]
        )
        self.assertEqual(
            results["dataset_sha256"]["recomputed"], metrics["dataset_sha256"]
        )

    def test_metrics_match_scikit_learn_on_its_predictions(self):
        results = run.Results.model_validate_json(self.results_path.read_text())
        labels = load_from_disk(str(self.data_dir / "test"))["label"]
        self.assertEqual(len(results.predictions), len(labels))
        self.assertAlmostEqual(
            results.test_accuracy, accuracy_score(labels, results.predictions)
        )
        self.assertAlmostEqual(
            results.test_f1_macro,
            f1_score(labels, results.predictions, average="macro", labels=range(3)),
        )
        self.assertEqual(set(results.test_f1_per_section), set(SECTIONS))

    def test_two_prediction_passes_agree(self):
        results = run.Results.model_validate_json(self.results_path.read_text())
        self.assertEqual(results.repeatability.label_mismatches, 0)
        self.assertLessEqual(results.repeatability.max_abs_logit_diff, 1e-5)

    def test_harness_refuses_to_overwrite_results(self):
        with self.assertRaises(FileExistsError):
            run.evaluate(self.run_dir, self.data_dir, self.root / "results")

    def test_gate_passes_a_run_that_meets_the_criteria(self):
        code, html_path = self.gate(self.results_path)
        self.assertEqual(code, 0)
        html = html_path.read_text()
        self.assertIn("DEPLOYABLE", html)
        self.assertNotIn("NOT DEPLOYABLE", html)
        self.assertIn('href="../../results/run.json"', html)

    def test_gate_fails_a_run_below_the_f1_minimum(self):
        code, html_path = self.gate(self.results_path, min_test_f1_macro=1.01)
        self.assertEqual(code, 1)
        self.assertIn("NOT DEPLOYABLE", html_path.read_text())

    def test_gate_refuses_results_from_another_eval_version(self):
        code, html_path = self.gate(
            self.results_path, eval_version=run.EVAL_VERSION + 1
        )
        self.assertEqual(code, 2)
        self.assertFalse(html_path.exists())

    def test_gate_refuses_results_from_another_test_split(self):
        code, html_path = self.gate(self.results_path, test_split_sha256="0" * 64)
        self.assertEqual(code, 2)
        self.assertFalse(html_path.exists())

    def test_gate_rejects_criteria_without_a_pinned_split_hash(self):
        with self.assertRaises(ValueError):
            report.load_criteria(write_criteria(self.root / "unpinned.toml"))

    def test_changed_weights_fail_reproducibility(self):
        results_path = self.evaluate_edited_run("tampered", weights_sha256="0" * 64)
        self.assertEqual(self.failed_checks(results_path), {"weights_sha256"})

    def test_dirty_commit_fails_reproducibility(self):
        results_path = self.evaluate_edited_run("dirty", git_commit="abc123-dirty")
        self.assertEqual(self.failed_checks(results_path), {"clean_commit"})


if __name__ == "__main__":
    unittest.main()
