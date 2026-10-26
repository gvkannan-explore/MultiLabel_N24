"""Tests for importing a run's metrics.json into a local MLflow store."""

import json
import tempfile
import unittest
from pathlib import Path

import mlflow
from test_baseline_model import fixture_splits, save_tiny_roberta, write_splits

from src.data import prepare_data
from src.track import import_run
from src.train import train_baseline


def fixture_metrics(output_dir: str) -> dict[str, object]:
    """A metrics.json as train_baseline writes it, shortened."""
    return {
        "dev_loss": 1.25,
        "dev_accuracy": 0.6,
        "dev_f1_macro": 0.5,
        "dev_runtime": 3.0,
        "dev_f1_per_section": {"Art & Design": 0.75, "Books": 0.25},
        "args": {"output_dir": output_dir, "lr": "2e-05", "seed": "42"},
        "git_commit": "abc123",
        "dataset_sha256": "d" * 64,
        "weights_sha256": "w" * 64,
        "log_history": [
            {"loss": 2.0, "grad_norm": 5.0, "learning_rate": 2e-5, "epoch": 0.5, "step": 50},
            {"loss": 1.5, "grad_norm": 4.0, "learning_rate": 1e-5, "epoch": 1.0, "step": 100},
            {"eval_loss": 1.25, "eval_f1_macro": 0.5, "epoch": 1.0, "step": 100},
        ],
    }  # fmt: skip


class ImportRunTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        mlflow.set_tracking_uri(f"sqlite:///{self.root / 'mlflow.db'}")

    def write_metrics(self, name: str) -> Path:
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(fixture_metrics(f"/news24-runs/{name}")))
        return path

    def test_logs_scores_params_and_tags(self):
        run_id = import_run.import_run(self.write_metrics("roberta-test"))

        assert run_id is not None
        run = mlflow.get_run(run_id)
        self.assertEqual(run.info.run_name, "roberta-test")
        self.assertEqual(run.info.status, "FINISHED")
        self.assertEqual(run.data.params["lr"], "2e-05")
        self.assertEqual(run.data.tags["git_commit"], "abc123")
        self.assertEqual(run.data.tags["weights_sha256"], "w" * 64)
        self.assertEqual(run.data.metrics["dev_f1_macro"], 0.5)
        self.assertEqual(run.data.metrics["dev_f1/Art and Design"], 0.75)
        self.assertEqual(run.data.metrics["dev_f1/Books"], 0.25)

    def test_logs_curves_by_step(self):
        run_id = import_run.import_run(self.write_metrics("roberta-test"))

        client = mlflow.MlflowClient()
        loss = client.get_metric_history(str(run_id), "loss")
        self.assertEqual([(m.step, m.value) for m in loss], [(50, 2.0), (100, 1.5)])
        eval_f1 = client.get_metric_history(str(run_id), "eval_f1_macro")
        self.assertEqual([(m.step, m.value) for m in eval_f1], [(100, 0.5)])

    def test_second_import_of_a_run_adds_nothing(self):
        path = self.write_metrics("roberta-test")
        import_run.import_run(path)

        self.assertIsNone(import_run.import_run(path))
        experiment = mlflow.get_experiment_by_name(import_run.EXPERIMENT)
        assert experiment is not None
        runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
        self.assertEqual(len(runs), 1)

    def test_imports_what_the_trainer_writes(self):
        (self.root / "news").mkdir()
        write_splits(self.root / "news", fixture_splits())
        prepared = prepare_data.prepare(self.root / "news", num_sections=3)
        prepared.save_to_disk(str(self.root / "prepared"))
        save_tiny_roberta(
            self.root / "model", [r["abstract"] for r in fixture_splits()["train"]]
        )
        argv = [
            "--data-dir", str(self.root / "prepared"),
            "--output-dir", str(self.root / "smoke-run"),
            "--model", str(self.root / "model"),
            "--batch-size", "4",
            "--max-steps", "2",
        ]  # fmt: skip
        metrics = train_baseline.train(train_baseline.parse_args(argv), "abc123")

        run_id = import_run.import_run(self.root / "smoke-run/metrics.json")

        assert run_id is not None
        run = mlflow.get_run(run_id)
        self.assertEqual(run.info.run_name, "smoke-run")
        self.assertEqual(run.data.metrics["dev_f1_macro"], metrics["dev_f1_macro"])
        self.assertIn("eval_f1_macro", run.data.metrics)

    def test_runs_with_other_names_are_kept_apart(self):
        first = import_run.import_run(self.write_metrics("roberta-a"))
        second = import_run.import_run(self.write_metrics("roberta-b"))

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
