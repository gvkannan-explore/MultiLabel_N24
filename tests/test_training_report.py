"""Tests for the PDF report a training run writes next to metrics.json."""

import tempfile
import unittest
from pathlib import Path

from n24.train import report
from n24.train.metrics import TrainingMetrics


def fixture_metrics() -> TrainingMetrics:
    """A metrics.json of a two-epoch run, as train_baseline writes it."""
    log_history = [
        {"loss": 2.0 - step / 1000, "grad_norm": 5.0, "learning_rate": 2e-5,
         "epoch": step / 100, "step": step}
        for step in range(50, 201, 50)
    ] + [
        {"eval_loss": 0.9, "eval_accuracy": 0.7, "eval_f1_macro": 0.6,
         "eval_runtime": 3.0, "epoch": 1.0, "step": 100},
        {"eval_loss": 0.95, "eval_accuracy": 0.72, "eval_f1_macro": 0.58,
         "eval_runtime": 3.0, "epoch": 2.0, "step": 200},
        {"train_runtime": 3725.0, "train_loss": 1.8, "epoch": 2.0, "step": 200},
    ]  # fmt: skip
    return TrainingMetrics(
        dev_loss=0.9,
        dev_accuracy=0.7,
        dev_f1_macro=0.6,
        dev_f1_per_section={"Art & Design": 0.9, "Books": 0.5, "Travel": 0.4},
        args={"output_dir": "/news24-runs/roberta-a", "model": "m",
              "epochs": "2", "lr": "2e-05", "batch_size": "32",
              "max_length": "128", "seed": "42", "max_steps": "-1"},
        git_commit="abc123",
        dataset_sha256="d" * 64,
        weights_sha256="w" * 64,
        log_history=log_history,
    )  # fmt: skip


class TrainingReportTests(unittest.TestCase):
    def test_ema_is_debiased(self):
        smoothed = report.ema([2.0, 2.0, 1.0])
        self.assertAlmostEqual(smoothed[0], 2.0)
        self.assertAlmostEqual(smoothed[1], 2.0)
        self.assertLess(smoothed[2], 2.0)
        self.assertGreater(smoothed[2], 1.0)

    def test_best_epoch_is_the_eval_with_the_highest_macro_f1(self):
        best = report.best_eval(fixture_metrics().log_history)
        self.assertEqual((best["epoch"], best["step"]), (1.0, 100))

    def test_writes_a_three_page_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_report.pdf"
            report.write_report(fixture_metrics(), path)
            pdf = path.read_bytes()
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertEqual(pdf.count(b"/Type /Page "), 3)


if __name__ == "__main__":
    unittest.main()
