"""Tests for the typed model of a run's metrics.json."""

import json
import tempfile
import unittest
from pathlib import Path

from test_training_report import fixture_metrics

from n24.train.metrics import TrainingMetrics


class TrainingMetricsTests(unittest.TestCase):
    def test_max_length_parses_from_the_recorded_args(self):
        self.assertEqual(fixture_metrics().max_length, 128)

    def test_reads_the_metrics_json_in_a_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            written = fixture_metrics().model_dump() | {"dev_runtime": 3.0}
            (run_dir / "metrics.json").write_text(json.dumps(written))
            self.assertEqual(TrainingMetrics.from_run_dir(run_dir), fixture_metrics())


if __name__ == "__main__":
    unittest.main()
