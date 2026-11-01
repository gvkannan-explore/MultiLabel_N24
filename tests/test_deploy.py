"""Tests for freezing a run into the deployed image.

They train a tiny RoBERTa for two steps on synthetic splits, offline, and
freeze it the way the Modal image build does.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from test_baseline_model import make_tiny_run

from n24.api.app import create_app
from n24.deploy.freeze import freeze_run
from n24.train.train_baseline import sha256_file


class FreezeRunTests(unittest.TestCase):
    _tmp: tempfile.TemporaryDirectory[str]
    trained_run: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        _, cls.trained_run = make_tiny_run(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        # A copy per test, so a test can edit the run without affecting others.
        self.run_dir = self.root / "volume" / "tiny-run"
        shutil.copytree(self.trained_run, self.run_dir)
        (self.run_dir / "training_report.pdf").write_bytes(b"%PDF")
        self.weights = sha256_file(self.run_dir / "best_model" / "model.safetensors")
        self.dest = self.root / "frozen"

    def test_copies_only_what_the_api_reads_and_serves_from_the_copy(self):
        freeze_run(self.run_dir, self.weights, self.dest)
        frozen = self.dest / "tiny-run"
        self.assertEqual(
            {path.name for path in frozen.iterdir()}, {"metrics.json", "best_model"}
        )
        shutil.rmtree(self.run_dir)  # serving must not need the volume
        client = TestClient(create_app(frozen))
        response = client.post("/predict", json={"abstracts": ["A match report."]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["run"], "tiny-run")

    def test_rejects_weights_other_than_the_gated_ones(self):
        with self.assertRaisesRegex(ValueError, "sha256"):
            freeze_run(self.run_dir, "0" * 64, self.dest)
        self.assertFalse(self.dest.exists())

    def test_rejects_a_checkpoint_that_is_not_an_encoder(self):
        config_path = self.run_dir / "best_model" / "config.json"
        config = json.loads(config_path.read_text())
        config["model_type"] = "qwen3"
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "qwen3"):
            freeze_run(self.run_dir, self.weights, self.dest)
        self.assertFalse(self.dest.exists())


if __name__ == "__main__":
    unittest.main()
