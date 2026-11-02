"""Tests for freezing a run into the deployed image.

They train a tiny RoBERTa for two steps on synthetic splits, offline, and
freeze it the way the Modal image build does.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from test_baseline_model import (
    fixture_splits,
    make_tiny_run,
    save_tiny_roberta,
    write_splits,
)

from n24.api.app import create_app
from n24.deploy.freeze import freeze_run
from n24.eval import run
from n24.train.train_baseline import parse_args, sha256_file, train


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

    def test_accepts_an_ultrabert_checkpoint(self):
        # dmis-lab/UltraBERT-150M saves model_type "ultrabert".
        config_path = self.run_dir / "best_model" / "config.json"
        config = json.loads(config_path.read_text())
        config["model_type"] = "ultrabert"
        config_path.write_text(json.dumps(config))
        freeze_run(self.run_dir, self.weights, self.dest)
        self.assertTrue((self.dest / "tiny-run" / "best_model").is_dir())

    def test_modernbert_trains_evaluates_and_serves_from_frozen_copy(self):
        from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

        from n24.data.prepare_data import prepare

        news = self.root / "news"
        news.mkdir()
        write_splits(news, fixture_splits())
        data_dir = self.root / "prepared"
        prepare(news, num_sections=3).save_to_disk(str(data_dir))
        model_dir = self.root / "modernbert"
        save_tiny_roberta(
            model_dir, [record["abstract"] for record in fixture_splits()["train"]]
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        ModernBertModel(
            ModernBertConfig(
                vocab_size=len(tokenizer),
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
                max_position_embeddings=128,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        ).save_pretrained(str(model_dir))
        run_dir = self.root / "modernbert-run"
        args = parse_args([
            "--model", str(model_dir),
            "--data-dir", str(data_dir),
            "--output-dir", str(run_dir),
            "--batch-size", "4",
            "--max-steps", "2",
            "--use-cpu",
        ])  # fmt: skip
        metrics = train(args, "abc123")
        results = run.evaluate(run_dir, data_dir, self.root / "results")
        abstracts = [record["abstract"] for record in fixture_splits()["test"]]
        eval_logits = run.predict(run_dir / "best_model", abstracts, args.max_length)
        freeze_run(run_dir, metrics.weights_sha256, self.dest)
        shutil.rmtree(run_dir)
        client = TestClient(create_app(self.dest / "modernbert-run"))
        response = client.post("/predict", json={"abstracts": abstracts})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(
            [body["sections"].index(item["section"]) for item in body["predictions"]],
            results.predictions,
        )
        np.testing.assert_allclose(
            [item["logits"] for item in body["predictions"]], eval_logits, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
