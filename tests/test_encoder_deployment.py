"""Offline coverage of additional encoders through the existing workflow."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from test_baseline_model import fixture_splits, save_tiny_roberta, write_splits
from transformers import ModernBertConfig, ModernBertModel

from n24.api.app import create_app
from n24.data.prepare_data import prepare
from n24.deploy.freeze import freeze_run
from n24.eval.run import compute_logits, load_model
from n24.train.train_baseline import parse_args, sha256_file, train


class ModernBertDeploymentTests(unittest.TestCase):
    def test_trained_encoder_evaluates_freezes_and_serves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "news").mkdir()
            write_splits(root / "news", fixture_splits())
            prepare(root / "news", num_sections=3).save_to_disk(str(root / "prepared"))
            model_dir = root / "model"
            save_tiny_roberta(
                model_dir, [row["abstract"] for row in fixture_splits()["train"]]
            )
            config = ModernBertConfig(
                vocab_size=300,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
                max_position_embeddings=128,
                pad_token_id=1,
                bos_token_id=0,
                eos_token_id=2,
                cls_token_id=0,
                sep_token_id=2,
            )
            ModernBertModel(config).save_pretrained(str(model_dir))
            run_dir = root / "run"
            args = parse_args(
                [
                    "--data-dir",
                    str(root / "prepared"),
                    "--output-dir",
                    str(run_dir),
                    "--model",
                    str(model_dir),
                    "--batch-size",
                    "4",
                    "--max-steps",
                    "2",
                    "--use-cpu",
                ]
            )
            metrics = train(args, "abc123")
            self.assertIsNone(metrics.gpu)
            self.assertEqual(metrics.args["use_cpu"], "True")
            abstracts = ["A match report.", "A book review."]
            tokenizer, model = load_model(run_dir / "best_model")
            logits = compute_logits(tokenizer, model, abstracts, max_length=128)
            self.assertEqual(logits.shape, (2, 3))
            self.assertTrue(np.isfinite(logits).all())
            freeze_run(
                run_dir,
                sha256_file(run_dir / "best_model/model.safetensors"),
                root / "frozen",
            )
            client = TestClient(create_app(root / "frozen/run"))
            response = client.post("/predict", json={"abstracts": abstracts})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["sections"], ["Books", "Sports", "Travel"])
            self.assertEqual(len(result["predictions"]), 2)
            np.testing.assert_allclose(
                [prediction["logits"] for prediction in result["predictions"]], logits
            )
            self.assertEqual(
                [prediction["section"] for prediction in result["predictions"]],
                [result["sections"][index] for index in logits.argmax(axis=-1)],
            )


if __name__ == "__main__":
    unittest.main()
