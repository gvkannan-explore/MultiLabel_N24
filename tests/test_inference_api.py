"""Tests for the inference API.

They train a tiny RoBERTa for two steps on synthetic splits, offline, score it
with the eval harness, and check that the API predicts what the harness does.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
from test_baseline_model import SECTIONS, make_tiny_run

from n24.api.app import create_app
from n24.eval import run


class InferenceApiTests(unittest.TestCase):
    _tmp: tempfile.TemporaryDirectory[str]
    run_dir: Path
    abstracts: list[str]
    results: run.Results
    client: TestClient

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        data_dir, cls.run_dir = make_tiny_run(root)
        cls.results = run.evaluate(cls.run_dir, data_dir, root / "results")
        cls.abstracts = TypeAdapter(list[str]).validate_python(
            list(load_from_disk(str(data_dir / "test"))["abstract"])
        )
        cls.client = TestClient(create_app(cls.run_dir))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def predict(self, abstracts: list[str]) -> dict:
        response = self.client.post("/predict", json={"abstracts": abstracts})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_names_the_run_and_its_sections(self):
        body = self.predict(self.abstracts[:1])
        self.assertEqual(body["run"], self.run_dir.name)
        self.assertEqual(body["sections"], sorted(SECTIONS))

    def test_predictions_match_the_eval_harness(self):
        body = self.predict(self.abstracts)
        section_ids = [
            body["sections"].index(prediction["section"])
            for prediction in body["predictions"]
        ]
        self.assertEqual(section_ids, self.results.predictions)
        max_length = int(
            run.RunMetrics.model_validate_json(
                (self.run_dir / "metrics.json").read_text()
            ).args["max_length"]
        )
        expected = run.predict(self.run_dir / "best_model", self.abstracts, max_length)
        np.testing.assert_allclose(
            [prediction["logits"] for prediction in body["predictions"]],
            expected,
            atol=1e-5,
        )

    def test_section_is_the_argmax_of_its_logits(self):
        body = self.predict(self.abstracts)
        for prediction in body["predictions"]:
            self.assertEqual(len(prediction["logits"]), len(body["sections"]))
            self.assertEqual(
                prediction["section"],
                body["sections"][int(np.argmax(prediction["logits"]))],
            )

    def test_one_prediction_per_abstract_in_order(self):
        body = self.predict(self.abstracts[:3])
        alone = [self.predict([abstract]) for abstract in self.abstracts[:3]]
        self.assertEqual(
            [prediction["section"] for prediction in body["predictions"]],
            [single["predictions"][0]["section"] for single in alone],
        )

    def test_rejects_an_empty_list(self):
        response = self.client.post("/predict", json={"abstracts": []})
        self.assertEqual(response.status_code, 422)

    def test_rejects_an_empty_abstract(self):
        response = self.client.post("/predict", json={"abstracts": [""]})
        self.assertEqual(response.status_code, 422)

    def test_rejects_a_request_without_abstracts(self):
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
