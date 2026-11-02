"""Tests for scoring external probes, on a tiny probe file written per test."""

import json
import tempfile
import unittest
from pathlib import Path

from n24.eval import probes

SECTIONS = ["Books", "Science", "Sports"]


class ProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def write_probes(self, rows: list[dict[str, str]]) -> Path:
        path = self.root / "examples.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def fixture_probes(self) -> Path:
        return self.write_probes(
            [
                {
                    "id": "E01",
                    "abstract": "A sample from an asteroid.",
                    "group": "external",
                    "expected_section": "Science",
                },
                {
                    "id": "E02",
                    "abstract": "Spain won the final.",
                    "group": "external",
                    "expected_section": "Sports",
                },
                {
                    "id": "A01",
                    "abstract": "A novel set in orbit.",
                    "group": "adversarial",
                    "expected_section": "Books",
                },
            ]
        )

    def test_load_keeps_only_external_probes(self):
        loaded = probes.load_probes(self.fixture_probes(), SECTIONS)
        self.assertEqual([probe.id for probe in loaded], ["E01", "E02"])

    def test_load_rejects_unknown_section(self):
        path = self.write_probes(
            [
                {
                    "id": "E01",
                    "abstract": "Text.",
                    "group": "external",
                    "expected_section": "Weather",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "Weather"):
            probes.load_probes(path, SECTIONS)

    def test_score_counts_correct_predictions(self):
        path = self.fixture_probes()
        loaded = probes.load_probes(path, SECTIONS)
        results = probes.score_probes("tiny", loaded, ["Science", "Books"], path)
        self.assertEqual((results.correct, results.total), (1, 2))
        self.assertEqual(results.predictions, {"E01": "Science", "E02": "Books"})

    def test_write_refuses_to_overwrite(self):
        path = self.fixture_probes()
        loaded = probes.load_probes(path, SECTIONS)
        results = probes.score_probes("tiny", loaded, ["Science", "Sports"], path)
        written = probes.write_probe_results(results, self.root / "results")
        self.assertEqual(
            probes.ProbeResults.model_validate_json(written.read_text()), results
        )
        with self.assertRaises(FileExistsError):
            probes.write_probe_results(results, self.root / "results")


if __name__ == "__main__":
    unittest.main()
