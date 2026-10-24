"""Tests for the baseline model's data preparation and trainer.

They run offline on tiny synthetic split files with three sections.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_data = load_script("prepare_data")

SECTIONS = ["Travel", "Books", "Sports"]


def record(section: str, n: int) -> dict:
    return {
        "section": section,
        "headline": f"  {section} headline\n{n} ",
        "abstract": f"Abstract about {section},\tnumber {n}.  ",
        "article": f"Body of {section}\n\nstory {n}.",
        "image": "ignored",
    }


def write_splits(directory: Path, splits: dict[str, list[dict]]) -> None:
    for split, records in splits.items():
        path = directory / prepare_data.SPLIT_FILES[split]
        path.write_text(json.dumps(records))


def fixture_splits(per_section: int = 4) -> dict[str, list[dict]]:
    return {
        split: [record(s, n) for n in range(per_section) for s in SECTIONS]
        for split in prepare_data.SPLIT_FILES
    }


class PrepareDataTests(unittest.TestCase):
    def prepare(self, splits):
        with tempfile.TemporaryDirectory() as tmp:
            write_splits(Path(tmp), splits)
            return prepare_data.prepare(Path(tmp), num_sections=len(SECTIONS))

    def test_writes_sorted_sections_and_matching_labels(self):
        prepared = self.prepare(fixture_splits())
        self.assertEqual(set(prepared), {"train", "dev", "test"})
        train = prepared["train"]
        self.assertEqual(train.column_names, ["headline", "abstract", "body", "label"])
        names = train.features["label"].names
        self.assertEqual(names, sorted(SECTIONS))
        for row, source in zip(train, fixture_splits()["train"], strict=True):
            self.assertEqual(names[row["label"]], source["section"])

    def test_collapses_whitespace_and_keeps_numbers(self):
        row = self.prepare(fixture_splits())["dev"][0]
        self.assertEqual(row["headline"], "Travel headline 0")
        self.assertEqual(row["abstract"], "Abstract about Travel, number 0.")
        self.assertEqual(row["body"], "Body of Travel story 0.")

    def test_rejects_section_missing_from_train(self):
        splits = fixture_splits()
        splits["dev"].append(record("Opinion", 0))
        with self.assertRaisesRegex(ValueError, "train does not have"):
            self.prepare(splits)

    def test_rejects_empty_abstract(self):
        splits = fixture_splits()
        splits["test"][2]["abstract"] = " \n "
        with self.assertRaisesRegex(ValueError, "test record 2 has an empty abstract"):
            self.prepare(splits)

    def test_rejects_wrong_section_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_splits(Path(tmp), fixture_splits())
            with self.assertRaisesRegex(ValueError, "Expected 24 sections"):
                prepare_data.prepare(Path(tmp))


if __name__ == "__main__":
    unittest.main()
