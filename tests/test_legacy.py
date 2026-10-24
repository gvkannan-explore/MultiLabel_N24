"""Characterization tests for the legacy code.

They record what the legacy code does today, including its failures, so that
style and type fixes can be checked for unchanged behavior. They run offline:
model and tokenizer downloads are patched out.
"""

import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch
from transformers import RobertaConfig, RobertaModel

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "news_classifier", ROOT / "scripts/news-classifier.py"
)
assert SPEC is not None and SPEC.loader is not None
news_classifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(news_classifier)

from src.preprocess import roberta_preprocessor  # noqa: E402
from src.utils.data import load_datajson  # noqa: E402


@contextmanager
def working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def tiny_roberta(*_args, **_kwargs):
    config = RobertaConfig(
        vocab_size=50,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=40,
    )
    return RobertaModel(config)


class StubTokenizer:
    def encode_plus(self, text, max_length, **_kwargs):
        ids = torch.arange(max_length).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class LoadDataJsonTests(unittest.TestCase):
    def write(self, records):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        with handle:
            json.dump(records, handle)
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_labels_follow_first_seen_section_order(self):
        path = self.write(
            [
                {"section": "sports", "article": "a"},
                {"section": "arts", "article": "b"},
                {"section": "sports", "article": "c"},
            ]
        )
        texts, labels, n_classes = load_datajson(path)
        self.assertEqual(list(texts), ["a", "b", "c"])
        self.assertEqual(labels, [0, 1, 0])
        self.assertEqual(n_classes, 2)

    def test_abstract_only_records_raise_key_error(self):
        path = self.write([{"section": "sports", "abstract": "a"}])
        with self.assertRaises(KeyError):
            load_datajson(path)


class NewsClassifierScriptTests(unittest.TestCase):
    def test_dataset_item_is_flattened_with_long_label(self):
        dataset = news_classifier.NewsDataset(
            ["text"], [3], StubTokenizer(), max_length=6
        )
        item = dataset[0]
        self.assertEqual(len(dataset), 1)
        self.assertEqual(item["input_ids"].shape, (6,))
        self.assertEqual(item["attention_mask"].shape, (6,))
        self.assertEqual(item["label"].dtype, torch.long)
        self.assertEqual(item["label"].item(), 3)

    def test_classifier_returns_one_logit_per_class(self):
        with mock.patch.object(
            news_classifier.RobertaModel, "from_pretrained", side_effect=tiny_roberta
        ):
            model = news_classifier.NewsClassifier(n_classes=5)
        model.eval()
        input_ids = torch.randint(3, 50, (2, 7))
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        self.assertEqual(output.shape, (2, 5))

    def test_main_fails_without_the_csv(self):
        with tempfile.TemporaryDirectory() as tmp, working_directory(tmp):
            with self.assertRaisesRegex(FileNotFoundError, "n24news.csv"):
                news_classifier.main()


class RobertaPreprocessorTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            roberta_preprocessor.RobertaTokenizer, "from_pretrained"
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.preprocessor = roberta_preprocessor.RoBERTaPreprocessor()

    def test_clean_text(self):
        text = (
            "Dr. Smith  paid 25 dollars — see www.example.com !\n"
            "Then café closed, mail bob@example.com ( ok )"
        )
        self.assertEqual(
            self.preprocessor.clean_text(text),
            "Dr. Smith paid NUM dollars - see! Then caf closed, mail (ok)",
        )

    def test_sentences_skip_abbreviations(self):
        processed = self.preprocessor.process_text(
            "Dr. Smith left. Mrs. Jones stayed! Then"
        )
        self.assertEqual(
            processed.sentences,
            ["Dr. Smith left.", "Mrs. Jones stayed!", "Then"],
        )
        self.assertEqual(processed.num_sentences, 3)


if __name__ == "__main__":
    unittest.main()
