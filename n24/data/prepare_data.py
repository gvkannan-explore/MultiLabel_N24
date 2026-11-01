"""Turn the N24News split files into the prepared dataset the trainer reads.

Writes a DatasetDict with train, dev, and test splits. Each row keeps the
headline, abstract, and body with whitespace collapsed, and a ClassLabel
`label` whose names are the sections, so the section names travel with the
data.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

SPLIT_FILES = {
    "train": "nytimes_train.json",
    "dev": "nytimes_dev.json",
    "test": "nytimes_test.json",
}
NUM_SECTIONS = 24


def clean(text: str) -> str:
    """Collapse whitespace only. Accents, numbers, and punctuation stay."""
    return re.sub(r"\s+", " ", text).strip()


def prepare(input_dir: Path, num_sections: int = NUM_SECTIONS) -> DatasetDict:
    records = {
        split: json.loads((input_dir / name).read_text(encoding="utf-8"))
        for split, name in SPLIT_FILES.items()
    }

    sections = sorted({record["section"] for record in records["train"]})
    if len(sections) != num_sections:
        raise ValueError(
            f"Expected {num_sections} sections in train, found {len(sections)}"
        )
    features = Features(
        {
            "headline": Value("string"),
            "abstract": Value("string"),
            "body": Value("string"),
            "label": ClassLabel(names=sections),
        }
    )

    splits = {}
    for split, split_records in records.items():
        rows = []
        for index, record in enumerate(split_records):
            if record["section"] not in sections:
                raise ValueError(
                    f"{split} record {index} has section {record['section']!r}, "
                    "which train does not have"
                )
            abstract = clean(record["abstract"])
            if not abstract:
                raise ValueError(f"{split} record {index} has an empty abstract")
            rows.append(
                {
                    "headline": clean(record["headline"]),
                    "abstract": abstract,
                    "body": clean(record["article"]),
                    "label": sections.index(record["section"]),
                }
            )
        splits[split] = Dataset.from_list(rows, features=features)
    return DatasetDict(splits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/news"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/prepared"))
    args = parser.parse_args()

    prepared = prepare(args.input_dir)
    prepared.save_to_disk(str(args.output_dir))

    names = prepared["train"].features["label"].names
    for split, dataset in prepared.items():
        print(f"{split}: {len(dataset)} rows")
    print(f"{len(names)} sections, records per section (train/dev/test):")
    counts = {split: Counter(dataset["label"]) for split, dataset in prepared.items()}
    for label, name in enumerate(names):
        per_split = "/".join(str(counts[split][label]) for split in SPLIT_FILES)
        print(f"  {name}: {per_split}")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
