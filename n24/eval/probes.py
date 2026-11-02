"""Score a model on the external probes and write its probe results file.

External probes are abstracts from outside N24News's time range with a
hand-inferred section (datasets/external_probes/). They are a small
diagnostic, kept apart from the test split and the deployment gate. This
module scores the baseline model; zero_shot.py's `probes` entrypoint scores
an LLM.

    uv run python -m n24.eval.probes --run-dir runs_modal/baseline-roberta-epoch3-Oct312026
"""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel
from rich import print
from rich.table import Table

from n24.eval.run import RunMetrics, predict
from n24.train.train_baseline import sha256_file

PROBES_PATH = Path("datasets/external_probes/examples.jsonl")
RESULTS_DIR = Path("results/probes")


class Probe(BaseModel):
    id: str
    abstract: str
    expected_section: str


class ModelConfig(BaseModel):
    """The field of a model's config.json that maps logits to sections."""

    id2label: dict[int, str]


class ProbeResults(BaseModel):
    run: str
    probes_sha256: str
    predictions: dict[str, str]
    correct: int
    total: int


def load_probes(path: Path, sections: list[str]) -> list[Probe]:
    """The external probes in path. Raises if a probe expects an unknown section."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    loaded = [Probe.model_validate(row) for row in rows if row["group"] == "external"]
    for probe in loaded:
        if probe.expected_section not in sections:
            raise ValueError(
                f"{probe.id} expects unknown section {probe.expected_section!r}"
            )
    return loaded


def score_probes(
    run: str, probes: list[Probe], predicted: list[str], probes_path: Path
) -> ProbeResults:
    """Score predicted sections, in probe order, and print one row per probe."""
    table = Table(title=f"{run}: external probes")
    for column in ("Probe", "Expected", "Predicted", "Correct"):
        table.add_column(column)
    correct = 0
    for probe, section in zip(probes, predicted, strict=True):
        hit = section == probe.expected_section
        correct += hit
        table.add_row(probe.id, probe.expected_section, section, "✓" if hit else "✗")
    print(table)
    print(f"{run}: {correct}/{len(probes)} correct")
    return ProbeResults(
        run=run,
        probes_sha256=sha256_file(probes_path),
        predictions={
            probe.id: section for probe, section in zip(probes, predicted, strict=True)
        },
        correct=correct,
        total=len(probes),
    )


def write_probe_results(results: ProbeResults, results_dir: Path) -> Path:
    """Write results_dir/<run>.json. Refuses to overwrite an existing file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{results.run}.json"
    with open(path, "x") as stream:
        stream.write(results.model_dump_json(indent=2) + "\n")
    print(f"-> {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--probes", type=Path, default=PROBES_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    model_dir = args.run_dir / "best_model"
    id2label = ModelConfig.model_validate_json(
        (model_dir / "config.json").read_text()
    ).id2label
    metrics = RunMetrics.model_validate_json(
        (args.run_dir / "metrics.json").read_text()
    )
    probes = load_probes(args.probes, list(id2label.values()))
    logits = predict(
        model_dir,
        [probe.abstract for probe in probes],
        int(metrics.args["max_length"]),
    )
    predicted = [id2label[int(i)] for i in logits.argmax(axis=-1)]
    results = score_probes(args.run_dir.name, probes, predicted, args.probes)
    write_probe_results(results, args.results_dir)


if __name__ == "__main__":
    main()
