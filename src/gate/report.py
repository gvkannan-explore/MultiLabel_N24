"""Decide whether a scored run may be deployed.

Reads a results file from src/eval/run.py and the pre-registered criteria,
checks accuracy, repeatability, and reproducibility, and writes an HTML report
whose rows link to their evidence files.

Exit codes: 0 deployable, 1 not deployable, 2 refused because the results
file's eval version or test split differs from the criteria.

    uv run python -m src.gate.report results/roberta-baseline.json
"""

import argparse
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, ConfigDict, Field
from rich import print
from rich.table import Table

from src.eval.run import Results


class AccuracyCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    min_test_f1_macro: float
    min_test_accuracy: float
    min_section_f1: float


class RepeatabilityCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    logits_atol: float


class Criteria(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    eval_version: int
    test_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    accuracy: AccuracyCriteria
    repeatability: RepeatabilityCriteria


@dataclass(frozen=True)
class Check:
    name: str
    group: str
    claim: str
    falsified_if: str
    observed: str
    passed: bool
    evidence: Path


def load_criteria(path: Path) -> Criteria:
    """Raises pydantic's ValidationError, a ValueError, for missing keys or a
    test split hash that is not pinned."""
    with open(path, "rb") as stream:
        return Criteria.model_validate(tomllib.load(stream))


def checks(results: Results, criteria: Criteria, results_path: Path) -> list[Check]:
    accuracy = criteria.accuracy
    worst_section, worst_f1 = min(
        results.test_f1_per_section.items(), key=lambda item: item[1]
    )
    repeat = results.repeatability
    atol = criteria.repeatability.logits_atol
    metrics_path = Path(results.run_dir) / "metrics.json"
    commit = results.git_commit
    return [
        Check(
            name="test_f1_macro",
            group="Accuracy",
            claim="Test macro F1 meets the minimum",
            falsified_if=f"test macro F1 < {accuracy.min_test_f1_macro}",
            observed=f"{results.test_f1_macro:.4f}",
            passed=results.test_f1_macro >= accuracy.min_test_f1_macro,
            evidence=results_path,
        ),
        Check(
            name="test_accuracy",
            group="Accuracy",
            claim="Test accuracy meets the minimum",
            falsified_if=f"test accuracy < {accuracy.min_test_accuracy}",
            observed=f"{results.test_accuracy:.4f}",
            passed=results.test_accuracy >= accuracy.min_test_accuracy,
            evidence=results_path,
        ),
        Check(
            name="section_f1",
            group="Accuracy",
            claim="Every section meets the minimum F1",
            falsified_if=f"any section's test F1 < {accuracy.min_section_f1}",
            observed=f"worst: {worst_section} {worst_f1:.4f}",
            passed=worst_f1 >= accuracy.min_section_f1,
            evidence=results_path,
        ),
        Check(
            name="repeatability",
            group="Repeatability",
            claim="Two prediction passes give the same output",
            falsified_if=f"any section differs, or any logit differs by more than {atol}",
            observed=(
                f"{repeat.label_mismatches} sections differ, "
                f"max logit difference {repeat.max_abs_logit_diff:.2e}"
            ),
            passed=repeat.label_mismatches == 0 and repeat.max_abs_logit_diff <= atol,
            evidence=results_path,
        ),
        Check(
            name="weights_sha256",
            group="Reproducibility",
            claim="The evaluated weights are the trained weights",
            falsified_if="the weights' sha256 differs from metrics.json",
            observed=results.weights_sha256.recomputed[:12],
            passed=results.weights_sha256.recorded == results.weights_sha256.recomputed,
            evidence=metrics_path,
        ),
        Check(
            name="dataset_sha256",
            group="Reproducibility",
            claim="The run trained on this prepared dataset",
            falsified_if="the prepared dataset's sha256 differs from metrics.json",
            observed=results.dataset_sha256.recomputed[:12],
            passed=results.dataset_sha256.recorded == results.dataset_sha256.recomputed,
            evidence=metrics_path,
        ),
        Check(
            name="clean_commit",
            group="Reproducibility",
            claim="The run's code is a recorded, clean commit",
            falsified_if="no git commit is recorded, or it ends in -dirty",
            observed=commit or "none recorded",
            passed=commit is not None and not commit.endswith("-dirty"),
            evidence=metrics_path,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path, help="A results file from src.eval.run")
    parser.add_argument("--criteria", type=Path, default=Path("gate/criteria.toml"))
    parser.add_argument("--reports-dir", type=Path, default=Path("gate/reports"))
    args = parser.parse_args(argv)

    criteria = load_criteria(args.criteria)
    results = Results.model_validate_json(args.results.read_text())
    # Eval comparability: never judge results from another eval or test split.
    if results.eval_version != criteria.eval_version:
        print(
            f"[bold red]Refused[/]: {args.results} has eval version "
            f"{results.eval_version}, the criteria {criteria.eval_version}. "
            "Rerun the eval harness."
        )
        return 2
    if results.test_split_sha256 != criteria.test_split_sha256:
        print(
            f"[bold red]Refused[/]: {args.results} was scored on a test split "
            "other than the one pinned in the criteria."
        )
        return 2

    gate_checks = checks(results, criteria, args.results)
    deployable = all(check.passed for check in gate_checks)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.reports_dir / f"{args.results.stem}.html"
    template = Environment(
        loader=FileSystemLoader(Path(__file__).parent),
        autoescape=select_autoescape(["html", "j2"]),
    ).get_template("report.html.j2")
    report_path.write_text(
        template.render(
            results=results,
            criteria=criteria,
            checks=gate_checks,
            deployable=deployable,
            links={
                path: os.path.relpath(path, args.reports_dir)
                for path in {args.results, args.criteria}
                | {check.evidence for check in gate_checks}
            },
            criteria_path=args.criteria,
            results_path=args.results,
        )
    )

    table = Table(title=f"Deployment gate: {results.run}")
    for column in ("Check", "Observed", "Falsified if", "Verdict"):
        table.add_column(column)
    for check in gate_checks:
        verdict = "[green]pass[/]" if check.passed else "[red]fail[/]"
        table.add_row(check.claim, check.observed, check.falsified_if, verdict)
    print(table)
    verdict = (
        "[bold green]DEPLOYABLE[/]" if deployable else "[bold red]NOT DEPLOYABLE[/]"
    )
    print(f"{verdict} -> {report_path}")
    return 0 if deployable else 1


if __name__ == "__main__":
    sys.exit(main())
