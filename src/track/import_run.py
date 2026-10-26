"""Log runs from their metrics.json into the local MLflow store.

Training never talks to MLflow. This command reads the metrics.json a run
wrote and logs it as one MLflow run named after the run's output directory.
A run that is already in the store is skipped, so the store is a cache:
delete it and import metrics/*.json again to rebuild it.

    uv run python -m src.track.import_run metrics/*.json
    uv run mlflow ui --backend-store-uri sqlite:///experiment_runs/mlflow.db
"""

import argparse
from pathlib import Path

import mlflow
from pydantic import BaseModel
from rich import print

EXPERIMENT = "baseline-model"


class TrainingMetrics(BaseModel):
    """The fields of a run's metrics.json that the import logs."""

    dev_loss: float
    dev_accuracy: float
    dev_f1_macro: float
    dev_f1_per_section: dict[str, float]
    args: dict[str, str]
    git_commit: str | None
    dataset_sha256: str
    weights_sha256: str
    log_history: list[dict[str, float]]


def import_run(path: Path) -> str | None:
    """Log one metrics.json into the current tracking store. Returns the new
    run's id, or None when a run with the same name is already there."""
    metrics = TrainingMetrics.model_validate_json(path.read_text())
    name = Path(metrics.args["output_dir"]).name
    mlflow.set_experiment(EXPERIMENT)
    if mlflow.search_runs(
        filter_string=f"attributes.run_name = '{name}'", output_format="list"
    ):
        return None
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(metrics.args)
        for entry in metrics.log_history:
            step = int(entry["step"])
            mlflow.log_metrics(
                {k: v for k, v in entry.items() if k not in ("step", "epoch")},
                step=step,
            )
        # MLflow refuses "&" in metric names, and "Art & Design" is a section.
        mlflow.log_metrics(
            {
                "dev_loss": metrics.dev_loss,
                "dev_accuracy": metrics.dev_accuracy,
                "dev_f1_macro": metrics.dev_f1_macro,
            }
            | {
                f"dev_f1/{section.replace('&', 'and')}": score
                for section, score in metrics.dev_f1_per_section.items()
            }
        )
        mlflow.set_tags(
            {
                "git_commit": str(metrics.git_commit),
                "dataset_sha256": metrics.dataset_sha256,
                "weights_sha256": metrics.weights_sha256,
            }
        )
    return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", type=Path, nargs="+", help="metrics.json files")
    parser.add_argument("--tracking-uri", default="sqlite:///experiment_runs/mlflow.db")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    for path in args.paths:
        if import_run(path) is None:
            print(f"[dim]{path}: already in {args.tracking_uri}[/dim]")
        else:
            print(f"[green]{path}: imported into {args.tracking_uri}[/green]")


if __name__ == "__main__":
    main()
