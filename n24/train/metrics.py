"""The typed fields of the metrics.json a training run writes.

Training, the training report, the eval harness, the inference API, external
probe scoring, and the MLflow import all read a run through this model.
"""

from pathlib import Path
from typing import Self

from pydantic import BaseModel


class TrainingMetrics(BaseModel):
    """The fields of a run's metrics.json. The trainer also writes its own dev
    metrics, such as dev_runtime; validation ignores them."""

    dev_loss: float
    dev_accuracy: float
    dev_f1_macro: float
    dev_f1_per_section: dict[str, float]
    args: dict[str, str]
    # Older runs may have recorded no commit; the deployment gate rejects them.
    git_commit: str | None
    # Runs trained before the trainer recorded the GPU have no gpu field.
    gpu: str | None = None
    dataset_sha256: str
    weights_sha256: str
    log_history: list[dict[str, float]]

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> Self:
        return cls.model_validate_json((run_dir / "metrics.json").read_text())

    @property
    def max_length(self) -> int:
        """The token limit the run trained with, so inference truncates the same way."""
        return int(self.args["max_length"])
