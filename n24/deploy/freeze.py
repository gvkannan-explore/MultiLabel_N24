"""Copy a gated run into the image the inference API is deployed from.

The Modal image build in n24/deploy/modal_app.py calls freeze_run, so the
deployed image holds its own copy of the checkpoint and never reads the runs
volume again.
"""

import json
import shutil
from pathlib import Path

from n24.train.train_baseline import sha256_file

# BERT-family encoders. Zero-shot LLM checkpoints are never deployed (ADR 0008).
ENCODER_MODEL_TYPES = frozenset(
    {
        "albert",
        "bert",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "roberta",
        "xlm-roberta",
    }
)


def freeze_run(run_dir: Path, weights_sha256: str, dest: Path) -> None:
    """Copy run_dir's metrics.json and best_model/ to dest/<run>/, after
    checking that best_model is a BERT-family encoder and that its weights are
    the ones the deployment gate judged. Raises ValueError otherwise."""
    best_model = run_dir / "best_model"
    model_type = json.loads((best_model / "config.json").read_text())["model_type"]
    if model_type not in ENCODER_MODEL_TYPES:
        raise ValueError(
            f"{run_dir.name} is a {model_type} checkpoint; only BERT-family "
            f"encoders deploy: {sorted(ENCODER_MODEL_TYPES)}"
        )
    actual = sha256_file(best_model / "model.safetensors")
    if actual != weights_sha256:
        raise ValueError(
            f"{run_dir.name}'s weights sha256 is {actual}, but the deployment "
            f"gate judged {weights_sha256}"
        )
    frozen = dest / run_dir.name
    shutil.copytree(best_model, frozen / "best_model")
    shutil.copy2(run_dir / "metrics.json", frozen / "metrics.json")
