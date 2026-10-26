"""Run src/train/train_baseline.py on a Modal GPU.

The image installs the locked dependencies from uv.lock. The prepared dataset
comes from the n24-data volume and each run is written to the n24-runs volume.

One-time setup:
    uv run modal setup
    uv run modal volume create n24-data
    uv run modal volume put n24-data datasets/prepared /prepared

Runs:
    uv run modal run -m src.train.modal_train --name smoke-1 --args "--max-steps 5"
    uv run modal run -m src.train.modal_train --name roberta-baseline
    uv run modal volume get n24-runs roberta-baseline runs_modal/
"""

import shlex

import modal

from src.train.train_baseline import git_commit, parse_args, train

app = modal.App("n24-baseline")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extra_options="--no-dev")
    # Cache the default model's PyTorch weights and tokenizer in the image so a
    # run does not download them again. The legacy name roberta-base fails here
    # with a 404, so this uses the canonical name.
    .run_commands(
        "hf download FacebookAI/roberta-base"
        " --include '*.json' --include '*.txt' --include '*.safetensors'"
    )
    .add_local_python_source("src")
)
data = modal.Volume.from_name("n24-data")
runs = modal.Volume.from_name("n24-runs", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=2 * 60 * 60,
    volumes={"/news24-data": data, "/news24-runs": runs},
)
def train_remote(argv: list[str], commit: str | None) -> dict:
    metrics = train(parse_args(argv), commit)
    runs.commit()
    return metrics


@app.local_entrypoint()
def main(name: str, args: str = "") -> None:
    """Train into /runs/<name> with extra trainer flags from --args."""
    argv = [
        "--data-dir", "/news24-data/data/prepared",
        "--output-dir", f"/news24-runs/{name}",
        *shlex.split(args),
    ]  # fmt: skip
    # Reject bad flags here, before a GPU starts.
    parse_args(argv)
    train_remote.remote(argv, git_commit())
