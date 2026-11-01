"""Serve one gated run with the inference API on Modal, from a frozen image.

The image build copies the run's checkpoint out of the n24-runs volume (see
n24/deploy/freeze.py), so a deployed version keeps serving the weights the
deployment gate judged even if the volume changes later. The endpoint
requires Modal proxy auth headers. bins/ci_tasks/deploy.sh sets N24_RUN and
N24_WEIGHTS_SHA256 from the results file before deploying:

    uv run modal deploy -m n24.deploy.modal_app
"""

import os
from pathlib import Path

import modal
from fastapi import FastAPI

from n24.api.app import create_app
from n24.deploy.freeze import freeze_run

RUN = os.environ["N24_RUN"]
WEIGHTS_SHA256 = os.environ["N24_WEIGHTS_SHA256"]
FROZEN = Path("/frozen")


def freeze(run: str, weights_sha256: str) -> None:
    freeze_run(Path("/news24-runs") / run, weights_sha256, FROZEN)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extra_options="--no-dev")
    # Before the freeze step: the build imports this module, which reads them.
    .env({"N24_RUN": RUN, "N24_WEIGHTS_SHA256": WEIGHTS_SHA256})
    .add_local_python_source("n24", copy=True)
    # The layer is keyed by its arguments, so a new run or new weights rebuild it.
    .run_function(
        freeze,
        volumes={"/news24-runs": modal.Volume.from_name("n24-runs")},
        kwargs={"run": RUN, "weights_sha256": WEIGHTS_SHA256},
    )
)
app = modal.App("n24-api", image=image)


@app.function()
@modal.asgi_app(requires_proxy_auth=True)
def web() -> FastAPI:
    return create_app(FROZEN / RUN)
