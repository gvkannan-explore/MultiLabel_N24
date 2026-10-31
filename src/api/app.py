import argparse
import threading
from pathlib import Path
from typing import Annotated

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, StringConstraints, TypeAdapter

from src.eval.run import RunMetrics, compute_logits, load_model


class PredictRequest(BaseModel):
    abstracts: list[Annotated[str, StringConstraints(min_length=1)]] = Field(
        min_length=1
    )


class Prediction(BaseModel):
    section: str
    logits: list[float]


class PredictResponse(BaseModel):
    run: str  # run directory name
    sections: list[str]  # Model config -> id2label
    predictions: list[Prediction]


def create_app(run_dir: Path) -> FastAPI:

    app = FastAPI()

    # Loading the heavy initializations before the predict/post call.
    torch.use_deterministic_algorithms(True)
    metrics = RunMetrics.model_validate_json((run_dir / "metrics.json").read_text())
    max_len = int(metrics.args["max_length"])

    # Give the model-directory, not the safe-tensors:
    tokenizer, model = load_model(model_dir=run_dir / "best_model")
    id2label = TypeAdapter(dict[int, str]).validate_python(model.config.id2label)
    sections: list[str] = [id2label[i] for i in range(len(id2label))]
    lock = threading.Lock()  # One model and tokenizer, shared by all the requests.

    @app.post("/predict")
    def predict_logits(request: PredictRequest) -> PredictResponse:
        with lock:
            logits = compute_logits(
                tokenizer=tokenizer,
                model=model,
                abstracts=request.abstracts,
                max_length=max_len,
            )
            section_ids = logits.argmax(axis=-1)  # shape (n,), one id per abstract

        return PredictResponse(
            run=run_dir.name,
            sections=sections,
            predictions=[
                Prediction(section=sections[idx], logits=row)
                for idx, row in zip(section_ids, logits)
            ],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(create_app(args.run_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
