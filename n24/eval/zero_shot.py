"""Score an instruction-tuned LLM zero-shot on the test split with vLLM.

The prompt lists the sections and gives the abstract (the model input, ADR
0004). vLLM decodes greedily and may answer only with a section name. The
test split is predicted twice on a Modal GPU, each pass in a fresh container,
and the number of sections that differ between the passes is recorded. Scores
use the same eval version and test split hash as the eval harness in
n24/eval/run.py, so they compare directly with trained runs.

The Modal image installs zero-shot/uv.lock, because vLLM pins torch and
transformers below the versions in the root uv.lock.

    uv run modal run -m n24.eval.zero_shot --model Qwen/Qwen3-0.6B
    uv run modal run -m n24.eval.zero_shot --model HuggingFaceTB/SmolLM2-360M-Instruct
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

import modal
import numpy as np
from huggingface_hub import HfApi

from n24.eval.run import TestScores, load_test_split, score_test_split, write_results
from n24.train.train_baseline import git_commit

SYSTEM = (
    "You classify news articles into exactly one section. "
    "Answer with the section name only."
)
USER = "Sections:\n{sections}\n\nAbstract:\n{abstract}"

Chat = list[dict[str, str]]
Predict = Callable[[list[Chat], list[str]], list[int]]

app = modal.App("n24-zero-shot")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir="zero-shot")
    .add_local_python_source("n24")
)


class ZeroShotResults(TestScores):
    model: str
    model_revision: str
    prompt_sha256: str
    git_commit: str | None
    label_mismatches: int


def build_messages(abstract: str, sections: list[str]) -> Chat:
    listing = "".join(f"- {section}\n" for section in sections)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER.format(sections=listing, abstract=abstract)},
    ]


def prompt_sha256() -> str:
    return hashlib.sha256(f"{SYSTEM}\n{USER}".encode()).hexdigest()


# One input per container, so each prediction pass loads the model fresh.
@app.function(image=image, gpu="L4", timeout=60 * 60, single_use_containers=True)
def predict_remote(
    model: str, revision: str, chats: list[Chat], sections: list[str]
) -> list[int]:
    """Section ids for each chat. The output is constrained to the section
    names, so every answer maps to a section."""
    from vllm import LLM, SamplingParams  # pyright: ignore[reportMissingImports]
    from vllm.sampling_params import (  # pyright: ignore[reportMissingImports]
        StructuredOutputsParams,
    )

    llm = LLM(model=model, revision=revision, seed=0, max_model_len=2048)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
        structured_outputs=StructuredOutputsParams(choice=sections),
    )
    # Qwen3 thinks before it answers unless this is off; other templates ignore it.
    outputs = llm.chat(chats, params, chat_template_kwargs={"enable_thinking": False})
    index = {section: i for i, section in enumerate(sections)}
    return [index[output.outputs[0].text] for output in outputs]


def evaluate(
    model: str,
    revision: str,
    data_dir: Path,
    results_dir: Path,
    predict: Predict,
    commit: str | None,
) -> ZeroShotResults:
    """Predict the test split twice and write results_dir/zs-<model name>.json.
    Refuses to overwrite an existing results file."""
    test = load_test_split(data_dir)
    sections = test.features["label"].names
    chats = [build_messages(abstract, sections) for abstract in test["abstract"]]
    first = np.array(predict(chats, sections))
    second = np.array(predict(chats, sections))
    run = f"zs-{model.split('/')[-1]}"
    results = ZeroShotResults(
        **score_test_split(run, data_dir, test, first).model_dump(),
        model=model,
        model_revision=revision,
        prompt_sha256=prompt_sha256(),
        git_commit=commit,
        label_mismatches=int(np.sum(first != second)),
    )
    write_results(results, results_dir)
    return results


@app.local_entrypoint()
def main(
    model: str, data_dir: str = "datasets/prepared", results_dir: str = "results"
) -> None:
    """Score one Hugging Face model, pinned to its current commit."""
    revision = HfApi().model_info(model).sha
    if revision is None:
        raise ValueError(f"Hugging Face returned no commit for {model}")
    evaluate(
        model=model,
        revision=revision,
        data_dir=Path(data_dir),
        results_dir=Path(results_dir),
        predict=lambda chats, sections: predict_remote.remote(
            model, revision, chats, sections
        ),
        # The commit also pins the vLLM version, through zero-shot/uv.lock.
        commit=git_commit(),
    )
