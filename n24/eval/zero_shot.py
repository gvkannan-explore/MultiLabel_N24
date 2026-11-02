"""Score an instruction-tuned LLM zero-shot on the test split with vLLM.

The prompt lists the sections and gives the abstract (the model input, ADR
0004). vLLM decodes greedily and may answer only with a section name. The
test split is predicted twice on a Modal GPU, each pass in a fresh container,
and the number of sections that differ between the passes is recorded. Scores
use the same eval version and test split hash as the eval harness in
n24/eval/run.py, so they compare directly with trained runs.

The Modal image installs zero-shot/uv.lock, because vLLM pins torch and
transformers below the versions in the root uv.lock.

    uv run modal run -m n24.eval.zero_shot::main --model Qwen/Qwen3-0.6B
    uv run modal run -m n24.eval.zero_shot::main --model HuggingFaceTB/SmolLM2-360M-Instruct
    uv run modal run -m n24.eval.zero_shot::main --model Qwen/Qwen3.5-9B --gpu L40S
    uv run modal run -m n24.eval.zero_shot::main --model Qwen/Qwen3.5-27B-FP8 --gpu L40S

Weights are cached in the n24-hf-cache volume, so only the first container
for a model downloads them.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import modal
import numpy as np
from huggingface_hub import HfApi
from rich import print
from rich.table import Table

from n24.eval.probes import (
    PROBES_PATH,
    RESULTS_DIR,
    load_probes,
    score_probes,
    write_probe_results,
)
from n24.eval.run import TestScores, load_test_split, score_test_split, write_results
from n24.train.train_baseline import git_commit

SYSTEM = (
    "You classify news articles into exactly one section. "
    "Answer with the section name only."
)
USER = "Sections:\n{sections}\n\nAbstract:\n{abstract}"

Chat = list[dict[str, str]]
Predict = Callable[[list[Chat], list[str]], list[int]]

# The first free GPU in the list. It fits models up to about 4B parameters in
# BF16; pass --gpu for larger ones.
DEFAULT_GPUS = ["L4", "A10G"]
HF_CACHE_DIR = "/hf-cache"

app = modal.App("n24-zero-shot")
hf_cache = modal.Volume.from_name("n24-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir="zero-shot")
    # FlashInfer's top-k/top-p sampler compiles with nvcc at startup, which
    # debian_slim lacks. Greedy decoding never uses it, so turn it off.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HOME": HF_CACHE_DIR})
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


def zero_shot_run_name(model: str) -> str:
    """The results file's run name: zs- and the model name without its org."""
    return f"zs-{model.split('/')[-1]}"


def prompt_sha256() -> str:
    return hashlib.sha256(f"{SYSTEM}\n{USER}".encode()).hexdigest()


def on_gpu[**P, R, O](
    function: modal.Function[P, R, O], gpu: str | None
) -> modal.Function[P, R, O]:
    """function on the given Modal GPU type, or on DEFAULT_GPUS if gpu is None."""
    return function if gpu is None else function.with_options(gpu=gpu)


# Any: vLLM installs only in the Modal image, so its types are unknown here.
def load_llm(model: str, revision: str) -> Any:
    """A vLLM engine for text-only, greedy, repeatable predictions."""
    from vllm import LLM  # pyright: ignore[reportMissingImports]

    # language_model_only skips the vision encoder of models such as Qwen3.5;
    # text-only models ignore it.
    return LLM(
        model=model,
        revision=revision,
        seed=0,
        max_model_len=2048,
        language_model_only=True,
    )


# One input per container, so each prediction pass loads the model fresh.
@app.function(
    image=image,
    gpu=DEFAULT_GPUS,
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=60 * 60,
    single_use_containers=True,
)
def predict_remote(
    model: str, revision: str, chats: list[Chat], sections: list[str]
) -> list[int]:
    """Section ids for each chat. The output is constrained to the section
    names, so every answer maps to a section."""
    from vllm import SamplingParams  # pyright: ignore[reportMissingImports]
    from vllm.sampling_params import (  # pyright: ignore[reportMissingImports]
        StructuredOutputsParams,
    )

    llm = load_llm(model, revision)
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
    commit: str,
) -> ZeroShotResults:
    """Predict the test split twice and write results_dir/zs-<model name>.json.
    Refuses to overwrite an existing results file."""
    test = load_test_split(data_dir)
    sections = test.features["label"].names
    chats = [build_messages(abstract, sections) for abstract in test["abstract"]]
    first_pass_section_ids = np.array(predict(chats, sections))
    second_pass_section_ids = np.array(predict(chats, sections))
    run = zero_shot_run_name(model)
    results = ZeroShotResults(
        **score_test_split(run, data_dir, test, first_pass_section_ids).model_dump(),
        model=model,
        model_revision=revision,
        prompt_sha256=prompt_sha256(),
        git_commit=commit,
        label_mismatches=int(np.sum(first_pass_section_ids != second_pass_section_ids)),
    )
    write_results(results, results_dir)

    table = Table(title=f"{run}: test split")
    table.add_column("Section")
    table.add_column("F1", justify="right")
    for section, score in results.test_f1_per_section.items():
        table.add_row(section, f"{score:.4f}")
    table.add_section()
    table.add_row("Macro F1", f"{results.test_f1_macro:.4f}")
    table.add_row("Accuracy", f"{results.test_accuracy:.4f}")
    table.add_row("Sections differing between passes", str(results.label_mismatches))
    print(table)
    return results


def pinned_revision(model: str) -> str:
    revision = HfApi().model_info(model).sha
    if revision is None:
        raise ValueError(f"Hugging Face returned no commit for {model}")
    return revision


@app.local_entrypoint()
def probes(
    model: str,
    probes_path: str = str(PROBES_PATH),
    data_dir: str = "datasets/prepared",
    results_dir: str = str(RESULTS_DIR),
    gpu: str | None = None,
) -> None:
    """Score the model on the external probes, with the test-split prompt.

    uv run modal run -m n24.eval.zero_shot::probes --model Qwen/Qwen3-0.6B
    """
    sections = load_test_split(Path(data_dir)).features["label"].names
    external = load_probes(Path(probes_path), sections)
    chats = [build_messages(probe.abstract, sections) for probe in external]
    ids = on_gpu(predict_remote, gpu).remote(
        model, pinned_revision(model), chats, sections
    )
    results = score_probes(
        zero_shot_run_name(model),
        external,
        [sections[i] for i in ids],
        Path(probes_path),
    )
    write_probe_results(results, Path(results_dir))


@app.local_entrypoint()
def main(
    model: str,
    data_dir: str = "datasets/prepared",
    results_dir: str = "results/llm",
    gpu: str | None = None,
) -> None:
    """Score one Hugging Face model, pinned to its current commit."""
    revision = pinned_revision(model)
    evaluate(
        model=model,
        revision=revision,
        data_dir=Path(data_dir),
        results_dir=Path(results_dir),
        predict=lambda chats, sections: on_gpu(predict_remote, gpu).remote(
            model, revision, chats, sections
        ),
        # The commit also pins the vLLM version, through zero-shot/uv.lock.
        commit=git_commit(),
    )
