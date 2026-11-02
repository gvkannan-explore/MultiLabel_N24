"""Write a training run's PDF training report from its metrics.json fields.

Page 1 is an executive summary of the kept epoch, page 2 the dev F1 of each
section, and page 3 the training curves, laid out like a Weights & Biases
workspace, with a dashed line at the kept epoch. The scores are on the dev
split; test split scores come from the eval harness and the deployment gate.
"""

from datetime import timedelta
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure, SubFigure
from matplotlib.patches import FancyBboxPatch

from n24.train.metrics import TrainingMetrics

INK, MUTED, RULE, PANEL = "#1a1c1f", "#6b7280", "#e5e7eb", "#f6f7f9"
TRAIN, EVAL, BELOW = "#7a5af8", "#0ea5a4", "#f59e0b"
# Landscape US Letter, so every page prints.
PAGE = (11, 8.5)


def ema(values: list[float], weight: float = 0.6) -> list[float]:
    """Weights & Biases' default smoothing: an exponential moving average,
    debiased so the first points are not pulled toward zero."""
    smoothed, last = [], 0.0
    for i, value in enumerate(values, start=1):
        last = weight * last + (1 - weight) * value
        smoothed.append(last / (1 - weight**i))
    return smoothed


def best_eval(log_history: list[dict[str, float]]) -> dict[str, float]:
    """The dev evaluation of the epoch the trainer kept: the one with the
    highest macro F1, as metric_for_best_model picks it."""
    return max(
        (e for e in log_history if "eval_f1_macro" in e),
        key=lambda e: e["eval_f1_macro"],
    )


def tile(fig: SubFigure, x: float, width: float, label: str, value: str) -> None:
    fig.patches.append(
        FancyBboxPatch(
            (x, 0.08), width, 0.84,
            boxstyle="round,pad=0,rounding_size=0.03",
            transform=fig.transSubfigure, facecolor=PANEL, edgecolor=RULE,
        )
    )  # fmt: skip
    fig.text(x + 0.015, 0.66, label, fontsize=10, color=MUTED)
    fig.text(x + 0.015, 0.24, value, fontsize=24, fontweight="bold", color=INK)


def table(fig: SubFigure, x: float, title: str, rows: dict[str, str]) -> None:
    fig.text(x, 0.94, title, fontsize=12, fontweight="bold", color=INK)
    for i, (key, value) in enumerate(rows.items()):
        y = 0.82 - i * 0.1
        fig.text(x, y, key, fontsize=9, color=MUTED)
        fig.text(x + 0.13, y, value, fontsize=9, color=INK, family="monospace")


def summary_page(metrics: TrainingMetrics) -> Figure:
    args = metrics.args
    log_history = metrics.log_history
    per_section = metrics.dev_f1_per_section
    best = best_eval(log_history)
    # The trainer's end-of-training summary is the entry with train_runtime.
    summary = next(e for e in log_history if "train_runtime" in e)
    ranked = sorted(per_section.items(), key=lambda item: item[1], reverse=True)

    fig = Figure(figsize=PAGE, layout="constrained", facecolor="white")
    # squeeze=False so the type is always an array of subfigures.
    head, tiles, notes, details = fig.subfigures(
        4, 1, squeeze=False, height_ratios=[1.1, 1.3, 1.9, 2.6]
    )[:, 0]
    head.text(
        0.03, 0.62, Path(args["output_dir"]).name,
        fontsize=26, fontweight="bold", color=INK,
    )  # fmt: skip
    head.text(
        0.03, 0.3,
        f"News section classifier  ·  {len(per_section)} sections"
        f"  ·  fine-tuned from {args['model']}",
        fontsize=12, color=MUTED,
    )  # fmt: skip
    head.text(
        0.03, 0.05,
        "Scores are on the dev split, which chose the kept epoch. Test split"
        " scores come from the eval harness and the deployment gate.",
        fontsize=9, color=MUTED, style="italic",
    )  # fmt: skip

    for i, (label, value) in enumerate(
        [
            ("Dev macro F1", f"{metrics.dev_f1_macro:.1%}"),
            ("Dev accuracy", f"{metrics.dev_accuracy:.1%}"),
            ("Dev loss", f"{metrics.dev_loss:.3f}"),
            (f"Kept epoch (step {int(best['step'])})", f"{best['epoch']:.3g}"),
        ]
    ):
        tile(tiles, 0.03 + i * 0.24, 0.22, label, value)

    for x, title, color, picks in [
        (0.03, "Strongest sections", EVAL, ranked[:5]),
        (0.51, "Weakest sections", BELOW, ranked[::-1][:5]),
    ]:
        notes.text(x, 0.9, title, fontsize=12, fontweight="bold", color=INK)
        for i, (section, score) in enumerate(picks):
            y = 0.72 - i * 0.16
            notes.text(
                x, y, f"{score:.1%}", fontsize=11, fontweight="bold", color=color
            )
            notes.text(x + 0.07, y, section, fontsize=11, color=INK)

    table(
        details, 0.03, "Training setup",
        {
            "Base model": args["model"],
            "Epochs": args["epochs"],
            "Steps": f"{int(summary['step'])}",
            "Learning rate": args["lr"],
            "Batch size": args["batch_size"],
            "Max tokens": args["max_length"],
            "Seed": args["seed"],
            "Training time": str(timedelta(seconds=round(summary["train_runtime"]))),
        },
    )  # fmt: skip
    table(
        details, 0.38, "Reproducibility",
        {
            "Git commit": str(metrics.git_commit),
            "Dataset sha256": metrics.dataset_sha256,
            "Weights sha256": metrics.weights_sha256,
        },
    )  # fmt: skip
    return fig


def sections_page(metrics: TrainingMetrics) -> Figure:
    per_section = metrics.dev_f1_per_section
    macro = metrics.dev_f1_macro
    ranked = sorted(per_section.items(), key=lambda item: item[1])

    fig = Figure(figsize=PAGE, layout="constrained", facecolor="white")
    fig.suptitle(
        "Dev F1 by section", x=0.02, ha="left",
        fontsize=18, fontweight="bold", color=INK,
    )  # fmt: skip
    ax = fig.add_subplot()
    names = [section for section, _ in ranked]
    scores = [score for _, score in ranked]
    bars = ax.barh(names, scores, color=[EVAL if s >= macro else BELOW for s in scores])
    ax.bar_label(bars, labels=[f"{s:.1%}" for s in scores], padding=3, fontsize=8)
    ax.axvline(macro, color=INK, linestyle="--", linewidth=1)
    ax.text(
        macro, len(names) - 0.4, f" macro F1 {macro:.1%}",
        fontsize=9, color=INK, va="bottom",
    )  # fmt: skip
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="x", color=RULE)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.tick_params(axis="y", colors=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    return fig


def curves_page(metrics: TrainingMetrics) -> Figure:
    log_history = metrics.log_history
    best_step = best_eval(log_history)["step"]

    def curve(name: str) -> tuple[list[int], list[float]]:
        points = [(int(e["step"]), e[name]) for e in log_history if name in e]
        return [step for step, _ in points], [value for _, value in points]

    fig = Figure(figsize=PAGE, layout="constrained", facecolor="white")
    fig.suptitle(
        "Training curves", x=0.02, ha="left",
        fontsize=18, fontweight="bold", color=INK,
    )  # fmt: skip
    sections = [
        ("train", TRAIN, ["loss", "learning_rate", "grad_norm"]),
        ("eval", EVAL, ["eval_loss", "eval_accuracy", "eval_f1_macro"]),
    ]
    for subfig, (section, color, names) in zip(
        fig.subfigures(2, 1, squeeze=False)[:, 0], sections, strict=True
    ):
        subfig.set_facecolor("#fafafa")
        subfig.suptitle(
            f"  {section}", x=0, ha="left", fontsize=12, fontweight="bold", color=INK
        )
        for ax, name in zip(subfig.subplots(1, len(names)), names, strict=True):
            steps, values = curve(name)
            ax.set_facecolor("white")
            if not steps:
                # Smoke runs stop before the first logging step.
                ax.text(0.5, 0.5, "No data logged", ha="center", color=MUTED)
                ax.set_xticks([])
                ax.set_yticks([])
            elif name in ("loss", "grad_norm"):
                # A faint raw line behind the smoothed one.
                ax.plot(steps, values, color=color, alpha=0.25, linewidth=1)
                ax.plot(steps, ema(values), color=color, linewidth=2)
            elif section == "eval":
                ax.plot(steps, values, color=color, linewidth=2, marker="o")
                ax.axvline(best_step, color=MUTED, linestyle="--", linewidth=1)
            else:
                ax.plot(steps, values, color=color, linewidth=2)
            ax.set_title(
                name if section == "eval" else f"train/{name}",
                loc="left", fontsize=10, color=INK,
            )  # fmt: skip
            ax.set_xlabel("train/global_step", fontsize=8, color=MUTED)
            ax.grid(color=RULE, linewidth=0.8)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(RULE)
    return fig


def write_report(metrics: TrainingMetrics, path: Path) -> None:
    """Write the three report pages for a run's metrics.json fields to path."""
    with PdfPages(path) as pdf:
        for page in (summary_page, sections_page, curves_page):
            pdf.savefig(page(metrics))
