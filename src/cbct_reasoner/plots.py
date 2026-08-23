"""Diagnostic figures for every pipeline stage.

Each figure answers a question you would otherwise have to trust:

* Is the dataset shaped the way the challenge describes?
* Does the prototype bank cover the corpus, and is it one finding per statement?
* Is the encoder learning, or is every fold flat at the prior?
* Did calibration find the expected structure - strict thresholds on rare
  statements, permissive on common ones?
* Where does the final score actually come from, and where does it break down?

Figures are written as PNGs under ``artifacts/plots/`` and uploaded with the rest
of the artifacts. They contain aggregate statistics and prototype text only -
never a patient's report and never voxels.

Style follows one convention throughout: a single categorical hue order applied
in fixed slot order, thin marks, solid hairline grid, and a legend plus direct
labels whenever more than one series shares an axis.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# --- design tokens ---------------------------------------------------------
# Validated categorical order (adjacent-pair safe for lines and bars). Scatter
# and other all-pairs forms use at most the first three slots.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"
SEQUENTIAL_MID = "#86b6ef"

DPI = 150


def _pyplot():
    """Import pyplot with a headless backend.

    Modal containers and CI have no display; matplotlib usually picks Agg on its
    own, but selecting it explicitly avoids a backend error surfacing at the very
    end of a long GPU run.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _style() -> dict[str, Any]:
    return {
        "figure.facecolor": SURFACE,
        "figure.dpi": DPI,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        # Solid hairline grid; dashed grids read as thresholds that are not there.
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK_SECONDARY,
        "font.size": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    }


def _new_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path)
    _pyplot().close(fig)
    return path


def _bar(ax, labels: Sequence[str], values: Sequence[float], *, color: str = SERIES[0]) -> None:
    """Nominal categories get one hue. A value-ramp here would double-encode length."""
    positions = np.arange(len(labels))
    ax.bar(positions, values, width=0.62, color=color)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Stage figures
# ---------------------------------------------------------------------------


def plot_dataset(entries: Sequence[Any], directory: Path) -> list[Path]:
    """Case counts, report multiplicity, and the length distributions we must match."""
    plt = _pyplot()

    from cbct_reasoner.metrics.official import tokenize
    from cbct_reasoner.text import split_phrases

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))

        centers = Counter(entry.center for entry in entries)
        names = sorted(centers)
        _bar(axes[0, 0], names, [centers[name] for name in names])
        axes[0, 0].set_title(f"Cases per centre  (n={len(entries)})")
        axes[0, 0].set_ylabel("cases")
        for index, name in enumerate(names):
            axes[0, 0].text(
                index,
                centers[name],
                f" {centers[name]}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_SECONDARY,
            )

        counts = Counter(len(entry.reports) for entry in entries)
        keys = sorted(counts)
        _bar(axes[0, 1], [str(key) for key in keys], [counts[key] for key in keys])
        axes[0, 1].set_title("Reports per case")
        axes[0, 1].set_xlabel("English reports")
        axes[0, 1].set_ylabel("cases")

        lengths = [len(tokenize(entry.reference)) for entry in entries]
        axes[1, 0].hist(lengths, bins=40, color=SERIES[0])
        axes[1, 0].set_title(f"Reference length  (median {int(np.median(lengths))} tokens)")
        axes[1, 0].set_xlabel("tokens")
        axes[1, 0].set_ylabel("cases")

        phrases = [len(split_phrases(entry.reference)) for entry in entries]
        axes[1, 1].hist(phrases, bins=range(0, max(phrases) + 2), color=SERIES[0])
        axes[1, 1].set_title(f"Phrases per reference  (median {int(np.median(phrases))})")
        axes[1, 1].set_xlabel("verifiable phrases")
        axes[1, 1].set_ylabel("cases")

        for ax in axes.flat:
            _new_axes(ax)
        fig.suptitle("ToothFairy4 dataset", fontsize=13, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        return [_save(fig, directory, "01_dataset.png")]


def plot_prototypes(bank: Any, labels: np.ndarray, directory: Path) -> list[Path]:
    """Prevalence spread and section composition of the learned label space."""
    plt = _pyplot()

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

        prevalence = np.sort(bank.prevalence)[::-1]
        axes[0].plot(np.arange(len(prevalence)), prevalence, color=SERIES[0])
        axes[0].fill_between(np.arange(len(prevalence)), prevalence, color=SERIES[0], alpha=0.12)
        axes[0].set_title(f"Statement prevalence  ({len(bank)} prototypes)")
        axes[0].set_xlabel("prototype, most common first")
        axes[0].set_ylabel("share of cases")
        axes[0].set_ylim(0, 1)

        sections = Counter(prototype.section for prototype in bank)
        names = [name for name, _ in sections.most_common()]
        axes[1].barh(
            np.arange(len(names))[::-1],
            [sections[name] for name in names],
            height=0.62,
            color=SERIES[0],
        )
        axes[1].set_yticks(np.arange(len(names))[::-1])
        axes[1].set_yticklabels(names)
        axes[1].grid(axis="y", visible=False)
        axes[1].set_title("Prototypes per report section")
        axes[1].set_xlabel("prototypes")

        per_case = labels.sum(axis=1)
        axes[2].hist(per_case, bins=30, color=SERIES[0])
        axes[2].set_title(f"Statements per case  (median {int(np.median(per_case))})")
        axes[2].set_xlabel("positive labels")
        axes[2].set_ylabel("cases")

        for ax in axes:
            _new_axes(ax)
        fig.tight_layout()
        return [_save(fig, directory, "02_prototypes.png")]


def plot_training(
    history: Sequence[dict[str, Any]], prior_map: float | None, directory: Path
) -> list[Path]:
    """Per-fold learning curves. Flat lines at the prior mean the encoder is not learning."""
    plt = _pyplot()

    if not history:
        return []

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

        for position, fold in enumerate(history):
            color = SERIES[position % len(SERIES)]
            rows = fold.get("history", [])
            if not rows:
                continue
            epochs = [row["epoch"] + 1 for row in rows]
            loss = [row["train_loss"] for row in rows]
            score = [row["val_map"] for row in rows]
            label = f"fold {fold.get('fold', position)}"
            axes[0].plot(epochs, loss, color=color, label=label)
            axes[1].plot(epochs, score, color=color, label=label)
            # Direct end labels: three of the categorical slots sit below 3:1 on a
            # light surface, so identity must not rest on hue alone.
            axes[1].annotate(
                label,
                (epochs[-1], score[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=7,
                color=INK_SECONDARY,
                va="center",
            )

        if prior_map is not None:
            axes[1].axhline(prior_map, color=INK_MUTED, linewidth=1.0)
            axes[1].annotate(
                "prior",
                (0.01, prior_map),
                xycoords=("axes fraction", "data"),
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=7,
                color=INK_MUTED,
            )

        axes[0].set_title("Training loss")
        axes[0].set_xlabel("epoch")
        axes[1].set_title("Validation mean average precision")
        axes[1].set_xlabel("epoch")
        axes[1].legend(loc="lower right", ncols=2)

        for ax in axes:
            _new_axes(ax)
        fig.tight_layout()
        return [_save(fig, directory, "03_training.png")]


def plot_calibration(
    calibration: dict[str, Any], thresholds: np.ndarray, bank: Any, directory: Path
) -> list[Path]:
    """Ascent trace and the threshold structure calibration is supposed to discover."""
    plt = _pyplot()

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

        trace = calibration.get("trace") or []
        if trace:
            axes[0].plot(np.arange(len(trace)), trace, color=SERIES[0], marker="o")
            baseline = calibration.get("baseline", {}).get("final")
            if baseline is not None:
                axes[0].axhline(baseline, color=INK_MUTED, linewidth=1.0)
                axes[0].annotate(
                    "fixed 0.5 threshold",
                    (0.02, baseline),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=INK_MUTED,
                )
        axes[0].set_title("Calibration objective")
        axes[0].set_xlabel("coordinate-ascent round")
        axes[0].set_ylabel("0.8·clinical + 0.2·captioning")

        # Single series, so all-pairs colour rules do not bind.
        axes[1].scatter(
            bank.prevalence,
            thresholds,
            s=26,
            color=SERIES[0],
            alpha=0.75,
            edgecolors=SURFACE,
            linewidths=1.0,
        )
        axes[1].set_title("Fitted threshold vs prevalence")
        axes[1].set_xlabel("prevalence in the training corpus")
        axes[1].set_ylabel("emission threshold")
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)

        objective = calibration.get("objective", {})
        keys = ("logical_precision", "logical_recall", "clinical", "bleu_4", "meteor", "final")
        present = [key for key in keys if key in objective]
        _bar(
            axes[2],
            [key.replace("logical_", "").replace("_", " ") for key in present],
            [objective[key] for key in present],
        )
        axes[2].set_title("Out-of-fold components")
        axes[2].set_ylim(0, 1)
        for index, key in enumerate(present):
            axes[2].text(
                index,
                objective[key],
                f"{objective[key]:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK_SECONDARY,
            )
        axes[2].tick_params(axis="x", labelrotation=30)

        for ax in axes:
            _new_axes(ax)
        fig.tight_layout()
        return [_save(fig, directory, "04_calibration.png")]


def plot_evaluation(
    predictions: dict[str, str],
    references: dict[str, str],
    centers: dict[str, str],
    directory: Path,
) -> list[Path]:
    """Per-case score distributions, per-centre breakdown, and length agreement."""
    plt = _pyplot()

    from cbct_reasoner.metrics.official import meteor_score, tokenize
    from cbct_reasoner.metrics.radfact import LexicalRadFact

    case_ids = sorted(set(predictions) & set(references))
    engine = LexicalRadFact()
    results = [
        engine.score_case(case_id, predictions[case_id], references[case_id])
        for case_id in case_ids
    ]
    f1 = np.asarray([item.logical_f1 for item in results])
    precision = np.asarray([item.logical_precision for item in results])
    recall = np.asarray([item.logical_recall for item in results])
    meteor = np.asarray(
        [meteor_score(predictions[case_id], references[case_id]) for case_id in case_ids]
    )
    predicted_length = np.asarray([len(tokenize(predictions[case_id])) for case_id in case_ids])
    reference_length = np.asarray([len(tokenize(references[case_id])) for case_id in case_ids])

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

        axes[0, 0].hist(f1, bins=30, color=SERIES[0])
        axes[0, 0].axvline(float(f1.mean()), color=INK_MUTED, linewidth=1.0)
        axes[0, 0].set_title(f"Per-case clinical F1  (mean {f1.mean():.3f}, surrogate)")
        axes[0, 0].set_xlabel("RadFact-surrogate F1")
        axes[0, 0].set_ylabel("cases")

        axes[0, 1].hist(meteor, bins=30, color=SERIES[0])
        axes[0, 1].axvline(float(meteor.mean()), color=INK_MUTED, linewidth=1.0)
        axes[0, 1].set_title(f"Per-case METEOR  (mean {meteor.mean():.4f})")
        axes[0, 1].set_xlabel("METEOR")
        axes[0, 1].set_ylabel("cases")

        # Two measures on one axis, same 0-1 scale - never a second y-axis.
        groups = sorted({centers.get(case_id, "?") for case_id in case_ids})
        width = 0.36
        positions = np.arange(len(groups))
        mean_precision = [
            float(np.mean([precision[i] for i, c in enumerate(case_ids) if centers.get(c) == g]))
            for g in groups
        ]
        mean_recall = [
            float(np.mean([recall[i] for i, c in enumerate(case_ids) if centers.get(c) == g]))
            for g in groups
        ]
        axes[1, 0].bar(
            positions - width / 2, mean_precision, width=width, color=SERIES[0], label="precision"
        )
        axes[1, 0].bar(
            positions + width / 2, mean_recall, width=width, color=SERIES[1], label="recall"
        )
        axes[1, 0].set_xticks(positions)
        axes[1, 0].set_xticklabels(groups)
        axes[1, 0].grid(axis="x", visible=False)
        axes[1, 0].set_title("Clinical precision and recall by centre")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend(loc="upper right", ncols=2)

        limit = max(reference_length.max(), predicted_length.max()) * 1.05
        axes[1, 1].plot([0, limit], [0, limit], color=GRID, linewidth=1.0)
        axes[1, 1].scatter(
            reference_length,
            predicted_length,
            s=18,
            color=SERIES[0],
            alpha=0.6,
            edgecolors=SURFACE,
            linewidths=0.8,
        )
        axes[1, 1].set_title("Report length agreement")
        axes[1, 1].set_xlabel("reference tokens")
        axes[1, 1].set_ylabel("generated tokens")

        for ax in axes.flat:
            _new_axes(ax)
        fig.tight_layout()
        return [_save(fig, directory, "05_evaluation.png")]


def plot_ablation(rows: Sequence[tuple[str, dict[str, float]]], directory: Path) -> list[Path]:
    """Compare decoder variants on the components that make up the ranking score."""
    plt = _pyplot()

    if not rows:
        return []

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        metrics = ("clinical", "captioning", "final")
        positions = np.arange(len(metrics))
        width = 0.8 / max(1, len(rows))

        for index, (name, values) in enumerate(rows):
            offset = (index - (len(rows) - 1) / 2) * width
            heights = [values.get(metric, 0.0) for metric in metrics]
            ax.bar(
                positions + offset,
                heights,
                width=width * 0.9,
                color=SERIES[index % len(SERIES)],
                label=name,
            )
            for position, height in zip(positions + offset, heights, strict=True):
                ax.text(
                    position,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=INK_SECONDARY,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels(["clinical (×0.8)", "captioning (×0.2)", "final score"])
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1)
        ax.set_title("Decoder variants, out-of-fold")
        ax.legend(loc="upper right", ncols=len(rows))
        _new_axes(ax)
        fig.tight_layout()
        return [_save(fig, directory, "06_ablation.png")]


def render_all(paths: Any) -> list[Path]:
    """Regenerate every figure the current artifacts support."""
    import json

    from cbct_reasoner.data.corpus import load_corpus
    from cbct_reasoner.decode.decoder import ReportDecoder
    from cbct_reasoner.prototypes import PrototypeBank, load_labels

    directory = Path(paths.artifacts) / "plots"
    written: list[Path] = []

    entries = load_corpus(paths.corpus)
    written += plot_dataset(entries, directory)

    if paths.prototypes.is_file() and paths.labels.is_file():
        bank = PrototypeBank.load(paths.prototypes)
        _, labels = load_labels(paths.labels)
        written += plot_prototypes(bank, labels, directory)
    else:
        bank = None

    history_path = paths.artifacts / "train_history.json"
    if history_path.is_file():
        written += plot_training(
            json.loads(history_path.read_text(encoding="utf-8")), None, directory
        )

    calibration_path = paths.artifacts / "calibration.json"
    if calibration_path.is_file() and bank is not None and paths.decoder.is_file():
        decoder = ReportDecoder.load(paths.decoder, bank)
        written += plot_calibration(
            json.loads(calibration_path.read_text(encoding="utf-8")),
            decoder.thresholds,
            bank,
            directory,
        )

    predictions_path = paths.artifacts / "oof_reports.json"
    if predictions_path.is_file():
        payload = json.loads(predictions_path.read_text(encoding="utf-8"))
        index = {entry.case_id: entry for entry in entries}
        written += plot_evaluation(
            payload["predictions"],
            {
                case_id: index[case_id].reference
                for case_id in payload["predictions"]
                if case_id in index
            },
            {
                case_id: index[case_id].center
                for case_id in payload["predictions"]
                if case_id in index
            },
            directory,
        )

    ablation_path = paths.artifacts / "ablation.json"
    if ablation_path.is_file():
        payload = json.loads(ablation_path.read_text(encoding="utf-8"))
        written += plot_ablation(list(payload.items()), directory)

    return written
