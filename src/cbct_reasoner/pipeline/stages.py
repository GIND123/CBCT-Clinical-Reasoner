"""End-to-end pipeline stages.

Each stage reads and writes files under ``Paths`` so any stage can be re-run in
isolation, resumed after a crash, or executed on a different machine (which is
how the Modal entrypoints reuse them). Every stage returns a JSON-serializable
summary that the CLI prints and the Hub uploader attaches to the artifact.

Order: ``prepare`` -> ``prototypes`` -> ``splits`` -> ``train`` -> ``calibrate``
-> ``evaluate`` -> ``package``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cbct_reasoner.config import ExperimentConfig, Paths
from cbct_reasoner.data.corpus import build_corpus, load_corpus, save_corpus
from cbct_reasoner.data.discovery import discover_cases, write_manifest
from cbct_reasoner.data.preprocess import is_cached, preprocess_volume, write_cache
from cbct_reasoner.data.splits import SplitPlan, build_splits
from cbct_reasoner.decode.calibrate import CalibrationScorer, calibrate, save_calibration
from cbct_reasoner.decode.decoder import DecoderSettings, ReportDecoder
from cbct_reasoner.decode.prior import prior_probabilities
from cbct_reasoner.metrics.score import score_reports
from cbct_reasoner.pipeline.bundle import InferenceBundle
from cbct_reasoner.prototypes import (
    PrototypeBank,
    build_bank,
    build_labels,
    load_labels,
    save_labels,
)


def _log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Stage 1: prepare
# ---------------------------------------------------------------------------


def prepare(
    paths: Paths,
    config: ExperimentConfig,
    *,
    force: bool = False,
    skip_errors: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Discover cases, build the text corpus, and cache normalized volumes."""
    paths.ensure()
    records = discover_cases(paths.data)
    if limit is not None:
        records = records[:limit]
    _log(f"[prepare] discovered {len(records)} cases in {paths.data}")

    write_manifest(records, paths.manifest)
    entries = build_corpus(records)
    save_corpus(entries, paths.corpus)
    _log(f"[prepare] wrote corpus with {sum(len(e.reports) for e in entries)} reports")

    failures: list[str] = []
    cached = 0
    started = time.time()
    for position, record in enumerate(records, start=1):
        if not force and is_cached(paths.cache, record.case_id, config.preprocess):
            cached += 1
            continue
        try:
            array, meta = preprocess_volume(
                record.volume_path, config.preprocess, case_id=record.case_id
            )
            write_cache(paths.cache, record.case_id, array, meta)
            cached += 1
        except Exception as error:
            failures.append(f"{record.case_id}: {type(error).__name__}: {error}")
            if not skip_errors:
                raise RuntimeError(
                    f"Preprocessing failed for {record.case_id}. "
                    f"Re-run with --skip-errors to continue past bad volumes.\n{error}"
                ) from error
        if position % 25 == 0 or position == len(records):
            elapsed = time.time() - started
            _log(f"[prepare] {position}/{len(records)} volumes ({elapsed:.0f}s)")

    summary = {
        "stage": "prepare",
        "cases": len(records),
        "cached": cached,
        "failures": failures,
        "centers": _counts(record.center for record in records),
        "cache_dir": str(paths.cache),
    }
    if failures:
        _log(f"[prepare] WARNING {len(failures)} volumes failed: {failures[:3]}")
    return summary


# ---------------------------------------------------------------------------
# Stage 2: prototypes
# ---------------------------------------------------------------------------


def prototypes(paths: Paths, config: ExperimentConfig) -> dict[str, Any]:
    """Cluster training phrases into the label space and build the target matrix."""
    entries = load_corpus(paths.corpus)
    bank = build_bank(entries, config.prototypes)
    bank.save(paths.prototypes)

    labels = build_labels(entries, bank)
    save_labels(paths.labels, [entry.case_id for entry in entries], labels)

    coverage = float(np.mean([_coverage(entry.all_phrases, bank) for entry in entries]))
    _log(
        f"[prototypes] {len(bank)} prototypes, mean labels/case={labels.sum(axis=1).mean():.1f}, "
        f"phrase coverage={coverage:.1%}"
    )
    return {
        "stage": "prototypes",
        "num_prototypes": len(bank),
        "mean_labels_per_case": float(labels.sum(axis=1).mean()),
        "phrase_coverage": coverage,
        "top_prototypes": [
            {"text": p.text, "prevalence": round(p.prevalence, 4), "section": p.section}
            for p in sorted(bank, key=lambda item: -item.prevalence)[:10]
        ],
    }


def _coverage(phrases: Sequence[str], bank: PrototypeBank) -> float:
    if not phrases:
        return 1.0
    return sum(bank.assign(phrase) is not None for phrase in phrases) / len(phrases)


# ---------------------------------------------------------------------------
# Stage 3: splits
# ---------------------------------------------------------------------------


def splits(
    paths: Paths, config: ExperimentConfig, *, strategy: str = "stratified"
) -> dict[str, Any]:
    entries = load_corpus(paths.corpus)
    plan = build_splits(
        [entry.case_id for entry in entries],
        [entry.center for entry in entries],
        strategy=strategy,
        n_folds=config.folds,
        seed=config.seed,
    )
    plan.save(paths.folds)
    _log(f"[splits] {plan.strategy}: {[len(fold.validation) for fold in plan]}")
    return {
        "stage": "splits",
        "strategy": plan.strategy,
        "folds": [
            {"name": fold.name, "train": len(fold.train), "validation": len(fold.validation)}
            for fold in plan
        ],
    }


# ---------------------------------------------------------------------------
# Stage 4: train
# ---------------------------------------------------------------------------


def train(
    paths: Paths,
    config: ExperimentConfig,
    *,
    only_folds: Sequence[int] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Train every fold and write the out-of-fold probability matrix."""
    from cbct_reasoner.models.trainer import resolve_device, save_history, train_fold

    entries = load_corpus(paths.corpus)
    bank = PrototypeBank.load(paths.prototypes)
    case_ids, labels = load_labels(paths.labels)
    labels_by_case = dict(zip(case_ids, labels, strict=True))
    plan = SplitPlan.load(paths.folds)

    order = [entry.case_id for entry in entries]
    position = {case_id: index for index, case_id in enumerate(order)}
    oof = np.zeros((len(order), len(bank)), dtype=np.float32)
    filled = np.zeros(len(order), dtype=bool)

    torch_device = resolve_device(device)
    _log(f"[train] device={torch_device} folds={len(plan)} prototypes={len(bank)}")

    results = []
    for fold in plan:
        if only_folds is not None and fold.index not in only_folds:
            continue
        result, probabilities = train_fold(
            config=config.encoder,
            fold_index=fold.index,
            train_ids=fold.train,
            validation_ids=fold.validation,
            labels_by_case=labels_by_case,
            cache_dir=paths.cache,
            prior=bank.prevalence,
            checkpoint_dir=paths.checkpoints,
            device=torch_device,
            expected_shape=tuple(config.preprocess.shape_zyx),
        )
        results.append(result)
        for row, case_id in enumerate(fold.validation):
            oof[position[case_id]] = probabilities[row]
            filled[position[case_id]] = True
        _log(
            f"[train] fold {fold.index} best mAP={result.best_score:.4f} "
            f"@ epoch {result.best_epoch + 1}"
        )

    if not filled.all():
        # Partial training is legitimate (resuming, or a single-fold debug run);
        # unfilled rows keep the prior so calibration still has a complete matrix.
        missing = int((~filled).sum())
        _log(f"[train] {missing} cases have no out-of-fold prediction; filling with the prior")
        oof[~filled] = bank.prevalence

    np.savez_compressed(
        paths.oof,
        case_ids=np.asarray(order, dtype=np.str_),
        probabilities=oof,
        covered=filled,
    )
    save_history(paths.artifacts / "train_history.json", results)
    return {
        "stage": "train",
        "device": str(torch_device),
        "folds": [result.to_dict() for result in results],
        "mean_val_map": float(np.mean([result.best_score for result in results]))
        if results
        else 0.0,
        "oof": str(paths.oof),
        "coverage": float(filled.mean()),
    }


def load_oof(paths: Paths) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not paths.oof.is_file():
        raise FileNotFoundError(
            f"No out-of-fold predictions at {paths.oof}. Run `cbct-reasoner train` first, "
            "or `cbct-reasoner calibrate --prior-only` to calibrate the image-free decoder."
        )
    with np.load(paths.oof, allow_pickle=False) as archive:
        return (
            [str(value) for value in archive["case_ids"].tolist()],
            archive["probabilities"].astype(np.float32),
            archive["covered"].astype(bool),
        )


# ---------------------------------------------------------------------------
# Stage 5: calibrate
# ---------------------------------------------------------------------------


def calibrate_decoder(
    paths: Paths,
    config: ExperimentConfig,
    *,
    prior_only: bool = False,
    rounds: int | None = None,
    refine_top: int | None = None,
) -> dict[str, Any]:
    """Fit per-prototype thresholds on out-of-fold probabilities."""
    entries = load_corpus(paths.corpus)
    bank = PrototypeBank.load(paths.prototypes)

    if prior_only:
        case_ids = [entry.case_id for entry in entries]
        probabilities = prior_probabilities(bank, len(entries))
    else:
        case_ids, probabilities, _ = load_oof(paths)

    index = {entry.case_id: entry for entry in entries}
    ordered = [index[case_id] for case_id in case_ids]
    scorer = CalibrationScorer(
        bank,
        [entry.reference for entry in ordered],
        reference_phrases=[entry.phrases for entry in ordered],
        clinical_weight=config.decode.clinical_weight,
        captioning_weight=config.decode.captioning_weight,
    )
    settings = DecoderSettings(
        min_sentences=config.decode.min_sentences, max_sentences=config.decode.max_sentences
    )
    result = calibrate(
        probabilities,
        scorer,
        bank,
        settings=settings,
        rounds=rounds if rounds is not None else config.decode.calibration_rounds,
        refine_top=refine_top if refine_top is not None else config.decode.refine_top,
    )

    decoder = ReportDecoder(bank, result.thresholds, settings=settings)
    decoder.save(paths.decoder)
    save_calibration(paths.artifacts / "calibration.json", result)
    _log(
        f"[calibrate] final={result.objective.final:.4f} "
        f"(baseline@0.5={result.baseline.final:.4f}) "
        f"clinical={result.objective.clinical:.4f} captioning={result.objective.captioning:.4f}"
    )
    return {
        "stage": "calibrate",
        "prior_only": prior_only,
        "objective": result.objective.to_dict(),
        "baseline": result.baseline.to_dict(),
        "decoder": str(paths.decoder),
    }


# ---------------------------------------------------------------------------
# Stage 6: evaluate
# ---------------------------------------------------------------------------


def evaluate(
    paths: Paths,
    config: ExperimentConfig,
    *,
    prior_only: bool = False,
    use_radfact_lite: bool = False,
    radfact_options: dict[str, Any] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Score out-of-fold reports, optionally with the organizer's real RadFact."""
    entries = load_corpus(paths.corpus)
    bank = PrototypeBank.load(paths.prototypes)
    decoder = ReportDecoder.load(paths.decoder, bank)

    if prior_only:
        case_ids = [entry.case_id for entry in entries]
        probabilities = prior_probabilities(bank, len(entries))
    else:
        case_ids, probabilities, _ = load_oof(paths)

    index = {entry.case_id: entry for entry in entries}
    predictions = {
        case_id: decoder.decode(row) for case_id, row in zip(case_ids, probabilities, strict=True)
    }
    references = {case_id: index[case_id].reference for case_id in case_ids}

    surrogate = score_reports(predictions, references)
    payload: dict[str, Any] = {
        "stage": "evaluate",
        "num_cases": len(case_ids),
        "surrogate": surrogate.to_dict(),
    }

    if use_radfact_lite:
        from cbct_reasoner.metrics.score import score_with_radfact_lite

        _log("[evaluate] running radfact_lite; this issues one LLM call per phrase")
        real = score_with_radfact_lite(predictions, references, **(radfact_options or {}))
        payload["radfact_lite"] = real.to_dict()

    destination = output or (paths.artifacts / "evaluation.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Generated reports only - never the references, which are patient text.
    (paths.artifacts / "oof_reports.json").write_text(
        json.dumps({"predictions": predictions}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    sample = predictions[case_ids[0]]
    _log(
        f"[evaluate] final(surrogate)={surrogate.final:.4f} "
        f"bleu4={surrogate.bleu_4:.4f} meteor={surrogate.meteor:.4f}"
    )
    _log(f"[evaluate] example report: {sample[:220]}{'...' if len(sample) > 220 else ''}")
    return payload


# ---------------------------------------------------------------------------
# Stage 7: package
# ---------------------------------------------------------------------------


def package(
    paths: Paths, config: ExperimentConfig, *, include_checkpoints: bool = True
) -> dict[str, Any]:
    """Assemble the directory the submission container copies to /opt/ml/model."""
    bank = PrototypeBank.load(paths.prototypes)
    decoder = ReportDecoder.load(paths.decoder, bank)
    checkpoints = tuple(sorted(paths.checkpoints.glob("fold*.pt"))) if include_checkpoints else ()

    evaluation_path = paths.artifacts / "evaluation.json"
    extra: dict[str, Any] = {}
    if evaluation_path.is_file():
        extra["evaluation"] = json.loads(evaluation_path.read_text(encoding="utf-8"))

    InferenceBundle.write(
        paths.bundle,
        bank=bank,
        decoder=decoder,
        config=config,
        checkpoints=checkpoints,
        extra=extra,
    )
    size = sum(path.stat().st_size for path in paths.bundle.rglob("*") if path.is_file())
    _log(
        f"[package] bundle at {paths.bundle} ({size / 1e6:.1f} MB, {len(checkpoints)} checkpoints)"
    )
    return {
        "stage": "package",
        "bundle": str(paths.bundle),
        "checkpoints": len(checkpoints),
        "size_mb": round(size / 1e6, 2),
    }


# ---------------------------------------------------------------------------
# Stage 8: ablation
# ---------------------------------------------------------------------------


def ablation(paths: Paths, config: ExperimentConfig) -> dict[str, Any]:
    """Score the image-free prior against the trained model on identical machinery.

    This is the control that decides whether the encoder earned its place. If the
    trained decoder does not beat the prior out-of-fold, the imaging path is
    measuring noise and the safe submission is the prior-only bundle.
    """
    entries = load_corpus(paths.corpus)
    bank = PrototypeBank.load(paths.prototypes)
    decoder = ReportDecoder.load(paths.decoder, bank)
    index = {entry.case_id: entry for entry in entries}

    variants: dict[str, dict[str, float]] = {}

    case_ids = [entry.case_id for entry in entries]
    references = {case_id: index[case_id].reference for case_id in case_ids}
    prior = prior_probabilities(bank, len(entries))
    variants["prior only"] = score_reports(
        {case_id: decoder.decode(row) for case_id, row in zip(case_ids, prior, strict=True)},
        references,
    ).to_dict()

    if paths.oof.is_file():
        oof_ids, probabilities, _ = load_oof(paths)
        variants["encoder + calibration"] = score_reports(
            {
                case_id: decoder.decode(row)
                for case_id, row in zip(oof_ids, probabilities, strict=True)
            },
            {case_id: index[case_id].reference for case_id in oof_ids},
        ).to_dict()

    (paths.artifacts / "ablation.json").write_text(
        json.dumps(variants, indent=2) + "\n", encoding="utf-8"
    )
    for name, values in variants.items():
        _log(f"[ablation] {name:24s} final={values['final']:.4f} clinical={values['clinical']:.4f}")
    return {"stage": "ablation", "variants": variants}


# ---------------------------------------------------------------------------
# Stage 9: figures
# ---------------------------------------------------------------------------


def figures(paths: Paths, config: ExperimentConfig) -> dict[str, Any]:
    """Render every diagnostic figure the current artifacts support."""
    from cbct_reasoner import plots

    written = plots.render_all(paths)
    _log(f"[figures] wrote {len(written)} figures to {paths.artifacts / 'plots'}")
    return {"stage": "figures", "plots": [str(path) for path in written]}


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
