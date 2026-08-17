"""Modal application for the ToothFairy4 pipeline.

The whole pipeline lives on one persistent Volume so stages are resumable and
independent:

    /vol/raw         the ToothFairy4 release (uploaded once)
    /vol/work        voxel cache, corpus, folds, out-of-fold probabilities
    /vol/artifacts   prototypes, checkpoints, decoder, bundle, metrics

Upload the dataset once, then run stages:

    modal volume create cbct-toothfairy4
    modal volume put cbct-toothfairy4 /local/toothfairy4 /raw
    modal run modal_app/app.py --stage inspect
    modal run modal_app/app.py --stage all

Folds train in parallel via ``Function.map``, so wall-clock time is roughly one
fold regardless of how many there are.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

APP_NAME = "cbct-clinical-reasoner"
VOLUME_NAME = os.environ.get("CBCT_VOLUME", "cbct-toothfairy4")
VOLUME_ROOT = "/vol"
GPU_TYPE = os.environ.get("CBCT_GPU", "A10G")
TRAIN_TIMEOUT = int(os.environ.get("CBCT_TRAIN_TIMEOUT", 6 * 60 * 60))

# ---------------------------------------------------------------------------
# Credentials: read .env locally and forward as an inline secret. This avoids a
# separate `modal secret create` step while keeping the token out of the image.
# ---------------------------------------------------------------------------
try:
    from cbct_reasoner.config import env as _env

    _SECRET_VALUES = {
        key: value
        for key, value in {
            "HF_TOKEN": _env("HF_TOKEN"),
            "HF_NAMESPACE": _env("HF_NAMESPACE"),
            "OPENAI_API_KEY": _env("OPENAI_API_KEY"),
        }.items()
        if value
    }
except Exception:  # pragma: no cover - the image itself has no .env
    _SECRET_VALUES = {}

secret = modal.Secret.from_dict(_SECRET_VALUES or {"HF_TOKEN": ""})
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(f"{APP_NAME}-hf-cache", create_if_missing=True)

# Local sources are attached LAST: Modal forbids a build step after
# `add_local_*`, and attaching them last also means editing repository code does
# not invalidate the (slow) dependency layer.
_base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Pinned to exactly what submission/requirements.txt and the Grand Challenge
    # base image (pytorch/pytorch:2.9.1) provide. A checkpoint trained against a
    # different torch or timm build can fail to load inside the algorithm
    # container, and that failure surfaces only at submission time.
    .pip_install(
        "numpy==2.1.3",
        "SimpleITK==2.4.1",
        "torch==2.9.1",
        "timm==1.0.19",
        "scikit-learn==1.6.1",
        "huggingface-hub>=0.24",
        "tqdm>=4.66",
        "nltk>=3.9",
        "matplotlib>=3.8",
    )
    .env(
        {
            "PYTHONPATH": "/root/src",
            "CBCT_WORK_DIR": VOLUME_ROOT,
            "HF_HOME": "/hf",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

_llm_base = _base.pip_install(
    "transformers>=4.44", "accelerate>=0.33", "peft>=0.12", "sentencepiece>=0.2"
)


def _with_sources(base: modal.Image) -> modal.Image:
    return base.add_local_dir(REPO_ROOT / "src", "/root/src").add_local_dir(
        REPO_ROOT / "configs", "/root/configs"
    )


image = _with_sources(_base)
llm_image = _with_sources(_llm_base)

app = modal.App(APP_NAME)

VOLUMES = {VOLUME_ROOT: volume, "/hf": hf_cache}


# ---------------------------------------------------------------------------
# Helpers executed inside the container
# ---------------------------------------------------------------------------


def _context(config_name: str | None):
    """Resolve Paths/ExperimentConfig against the mounted volume."""
    from cbct_reasoner.config import ExperimentConfig, Paths

    paths = Paths().with_root(VOLUME_ROOT).ensure()
    config_path = Path("/root/configs") / (config_name or "toothfairy4.json")
    config = ExperimentConfig.load(config_path if config_path.is_file() else None)
    return paths, config


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


@app.function(image=image, volumes=VOLUMES, timeout=900)
def inspect() -> dict:
    from cbct_reasoner.data.discovery import inspect_layout

    paths, _ = _context(None)
    if not paths.data.exists():
        return {
            "ok": False,
            "reason": f"{paths.data} is empty",
            "hint": f"modal volume put {VOLUME_NAME} /local/toothfairy4 /raw",
        }
    report = inspect_layout(paths.data)
    print(report.render(), flush=True)
    return report.to_dict()


@app.function(image=image, volumes=VOLUMES, timeout=8 * 60 * 60, cpu=8.0, memory=32768)
def prepare(config_name: str | None = None, force: bool = False, skip_errors: bool = True) -> dict:
    from cbct_reasoner import pipeline

    paths, config = _context(config_name)
    summary = pipeline.prepare(paths, config, force=force, skip_errors=skip_errors)
    volume.commit()
    return summary


@app.function(image=image, volumes=VOLUMES, timeout=3600, cpu=4.0, memory=16384)
def prototypes(config_name: str | None = None) -> dict:
    from cbct_reasoner import pipeline

    paths, config = _context(config_name)
    summary = pipeline.prototypes(paths, config)
    volume.commit()
    return summary


@app.function(image=image, volumes=VOLUMES, timeout=900)
def splits(config_name: str | None = None, strategy: str = "stratified") -> dict:
    from cbct_reasoner import pipeline

    paths, config = _context(config_name)
    summary = pipeline.splits(paths, config, strategy=strategy)
    volume.commit()
    return summary


@app.function(image=image, volumes=VOLUMES, gpu=GPU_TYPE, timeout=TRAIN_TIMEOUT, memory=32768)
def train_fold(fold_index: int, config_name: str | None = None) -> dict:
    """Train one fold. Called through ``.map`` so folds run concurrently."""
    import numpy as np

    from cbct_reasoner.data.splits import SplitPlan
    from cbct_reasoner.models.trainer import resolve_device
    from cbct_reasoner.models.trainer import train_fold as run_fold
    from cbct_reasoner.prototypes import PrototypeBank, load_labels

    paths, config = _context(config_name)
    bank = PrototypeBank.load(paths.prototypes)
    case_ids, labels = load_labels(paths.labels)
    plan = SplitPlan.load(paths.folds)
    fold = next(item for item in plan if item.index == fold_index)

    result, probabilities = run_fold(
        config=config.encoder,
        fold_index=fold_index,
        train_ids=fold.train,
        validation_ids=fold.validation,
        labels_by_case=dict(zip(case_ids, labels, strict=True)),
        cache_dir=paths.cache,
        prior=bank.prevalence,
        checkpoint_dir=paths.checkpoints,
        device=resolve_device(),
        expected_shape=tuple(config.preprocess.shape_zyx),
    )
    # Each fold writes its own slice; `collect_oof` assembles the full matrix so
    # concurrent folds never race on one file.
    slice_path = paths.work / f"oof_fold{fold_index}.npz"
    np.savez_compressed(
        slice_path,
        case_ids=np.asarray(list(fold.validation), dtype=np.str_),
        probabilities=probabilities.astype(np.float32),
    )
    volume.commit()
    return result.to_dict()


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def collect_oof(config_name: str | None = None) -> dict:
    """Merge per-fold probability slices into the single out-of-fold matrix."""
    import numpy as np

    from cbct_reasoner.data.corpus import load_corpus
    from cbct_reasoner.prototypes import PrototypeBank

    paths, _ = _context(config_name)
    entries = load_corpus(paths.corpus)
    bank = PrototypeBank.load(paths.prototypes)
    order = [entry.case_id for entry in entries]
    position = {case_id: index for index, case_id in enumerate(order)}

    matrix = np.tile(bank.prevalence.astype(np.float32), (len(order), 1))
    covered = np.zeros(len(order), dtype=bool)
    slices = sorted(paths.work.glob("oof_fold*.npz"))
    for path in slices:
        with np.load(path, allow_pickle=False) as archive:
            for row, case_id in enumerate(str(v) for v in archive["case_ids"].tolist()):
                if case_id in position:
                    matrix[position[case_id]] = archive["probabilities"][row]
                    covered[position[case_id]] = True

    np.savez_compressed(
        paths.oof, case_ids=np.asarray(order, dtype=np.str_), probabilities=matrix, covered=covered
    )
    volume.commit()
    return {"stage": "collect_oof", "slices": len(slices), "coverage": float(covered.mean())}


@app.function(image=image, volumes=VOLUMES, timeout=4 * 60 * 60, cpu=8.0, memory=32768)
def calibrate(
    config_name: str | None = None, prior_only: bool = False, rounds: int | None = None
) -> dict:
    from cbct_reasoner import pipeline

    paths, config = _context(config_name)
    summary = pipeline.calibrate_decoder(paths, config, prior_only=prior_only, rounds=rounds)
    volume.commit()
    return summary


@app.function(image=image, volumes=VOLUMES, timeout=2 * 60 * 60, cpu=4.0, memory=16384)
def evaluate(config_name: str | None = None, prior_only: bool = False) -> dict:
    from cbct_reasoner import pipeline

    paths, config = _context(config_name)
    summary = pipeline.evaluate(paths, config, prior_only=prior_only)
    volume.commit()
    return summary


@app.function(image=image, volumes=VOLUMES, timeout=2 * 60 * 60, cpu=4.0, memory=16384)
def ablation(config_name: str | None = None) -> dict:
    from cbct_reasoner import pipeline

    paths, config = _context(config_name)
    summary = pipeline.ablation(paths, config)
    volume.commit()
    return summary


@app.function(image=image, volumes=VOLUMES, timeout=3600, cpu=4.0, memory=16384)
def figures(config_name: str | None = None) -> dict:
    from cbct_reasoner import pipeline

    paths, config = _context(config_name)
    summary = pipeline.figures(paths, config)
    volume.commit()
    return summary


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def package(config_name: str | None = None, include_checkpoints: bool = True) -> dict:
    from cbct_reasoner import pipeline

    paths, config = _context(config_name)
    summary = pipeline.package(paths, config, include_checkpoints=include_checkpoints)
    volume.commit()
    return summary


@app.function(image=image, volumes=VOLUMES, secrets=[secret], timeout=3 * 60 * 60)
def push_hub(namespace: str | None = None, project: str = APP_NAME, private: bool = True) -> dict:
    from cbct_reasoner.hub import push_all

    paths, _ = _context(None)
    summary_path = paths.artifacts / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return push_all(paths, namespace=namespace, project=project, private=private, summary=summary)


@app.function(image=llm_image, volumes=VOLUMES, gpu=GPU_TYPE, timeout=TRAIN_TIMEOUT)
def train_narrative(config_name: str | None = None) -> dict:
    """Optional: LoRA fine-tune the findings-to-prose renderer."""
    from cbct_reasoner.data.corpus import load_corpus
    from cbct_reasoner.models.llm import build_sft_dataset, save_sft_dataset, train_adapter

    paths, config = _context(config_name)
    entries = load_corpus(paths.corpus)
    examples = build_sft_dataset([entry.reference for entry in entries])
    save_sft_dataset(examples, paths.work / "llm_sft.jsonl")
    destination = train_adapter(examples, config.llm, paths.artifacts / "narrative_adapter")
    volume.commit()
    return {"stage": "train_narrative", "examples": len(examples), "adapter": str(destination)}


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def download_bundle() -> bytes:
    """Return the packaged bundle as a tar archive for local Docker builds."""
    import io
    import tarfile

    paths, _ = _context(None)
    if not paths.bundle.is_dir():
        raise RuntimeError("No bundle on the volume; run the `package` stage first")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(paths.bundle, arcname="model")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    stage: str = "all",
    config_name: str = "toothfairy4.json",
    strategy: str = "stratified",
    folds: str = "",
    prior_only: bool = False,
    push: bool = False,
    force: bool = False,
    output: str = "",
) -> None:
    """Run one stage or the whole pipeline.

    ``--stage all`` runs prepare -> prototypes -> splits -> train (parallel folds)
    -> collect-oof -> calibrate -> evaluate -> ablation -> figures -> package.

    ``--stage post`` runs everything after training, for the common case where the
    voxel cache and label space were built locally and uploaded.
    """
    selected = [int(value) for value in folds.split(",") if value.strip()] or None
    results: dict[str, object] = {}

    def show(name: str, payload: object) -> None:
        results[name] = payload
        print(f"\n===== {name} =====\n{json.dumps(payload, indent=2, default=str)}", flush=True)

    if stage in {"inspect"}:
        show("inspect", inspect.remote())
        return
    if stage == "download-bundle":
        target = Path(output or "submission/model.tar.gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(download_bundle.remote())
        print(f"wrote {target}")
        return
    if stage == "push":
        show("push", push_hub.remote())
        return
    if stage == "train-narrative":
        show("train_narrative", train_narrative.remote(config_name))
        return

    # "post" resumes after training when the cache, corpus, prototypes and folds
    # were prepared locally and only the GPU-side work runs on Modal.
    post = stage == "post"

    if stage in {"all", "prepare"}:
        show("prepare", prepare.remote(config_name, force))
    if stage in {"all", "prototypes"}:
        show("prototypes", prototypes.remote(config_name))
    if stage in {"all", "splits"}:
        show("splits", splits.remote(config_name, strategy))
    if stage in {"all", "train"} and not prior_only:
        plan = results.get("splits") or splits.remote(config_name, strategy)
        indices = selected or list(range(len(plan["folds"])))  # type: ignore[index]
        show("train", list(train_fold.map(indices, kwargs={"config_name": config_name})))
        show("collect_oof", collect_oof.remote(config_name))
    if post or stage in {"all", "calibrate"}:
        show("calibrate", calibrate.remote(config_name, prior_only))
    if post or stage in {"all", "evaluate"}:
        show("evaluate", evaluate.remote(config_name, prior_only))
    if post or stage in {"all", "ablation"}:
        show("ablation", ablation.remote(config_name))
    if post or stage in {"all", "figures"}:
        show("figures", figures.remote(config_name))
    if post or stage in {"all", "package"}:
        show("package", package.remote(config_name, not prior_only))
    if push:
        show("push", push_hub.remote())

    final = results.get("evaluate", {})
    if isinstance(final, dict) and "surrogate" in final:
        print(f"\nFINAL (surrogate) = {final['surrogate']['final']:.4f}")
