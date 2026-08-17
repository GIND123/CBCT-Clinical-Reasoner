"""Hugging Face Hub integration.

Reads the token from ``.env`` (the ``hf`` key) or the environment, then keeps
three repositories in sync:

``<ns>/cbct-clinical-reasoner``
    Model repo: the inference bundle, fold checkpoints, calibration report.
``<ns>/cbct-clinical-reasoner-data``
    Dataset repo: manifest, corpus, folds, labels, out-of-fold probabilities.
``<ns>/cbct-clinical-reasoner-runs``
    Dataset repo: metric JSON and training histories for run-to-run comparison.

**Everything is created private by default and pushing publicly requires an
explicit flag.** The prototype bank stores clinical sentences copied verbatim
from ToothFairy4 reports and the corpus stores the reports themselves; both are
patient-derived text released under an access-controlled agreement. Publishing
them would redistribute the dataset. Check the ToothFairy4 terms before ever
passing ``--public``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cbct_reasoner.config import Paths, env, require_env

MODEL_SUFFIX = ""
DATA_SUFFIX = "-data"
RUNS_SUFFIX = "-runs"
DEFAULT_PROJECT = "cbct-clinical-reasoner"

#: Never uploaded, regardless of which directory is pushed.
IGNORE_PATTERNS = (
    "*.nii",
    "*.nii.gz",
    "*.mha",
    "*.mhd",
    "*.nrrd",
    "cache/*",
    "**/cache/**",
    "*.tmp",
    "__pycache__/*",
    ".git/*",
)


@dataclass(frozen=True, slots=True)
class HubConfig:
    namespace: str
    project: str = DEFAULT_PROJECT
    private: bool = True

    @property
    def model_repo(self) -> str:
        return f"{self.namespace}/{self.project}{MODEL_SUFFIX}"

    @property
    def data_repo(self) -> str:
        return f"{self.namespace}/{self.project}{DATA_SUFFIX}"

    @property
    def runs_repo(self) -> str:
        return f"{self.namespace}/{self.project}{RUNS_SUFFIX}"


def resolve_token() -> str:
    return require_env("HF_TOKEN")


def resolve_namespace(explicit: str | None = None) -> str:
    """Namespace from the flag, the environment, or the token's own account."""
    if explicit:
        return explicit
    configured = env("HF_NAMESPACE")
    if configured:
        return configured
    from huggingface_hub import HfApi

    identity = HfApi(token=resolve_token()).whoami()
    name = identity.get("name")
    if not name:
        raise RuntimeError("Could not determine the Hugging Face namespace; set HF_NAMESPACE")
    return str(name)


class HubClient:
    """Thin wrapper over ``HfApi`` with safe defaults for clinical artifacts."""

    def __init__(self, config: HubConfig, *, token: str | None = None) -> None:
        from huggingface_hub import HfApi

        self.config = config
        self.token = token or resolve_token()
        self.api = HfApi(token=self.token)

    @classmethod
    def create(
        cls, *, namespace: str | None = None, project: str = DEFAULT_PROJECT, private: bool = True
    ) -> HubClient:
        return cls(
            HubConfig(namespace=resolve_namespace(namespace), project=project, private=private)
        )

    def ensure_repo(self, repo_id: str, repo_type: str) -> str:
        self.api.create_repo(
            repo_id=repo_id, repo_type=repo_type, private=self.config.private, exist_ok=True
        )
        return repo_id

    def upload_folder(
        self,
        folder: str | Path,
        *,
        repo_id: str,
        repo_type: str = "model",
        path_in_repo: str = ".",
        message: str = "Update artifacts",
        allow_patterns: tuple[str, ...] | None = None,
    ) -> str:
        source = Path(folder)
        if not source.is_dir():
            raise FileNotFoundError(f"Nothing to upload: {source} does not exist")
        self.ensure_repo(repo_id, repo_type)
        self.api.upload_folder(
            folder_path=str(source),
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
            commit_message=message,
            ignore_patterns=list(IGNORE_PATTERNS),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
        )
        return f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"

    def upload_file(
        self,
        path: str | Path,
        *,
        repo_id: str,
        repo_type: str = "model",
        path_in_repo: str | None = None,
        message: str = "Update file",
    ) -> str:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Nothing to upload: {source} does not exist")
        self.ensure_repo(repo_id, repo_type)
        self.api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=path_in_repo or source.name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=message,
        )
        return f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"

    def download(
        self,
        repo_id: str,
        *,
        repo_type: str = "model",
        destination: str | Path,
        subfolder: str | None = None,
    ) -> Path:
        from huggingface_hub import snapshot_download

        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(target),
            token=self.token,
            allow_patterns=[f"{subfolder}/*"] if subfolder else None,
        )
        return target


def write_model_card(paths: Paths, config: HubConfig, summary: dict[str, Any]) -> Path:
    """Emit a README the Hub renders, including the data-use caveat."""
    evaluation = summary.get("evaluation", {}).get("surrogate", {})
    card = f"""---
library_name: cbct-clinical-reasoner
tags:
  - medical-imaging
  - cbct
  - report-generation
  - toothfairy4
  - odin2026
---

# {config.project}

Finding predictor and calibrated report decoder for **ODIN 2026 Task 1
(ToothFairy4)**: maxillofacial and surgical report generation from a 3D CBCT
volume.

## Contents

| Path | Description |
|---|---|
| `bundle/prototypes.json` | Sentence-prototype label space built from the training corpus |
| `bundle/decoder.json` | Per-prototype thresholds calibrated on out-of-fold predictions |
| `bundle/checkpoints/` | Per-fold encoder checkpoints |
| `bundle/config.json` | Preprocessing and training configuration |
| `bundle/fallback_report.txt` | Prior-only report used if inference fails |

## Reported development metrics

These are **out-of-fold** numbers computed with the repository's offline RadFact
surrogate, not official challenge scores.

```json
{json.dumps(evaluation, indent=2) if evaluation else "{}"}
```

BLEU-4 and METEOR are reimplementations of the grader's own local
implementations and match NLTK to machine precision. The clinical component is a
lexical surrogate; the real number requires `radfact_lite` with an LLM backend.

## Intended use and limitations

Research artifact for a benchmark. **Not a medical device.** Generated text is a
draft for review by a qualified clinician and must not be used for patient care.

## Data provenance

Derived from the access-controlled ToothFairy4 release. The prototype bank
contains sentences taken verbatim from clinical reports, so this repository is
private by default and must not be made public without checking the ToothFairy4
data-use agreement.
"""
    output = paths.artifacts / "MODEL_CARD.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(card, encoding="utf-8")
    return output


def push_all(
    paths: Paths,
    *,
    namespace: str | None = None,
    project: str = DEFAULT_PROJECT,
    private: bool = True,
    include_data: bool = True,
    summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Push the bundle, artifacts, and derived data to the Hub."""
    client = HubClient.create(namespace=namespace, project=project, private=private)
    config = client.config
    urls: dict[str, str] = {}

    if paths.bundle.is_dir():
        urls["bundle"] = client.upload_folder(
            paths.bundle,
            repo_id=config.model_repo,
            repo_type="model",
            path_in_repo="bundle",
            message="Update inference bundle",
        )
        card = write_model_card(paths, config, summary or {})
        client.upload_file(
            card, repo_id=config.model_repo, repo_type="model", path_in_repo="README.md"
        )

    plots_dir = paths.artifacts / "plots"
    if plots_dir.is_dir():
        urls["plots"] = client.upload_folder(
            plots_dir,
            repo_id=config.model_repo,
            repo_type="model",
            path_in_repo="plots",
            message="Update diagnostic figures",
        )

    for name in ("calibration.json", "evaluation.json", "train_history.json", "ablation.json"):
        candidate = paths.artifacts / name
        if candidate.is_file():
            urls["runs"] = client.upload_file(
                candidate,
                repo_id=config.runs_repo,
                repo_type="dataset",
                path_in_repo=name,
                message=f"Update {name}",
            )

    if include_data and paths.work.is_dir():
        derived = [
            path
            for path in (paths.manifest, paths.corpus, paths.folds, paths.labels, paths.oof)
            if path.is_file()
        ]
        if derived:
            for path in derived:
                urls["data"] = client.upload_file(
                    path,
                    repo_id=config.data_repo,
                    repo_type="dataset",
                    path_in_repo=path.name,
                    message=f"Update {path.name}",
                )
    return urls
