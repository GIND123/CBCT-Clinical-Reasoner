"""Environment, path, and experiment configuration.

Secrets are read from a local ``.env`` file (never committed) or the process
environment. The ToothFairy4 release is clinical data, so nothing in this module
ever copies report text or voxels into a shared location by default.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

#: Accepted spellings for each logical secret, in priority order. The bare ``hf``
#: key exists because the supplied ``.env`` uses it.
ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "HF_TOKEN": ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf", "HF"),
    "HF_NAMESPACE": ("HF_NAMESPACE", "HF_USER", "HF_ORG"),
    "OPENAI_API_KEY": ("OPENAI_API_KEY",),
    "RADFACT_BASE_URL": ("RADFACT_BASE_URL", "OPENAI_BASE_URL"),
    "RADFACT_MODEL": ("RADFACT_MODEL",),
    "TOOTHFAIRY_DATA": ("TOOTHFAIRY_DATA", "CBCT_DATA_ROOT", "RAW_DATA_DIR"),
    "CBCT_WORK_DIR": ("CBCT_WORK_DIR",),
}

_ENV_CACHE: dict[str, str] | None = None


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Parse a ``KEY=value`` file into a dictionary and export it to ``os.environ``.

    Values may be quoted; ``export`` prefixes and ``#`` comments are tolerated.
    Existing process environment variables win unless ``override`` is set, which
    keeps CI and Modal secrets authoritative over a stale local file.
    """
    global _ENV_CACHE
    location = Path(path) if path is not None else DEFAULT_ENV_FILE
    parsed: dict[str, str] = {}
    if location.is_file():
        for raw in location.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            parsed[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    _ENV_CACHE = parsed
    return parsed


def env(name: str, default: str | None = None) -> str | None:
    """Resolve a logical setting through its aliases, loading ``.env`` on demand."""
    if _ENV_CACHE is None:
        load_dotenv()
    for alias in ENV_ALIASES.get(name, (name,)):
        value = os.environ.get(alias) or (_ENV_CACHE or {}).get(alias)
        if value:
            return value
    return default


def require_env(name: str) -> str:
    value = env(name)
    if not value:
        aliases = " / ".join(ENV_ALIASES.get(name, (name,)))
        raise RuntimeError(f"Missing required setting {name}. Set one of: {aliases} in .env")
    return value


@dataclass(frozen=True, slots=True)
class Paths:
    """Every filesystem location the pipeline writes to.

    ``work`` holds derived clinical data (voxel caches, report text, folds) and is
    git-ignored. ``artifacts`` holds shareable, de-identified model outputs.
    """

    root: Path = REPO_ROOT
    data: Path = REPO_ROOT / "data" / "raw"
    work: Path = REPO_ROOT / "work"
    artifacts: Path = REPO_ROOT / "artifacts"

    @property
    def cache(self) -> Path:
        return self.work / "cache"

    @property
    def manifest(self) -> Path:
        return self.work / "manifest.jsonl"

    @property
    def corpus(self) -> Path:
        return self.work / "corpus.jsonl"

    @property
    def folds(self) -> Path:
        return self.work / "folds.json"

    @property
    def prototypes(self) -> Path:
        return self.artifacts / "prototypes.json"

    @property
    def labels(self) -> Path:
        return self.work / "labels.npz"

    @property
    def checkpoints(self) -> Path:
        return self.artifacts / "checkpoints"

    @property
    def oof(self) -> Path:
        return self.work / "oof.npz"

    @property
    def decoder(self) -> Path:
        return self.artifacts / "decoder.json"

    @property
    def bundle(self) -> Path:
        return self.artifacts / "bundle"

    def with_root(self, root: str | Path) -> Paths:
        """Re-anchor every derived path under ``root`` (used by Modal volumes)."""
        base = Path(root)
        return Paths(root=base, data=base / "raw", work=base / "work", artifacts=base / "artifacts")

    def ensure(self) -> Paths:
        for directory in (self.work, self.artifacts, self.cache, self.checkpoints):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def default_paths() -> Paths:
    """Paths honouring ``TOOTHFAIRY_DATA`` and ``CBCT_WORK_DIR`` overrides."""
    paths = Paths()
    work_override = env("CBCT_WORK_DIR")
    if work_override:
        paths = paths.with_root(work_override)
    data_override = env("TOOTHFAIRY_DATA")
    if data_override:
        paths = replace(paths, data=Path(data_override))
    return paths


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Volume normalization applied identically at train and inference time."""

    spacing_mm: tuple[float, float, float] = (0.8, 0.8, 0.8)
    shape_zyx: tuple[int, int, int] = (96, 192, 192)
    clip_percentiles: tuple[float, float] = (0.5, 99.5)
    orientation: str = "LPS"
    dtype: str = "float16"


@dataclass(frozen=True, slots=True)
class PrototypeConfig:
    """Sentence-prototype bank construction."""

    max_prototypes: int = 192
    min_support: int = 6
    embedder: str = "tfidf"
    linkage_threshold: float = 0.62
    max_sentence_words: int = 60


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """CBCT encoder and multi-label finding head."""

    backbone: str = "slice2d"
    timm_model: str = "convnext_tiny"
    width: int = 32
    dropout: float = 0.2
    epochs: int = 40
    batch_size: int = 4
    accumulate: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.02
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.02
    focal_gamma_negative: float = 2.0
    focal_gamma_positive: float = 0.0
    ema_decay: float = 0.999
    amp: bool = True
    num_workers: int = 4
    seed: int = 2026


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """Optional findings-to-narrative renderer."""

    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    enabled: bool = False
    lora_rank: int = 32
    lora_alpha: int = 64
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_length: int = 1536
    max_new_tokens: int = 512


@dataclass(frozen=True, slots=True)
class DecodeConfig:
    """Report assembly and threshold calibration."""

    strategy: str = "prototype"
    clinical_weight: float = 0.8
    captioning_weight: float = 0.2
    calibration_rounds: int = 6
    min_sentences: int = 6
    max_sentences: int = 26
    mbr_candidates: int = 24


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level configuration; serialized next to every artifact for provenance."""

    name: str = "cbct-reasoner-v1"
    seed: int = 2026
    folds: int = 5
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    prototypes: PrototypeConfig = field(default_factory=PrototypeConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        sections = {
            "preprocess": PreprocessConfig,
            "prototypes": PrototypeConfig,
            "encoder": EncoderConfig,
            "llm": LlmConfig,
            "decode": DecodeConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in payload.items():
            if key in sections:
                section = sections[key]
                allowed = {f for f in section.__dataclass_fields__}
                unknown = set(value) - allowed
                if unknown:
                    raise ValueError(f"Unknown {key} options: {sorted(unknown)}")
                kwargs[key] = section(**{k: _coerce(v) for k, v in value.items()})
            elif key in cls.__dataclass_fields__:
                kwargs[key] = value
            else:
                raise ValueError(f"Unknown configuration key: {key!r}")
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None) -> ExperimentConfig:
        if path is None:
            return cls()
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def _coerce(value: Any) -> Any:
    """JSON round-trips tuples as lists; dataclass fields expect tuples back."""
    return tuple(value) if isinstance(value, list) else value
