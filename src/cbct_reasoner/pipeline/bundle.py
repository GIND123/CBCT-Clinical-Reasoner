"""The self-contained inference bundle shipped inside the submission container.

One directory holds everything inference needs: the prototype bank, the
calibrated decoder, the fold checkpoints, the preprocessing configuration, and a
precomputed fallback report.

The fallback matters more than it looks. The challenge treats a missing result as
a zero-character report, which scores zero on every metric - so any unhandled
exception on one unusual volume costs 2% of the final score outright. This bundle
therefore degrades in stages: full ensemble, then a single fold, then the
prior-only report, and it only propagates an error if it cannot even write text.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cbct_reasoner.config import ExperimentConfig, PreprocessConfig
from cbct_reasoner.decode.decoder import ReportDecoder
from cbct_reasoner.prototypes import PrototypeBank

BUNDLE_VERSION = 2
PROTOTYPES_NAME = "prototypes.json"
DECODER_NAME = "decoder.json"
CONFIG_NAME = "config.json"
FALLBACK_NAME = "fallback_report.txt"
MANIFEST_NAME = "bundle.json"
CHECKPOINT_DIR = "checkpoints"


class InferenceBundle:
    """Loads a bundle directory and turns a CBCT volume into a report."""

    def __init__(
        self,
        root: str | Path,
        *,
        bank: PrototypeBank,
        decoder: ReportDecoder,
        preprocess: PreprocessConfig,
        fallback_report: str,
        checkpoints: tuple[Path, ...] = (),
    ) -> None:
        self.root = Path(root)
        self.bank = bank
        self.decoder = decoder
        self.preprocess = preprocess
        self.fallback_report = fallback_report
        self.checkpoints = checkpoints
        self._models: list[Any] | None = None

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, root: str | Path) -> InferenceBundle:
        directory = Path(root)
        if not directory.is_dir():
            raise FileNotFoundError(f"Bundle directory does not exist: {directory}")

        bank = PrototypeBank.load(directory / PROTOTYPES_NAME)
        decoder = ReportDecoder.load(directory / DECODER_NAME, bank)
        config = ExperimentConfig.load(directory / CONFIG_NAME)
        fallback_path = directory / FALLBACK_NAME
        fallback = (
            fallback_path.read_text(encoding="utf-8").strip()
            if fallback_path.is_file()
            else decoder.decode(bank.prevalence)
        )
        checkpoints = tuple(sorted((directory / CHECKPOINT_DIR).glob("fold*.pt")))
        return cls(
            directory,
            bank=bank,
            decoder=decoder,
            preprocess=config.preprocess,
            fallback_report=fallback,
            checkpoints=checkpoints,
        )

    @staticmethod
    def write(
        root: str | Path,
        *,
        bank: PrototypeBank,
        decoder: ReportDecoder,
        config: ExperimentConfig,
        checkpoints: tuple[Path, ...] = (),
        fallback_report: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        directory = Path(root)
        (directory / CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        bank.save(directory / PROTOTYPES_NAME)
        decoder.save(directory / DECODER_NAME)
        config.save(directory / CONFIG_NAME)
        (directory / FALLBACK_NAME).write_text(
            (fallback_report or decoder.decode(bank.prevalence)).strip() + "\n", encoding="utf-8"
        )

        copied: list[str] = []
        for checkpoint in checkpoints:
            target = directory / CHECKPOINT_DIR / Path(checkpoint).name
            if Path(checkpoint).resolve() != target.resolve():
                target.write_bytes(Path(checkpoint).read_bytes())
            copied.append(target.name)

        (directory / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "bundle_version": BUNDLE_VERSION,
                    "num_prototypes": len(bank),
                    "num_training_cases": bank.num_cases,
                    "checkpoints": copied,
                    "preprocess": asdict(config.preprocess),
                    **(extra or {}),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return directory

    # -- inference ---------------------------------------------------------

    def _load_models(self) -> list[Any]:
        if self._models is not None:
            return self._models
        models: list[Any] = []
        if self.checkpoints:
            try:
                from cbct_reasoner.models.trainer import load_checkpoint, resolve_device

                device = resolve_device()
                for checkpoint in self.checkpoints:
                    try:
                        model, _ = load_checkpoint(checkpoint, device=device)
                        models.append((model, device))
                    except Exception as error:  # pragma: no cover - defensive
                        print(
                            f"warning: skipping checkpoint {checkpoint.name}: {error}", flush=True
                        )
            except ImportError as error:  # pragma: no cover - CPU-only bundle
                print(f"warning: torch unavailable, using prior decoder ({error})", flush=True)
        self._models = models
        return models

    def probabilities(self, volume_path: str | Path) -> np.ndarray:
        """Ensemble finding probabilities; falls back to the corpus prior."""
        models = self._load_models()
        if not models:
            return self.bank.prevalence.astype(np.float32)

        import torch

        from cbct_reasoner.data.preprocess import preprocess_volume

        array, meta = preprocess_volume(volume_path, self.preprocess)
        volume = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None]
        meta_tensor = torch.from_numpy(meta.to_vector())[None]

        outputs: list[np.ndarray] = []
        for model, device in models:
            with torch.no_grad():
                logits = model(volume.to(device), meta_tensor.to(device))
                outputs.append(torch.sigmoid(logits.float()).cpu().numpy()[0])
        return np.mean(outputs, axis=0).astype(np.float32)

    def predict(self, volume_path: str | Path) -> str:
        """Generate one report, degrading to the fallback rather than failing."""
        try:
            return self.decoder.decode(self.probabilities(volume_path))
        except Exception as error:  # pragma: no cover - defensive by design
            print(
                f"warning: falling back to prior report ({type(error).__name__}: {error})",
                flush=True,
            )
            return self.fallback_report
