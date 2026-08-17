"""Fold training loop for the finding predictor.

Produces two things per fold: a checkpoint for inference, and out-of-fold
probabilities for every validation case. The OOF matrix is what the decoder
calibrates on - thresholds tuned on in-sample probabilities are wildly
optimistic, and on a 630-case dataset that mistake is worth more than any
architecture change.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cbct_reasoner.config import EncoderConfig
from cbct_reasoner.data.dataset import AugmentConfig, CBCTDataset, build_loader
from cbct_reasoner.data.preprocess import META_DIM
from cbct_reasoner.models.losses import AsymmetricLoss, ModelEma, average_precision
from cbct_reasoner.models.network import FindingNet, build_network

CHECKPOINT_VERSION = 2


@dataclass
class FoldResult:
    fold: int
    best_epoch: int
    best_score: float
    history: list[dict[str, float]] = field(default_factory=list)
    checkpoint: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_device(preference: str | None = None) -> torch.device:
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def curve(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, curve)


@torch.no_grad()
def predict_probabilities(
    model: nn.Module, dataset: CBCTDataset, *, device: torch.device, batch_size: int = 4
) -> np.ndarray:
    model.eval()
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    for batch in loader:
        volume = batch["volume"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(volume, meta)
        outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)


def train_fold(
    *,
    config: EncoderConfig,
    fold_index: int,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    labels_by_case: dict[str, np.ndarray],
    cache_dir: str | Path,
    prior: np.ndarray,
    checkpoint_dir: str | Path,
    device: torch.device | None = None,
    progress: bool = True,
    expected_shape: tuple[int, int, int] | None = None,
) -> tuple[FoldResult, np.ndarray]:
    """Train one fold and return its result plus validation probabilities."""
    device = device or resolve_device()
    torch.manual_seed(config.seed + fold_index)
    np.random.seed(config.seed + fold_index)

    num_prototypes = len(prior)
    train_labels = np.stack([labels_by_case[case_id] for case_id in train_ids])
    validation_labels = np.stack([labels_by_case[case_id] for case_id in validation_ids])

    train_set = CBCTDataset(
        train_ids,
        cache_dir,
        train_labels,
        augment=AugmentConfig(enabled=True),
        seed=config.seed,
        expected_shape=expected_shape,
    )
    validation_set = CBCTDataset(
        validation_ids, cache_dir, validation_labels, expected_shape=expected_shape
    )

    model = build_network(
        backbone=config.backbone,
        num_prototypes=num_prototypes,
        prior=prior,
        meta_dim=META_DIM,
        timm_model=config.timm_model,
        width=config.width,
        dropout=config.dropout,
    ).to(device)

    criterion = AsymmetricLoss(
        gamma_negative=config.focal_gamma_negative,
        gamma_positive=config.focal_gamma_positive,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = build_loader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=len(train_ids) > config.batch_size,
    )
    steps_per_epoch = max(1, math.ceil(len(loader) / config.accumulate))
    scheduler = _scheduler(optimizer, steps_per_epoch * config.epochs, config.warmup_ratio)
    scaler = torch.amp.GradScaler(device.type, enabled=config.amp and device.type == "cuda")
    ema = ModelEma(model, decay=config.ema_decay)

    checkpoint_path = Path(checkpoint_dir) / f"fold{fold_index}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    validation_targets = torch.from_numpy(validation_labels)
    best_score = -1.0
    best_epoch = -1
    best_probabilities = np.zeros_like(validation_labels)
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        started = time.time()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            volume = batch["volume"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            targets = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = criterion(model(volume, meta), targets) / config.accumulate
            scaler.scale(loss).backward()
            running += float(loss.item()) * config.accumulate

            if (step + 1) % config.accumulate == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)

        probabilities = predict_probabilities(
            ema.module, validation_set, device=device, batch_size=config.batch_size
        )
        score = average_precision(torch.from_numpy(probabilities), validation_targets)
        entry = {
            "epoch": float(epoch),
            "train_loss": running / max(1, len(loader)),
            "val_map": score,
            "seconds": time.time() - started,
        }
        history.append(entry)
        if progress:
            print(
                f"[fold {fold_index}] epoch {epoch + 1}/{config.epochs} "
                f"loss={entry['train_loss']:.4f} val_mAP={score:.4f} "
                f"({entry['seconds']:.1f}s)",
                flush=True,
            )

        if score > best_score:
            best_score, best_epoch = score, epoch
            best_probabilities = probabilities
            save_checkpoint(
                checkpoint_path,
                model=ema.module,
                config=config,
                num_prototypes=num_prototypes,
                prior=prior,
                fold=fold_index,
                epoch=epoch,
                score=score,
            )

    result = FoldResult(
        fold=fold_index,
        best_epoch=best_epoch,
        best_score=best_score,
        history=history,
        checkpoint=str(checkpoint_path),
    )
    return result, best_probabilities


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    config: EncoderConfig,
    num_prototypes: int,
    prior: np.ndarray,
    fold: int,
    epoch: int,
    score: float,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "state_dict": model.state_dict(),
            "encoder_config": asdict(config),
            "num_prototypes": num_prototypes,
            "prior": np.asarray(prior, dtype=np.float32),
            "fold": fold,
            "epoch": epoch,
            "score": score,
        },
        output,
    )
    return output


def load_checkpoint(
    path: str | Path, *, device: torch.device | None = None, pretrained_backbone: bool = False
) -> tuple[FindingNet, dict[str, object]]:
    """Rebuild a network from a checkpoint.

    ``pretrained_backbone`` stays False by design: the saved state dict already
    holds the fine-tuned weights, and the inference container has no network
    access to fetch ImageNet weights it would immediately overwrite.
    """
    device = device or resolve_device()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if int(payload.get("checkpoint_version", 0)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version {payload.get('checkpoint_version')} != {CHECKPOINT_VERSION}"
        )
    config = EncoderConfig(**payload["encoder_config"])
    model = build_network(
        backbone=config.backbone,
        num_prototypes=int(payload["num_prototypes"]),
        prior=np.asarray(payload["prior"], dtype=np.float32),
        meta_dim=META_DIM,
        timm_model=config.timm_model,
        width=config.width,
        dropout=config.dropout,
        pretrained=pretrained_backbone,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def save_history(path: str | Path, results: Sequence[FoldResult]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([result.to_dict() for result in results], indent=2) + "\n", encoding="utf-8"
    )
    return output
