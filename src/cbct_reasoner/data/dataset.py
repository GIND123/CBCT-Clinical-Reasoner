"""Torch dataset over the preprocessed voxel cache.

Augmentation deliberately omits left-right mirroring. Roughly a third of the
finding vocabulary is lateralized ("mucosal thickening in the left maxillary
sinus"), so a mirror flip silently inverts the target unless every laterality
label is swapped with it. Losing one cheap augmentation is a far better trade
than training the model to confuse sides, which is also the error clinicians
penalise hardest in the Phase-2 arena.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cbct_reasoner.data.preprocess import read_cache


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    enabled: bool = True
    shift_voxels: int = 6
    scale_range: tuple[float, float] = (0.92, 1.08)
    gamma_range: tuple[float, float] = (0.8, 1.25)
    noise_std: float = 0.02
    dropout_prob: float = 0.1
    intensity_shift: float = 0.06


class CBCTDataset(Dataset):
    """Yields ``(volume, meta, labels)`` for one case."""

    def __init__(
        self,
        case_ids: Sequence[str],
        cache_dir: str | Path,
        labels: np.ndarray | None = None,
        *,
        augment: AugmentConfig | None = None,
        seed: int = 2026,
    ) -> None:
        if labels is not None and len(labels) != len(case_ids):
            raise ValueError("labels must have one row per case id")
        self.case_ids = list(case_ids)
        self.cache_dir = Path(cache_dir)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.augment = augment or AugmentConfig(enabled=False)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case_id = self.case_ids[index]
        array, meta = read_cache(self.cache_dir, case_id)
        volume = np.asarray(array, dtype=np.float32)

        if self.augment.enabled:
            volume = self._augment(volume, index)

        sample = {
            "volume": torch.from_numpy(np.ascontiguousarray(volume))[None],
            "meta": torch.from_numpy(meta.to_vector()),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.labels is not None:
            sample["labels"] = torch.from_numpy(self.labels[index])
        return sample

    def _augment(self, volume: np.ndarray, index: int) -> np.ndarray:
        rng = np.random.default_rng(
            (self.seed * 1_000_003 + index * 7919 + torch.randint(0, 2**31 - 1, (1,)).item())
            % (2**32)
        )
        config = self.augment

        if config.shift_voxels > 0:
            shift = rng.integers(-config.shift_voxels, config.shift_voxels + 1, size=3)
            volume = np.roll(volume, shift=tuple(int(v) for v in shift), axis=(0, 1, 2))

        low, high = config.scale_range
        volume = volume * float(rng.uniform(low, high))

        gamma_low, gamma_high = config.gamma_range
        volume = np.clip(volume, 0.0, None) ** float(rng.uniform(gamma_low, gamma_high))

        if config.intensity_shift > 0:
            volume = volume + float(rng.uniform(-config.intensity_shift, config.intensity_shift))
        if config.noise_std > 0:
            volume = volume + rng.normal(0.0, config.noise_std, size=volume.shape).astype(
                np.float32
            )
        if config.dropout_prob > 0 and rng.random() < config.dropout_prob:
            # Occlude a random slab; CBCT fields of view are frequently truncated.
            axis = int(rng.integers(0, 3))
            width = int(volume.shape[axis] * rng.uniform(0.05, 0.15))
            start = int(rng.integers(0, max(1, volume.shape[axis] - width)))
            slicer: list[slice] = [slice(None)] * 3
            slicer[axis] = slice(start, start + width)
            volume[tuple(slicer)] = 0.0

        return np.clip(volume, 0.0, 1.5).astype(np.float32)


def build_loader(
    dataset: CBCTDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
