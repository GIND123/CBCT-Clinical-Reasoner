"""Cross-validation splits.

Two rules are non-negotiable for this dataset:

1. **Group by case.** A patient with three reports must sit entirely on one side
   of every split, or the model memorises the text through a leaked twin.
2. **Report per centre.** The hidden test set comes from an independent centre,
   so a random split measures in-domain interpolation and will overstate the
   score. ``leave_one_center_out`` is the honest estimate; stratified K-fold is
   for model selection where fold count matters more than realism.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Fold:
    index: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "train": list(self.train),
            "validation": list(self.validation),
        }


@dataclass(frozen=True, slots=True)
class SplitPlan:
    strategy: str
    folds: tuple[Fold, ...]

    def __len__(self) -> int:
        return len(self.folds)

    def __iter__(self):
        return iter(self.folds)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"strategy": self.strategy, "folds": [fold.to_dict() for fold in self.folds]}
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> SplitPlan:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(
            strategy=str(payload["strategy"]),
            folds=tuple(
                Fold(
                    index=int(item["index"]),
                    name=str(item["name"]),
                    train=tuple(item["train"]),
                    validation=tuple(item["validation"]),
                )
                for item in payload["folds"]
            ),
        )

    def validation_of(self, case_id: str) -> int | None:
        for fold in self.folds:
            if case_id in fold.validation:
                return fold.index
        return None


def stratified_group_folds(
    case_ids: Sequence[str], centers: Sequence[str], *, n_folds: int = 5, seed: int = 2026
) -> SplitPlan:
    """Balanced K-fold that keeps centre proportions equal across folds."""
    if len(case_ids) != len(centers):
        raise ValueError("case_ids and centers must have the same length")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(case_ids) < n_folds:
        raise ValueError(f"Need at least {n_folds} cases to build {n_folds} folds")

    rng = np.random.default_rng(seed)
    by_center: dict[str, list[str]] = defaultdict(list)
    for case_id, center in zip(case_ids, centers, strict=True):
        by_center[center].append(case_id)

    assignment: dict[str, int] = {}
    for center in sorted(by_center):
        members = sorted(by_center[center])
        rng.shuffle(members)  # type: ignore[arg-type]
        # Round-robin from a rotating offset so small centres do not all land in fold 0.
        offset = rng.integers(0, n_folds)
        for position, case_id in enumerate(members):
            assignment[case_id] = int((position + offset) % n_folds)

    folds = []
    for index in range(n_folds):
        validation = tuple(sorted(c for c, f in assignment.items() if f == index))
        train = tuple(sorted(c for c, f in assignment.items() if f != index))
        folds.append(Fold(index=index, train=train, validation=validation, name=f"fold{index}"))
    return SplitPlan(strategy=f"stratified-group-{n_folds}fold", folds=tuple(folds))


def leave_one_center_out(case_ids: Sequence[str], centers: Sequence[str]) -> SplitPlan:
    """Hold out one acquisition centre at a time - the external-validation estimate."""
    if len(case_ids) != len(centers):
        raise ValueError("case_ids and centers must have the same length")
    unique = sorted(set(centers))
    if len(unique) < 2:
        raise ValueError("leave-one-center-out needs at least two centers")
    folds = []
    for index, center in enumerate(unique):
        validation = tuple(sorted(c for c, g in zip(case_ids, centers, strict=True) if g == center))
        train = tuple(sorted(c for c, g in zip(case_ids, centers, strict=True) if g != center))
        folds.append(Fold(index=index, train=train, validation=validation, name=f"center-{center}"))
    return SplitPlan(strategy="leave-one-center-out", folds=tuple(folds))


def build_splits(
    case_ids: Sequence[str],
    centers: Sequence[str],
    *,
    strategy: str = "stratified",
    n_folds: int = 5,
    seed: int = 2026,
) -> SplitPlan:
    if strategy == "stratified":
        return stratified_group_folds(case_ids, centers, n_folds=n_folds, seed=seed)
    if strategy in {"center", "loco", "leave-one-center-out"}:
        return leave_one_center_out(case_ids, centers)
    raise ValueError(f"Unknown split strategy: {strategy!r}")
