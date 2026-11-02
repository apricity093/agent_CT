"""Deterministic projection splits for measurement-only validation."""
from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

@dataclass(frozen=True)
class ProjectionSplit:
    """Deterministic angle-fold split for measurement-only validation.

    Indices refer to the original view axis.  Every fold is a validation
    subset and its complement is the corresponding training subset; no new
    angles are synthesized, so a limited-angle case keeps its original
    angular support.
    """

    case_id: str
    protocol_version: str
    validation_folds: tuple[tuple[int, ...], ...]
    sorted_view_indices: tuple[int, ...]

    @property
    def num_views(self) -> int:
        return len(self.sorted_view_indices)

    @property
    def fold_count(self) -> int:
        return len(self.validation_folds)

    def training_indices(self, fold: int) -> tuple[int, ...]:
        if fold < 0 or fold >= self.fold_count:
            raise IndexError(f"fold must be between 0 and {self.fold_count - 1}")
        held_out = set(self.validation_folds[fold])
        return tuple(index for index in range(self.num_views) if index not in held_out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "protocol_version": self.protocol_version,
            "num_views": self.num_views,
            "fold_count": self.fold_count,
            "sorted_view_indices": list(self.sorted_view_indices),
            "validation_folds": [list(fold) for fold in self.validation_folds],
            "training_folds": [list(self.training_indices(index)) for index in range(self.fold_count)],
        }


def make_heldout_projection_split(
    case_id: str,
    angles_or_num_views: Sequence[float] | int,
    *,
    folds: int = 3,
    protocol_version: str = "heldout_projection_cv/v1",
) -> ProjectionSplit:
    """Create a stable, interleaved angle split shared by all candidates.

    Angles are sorted modulo ``pi`` before assigning validation folds in
    round-robin order.  Passing an integer is a convenience for synthetic
    tests and uses the original index order.
    """

    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be a non-empty string")
    if isinstance(angles_or_num_views, bool):
        raise ValueError("angles_or_num_views must be a sequence or integer")
    if isinstance(angles_or_num_views, int):
        if angles_or_num_views <= 0:
            raise ValueError("num_views must be positive")
        pairs = [(float(index), index) for index in range(angles_or_num_views)]
    else:
        pairs = []
        try:
            for index, angle in enumerate(angles_or_num_views):
                value = float(angle)
                if not isfinite(value):
                    raise ValueError("angles must be finite")
                pairs.append((value % 3.141592653589793, index))
        except TypeError as error:
            raise ValueError("angles_or_num_views must be a sequence or integer") from error
    if not pairs:
        raise ValueError("at least one view is required")
    fold_count = int(folds)
    if isinstance(folds, bool) or fold_count < 2 or fold_count > len(pairs):
        raise ValueError("folds must be at least 2 and no greater than num_views")
    ordered = tuple(index for _angle, index in sorted(pairs, key=lambda item: (item[0], item[1])))
    validation = tuple(
        tuple(ordered[position] for position in range(fold_index, len(ordered), fold_count))
        for fold_index in range(fold_count)
    )
    return ProjectionSplit(
        case_id=case_id,
        protocol_version=str(protocol_version),
        validation_folds=validation,
        sorted_view_indices=ordered,
    )
