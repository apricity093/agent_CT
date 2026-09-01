"""Stable CT benchmark-case seam used by tests, examples, and agents.

Numeric arrays live in HDF5 while searchable provenance and geometry live in
JSON.  Callers normally need only ``list_ct_cases``, ``load_ct_case``, and
``evaluate_ct_case``; storage validation, checksums, dtype migration, and
metric collection stay behind this module interface.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .protocol import ProjectionSplit, make_heldout_projection_split


SCHEMA_VERSION = "1.0"
DATA_ROOT_ENV = "INVFRAMEWORK_CT_CASE_ROOT"


def _require_h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "CT benchmark cases require h5py. Install h5py or use generated "
            "in-memory operator contract fixtures."
        ) from error
    return h5py


def _default_data_root() -> Path:
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "test" / "data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value)))


@dataclass(frozen=True)
class CTTestCase:
    """One operator-ready, batched CT benchmark case.

    ``truth`` has shape ``(B, *domain_shape)``. ``measurement`` and
    ``measurement_clean`` have shape ``(B, *range_shape)``.  For a public
    staged case, ``measurement_clean`` may intentionally alias the observed
    measurement because the clean signal is withheld.  Geometry and metadata
    are JSON-compatible mappings so agents can inspect them without loading
    the HDF5 arrays.
    """

    case_id: str
    truth: torch.Tensor
    measurement_clean: torch.Tensor
    measurement: torch.Tensor
    geometry: Mapping[str, Any]
    metadata: Mapping[str, Any]
    roi_mask: torch.Tensor | None = None
    valid_measurement_mask: torch.Tensor | None = None
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.case_id or self.case_id.strip() != self.case_id:
            raise ValueError("case_id must be a non-empty normalized string.")
        for name, tensor in (
            ("truth", self.truth),
            ("measurement_clean", self.measurement_clean),
            ("measurement", self.measurement),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if tensor.ndim < 2 or tensor.shape[0] <= 0:
                raise ValueError(f"{name} must be a non-empty batched tensor.")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values.")
        if self.measurement_clean.shape != self.measurement.shape:
            raise ValueError("measurement_clean and measurement must have equal shapes.")
        if self.truth.shape[0] != self.measurement.shape[0]:
            raise ValueError("truth and measurement batch sizes must agree.")

        domain_shape = tuple(int(v) for v in self.geometry.get("domain_shape", ()))
        range_shape = tuple(int(v) for v in self.geometry.get("range_shape", ()))
        if domain_shape != tuple(self.truth.shape[1:]):
            raise ValueError(
                f"geometry domain_shape {domain_shape} does not match truth "
                f"shape {tuple(self.truth.shape[1:])}."
            )
        if range_shape != tuple(self.measurement.shape[1:]):
            raise ValueError(
                f"geometry range_shape {range_shape} does not match measurement "
                f"shape {tuple(self.measurement.shape[1:])}."
            )
        if self.roi_mask is not None and tuple(self.roi_mask.shape) not in {
            tuple(self.truth.shape),
            tuple(self.truth.shape[1:]),
        }:
            raise ValueError("roi_mask must match the batched or per-sample truth shape.")
        if self.valid_measurement_mask is not None and tuple(
            self.valid_measurement_mask.shape
        ) not in {tuple(self.measurement.shape), tuple(self.measurement.shape[1:])}:
            raise ValueError(
                "valid_measurement_mask must match the batched or per-sample "
                "measurement shape."
            )

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.metadata.get("capability_tags", ()))

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "CTTestCase":
        """Return a case migrated to ``device``/``dtype`` without mutating it."""

        def migrate(tensor: torch.Tensor | None, *, is_mask: bool = False):
            if tensor is None:
                return None
            target_dtype = tensor.dtype if is_mask or dtype is None else dtype
            return tensor.to(device=device, dtype=target_dtype)

        return replace(
            self,
            truth=migrate(self.truth),
            measurement_clean=migrate(self.measurement_clean),
            measurement=migrate(self.measurement),
            roi_mask=migrate(self.roi_mask, is_mask=True),
            valid_measurement_mask=migrate(
                self.valid_measurement_mask, is_mask=True
            ),
        )


def restrict_ct_case(
    case: CTTestCase,
    view_indices: Sequence[int],
    *,
    case_id: str | None = None,
    partition: str | None = None,
) -> CTTestCase:
    """Return a CT case restricted to an existing subset of projection views."""

    indices = tuple(int(value) for value in view_indices)
    if not indices:
        raise ValueError("view_indices must contain at least one view")
    num_views = int(case.measurement.shape[-2])
    if len(set(indices)) != len(indices) or any(index < 0 or index >= num_views for index in indices):
        raise ValueError("view_indices must be unique indices within the case view axis")
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=case.measurement.device)

    def select_views(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None or tensor.ndim < 2:
            return tensor
        return tensor.index_select(-2, index_tensor.to(device=tensor.device))

    geometry = _json_copy(case.geometry)
    if "angles_rad" in geometry:
        angles = list(geometry["angles_rad"])
        geometry["angles_rad"] = [angles[index] for index in indices]
    range_shape = list(geometry.get("range_shape", ()))
    if len(range_shape) >= 2:
        range_shape[-2] = len(indices)
        geometry["range_shape"] = range_shape
    metadata = _json_copy(case.metadata)
    metadata["source_case_id"] = case.case_id
    metadata["view_indices"] = list(indices)
    if partition is not None:
        metadata["projection_partition"] = partition
    return replace(
        case,
        case_id=case_id or case.case_id,
        measurement_clean=select_views(case.measurement_clean),
        measurement=select_views(case.measurement),
        valid_measurement_mask=select_views(case.valid_measurement_mask),
        geometry=geometry,
        metadata=metadata,
    )


def make_heldout_ct_splits(
    case: CTTestCase,
    *,
    folds: int = 3,
    protocol_version: str = "heldout_projection_cv/v1",
) -> tuple[ProjectionSplit, tuple[tuple[CTTestCase, CTTestCase], ...]]:
    """Create train/held-out CT case pairs using only observed angle views."""

    angles = case.geometry.get("angles_rad")
    if angles is None:
        angles = int(case.measurement.shape[-2])
    split = make_heldout_projection_split(
        case.case_id,
        angles,
        folds=folds,
        protocol_version=protocol_version,
    )
    pairs = tuple(
        (
            restrict_ct_case(
                case,
                split.training_indices(fold),
                case_id=f"{case.case_id}::fold{fold}::train",
                partition=f"train_fold_{fold}",
            ),
            restrict_ct_case(
                case,
                split.validation_folds[fold],
                case_id=f"{case.case_id}::fold{fold}::heldout",
                partition=f"heldout_fold_{fold}",
            ),
        )
        for fold in range(split.fold_count)
    )
    return split, pairs


@dataclass(frozen=True)
class CTCaseEvaluation:
    """Reconstruction plus portable scalar metrics for one case."""

    reconstruction: torch.Tensor
    metrics: Mapping[str, float | str]


def _read_catalog(data_root: Path) -> list[dict[str, Any]]:
    path = data_root / "catalog.json"
    if not path.exists():
        raise FileNotFoundError(f"CT benchmark catalog does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported catalog schema_version={payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}."
        )
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("catalog.json must contain a cases list.")
    ids = [record.get("case_id") for record in cases]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("catalog case_id values must be unique and non-empty.")
    return cases


def list_ct_cases(
    filters: Mapping[str, Any] | None = None,
    *,
    data_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List agent-readable case records, optionally filtered by fields/tags."""

    root = Path(data_root).expanduser().resolve() if data_root else _default_data_root()
    records = _read_catalog(root)
    requested = dict(filters or {})

    def matches(record: Mapping[str, Any]) -> bool:
        for key, expected in requested.items():
            if key in {"tag", "tags"}:
                wanted = {str(expected)} if isinstance(expected, str) else {
                    str(value) for value in expected
                }
                if not wanted.issubset(set(record.get("capability_tags", ()))):
                    return False
            elif record.get(key) != expected:
                return False
        return True

    return [_json_copy(record) for record in records if matches(record)]


def _case_record(case_id: str, root: Path) -> dict[str, Any]:
    records = [record for record in _read_catalog(root) if record["case_id"] == case_id]
    if not records:
        available = ", ".join(record["case_id"] for record in _read_catalog(root))
        raise KeyError(f"Unknown CT case {case_id!r}; available: {available}")
    return records[0]


def load_ct_case(
    case_id: str,
    *,
    data_root: str | Path | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    verify_checksum: bool = True,
) -> CTTestCase:
    """Load and validate one versioned HDF5/JSON CT case."""

    root = Path(data_root).expanduser().resolve() if data_root else _default_data_root()
    record = _case_record(case_id, root)
    case_dir = (root / record["path"]).resolve()
    manifest_path = case_dir / "case.json"
    arrays_path = case_dir / "arrays.h5"
    if not manifest_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(f"Incomplete CT case directory: {case_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported case schema in {manifest_path}.")
    if manifest.get("case_id") != case_id:
        raise ValueError("Catalog and case manifest case_id values disagree.")
    expected_digest = manifest.get("sha256", {}).get("arrays.h5")
    if verify_checksum and expected_digest and _sha256(arrays_path) != expected_digest:
        raise ValueError(f"Checksum mismatch for {arrays_path}")

    h5py = _require_h5py()
    with h5py.File(arrays_path, "r") as handle:
        truth = torch.from_numpy(np.asarray(handle["truth/x"], dtype=np.float32))
        observed = torch.from_numpy(
            np.asarray(handle["measurement/y_observed"], dtype=np.float32)
        )
        clean_dataset = handle.get("measurement/y_clean")
        clean = observed if clean_dataset is None else torch.from_numpy(
            np.asarray(clean_dataset, dtype=np.float32)
        )
        roi = (
            torch.from_numpy(np.asarray(handle["masks/roi"], dtype=bool))
            if "masks/roi" in handle
            else None
        )
        valid = (
            torch.from_numpy(
                np.asarray(handle["masks/valid_measurement"], dtype=bool)
            )
            if "masks/valid_measurement" in handle
            else None
        )

    case = CTTestCase(
        case_id=case_id,
        truth=truth,
        measurement_clean=clean,
        measurement=observed,
        geometry=manifest["geometry"],
        metadata=manifest,
        roi_mask=roi,
        valid_measurement_mask=valid,
        source_path=case_dir,
    )
    return case.to(device=device, dtype=dtype)


def write_ct_case(
    case: CTTestCase,
    case_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one authored case and return its catalog record."""

    destination = Path(case_dir).expanduser().resolve()
    arrays_path = destination / "arrays.h5"
    manifest_path = destination / "case.json"
    if not overwrite and (arrays_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"CT case already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    h5py = _require_h5py()
    with h5py.File(arrays_path, "w") as handle:
        handle.create_dataset(
            "truth/x", data=case.truth.detach().cpu().numpy(), compression="gzip"
        )
        handle.create_dataset(
            "measurement/y_clean",
            data=case.measurement_clean.detach().cpu().numpy(),
            compression="gzip",
        )
        handle.create_dataset(
            "measurement/y_observed",
            data=case.measurement.detach().cpu().numpy(),
            compression="gzip",
        )
        if case.roi_mask is not None:
            handle.create_dataset(
                "masks/roi", data=case.roi_mask.detach().cpu().numpy()
            )
        if case.valid_measurement_mask is not None:
            handle.create_dataset(
                "masks/valid_measurement",
                data=case.valid_measurement_mask.detach().cpu().numpy(),
            )

    manifest = _json_copy(case.metadata)
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "case_id": case.case_id,
            "geometry": _json_copy(case.geometry),
            "sha256": {"arrays.h5": _sha256(arrays_path)},
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "case_id": case.case_id,
        "path": destination.name,
        "dimension": manifest.get("dimension"),
        "geometry_type": case.geometry.get("type"),
        "measurement_kind": manifest.get("measurement", {}).get("kind"),
        "noise_model": manifest.get("measurement", {}).get("noise_model"),
        "reference_kind": manifest.get("provenance", {}).get("reference_kind"),
        "capability_tags": list(manifest.get("capability_tags", ())),
    }


def evaluate_ct_case(solver, operator, case: CTTestCase) -> CTCaseEvaluation:
    """Evaluate injected solver/operator dependencies on one loaded case."""

    if tuple(operator.domain_shape) != tuple(case.truth.shape[1:]):
        raise ValueError("operator.domain_shape does not match the CT case.")
    if tuple(operator.range_shape) != tuple(case.measurement.shape[1:]):
        raise ValueError("operator.range_shape does not match the CT case.")
    reconstruction = solver.solve(case.measurement, operator)
    if reconstruction.shape != case.truth.shape:
        raise ValueError("solver returned a reconstruction with the wrong shape.")
    predicted = operator.forward(reconstruction).detach()
    truth = case.truth
    residual = predicted - case.measurement
    truth_norm = truth.norm().clamp_min(1e-12)
    measurement_norm = case.measurement.norm().clamp_min(1e-12)
    rmse = torch.sqrt((reconstruction - truth).square().mean())
    data_range = float(case.metadata.get("ground_truth", {}).get("data_range", 1.0))
    psnr = 20.0 * torch.log10(
        torch.as_tensor(data_range, device=rmse.device, dtype=rmse.dtype)
        / rmse.clamp_min(1e-12)
    )
    metrics: dict[str, float | str] = {
        "case_id": case.case_id,
        "relative_error": float(((reconstruction - truth).norm() / truth_norm).item()),
        "data_residual": float((residual.norm() / measurement_norm).item()),
        "rmse": float(rmse.item()),
        "psnr": float(psnr.item()),
    }
    return CTCaseEvaluation(reconstruction=reconstruction, metrics=metrics)
