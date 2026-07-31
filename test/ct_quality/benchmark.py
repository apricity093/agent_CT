"""Artifact writer for the high-resolution synthetic CT quality cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import torch

from inv_framework.benchmarks import CTTestCase, evaluate_ct_case
from inv_framework.utils.metrics import ssim


QUALITY_CASE_IDS = (
    "parallel_2d/tissue_breast_dense_clean_128",
    "parallel_2d/tissue_breast_sparse_poisson_128",
    "parallel_2d/tissue_breast_limited_angle_128",
)


def _roi_metrics(case: CTTestCase, reconstruction: torch.Tensor) -> dict[str, float]:
    mask = case.roi_mask
    if mask is None:
        mask = torch.ones_like(case.truth, dtype=torch.bool)
    elif mask.ndim == case.truth.ndim - 1:
        mask = mask.unsqueeze(0)
    error = reconstruction - case.truth
    roi_error = error[mask]
    rmse = torch.sqrt(roi_error.square().mean())
    data_range = float(case.metadata["ground_truth"]["data_range"])
    roi_psnr = 20.0 * torch.log10(
        torch.as_tensor(data_range, dtype=rmse.dtype, device=rmse.device)
        / rmse.clamp_min(1e-12)
    )
    reconstruction_for_ssim = reconstruction.clamp(0.0, data_range)
    return {
        "roi_rmse": float(rmse.item()),
        "roi_psnr": float(roi_psnr.item()),
        "ssim": float(
            ssim(reconstruction_for_ssim, case.truth, data_range=data_range)
            .mean()
            .item()
        ),
    }


def _plot_quality_summary(
    output_path: Path,
    rows: list[tuple[CTTestCase, torch.Tensor, dict[str, float | str]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(rows), 4, figsize=(14, 3.6 * len(rows)))
    axes = axes.reshape(len(rows), 4)
    for row_index, (case, reconstruction, metrics) in enumerate(rows):
        truth = case.truth[0, 0].detach().cpu()
        observed = case.measurement[0, 0].detach().cpu()
        result = reconstruction[0, 0].detach().cpu()
        display_scale = float(
            case.metadata["ground_truth"].get("display_scale_to_cm_inverse", 1.0)
        )
        display_window = case.metadata["ground_truth"].get(
            "display_window_cm_inverse", [0.0, float(truth.max() * display_scale)]
        )
        vmin, vmax = (float(value) for value in display_window)
        truth_display = truth * display_scale
        result_display = result * display_scale
        error_display = (result - truth).abs() * display_scale

        truth_axis, sino_axis, result_axis, error_axis = axes[row_index]
        truth_image = truth_axis.imshow(
            truth_display, cmap="gray", vmin=vmin, vmax=vmax
        )
        truth_axis.set_title("Truth (cm$^{-1}$)")
        sino_axis.imshow(observed, cmap="magma", aspect="auto")
        sino_axis.set_title("Observed sinogram")
        result_axis.imshow(result_display, cmap="gray", vmin=vmin, vmax=vmax)
        result_axis.set_title(
            f"FBP | PSNR {metrics['roi_psnr']:.2f} dB | SSIM {metrics['ssim']:.3f}"
        )
        error_limit = max(float(torch.quantile(error_display, 0.995)), 1e-6)
        error_image = error_axis.imshow(
            error_display, cmap="inferno", vmin=0.0, vmax=error_limit
        )
        error_axis.set_title("Absolute error (cm$^{-1}$)")
        for axis in (truth_axis, result_axis, error_axis):
            axis.axis("off")
        sino_axis.set_xlabel("detector bin")
        sino_axis.set_ylabel("view")
        figure.colorbar(truth_image, ax=truth_axis, fraction=0.046, pad=0.03)
        figure.colorbar(error_image, ax=error_axis, fraction=0.046, pad=0.03)

        acquisition = case.metadata["measurement"]["noise_model"]
        coverage = float(case.geometry["angular_coverage_deg"])
        figure.text(
            0.01,
            1.0 - (row_index + 0.48) / len(rows),
            f"{case.case_id.split('/')[-1]}\n"
            f"{case.measurement.shape[-2]} views | {coverage:.1f} deg | {acquisition}",
            rotation=90,
            va="center",
            ha="left",
            fontsize=8,
        )

    figure.suptitle(
        "invframework synthetic tissue CT quality benchmark\n"
        "ASTRA reference projections -> invframework ParallelBeamRadon2D + FBP",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.035, 0.0, 1.0, 0.95))
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def write_quality_artifacts(
    cases: Iterable[CTTestCase],
    *,
    operator_factory: Callable[[CTTestCase], object],
    solver_factory: Callable[[CTTestCase], object],
    output_dir: str | Path,
) -> dict:
    """Evaluate injected dependencies and write one portable quality bundle."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    tensor_rows = {}
    plot_rows = []
    for case in cases:
        operator = operator_factory(case)
        solver = solver_factory(case)
        evaluation = evaluate_ct_case(solver, operator, case)
        metrics = dict(evaluation.metrics)
        metrics.update(_roi_metrics(case, evaluation.reconstruction))
        record = {
            "case_id": case.case_id,
            "geometry": {
                "views": int(case.measurement.shape[-2]),
                "detectors": int(case.measurement.shape[-1]),
                "angular_coverage_deg": float(case.geometry["angular_coverage_deg"]),
            },
            "noise_model": case.metadata["measurement"]["noise_model"],
            "reference_kind": case.metadata["provenance"]["reference_kind"],
            "reference_generator": case.metadata["provenance"]["generator"],
            "metrics": metrics,
        }
        summary_rows.append(record)
        tensor_rows[case.case_id] = {
            "truth": case.truth.detach().cpu(),
            "measurement": case.measurement.detach().cpu(),
            "reconstruction": evaluation.reconstruction.detach().cpu(),
        }
        plot_rows.append((case, evaluation.reconstruction, metrics))

    summary = {
        "schema_version": 1,
        "benchmark_kind": "synthetic_quality",
        "data_statement": "Synthetic tissue-like phantom; not patient data.",
        "inverse_crime_control": (
            "ASTRA line-projector measurements are reconstructed with the "
            "invframework grid-sample ParallelBeamRadon2D operator."
        ),
        "cases": summary_rows,
    }
    (destination / "synthetic_quality.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(tensor_rows, destination / "synthetic_quality.pt")
    _plot_quality_summary(destination / "synthetic_quality.png", plot_rows)
    return summary
