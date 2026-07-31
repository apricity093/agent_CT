from __future__ import annotations

import json

import pytest
import torch

from .baselines import CALIBRATION_ENVIRONMENT
from .helpers import jsonable_metrics, run_optional_astra_fdk_benchmark, run_solver_2d


@pytest.mark.benchmark
def test_write_ct_benchmark_artifacts(
    ct_case_2d,
    ct_solvers_2d,
    benchmark_artifact_dir,
):
    metrics = {}
    reconstructions = {}
    for name, solver in ct_solvers_2d.items():
        reconstruction, algorithm_metrics = run_solver_2d(name, solver, ct_case_2d)
        metrics[name] = jsonable_metrics(algorithm_metrics)
        reconstructions[name] = reconstruction.detach().cpu()

    fdk_reconstruction, fdk_metrics = run_optional_astra_fdk_benchmark()
    metrics["fdk"] = jsonable_metrics(fdk_metrics)
    if fdk_reconstruction is not None:
        torch.save(fdk_reconstruction, benchmark_artifact_dir / "fdk_reconstruction.pt")

    summary = {
        "schema_version": 1,
        "case": {
            "image_shape": list(ct_case_2d["truth"].shape[1:]),
            "measurement_shape": list(ct_case_2d["measurement"].shape[1:]),
            "noise_fraction": ct_case_2d["noise_fraction"],
            "seed": ct_case_2d["seed"],
        },
        "algorithms": metrics,
        "calibration_environment": CALIBRATION_ENVIRONMENT,
        "geometry_note": "FDK uses a separate 3D cone-beam geometry and is never compared pointwise to 2D results.",
    }
    (benchmark_artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (benchmark_artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    torch.save(
        {"truth": ct_case_2d["truth"].cpu(), "reconstructions": reconstructions},
        benchmark_artifact_dir / "reconstruction.pt",
    )

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt is not None:
        names = list(reconstructions)
        figure, axes = plt.subplots(2, 4, figsize=(10, 5))
        panels = [("truth", ct_case_2d["truth"].cpu())] + [
            (name, reconstructions[name]) for name in names
        ]
        for axis, (name, image) in zip(axes.flat, panels):
            axis.imshow(image[0, 0], cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(name)
            axis.axis("off")
        for axis in axes.flat[len(panels) :]:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(benchmark_artifact_dir / "reconstruction.png", dpi=120)
        plt.close(figure)

    assert (benchmark_artifact_dir / "metrics.json").exists()
    assert (benchmark_artifact_dir / "summary.json").exists()
    assert (benchmark_artifact_dir / "reconstruction.pt").exists()
