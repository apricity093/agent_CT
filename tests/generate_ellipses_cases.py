"""Convert ct-ellipses-new-20261101 reference images into CT cases."""

import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inv_framework.benchmarks import CTTestCase, write_ct_case
from inv_framework.ct.measurement import simulate_observation
from inv_framework.ct.backends import ParallelBeamRadon2D


def generate(archive: Path, data_root: Path, sample: int = 0):
    """Use a fixed test image for four matched acquisition scenarios."""
    source = data_root / "reference_images"
    source.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        prefix = "ct-ellipses-new-20261101/"
        for split in ("train", "val", "test"):
            array = np.load(
                io.BytesIO(bundle.extractfile(prefix + split + ".npy").read())
            )
            np.save(source / (split + ".npy"), array)
    # The source images are unitless; a fixed attenuation scale keeps line
    # integrals in a useful range for photon-count experiments.
    truth = (
        torch.from_numpy(np.load(source / "test.npy")[sample]).float()[None, None]
        * 0.02
    )
    size = truth.shape[-1]
    records = []
    catalog = data_root / "catalog.json"
    if catalog.exists():
        records = [
            r
            for r in json.loads(catalog.read_text())["cases"]
            if r["geometry_type"] == "cone_3d"
        ]
    scenarios = (
        ("dense_clean", 180, 180),
        ("sparse_gaussian", 32, 180),
        ("sparse_poisson", 32, 180),
        ("limited_angle", 60, 120),
    )
    for index, (name, views, degrees) in enumerate(scenarios):
        angles = torch.arange(views, dtype=torch.float32) * (
            degrees * torch.pi / 180 / views
        )
        operator = ParallelBeamRadon2D(image_size=size, angles=angles, device="cpu")
        clean = operator.forward(truth).detach()
        observed, public = clean.clone(), {}
        measurement = dict(
            kind="line_integral", observation_domain="line_integral", noise_model="none"
        )
        seed = 20261101 + index
        if name == "sparse_gaussian":
            observed, public, measurement = simulate_observation(
                clean,
                model="gaussian",
                seed=seed,
                sigma=float(clean.square().mean().sqrt()) * 0.01,
            )
        elif name == "sparse_poisson":
            observed, public, measurement = simulate_observation(
                clean, model="poisson", seed=seed, I0=100000.0
            )
        measurement["kind"] = "line_integral"
        case_id = "parallel_2d__ellipses_" + name
        geometry = dict(
            type="parallel_2d",
            domain_shape=[1, size, size],
            range_shape=[1, views, size],
            angles_rad=angles.tolist(),
            detector_count=size,
            detector_spacing=2 / size,
            image_shape=[size, size],
            image_layout=["channel", "y", "x"],
            measurement_layout=["channel", "angle", "detector"],
            length_unit="normalized_image_coordinate",
        )
        metadata = dict(
            dimension=2,
            modality="xray_ct",
            measurement=measurement,
            ground_truth=dict(
                quantity="linear_attenuation",
                unit="normalized_inverse_length",
                data_range=0.02,
            ),
            provenance=dict(
                dataset="ct-ellipses-new-20261101",
                split="test",
                sample_index=sample,
                attenuation_scale=0.02,
                reference_kind="model_matched",
                generator="inv_framework.ct.backends.ParallelBeamRadon2D",
            ),
            capability_tags=["2d", "parallel", "ground_truth", name],
        )
        case = CTTestCase(
            case_id=case_id,
            truth=truth,
            measurement_clean=clean,
            measurement=observed,
            geometry=geometry,
            metadata=metadata,
            roi_mask=truth > 0,
            public_measurement=public,
        )
        record = write_ct_case(case, data_root / "cases" / case_id, overwrite=True)
        record["path"] = "cases/" + case_id
        records.append(record)
    catalog.write_text(
        json.dumps({"schema_version": "1.0", "cases": records}, indent=2) + "\n"
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "tests/data")
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    generate(args.archive, args.data_root, args.sample)
