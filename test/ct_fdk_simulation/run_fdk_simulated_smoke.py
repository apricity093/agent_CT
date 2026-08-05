"""Command-line entry point for the runtime-generated cone-beam FDK smoke case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .simulation import FDKSimulationConfig, run_simulated_fdk, validate_result, write_artifacts
except ImportError:
    from simulation import FDKSimulationConfig, run_simulated_fdk, validate_result, write_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a clean 3D modified Shepp-Logan cone-beam case and run FDK."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory that will receive small diagnostics and reconstruction slices.",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device used by ASTRA FDK.")
    args = parser.parse_args()

    result = run_simulated_fdk(FDKSimulationConfig(), device=args.device)
    validate_result(result)
    metrics = write_artifacts(result, args.output_dir)
    print(
        "FDK simulated smoke succeeded: "
        f"truth={tuple(result.truth.shape)} measurement={tuple(result.measurement.shape)} "
        f"reconstruction={tuple(result.reconstruction.shape)} runtime={metrics['runtime_seconds']:.3f}s"
    )


if __name__ == "__main__":
    main()
