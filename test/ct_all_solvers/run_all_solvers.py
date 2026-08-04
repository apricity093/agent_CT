"""Command-line entry point for the traditional CT solver benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import BenchmarkConfig, run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every traditional inv_framework CT reconstruction solver."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive metrics, tensors, figures, and manifest.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda.")
    parser.add_argument(
        "--profile",
        default=None,
        help="Existing calibration_profile.json to reuse instead of recalibrating.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate parameters on the held-out 64x64 pilot before formal cases.",
    )
    parser.add_argument(
        "--require-fdk",
        action="store_true",
        help="Fail when the ASTRA CUDA FDK backend is unavailable.",
    )
    parser.add_argument(
        "--skip-fdk",
        action="store_true",
        help="Skip the optional local FDK branch; not valid for the required remote run.",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    if args.profile and args.calibrate:
        parser.error("--profile and --calibrate are mutually exclusive")
    run = run_benchmark(
        BenchmarkConfig(
            output_dir=args.output_dir,
            device=args.device,
            profile_path=args.profile,
            calibrate=args.calibrate,
            require_fdk=args.require_fdk,
            include_fdk=not args.skip_fdk,
            seed=args.seed,
        )
    )
    successful = [record for record in run.records if record["status"] == "success"]
    unavailable = [record for record in run.records if record["status"] != "success"]
    print(f"wrote {len(run.records)} records to {run.output_dir}")
    print(f"successful={len(successful)} unavailable={len(unavailable)}")
    for record in unavailable:
        print(f"{record['algorithm']}: {record.get('reason', record['status'])}")


if __name__ == "__main__":
    main()
