"""Command-line interface for CT reconstruction and benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from inv_framework.benchmarks import list_ct_cases
from inv_framework.ct_runtime import (
    ConfigError,
    SOLVER_SPECS,
    evaluate_run,
    run_case,
    run_suite,
    solver_records,
    validate_cases,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="invct", description="inv_framework CT reconstruction CLI.")
    commands = parser.add_subparsers(dest="command")

    solvers = commands.add_parser("list-solvers", help="List runnable traditional CT solvers.")
    solvers.add_argument("--json", action="store_true", dest="as_json")

    data = commands.add_parser("data", help="Inspect and validate the CT case catalog.")
    data_commands = data.add_subparsers(dest="data_command")
    data_list = data_commands.add_parser("list", help="List catalog cases.")
    data_list.add_argument("--root")
    data_list.add_argument("--tag", action="append", default=[])
    data_list.add_argument("--dimension", type=int, choices=(2, 3))
    data_list.add_argument("--geometry")
    data_list.add_argument("--json", action="store_true", dest="as_json")
    data_show = data_commands.add_parser("show", help="Show one catalog record.")
    data_show.add_argument("case_id")
    data_show.add_argument("--root")
    data_validate = data_commands.add_parser("validate", help="Load cases and verify checksums.")
    data_validate.add_argument("case_id", nargs="?")
    data_validate.add_argument("--root")

    run = commands.add_parser("run", help="Run one solver on one catalog case.")
    run.add_argument("solver", choices=tuple(SOLVER_SPECS))
    run.add_argument("--case", required=True, dest="case_id")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--data-root")
    run.add_argument("--overwrite", action="store_true")

    evaluate = commands.add_parser("eval", help="Evaluate a saved run without rerunning its solver.")
    evaluate.add_argument("--run", required=True, dest="run_dir")
    evaluate.add_argument("--protocol", required=True)

    bench = commands.add_parser("bench", help="Run a YAML benchmark suite.")
    bench.add_argument("--suite", required=True)
    return parser


def _print_records(records: list[dict], columns: tuple[str, ...]) -> None:
    widths = {name: max(len(name), *(len(str(record.get(name, ""))) for record in records)) for name in columns}
    print("  ".join(name.ljust(widths[name]) for name in columns))
    print("  ".join("-" * widths[name] for name in columns))
    for record in records:
        print("  ".join(str(record.get(name, "")).ljust(widths[name]) for name in columns))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "list-solvers":
            records = solver_records()
            if args.as_json:
                print(json.dumps(records, indent=2))
            else:
                printable = [{"name": item["name"], "dimension": ",".join(map(str, item["dimensions"])), "geometry": ",".join(item["geometry_types"]), "backend": item["backend"] or "built-in"} for item in records]
                _print_records(printable, ("name", "dimension", "geometry", "backend"))
            return 0
        if args.command == "data":
            if args.data_command is None:
                parser.parse_args(["data", "--help"])
                return 0
            if args.data_command == "list":
                filters = {}
                if args.tag:
                    filters["tags"] = args.tag
                if args.dimension is not None:
                    filters["dimension"] = args.dimension
                if args.geometry:
                    filters["geometry_type"] = args.geometry
                records = list_ct_cases(filters, data_root=args.root)
                if args.as_json:
                    print(json.dumps(records, indent=2))
                else:
                    _print_records(records, ("case_id", "dimension", "geometry_type", "noise_model"))
                return 0
            if args.data_command == "show":
                records = list_ct_cases(data_root=args.root)
                matches = [record for record in records if record["case_id"] == args.case_id]
                if not matches:
                    raise ConfigError(f"unknown CT case: {args.case_id!r}")
                print(json.dumps(matches[0], indent=2))
                return 0
            results = validate_cases(args.case_id, data_root=args.root)
            _print_records(results, ("case_id", "status", "truth_shape", "measurement_shape"))
            return 0
        if args.command == "run":
            result = run_case(args.solver, args.case_id, args.config, args.out, device=args.device, data_root=args.data_root, overwrite=args.overwrite)
            print(result["output_dir"])
            return 0 if result["status"] == "success" else 1
        if args.command == "eval":
            result = evaluate_run(args.run_dir, args.protocol)
            print(f"{args.run_dir}/evaluation.md")
            return 0 if result["passed"] else 1
        if args.command == "bench":
            result = run_suite(args.suite)
            print(result["output_root"])
            return 0 if result["evaluation"]["passed"] else 1
    except (ConfigError, FileNotFoundError, KeyError, ValueError) as error:
        print(f"invct: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"invct: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
