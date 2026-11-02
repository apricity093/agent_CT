"""File output and suite support for established CLI callers."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from inv_framework.api.execution import (
    ConfigError,
    InvalidParametersError,
    NumericalFailure,
    RESULT_FILENAME,
    SCHEMA_VERSION,
    _check_keys,
    _failure,
    _metrics,
    _require_mapping,
    _structured_failure,
    _write_json,
    execute_case,
    list_ct_cases,
    load_algorithm_config,
    load_ct_case,
    load_yaml,
)

def _load_parameter_sources(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"parameter source file does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot parse parameter source file {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ConfigError("parameter source file must contain a JSON object")
    return {str(key): str(value) for key, value in payload.items()}



def _load_stopping_policy(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ConfigError("stopping policy must be a JSON object")
    if payload.get("schema_version") != "1.0" or payload.get("mode") != "solver_native_and_endpoint":
        raise ConfigError("unsupported stopping policy schema_version or mode")
    return dict(payload)



def _prepare_output(path: str | Path, overwrite: bool) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ConfigError(f"output path must be a directory and not a symlink: {destination}")
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise ConfigError(f"output directory is not empty: {destination}; use --overwrite")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    return destination



def _plot_comparison(
    path: Path,
    truth: torch.Tensor,
    reconstruction: torch.Tensor,
    dimension: int,
    *,
    truth_available: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    reconstruction_image = reconstruction[0, 0] if dimension == 2 else reconstruction[0, reconstruction.shape[1] // 2]
    if truth_available:
        truth_image = truth[0, 0] if dimension == 2 else truth[0, truth.shape[1] // 2]
        images = (truth_image, reconstruction_image)
        titles = ("Reference", "Reconstruction")
    else:
        images = (reconstruction_image,)
        titles = ("Reconstruction (truth withheld)",)
    vmin = float(torch.stack([image.min() for image in images]).min().item())
    vmax = float(torch.stack([image.max() for image in images]).max().item())
    if vmin == vmax:
        vmax = vmin + 1.0
    figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), squeeze=False, constrained_layout=True)
    for axis, image, title in zip(axes[0], images, titles):
        axis.imshow(image.detach().cpu().numpy(), cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)



def run_case(
    solver_name: str,
    case_id: str,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    data_root: str | Path | None = None,
    overwrite: bool = False,
    observation_domain: str | None = None,
    problem_constraints: Mapping[str, Any] | None = None,
    max_iterations: int | None = None,
    parameter_sources_path: str | Path | None = None,
    stopping_policy_path: str | Path | None = None,
    stopping_policy: Mapping[str, Any] | None = None,
    parameter_overrides: Mapping[str, Any] | None = None,
    parameter_sources: Mapping[str, str] | None = None,
    fit_view_indices: Sequence[int] | None = None,
    heldout_view_indices: Sequence[int] | None = None,
    split_metadata: Mapping[str, Any] | None = None,
    fixed_compute: bool = False,
) -> dict[str, Any]:
    """Run the shared core and write the established CLI artifacts."""
    destination = _prepare_output(output_dir, overwrite)
    try:
        name, parameters, _ = load_algorithm_config(config_path, solver_name)
        sources = _load_parameter_sources(parameter_sources_path)
        sources.update(parameter_sources or {})
        result = execute_case(
            name, case_id, parameters, device=device, data_root=data_root,
            observation_domain=observation_domain, problem_constraints=problem_constraints,
            max_iterations=max_iterations,
            stopping_policy=stopping_policy if stopping_policy is not None else _load_stopping_policy(stopping_policy_path),
            parameter_overrides=parameter_overrides, parameter_sources=sources,
            fit_view_indices=fit_view_indices, heldout_view_indices=heldout_view_indices,
            split_metadata=split_metadata, fixed_compute=fixed_compute,
        )
    except InvalidParametersError as error:
        return _structured_failure(destination, status="invalid_parameters", error=error, algorithm=solver_name, validation=getattr(error, "validation", None))
    except Exception as error:
        _failure(destination, "failed", error)
        raise
    result["output_dir"] = destination
    partial = result.pop("partial_reconstruction", None)
    if partial is not None:
        torch.save(partial, destination / "partial_reconstruction.pt")
    bundle = result.pop("bundle", None)
    if bundle is not None:
        torch.save(bundle, destination / RESULT_FILENAME)
        _write_json(destination / "metrics.json", result["metrics"])
        _plot_comparison(destination / "comparison.png", bundle["truth"], bundle["reconstruction"], bundle["dimension"], truth_available=bundle["ground_truth_available"])
    if "diagnostics" in result:
        _write_json(destination / "diagnostics.json", result["diagnostics"])
    if result["status"] != "success":
        _failure(destination, result["status"], NumericalFailure(result.get("message", result.get("stopping_reason", "execution failed"))), algorithm=solver_name)
    return result



def _load_result(path: Path) -> dict[str, Any]:
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    required = {"solver", "case_id", "dimension", "data_range", "reconstruction", "truth", "measurement", "predicted_measurement"}
    if not isinstance(bundle, Mapping) or not required.issubset(bundle):
        raise ConfigError(f"invalid reconstruction artifact: {path}")
    return dict(bundle)



def load_protocol(path: str | Path) -> tuple[dict[str, Any], Path]:
    payload, source = load_yaml(path)
    _check_keys(payload, {"schema_version", "name", "expected_statuses", "required_metrics", "thresholds", "min_records"}, "protocol")
    if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ConfigError(f"protocol schema_version must be {SCHEMA_VERSION}.")
    statuses = payload.get("expected_statuses", ["success"])
    required = payload.get("required_metrics", ["relative_error", "rmse", "psnr", "ssim", "data_residual"])
    thresholds = _require_mapping(payload.get("thresholds", {}), "protocol thresholds")
    if not isinstance(statuses, list) or not all(isinstance(v, str) for v in statuses):
        raise ConfigError("protocol expected_statuses must be a list of strings.")
    if not isinstance(required, list) or not all(isinstance(v, str) for v in required):
        raise ConfigError("protocol required_metrics must be a list of strings.")
    normalized: dict[str, dict[str, float]] = {}
    for metric, limits in thresholds.items():
        limits = _require_mapping(limits, f"threshold {metric}")
        _check_keys(limits, {"min", "max"}, f"threshold {metric}")
        if not limits:
            raise ConfigError(f"threshold {metric} must define min or max.")
        normalized[str(metric)] = {key: float(value) for key, value in limits.items()}
    payload["expected_statuses"] = statuses
    payload["required_metrics"] = required
    payload["thresholds"] = normalized
    payload["min_records"] = int(payload.get("min_records", 1))
    return payload, source



def evaluate_metrics(records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    allowed = set(protocol["expected_statuses"])
    for index, original in enumerate(records):
        record = dict(original)
        failures: list[str] = []
        if record.get("status") not in allowed:
            failures.append(f"status {record.get('status')!r} not in {sorted(allowed)}")
        for metric in protocol["required_metrics"]:
            if record.get(metric) is None:
                failures.append(f"missing metric {metric}")
        for metric, limits in protocol["thresholds"].items():
            value = record.get(metric)
            if value is None:
                failures.append(f"missing threshold metric {metric}")
                continue
            numeric = float(value)
            if "min" in limits and numeric < limits["min"]:
                failures.append(f"{metric}={numeric:.6g} < {limits['min']:.6g}")
            if "max" in limits and numeric > limits["max"]:
                failures.append(f"{metric}={numeric:.6g} > {limits['max']:.6g}")
        checks.append({"index": index, "solver": record.get("solver"), "case_id": record.get("case_id"), "passed": not failures, "failures": failures})
    minimum_met = len(records) >= int(protocol["min_records"])
    return {"schema_version": SCHEMA_VERSION, "passed": bool(records) and minimum_met and all(item["passed"] for item in checks), "record_count": len(records), "minimum_records_met": minimum_met, "checks": checks}



def evaluate_run(run_dir: str | Path, protocol_path: str | Path) -> dict[str, Any]:
    directory = Path(run_dir).expanduser().resolve()
    bundle = _load_result(directory / RESULT_FILENAME)
    runtime = None
    metrics_path = directory / "metrics.json"
    if metrics_path.is_file():
        runtime = json.loads(metrics_path.read_text(encoding="utf-8")).get(
            "runtime_seconds"
        )
    metrics = _metrics(bundle, runtime)
    protocol, source = load_protocol(protocol_path)
    evaluation = evaluate_metrics([metrics], protocol)
    evaluation.update({"protocol": str(source), "metrics": metrics})
    _write_json(directory / "evaluation.json", evaluation)
    status = "通过" if evaluation["passed"] else "未通过"
    lines = [
        "# CT 重建评估",
        "",
        f"- 结果：**{status}**",
        f"- Solver：`{metrics['solver']}`",
        f"- Case：`{metrics['case_id']}`",
        "",
        "## 指标",
        "",
    ]
    for name in (
        "relative_error",
        "rmse",
        "psnr",
        "ssim",
        "data_residual",
        "runtime_seconds",
    ):
        if name in metrics:
            lines.append(f"- `{name}`：{metrics[name]:.8g}")
    failures = evaluation["checks"][0]["failures"]
    if failures:
        lines.extend(["", "## 未通过原因", ""] + [f"- {value}" for value in failures])
    (directory / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return evaluation



def _resolve_relative(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()



def validate_cases(
    case_id: str | None = None, *, data_root: str | Path | None = None
) -> list[dict[str, Any]]:
    records = list_ct_cases(data_root=data_root)
    if case_id is not None:
        records = [record for record in records if record["case_id"] == case_id]
        if not records:
            raise ConfigError(f"unknown CT case: {case_id!r}")
    results = []
    for record in records:
        loaded = load_ct_case(record["case_id"], data_root=data_root)
        results.append(
            {
                "case_id": loaded.case_id,
                "status": "valid",
                "truth_shape": list(loaded.truth.shape),
                "measurement_shape": list(loaded.measurement.shape),
            }
        )
    return results



def load_suite(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a fixed-parameter benchmark suite."""
    suite, source = load_yaml(path)
    if suite.get("benchmark_protocol", "fixed_defaults") not in {
        "fixed_defaults",
        "fixed_defaults/v1",
    }:
        raise ConfigError("Only fixed_defaults suites are supported")
    if not suite.get("groups"):
        raise ConfigError("suite requires groups")
    return suite, source



def run_suite(path: str | Path) -> dict[str, Any]:
    suite, source = load_suite(path)
    base = source.parent
    destination = _prepare_output(
        _resolve_relative(base, suite["output_root"]), suite.get("overwrite", False)
    )
    root = (
        _resolve_relative(base, suite["data_root"]) if suite.get("data_root") else None
    )
    records = []
    for group in suite["groups"]:
        if group.get("budget") or group.get(
            "benchmark_protocol", "fixed_defaults"
        ) not in {"fixed_defaults", "fixed_defaults/v1"}:
            raise ConfigError(
                "Operator-call billing and tuning protocols are no longer supported"
            )
        for algorithm in group["algorithms"]:
            for case_id in group["cases"]:
                try:
                    result = run_case(
                        algorithm["name"],
                        case_id,
                        _resolve_relative(base, algorithm["config"]),
                        destination / algorithm["name"] / case_id.replace("/", "__"),
                        device=suite.get("device", "cpu"),
                        data_root=root,
                        observation_domain=group.get("observation_domain"),
                        problem_constraints=group.get("problem_constraints"),
                        max_iterations=group.get("max_iterations"),
                    )
                except Exception as error:
                    if not suite.get("continue_on_error", False):
                        raise
                    result = {"status": "failed", "message": str(error)}
                records.append(
                    {
                        "solver": algorithm["name"],
                        "case_id": case_id,
                        "status": result["status"],
                        "message": result.get("message"),
                        **result.get("metrics", {}),
                    }
                )
    evaluation = (
        evaluate_metrics(
            records, load_protocol(_resolve_relative(base, suite["protocol"]))[0]
        )
        if suite.get("protocol")
        else {"passed": all(r["status"] == "success" for r in records)}
    )
    _write_json(
        destination / "benchmark.json", {"records": records, "evaluation": evaluation}
    )
    return {"output_root": destination, "records": records, "evaluation": evaluation}
