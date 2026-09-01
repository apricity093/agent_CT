from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from inv_framework.cli import main
from inv_framework.ct_runtime import SOLVER_SPECS, _load_result, run_suite


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "test" / "data"
CASE_ID = "parallel_2d/disk_analytic_32"


def _yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _algorithm(tmp_path: Path, name: str, parameters: dict | None = None) -> Path:
    return _yaml(
        tmp_path / f"{name}.yaml",
        {"schema_version": 1, "name": name, "parameters": parameters or {}},
    )


def _protocol(tmp_path: Path, *, psnr_min: float = 0.0, min_records: int = 1) -> Path:
    return _yaml(
        tmp_path / f"protocol_{psnr_min}_{min_records}.yaml",
        {
            "schema_version": 1,
            "name": "test",
            "expected_statuses": ["success"],
            "min_records": min_records,
            "required_metrics": [
                "relative_error",
                "rmse",
                "psnr",
                "ssim",
                "data_residual",
                "runtime_seconds",
            ],
            "thresholds": {
                "psnr": {"min": psnr_min},
                "ssim": {"min": -1.0, "max": 1.0},
            },
        },
    )


def test_help_and_registry_are_stable(capsys):
    assert main([]) == 0
    assert "list-solvers" in capsys.readouterr().out
    assert main(["list-solvers", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [record["name"] for record in payload] == [
        "fbp",
        "sirt",
        "landweber",
        "cgls",
        "lsqr",
        "sart",
        "os_sart",
        "mlem",
        "osem",
        "tikhonov",
        "tv_fista",
        "fdk",
    ]
    assert tuple(SOLVER_SPECS) == tuple(record["name"] for record in payload)


def test_data_list_show_and_validate(capsys):
    assert main(["data", "list", "--root", str(DATA_ROOT), "--dimension", "2", "--tag", "analytic", "--json"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert [record["case_id"] for record in records] == [CASE_ID]
    assert main(["data", "show", CASE_ID, "--root", str(DATA_ROOT)]) == 0
    assert json.loads(capsys.readouterr().out)["geometry_type"] == "parallel_2d"
    assert main(["data", "validate", CASE_ID, "--root", str(DATA_ROOT)]) == 0
    assert "valid" in capsys.readouterr().out


def test_algorithm_config_is_strict(tmp_path, capsys):
    config = _yaml(
        tmp_path / "bad.yaml",
        {"schema_version": 1, "name": "fbp", "parameters": {}, "unexpected": True},
    )
    assert main(["run", "fbp", "--case", CASE_ID, "--config", str(config), "--out", str(tmp_path / "run")]) == 2
    assert "unknown field" in capsys.readouterr().err
    assert (tmp_path / "run" / "failure_report.md").is_file()

    wrong_type = _algorithm(tmp_path, "sirt", {"num_iterations": "25"})
    assert main(["run", "sirt", "--case", CASE_ID, "--config", str(wrong_type), "--out", str(tmp_path / "wrong_type")]) == 2
    assert "must be an integer" in capsys.readouterr().err


def test_run_eval_and_overwrite_contract(tmp_path, capsys):
    config = _algorithm(tmp_path, "fbp")
    run_dir = tmp_path / "run"
    args = ["run", "fbp", "--case", CASE_ID, "--config", str(config), "--out", str(run_dir), "--data-root", str(DATA_ROOT)]
    assert main(args) == 0
    capsys.readouterr()
    expected = {"reconstruction.pt", "metrics.json", "manifest.json", "comparison.png", "artifacts.sha256"}
    assert expected.issubset(path.name for path in run_dir.iterdir())
    bundle = _load_result(run_dir / "reconstruction.pt")
    assert bundle["reconstruction"].shape == bundle["truth"].shape == (1, 1, 32, 32)
    assert bundle["measurement"].shape == bundle["predicted_measurement"].shape
    assert torch.isfinite(bundle["reconstruction"]).all()

    assert main(args) == 2
    assert "not empty" in capsys.readouterr().err
    assert main(args + ["--overwrite"]) == 0
    capsys.readouterr()

    passing = _protocol(tmp_path, psnr_min=10.0)
    assert main(["eval", "--run", str(run_dir), "--protocol", str(passing)]) == 0
    capsys.readouterr()
    assert json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))["passed"] is True
    assert "结果：**通过**" in (run_dir / "evaluation.md").read_text(encoding="utf-8")

    failing = _protocol(tmp_path, psnr_min=1000.0)
    assert main(["eval", "--run", str(run_dir), "--protocol", str(failing)]) == 1
    capsys.readouterr()
    assert json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))["passed"] is False


def test_geometry_mismatch_and_fdk_backend_gate(tmp_path, capsys):
    fdk = _algorithm(tmp_path, "fdk")
    mismatch = tmp_path / "mismatch"
    assert main(["run", "fdk", "--case", CASE_ID, "--config", str(fdk), "--out", str(mismatch), "--data-root", str(DATA_ROOT)]) == 2
    assert "supports dimension" in capsys.readouterr().err

    unavailable = tmp_path / "unavailable"
    assert main(["run", "fdk", "--case", "cone_3d/spheres_astra_12", "--config", str(fdk), "--out", str(unavailable), "--data-root", str(DATA_ROOT), "--device", "cpu"]) == 1
    capsys.readouterr()
    failure = json.loads((unavailable / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "unavailable"
    assert "requires --device cuda" in failure["message"]


def test_small_benchmark_suite(tmp_path, capsys):
    fbp = _algorithm(tmp_path, "fbp")
    sirt = _algorithm(tmp_path, "sirt", {"num_iterations": 2})
    protocol = _protocol(tmp_path, psnr_min=-100.0, min_records=2)
    output_root = tmp_path / "suite_output"
    suite = _yaml(
        tmp_path / "suite.yaml",
        {
            "schema_version": 1,
            "name": "small",
            "data_root": str(DATA_ROOT),
            "output_root": str(output_root),
            "device": "cpu",
            "continue_on_error": True,
            "protocol": str(protocol),
            "groups": [
                {
                    "algorithms": [
                        {"name": "fbp", "config": str(fbp)},
                        {"name": "sirt", "config": str(sirt)},
                    ],
                    "cases": [CASE_ID],
                }
            ],
        },
    )
    assert main(["bench", "--suite", str(suite)]) == 0
    capsys.readouterr()
    metrics = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["records"]) == 2
    assert {record["solver"] for record in metrics["records"]} == {"fbp", "sirt"}
    assert json.loads((output_root / "evaluation.json").read_text(encoding="utf-8"))["passed"] is True
    assert (output_root / "metrics.csv").is_file()
    assert (output_root / "report.md").is_file()
    assert (output_root / "artifacts.sha256").is_file()


def test_equal_call_suite_keeps_budget_failures_as_records(tmp_path):
    fbp = _algorithm(tmp_path, "fbp")
    sirt = _algorithm(tmp_path, "sirt", {"num_iterations": 1})
    output_root = tmp_path / "equal_output"
    suite = _yaml(
        tmp_path / "equal_suite.yaml",
        {
            "schema_version": 1,
            "name": "equal",
            "data_root": str(DATA_ROOT),
            "output_root": str(output_root),
            "device": "cpu",
            "benchmark_protocol": "equal_operator_calls",
            "budget": {
                "protocol": "equal_operator_calls",
                "max_forward_calls": 2,
                "max_adjoint_calls": 2,
            },
            "groups": [{
                "algorithms": [
                    {"name": "fbp", "config": str(fbp)},
                    {"name": "sirt", "config": str(sirt)},
                ],
                "cases": [CASE_ID],
            }],
        },
    )

    result = run_suite(suite)
    records = {record["solver"]: record for record in result["records"]}
    assert records["fbp"]["status"] == "success"
    assert records["sirt"]["status"] == "resource_exhausted"
    assert records["sirt"]["failure_reason"]
    assert records["sirt"]["tuning_protocol"] == "equal_operator_calls"


def test_heldout_suite_materializes_shared_fold_hash(tmp_path):
    fbp = _algorithm(tmp_path, "fbp")
    output_root = tmp_path / "heldout_output"
    suite = _yaml(
        tmp_path / "heldout_suite.yaml",
        {
            "schema_version": 1,
            "name": "heldout",
            "data_root": str(DATA_ROOT),
            "output_root": str(output_root),
            "device": "cpu",
            "groups": [{
                "heldout_split": {"folds": 3, "protocol_version": "heldout_projection_cv/v1"},
                "algorithms": [{"name": "fbp", "config": str(fbp)}],
                "cases": [CASE_ID],
            }],
        },
    )

    result = run_suite(suite)
    records = result["records"]
    assert len(records) == 3
    assert {record["split_fold"] for record in records} == {0, 1, 2}
    assert len({record["split_sha256"] for record in records}) == 1
    assert all(record.get("held_out_projection_residual") is not None for record in records)
    assert set(result["comparison"]["fairness"]) == {
        f"{CASE_ID}::fold_0",
        f"{CASE_ID}::fold_1",
        f"{CASE_ID}::fold_2",
    }


def test_oracle_suite_calibrates_on_disjoint_development_case(tmp_path):
    sirt = _algorithm(tmp_path, "sirt", {"num_iterations": 1})
    output_root = tmp_path / "oracle_output"
    suite = _yaml(
        tmp_path / "oracle_suite.yaml",
        {
            "schema_version": 1,
            "name": "oracle",
            "data_root": str(DATA_ROOT),
            "output_root": str(output_root),
            "device": "cpu",
            "benchmark_protocol": "oracle_calibration",
            "budget": {
                "protocol": "oracle_calibration",
                "tuning_trials": 2,
                "tuning_max_forward_calls": 1000,
                "tuning_max_adjoint_calls": 1000,
            },
            "groups": [{
                "calibration_cases": ["parallel_2d/shepp_logan_dense_clean_32"],
                "oracle_metric": "psnr",
                "algorithms": [{
                    "name": "sirt",
                    "config": str(sirt),
                    "parameter_grid": [
                        {"num_iterations": 1},
                        {"num_iterations": 2},
                    ],
                }],
                "cases": [CASE_ID],
            }],
        },
    )

    result = run_suite(suite)
    assert len(result["calibration_records"]) == 2
    assert len(result["records"]) == 1
    assert all(record["oracle_phase"] == "calibration" for record in result["calibration_records"])
    assert result["oracle_selection"]["group_0/sirt"]["selected_trial_index"] in {0, 1}
    assert result["records"][0]["oracle_phase"] == "final"
    assert result["records"][0]["parameter_source"] == "oracle_calibration_development_only"
    benchmark = json.loads((output_root / "benchmark.json").read_text(encoding="utf-8"))
    assert len(benchmark["calibration_records"]) == 2
    assert (output_root / "calibration_metrics.csv").is_file()
