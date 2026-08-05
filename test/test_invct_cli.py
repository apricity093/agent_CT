from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from inv_framework.cli import main
from inv_framework.ct_runtime import SOLVER_SPECS, _load_result


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
