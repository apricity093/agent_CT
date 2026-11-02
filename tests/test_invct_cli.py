from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from inv_framework.cli import main
from inv_framework.api.execution import SOLVER_SPECS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "tests" / "data"
CASE_ID = "parallel_2d__ellipses_dense_clean"


def _yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _algorithm(tmp_path: Path, name: str, parameters: dict | None = None) -> Path:
    return _yaml(
        tmp_path / f"{name}.yaml",
        {"schema_version": 1, "name": name, "parameters": parameters or {}},
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
        "fista",
        "admm",
        "pdhg",
        "constrained_pdhg",
        "asd_pocs_tv",
        "fdk",
    ]
    assert tuple(SOLVER_SPECS) == tuple(record["name"] for record in payload)


def test_data_list_show_and_validate(capsys):
    assert main(["data", "list", "--root", str(DATA_ROOT), "--dimension", "2", "--tag", "dense_clean", "--json"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert [record["case_id"] for record in records] == [CASE_ID]
    assert main(["data", "show", CASE_ID, "--root", str(DATA_ROOT)]) == 0
    assert json.loads(capsys.readouterr().out)["geometry_type"] == "parallel_2d"
    assert main(["data", "validate", CASE_ID, "--root", str(DATA_ROOT)]) == 0
    assert "valid" in capsys.readouterr().out


@pytest.mark.parametrize("as_json", [False, True])
def test_data_list_without_matching_cases(capsys, as_json):
    args = ["data", "list", "--root", str(DATA_ROOT), "--tag", "quality"]
    if as_json:
        args.append("--json")
    assert main(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if as_json:
        assert json.loads(captured.out) == []
    else:
        lines = captured.out.splitlines()
        assert lines[0].split() == ["case_id", "dimension", "geometry_type", "noise_model"]
        assert len(lines) == 2


def test_fbp_cli_writes_legacy_outputs(tmp_path, capsys):
    output = tmp_path / "fbp"
    assert main(["run", "fbp", "--case", CASE_ID,
                 "--config", str(PROJECT_ROOT / "configs/algorithms/fbp.yaml"),
                 "--out", str(output), "--device", "cpu"]) == 0
    capsys.readouterr()
    diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["iterations_completed"] == 0
    assert (output / "reconstruction.pt").is_file()


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
