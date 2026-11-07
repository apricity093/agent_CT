"""Framework-independent file references, discovery resources and observations."""
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any
import numpy as np
import yaml


def read_array(value, base_dir=None):
    """Read an array or an explicit {path, key?} NPY/NPZ reference."""
    if not isinstance(value, dict):
        return np.asarray(value)
    if set(value) - {"path", "key"} or "path" not in value:
        raise ValueError("array reference requires path and optional NPZ key")
    path = Path(value["path"])
    if not path.is_absolute():
        path = Path(base_dir or Path.cwd()) / path
    if path.suffix == ".npy":
        if "key" in value:
            raise ValueError("NPY references do not accept key")
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        if "key" not in value:
            raise ValueError("NPZ references require key")
        with np.load(path, allow_pickle=False) as archive:
            return archive[value["key"]]
    raise ValueError("array files must be .npy or .npz")


def planned_capabilities(domain):
    directory = files("configs").joinpath(domain, "algorithms")
    records = [yaml.safe_load(item.read_text(encoding="utf-8"))
               for item in sorted(directory.iterdir(), key=lambda item: item.name)
               if item.name.endswith(".yaml")]
    return {"modality": domain,
            "algorithms": [r for r in records if r["implementation_status"] == "implemented"],
            "planned_algorithms": [r for r in records if r["implementation_status"] != "implemented"]}


def resolve_parameters(domain, algorithm, supplied):
    records = planned_capabilities(domain)
    for record in records["algorithms"]:
        if record["name"] == algorithm:
            defaults = record["parameters"]
            if set(supplied) - defaults.keys():
                raise ValueError(f"Unknown parameters: {set(supplied) - defaults.keys()}")
            params = {**defaults, **supplied}
            for name, rule in record["parameter_schema"].items():
                value = params[name]
                if value is None and rule.get("nullable"):
                    continue
                kind = rule["type"]
                valid = ((kind == "float" and isinstance(value, (int, float)) and not isinstance(value, bool))
                         or (kind == "int" and isinstance(value, int) and not isinstance(value, bool))
                         or (kind == "str" and isinstance(value, str)))
                if not valid or (kind in {"float", "int"} and not np.isfinite(value)):
                    raise ValueError(f"Invalid {name}")
                if "minimum" in rule and value < rule["minimum"]:
                    raise ValueError(f"{name} must be >= {rule['minimum']}")
                if "choices" in rule and value not in rule["choices"]:
                    raise ValueError(f"Invalid {name}: {value}")
            return params
    if any(r["name"] == algorithm for r in records["planned_algorithms"]):
        raise NotImplementedError(f"{algorithm} is not implemented")
    raise ValueError(f"Unknown {domain} algorithm: {algorithm}")


@dataclass
class ExperimentResult:
    status: str
    modality: str
    algorithm: str
    effective_params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    runtime_sec: float = 0.0
    iterations: int = 0
    converged: bool = False
    stopping_reason: str = "not_implemented"
    outputs: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, str] | None = None
