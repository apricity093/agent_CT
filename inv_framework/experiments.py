"""Framework-independent file references, discovery resources and observations."""
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any
import numpy as np
import yaml


COMPOSABLE_METHODS = ('admm', 'pdhg', 'fista', 'landweber', 'cgls', 'lsqr', 'cg')

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


def planned_capabilities(domain, context=None):
    directory = files("configs").joinpath(domain, "algorithms")
    records = [yaml.safe_load(item.read_text(encoding="utf-8"))
               for item in sorted(directory.iterdir(), key=lambda item: item.name)
               if item.name.endswith(".yaml")]
    from inv_framework.algorithm_catalog import describe_algorithm, resolve_recommendations
    for record in records:
        record.update(describe_algorithm(domain, record))
        resolve_recommendations(domain, record, context)
    return {"modality": domain,
            "algorithms": [r for r in records if r["implementation_status"] == "implemented"],
            "planned_algorithms": [r for r in records if r["implementation_status"] != "implemented"]}


def normalize_ct_parameters(supplied, algorithm=None):
    """Normalize one precedence layer; reject contradictions within that layer."""
    values = dict(supplied)
    aliases = {"max_iter": "num_iterations", "num_subsets": "subset_count", "lam": "strength", "fidelity": "data_fidelity"}
    if algorithm in {"fista", "admm", "pdhg"}:
        aliases["reg_strength"] = "strength"
    elif algorithm == "tikhonov":
        aliases.update(lam="reg_strength", strength="reg_strength")
    elif algorithm == "landweber":
        aliases["learning_rate"] = "step_size"
    for alias, canonical in aliases.items():
        if alias in values:
            value = values.pop(alias)
            if value is None and alias in {"reg_strength", "strength"}:
                continue
            if canonical in values and values[canonical] != value:
                raise ValueError(f"{alias} and {canonical} conflict")
            values[canonical] = value
    return values


def resolve_fidelity(record, selected, model, inputs):
    """Resolve reconstruction assumptions without changing generation metadata."""
    contract = record["fidelity"]
    if contract["mode"] == "not_applicable":
        if selected is not None:
            raise ValueError(f"{record['name']}: optimization fidelity is not applicable")
        return None
    if selected is None:
        selected = contract["default"]
        if record.get("modality") == "deconvolution" and record["name"] not in COMPOSABLE_METHODS and "noise_model" in model:
            noise = model["noise_model"]
            if noise not in {"gaussian", "poisson"}:
                raise ValueError("Choose an explicit fidelity for an unknown generation model")
            selected = "poisson_nll" if noise == "poisson" else "l2"
    aliases = {"gaussian": "gaussian_nll", "poisson": "poisson_nll"}
    selected = aliases.get(selected, selected)
    if selected not in contract["supported"]:
        raise ValueError(f"{record['name']}: unsupported fidelity {selected}; {contract['reason']}")
    if selected == "wls" and "data_weights" not in inputs:
        raise ValueError("wls requires explicit inputs.data_weights (inverse variance)")
    return selected


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
                         or (kind == "str" and isinstance(value, str))
                         or (kind == "bool" and isinstance(value, bool)))
                if not valid or (kind in {"float", "int"} and not np.isfinite(value)):
                    raise ValueError(f"Invalid {name}")
                if "minimum" in rule and (value < rule["minimum"] or (rule.get("exclusive_minimum") and value == rule["minimum"])):
                    raise ValueError(f"{name} must be {'>' if rule.get('exclusive_minimum') else '>='} {rule['minimum']}")
                if "maximum" in rule and (value > rule["maximum"] or (rule.get("exclusive_maximum") and value == rule["maximum"])):
                    raise ValueError(f"{name} must be {'<' if rule.get('exclusive_maximum') else '<='} {rule['maximum']}")
                if "choices" in rule and value not in rule["choices"]:
                    if name == "regularizer" and value == "tv_haar":
                        raise ValueError("TV+Haar is implemented for ADMM only; select algorithm=admm")
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
