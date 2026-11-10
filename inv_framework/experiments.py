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
    from inv_framework.algorithm_catalog import describe_algorithm, resolve_recommendations, describe_execution
    for record in records:
        record.update(describe_algorithm(domain, record))
        resolve_recommendations(domain, record, context)
        describe_execution(domain, record, context)
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
    output_artifacts: dict[str, Any] = field(default_factory=dict)


def execution_controls(domain, algorithm, defaults):
    """Supported top-level fields mapped to existing outer algorithm parameters."""
    policy = {}
    budget = 'num_iterations' if 'num_iterations' in defaults else 'max_iter' if 'max_iter' in defaults else None
    if budget:
        policy['max_iterations'] = budget
    if 'tolerance' in defaults:
        policy['tolerance'] = 'tolerance'
    # LSQR's two native thresholds retain their distinct names.
    for key in ('atol', 'btol', 'tol'):
        if key in defaults:
            policy[key] = key
    constraints = {}
    if domain == 'deconvolution':
        if algorithm in ('fista', 'admm', 'pdhg'):
            constraints = dict(min_value='min_value', max_value='max_value')
        elif algorithm == 'tv_admm':
            constraints = dict(nonnegative='nonnegative')
        elif algorithm in ('richardson_lucy', 'rl_tv'):
            constraints = dict(nonnegative=None)  # intrinsic, cannot be disabled
    return dict(stopping_policy=policy, problem_constraints=constraints)


def resolve_execution_parameters(domain, request, result):
    record = next((r for r in planned_capabilities(domain)['algorithms'] if r['name'] == request.algorithm), None)
    if record is None:
        return resolve_parameters(domain, request.algorithm, request.params)
    mappings = execution_controls(domain, request.algorithm, record['parameters'])
    supplied = dict(request.params)
    overrides = {}
    for group, mapping in mappings.items():
        values = getattr(request, group, None)
        if values is not None and not isinstance(values, dict):
            raise ValueError(f'{group} must be a mapping')
        values = values or {}
        unknown = set(values) - mapping.keys()
        if unknown:
            raise ValueError(f'{request.algorithm}: unsupported {group} fields: {sorted(unknown)}')
        for key, value in values.items():
            target = mapping[key]
            if target is None:
                if value is not True:
                    raise ValueError('Multiplicative Poisson updates require nonnegative=true')
                continue
            if target in supplied and supplied[target] != value:
                overrides[target] = dict(previous=supplied[target], effective=value, source=group)
            supplied[target] = value
    p = resolve_parameters(domain, request.algorithm, supplied)
    constraints = {key: p[target] if target else True for key, target in mappings['problem_constraints'].items()}
    if request.algorithm == 'tv_admm' and constraints.get('nonnegative') is None:
        constraints['nonnegative'] = (request.fidelity == 'poisson_nll' or
            (request.fidelity is None and request.model.get('noise_model') == 'poisson'))
        p['nonnegative'] = constraints['nonnegative']
    if domain == 'mri' and (p.get('min_value') is not None or p.get('max_value') is not None):
        raise ValueError('Complex MRI does not accept real Box constraints')
    result.diagnostics.update(effective_stopping_policy={key: p[target] for key, target in mappings['stopping_policy'].items()},
                              effective_problem_constraints=constraints, parameter_overrides=overrides)
    return p


def describe_outputs(result, model):
    """Describe actual arrays, including calibration and blind-task auxiliary outputs."""
    for name, value in result.outputs.items():
        array = np.asarray(value)
        if result.modality == 'mri':
            spatial = ['z','y','x'][-len(model.get('spatial_shape', [1,1])):]
            axes = (['batch','coil'] + spatial if name in ('kspace','sensitivity_maps')
                    else ['batch'] + spatial)
            if len(axes) != array.ndim:
                axes = [f'axis_{i}' for i in range(array.ndim)]
            representation = 'complex' if np.iscomplexobj(array) else 'magnitude' if name == 'reconstruction' else 'real'
            units = 'dimensionless' if name in ('sensitivity_maps','eigenvalues') else 'arbitrary; physical units unspecified'
            scale = 'native FFT/model scale; no intensity rescaling'
        elif result.modality == 'deconvolution':
            axes = ['kernel_y','kernel_x'] if name == 'psf' else model.get('axes', [])
            representation = 'real'
            units = 'dimensionless' if name == 'psf' else 'unknown; object intensity in declared observation model'
            scale = 'unit-sum PSF' if name == 'psf' else 'object x in gain*H(x)+background; no post-rescaling'
        else:
            axes = ['batch','z','y','x'] if array.ndim == 4 and (model.get('geometry',{}).get('type') == 'cone_3d' or result.diagnostics.get('dimension') == 3) else ['batch','channel','y','x']
            representation, units, scale = 'real', 'unknown; attenuation model units unspecified', 'native reconstruction scale'
        if len(axes) != array.ndim:
            axes = [f'axis_{i}' for i in range(array.ndim)]
        result.output_artifacts[name] = dict(path=None, shape=list(array.shape), dtype=str(array.dtype),
            axes=axes, representation=representation, units=units, scale=scale)
    if result.modality != 'ct' and result.outputs:
        # Native NumPy paths promote to double; shared solvers explicitly use double tensors.
        result.diagnostics.setdefault('computation_precision',
            result.diagnostics.get('compute_dtype', 'float32' if result.algorithm == 'varnet' else
                'complex128' if result.modality == 'mri' else 'float64'))
    result.diagnostics['output_dtypes'] = {k: v['dtype'] for k,v in result.output_artifacts.items()}
    if any(not np.isfinite(a).all() for a in result.outputs.values()):
        result.status = 'failed'
        result.converged = False
        result.stopping_reason = 'numerical_error'
        result.error = dict(code='numerical_error', message='Nonfinite output')
    return result
