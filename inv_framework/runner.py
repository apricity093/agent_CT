"""One explicit experiment, with no planning, search or retry policy."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import yaml
from inv_framework.experiments import ExperimentResult


def _resolve_paths(config, base):
    """Resolve only declared file references; ordinary model strings are untouched."""
    from copy import deepcopy
    config = deepcopy(config)
    def reference(value):
        if isinstance(value, dict) and isinstance(value.get('path'), str):
            value['path'] = str((base / value['path']).resolve())
    for key, value in config.get('inputs', {}).items():
        if key == 'public_measurement':
            for item in value.values():
                reference(item)
        elif key not in {'geometry', 'measurement', 'data_range'}:
            reference(value)
    reference((config.get('evaluation') or {}).get('reference'))
    checkpoint = config.get('params', {}).get('checkpoint')
    if checkpoint:
        config['params']['checkpoint'] = str((base / checkpoint).resolve())
    if config.get('data_root'):
        config['data_root'] = str((base / config['data_root']).resolve())
    return config


def _execute(config, *, params=None):
    """Run a mapping or YAML/JSON file. Explicit params override file params."""
    allowed = {"modality", "algorithm", "task_type", "inputs", "model", "params", "device", "backend", "case_id", "data_root", "evaluation", "stopping_policy", "problem_constraints", "fidelity"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("unsupported experiment fields")
    domain = config["modality"]
    file_params, explicit_params = config.get("params", {}), params or {}
    if domain == "ct":
        from inv_framework.experiments import normalize_ct_parameters
        file_params = normalize_ct_parameters(file_params, config["algorithm"])
        if config.get("fidelity") is not None:
            if "data_fidelity" in file_params and file_params["data_fidelity"] != config["fidelity"]:
                raise ValueError("experiment fidelity conflicts with params.data_fidelity")
            file_params["data_fidelity"] = config["fidelity"]
        explicit_params = normalize_ct_parameters(explicit_params, config["algorithm"])
    effective = {**file_params, **explicit_params}
    if domain == "ct":
        if config.get("task_type", "reconstruction") != "reconstruction":
            raise ValueError("CT task_type must be reconstruction")
        from inv_framework.ct.api import CTTaskRequest, run_ct_task
        inputs = {**config["inputs"], **config["model"]} if "inputs" in config else None
        if inputs is not None:
            inputs.pop("truth", None)
            inputs.pop("data_range", None)
        evaluation = config.get("evaluation")
        if evaluation and inputs is None:
            raise ValueError("evaluation references require external CT inputs")
        request = CTTaskRequest(config.get("case_id", "external"), config["algorithm"],
            params=effective, inputs=inputs, **{key: config[key] for key in
            ("backend", "device", "data_root", "stopping_policy", "problem_constraints") if key in config})
        ct = run_ct_task(request)
        result = ExperimentResult(ct.status, domain, config["algorithm"], ct.effective_params,
            ct.metrics, ct.runtime_sec, ct.iterations, ct.converged, ct.stopping_reason,
            {} if ct.reconstruction is None else {"reconstruction": ct.reconstruction},
            ct.diagnostics, None if ct.error_message is None else
            {"code": ct.reason_code, "message": ct.error_message})
        if result.status == "success" and evaluation and "reference" in evaluation:
            import torch
            from inv_framework.experiments import read_array
            from inv_framework.utils.metrics import ct_image_metrics
            image = torch.as_tensor(result.outputs["reconstruction"])
            reference = torch.as_tensor(read_array(evaluation["reference"]))
            data_range = float(evaluation["data_range"])
            if reference.shape != image.shape or reference.is_complex() or not torch.isfinite(reference).all():
                raise ValueError("CT evaluation reference must be finite, real and match reconstruction")
            if not np.isfinite(data_range) or data_range <= 0:
                raise ValueError("evaluation.data_range must be positive")
            dimension = 3 if inputs['geometry']['type'] == 'cone_3d' else 2
            result.metrics.update(ct_image_metrics(image, reference, data_range, dimension))
            result.diagnostics['evaluation'] = dict(reference_provided=True,
                stage='post_reconstruction', data_range=data_range,
                ssim_definition=result.metrics['ssim_definition'])
    elif domain in {"mri", "deconvolution"}:
        if domain == "mri":
            from inv_framework.mri.api import MRITaskRequest as Request, run_mri_task as run
        else:
            from inv_framework.deconvolution.api import DeconvolutionTaskRequest as Request, run_deconvolution_task as run
        result = run(Request(config["algorithm"], config.get("inputs", {}), config.get("model", {}),
            params=effective, task_type=config.get("task_type", "reconstruction"), device=config.get("device", "cpu"), fidelity=config.get("fidelity"),
            stopping_policy=config.get("stopping_policy"), problem_constraints=config.get("problem_constraints")))
        if config.get("evaluation") and config.get("task_type") == "calibration":
            raise ValueError("Image-reference evaluation applies to reconstruction, not calibration maps")
        if result.status == "success" and (config.get("evaluation") or {}).get("reference") is not None:
            from inv_framework.evaluation.array_metrics import evaluate_array
            image = result.outputs["reconstruction"]
            axes = tuple(range(1, image.ndim)) if domain == "mri" else tuple(
                i for i, name in enumerate(config["model"]["axes"]) if name != "channel")
            result.metrics = evaluate_array(image, config["evaluation"], spatial_axes=axes, mri=domain == "mri")
    else:
        raise ValueError(f"Unknown modality: {domain}")
    result.diagnostics["call_parameter_overrides"] = {
        key: dict(previous=file_params[key], effective=value, source="call_params")
        for key, value in explicit_params.items() if key in file_params and file_params[key] != value}
    return result


def run_experiment(config, output_dir=None, *, params=None):
    """Run a mapping or file; file references are relative to its directory.

    Parameter precedence: defaults < config params < call params < top-level controls.
    Read/write errors retain their original exception. Execution/request failures
    are structured results and are persisted when an output directory is supplied.
    """
    import time
    from inv_framework.experiments import describe_outputs
    base = None
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        content = path.read_text(encoding='utf-8')
        config = json.loads(content) if path.suffix == '.json' else yaml.safe_load(content)
        base = path.parent
    started = time.perf_counter()
    try:
        if isinstance(config, dict):
            for section in ('inputs', 'model', 'params', 'evaluation'):
                if section in config and not isinstance(config[section], dict):
                    raise ValueError(f'{section} must be a mapping')
            if base is not None:
                config = _resolve_paths(config, base)
        result = _execute(config, params=params)
    except (ValueError, KeyError, TypeError, OSError, NotImplementedError) as error:
        info = config if isinstance(config, dict) else {}
        result = ExperimentResult('failed', info.get('modality', 'unknown'), info.get('algorithm', 'unknown'),
            stopping_reason='invalid_request', error=dict(code='invalid_request', message=str(error)),
            runtime_sec=time.perf_counter()-started)
    describe_outputs(result, config.get('model', {}) if isinstance(config, dict) else {})
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        payload = asdict(result)
        payload['outputs'] = {}
        for name, value in result.outputs.items():
            path = destination / f'{name}.npy'
            np.save(path, value, allow_pickle=False)
            payload['outputs'][name] = path.name
            payload['output_artifacts'][name]['path'] = path.name
        (destination / 'result.json').write_text(json.dumps(_finite_json(payload), indent=2, default=_json_scalar, allow_nan=False), encoding='utf-8')
    return result


def _finite_json(value):
    """Unavailable numerical diagnostics are null in strict JSON."""
    if isinstance(value, dict):
        return {k: _finite_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_experiment(args.config, args.output)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"Invalid experiment: {error}\n")
    print(json.dumps({"status": result.status, "stopping_reason": result.stopping_reason}))
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
