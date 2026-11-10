"""One short ASTRA lifecycle acceptance run with Windows process GPU memory."""
import os
import subprocess
import argparse
import contextlib
import io
import json
from pathlib import Path
import numpy as np
import astra
import torch
from inv_framework.api import CTTaskRequest, run_ct_task


def process_memory():
    query = ("(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory "
             f"-Filter \"Name LIKE 'pid_{os.getpid()}_%'\" | "
             "Measure-Object DedicatedUsage -Sum).Sum")
    try:
        value = subprocess.check_output(["powershell", "-NoProfile", "-Command", query], text=True)
    except subprocess.CalledProcessError as error:
        print('Process memory counter unavailable:', error, flush=True)
        return None
    return int(value.strip()) if value.strip() else None


def objects():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        for manager in (astra.data2d, astra.data3d, astra.projector, astra.algorithm, astra.matrix):
            manager.info()
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    torch.set_num_threads(4)
    request = CTTaskRequest("cone_3d/spheres_astra_12", "fdk", backend="astra", device="cuda")
    baseline = objects()
    for _ in range(3):
        warmup = run_ct_task(request)
        assert warmup.status == "success", warmup.error_message
        del warmup
    assert objects() == baseline
    torch.cuda.synchronize()
    samples = [(0, process_memory())]
    for i in range(1, 51):
        result = run_ct_task(request)
        assert result.status == "success", result.error_message
        assert result.reconstruction.shape == (1, 12, 12, 12)
        assert np.isfinite(result.reconstruction).all()
        del result
        torch.cuda.synchronize()
        assert objects() == baseline, f'ASTRA objects retained at call {i}'
        samples.append((i, process_memory()))
        print(i, samples[-1][1], flush=True)
    args.output.write_text(json.dumps(dict(tool='WDDM GPUProcessMemory DedicatedUsage bytes',
        pid=os.getpid(), warmup=3, formal_calls=50, request=vars(request), samples=samples,
        astra_objects_before=baseline, astra_objects_after=objects()), indent=2), encoding='utf-8')
    print("WDDM process dedicated GPU bytes:", samples, flush=True)
    print("50 ASTRA tasks completed; compare warmed samples for sustained growth.", flush=True)

if __name__ == "__main__":
    main()
