"""One short ASTRA lifecycle acceptance run with Windows process GPU memory."""
import os
import subprocess
import torch
from inv_framework.api import CTTaskRequest, run_ct_task


def process_memory():
    query = ("(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory "
             f"-Filter \"Name LIKE 'pid_{os.getpid()}_%'\" | "
             "Measure-Object DedicatedUsage -Sum).Sum")
    value = subprocess.check_output(["powershell", "-NoProfile", "-Command", query], text=True)
    return int(value.strip())


def main():
    request = CTTaskRequest("parallel_2d__ellipses_dense_clean", "fbp", backend="astra", device="cuda")
    warmup = run_ct_task(request)
    assert warmup.status == "success", warmup.error_message
    torch.cuda.synchronize()
    samples = [(0, process_memory())]
    for i in range(1, 51):
        result = run_ct_task(request)
        assert result.status == "success", result.error_message
        if i % 10 == 0:
            torch.cuda.synchronize()
            samples.append((i, process_memory()))
    print("WDDM process dedicated GPU bytes:", samples, flush=True)
    print("50 ASTRA tasks completed; compare warmed samples for sustained growth.", flush=True)

if __name__ == "__main__":
    main()
