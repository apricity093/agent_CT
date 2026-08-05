#requires -Version 5.1
<#
.SYNOPSIS
Synchronize the current worktree into the existing remote ct_agent project and
run the simulated cone-beam FDK smoke workflow.

.DESCRIPTION
The archive is uploaded temporarily to /tmp and extracted over the existing
project. No sibling worktree is created below /home/gaodongxu/autodl-tmp.
#>

[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$RemoteHost = "202.121.181.119",

    [ValidateNotNullOrEmpty()]
    [string]$RemoteProjectDirectory = "/home/gaodongxu/autodl-tmp/ct_agent",

    [ValidateNotNullOrEmpty()]
    [string]$RemotePython = "/home/gaodongxu/miniconda3/envs/fno/bin/python",

    [ValidateNotNullOrEmpty()]
    [string]$RemoteRunName = "fdk_simulated_smoke_20260805_run1",

    [string]$OutputDirectory,

    [ValidateSet("cuda")]
    [string]$Device = "cuda",

    [ValidateRange(5, 300)]
    [int]$ConnectionTimeoutSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-SafeRemoteValue {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Pattern
    )
    if ($Value -notmatch $Pattern) {
        throw "$Name contains characters that are unsafe for the remote shell: $Value"
    }
}

function Invoke-RemoteBash {
    param(
        [Parameter(Mandatory = $true)][string]$Step,
        [Parameter(Mandatory = $true)][string]$ScriptText
    )
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($ScriptText))
    $command = "printf '%s' '$encoded' | base64 --decode | bash"
    $output = @(& ssh.exe "-o" "BatchMode=yes" "-o" "ConnectTimeout=$ConnectionTimeoutSeconds" $RemoteHost $command 2>&1)
    $exitCode = $LASTEXITCODE
    $text = ($output | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
    if ($text) {
        Write-Host "[$Step]"
        Write-Host $text
    }
    if ($exitCode -ne 0) {
        throw "$Step failed with ssh exit code $exitCode."
    }
    return $text
}

function Invoke-Scp {
    param(
        [Parameter(Mandatory = $true)][string]$Step,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host "[$Step]"
    & scp.exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with scp exit code $LASTEXITCODE."
    }
}

function Assert-ArtifactChecksums {
    param(
        [Parameter(Mandatory = $true)][string]$ArtifactDirectory,
        [Parameter(Mandatory = $true)][string]$ChecksumManifest
    )
    if (-not (Test-Path -LiteralPath $ArtifactDirectory -PathType Container)) {
        throw "Downloaded artifact directory is missing: $ArtifactDirectory"
    }
    if (-not (Test-Path -LiteralPath $ChecksumManifest -PathType Leaf)) {
        throw "Downloaded checksum manifest is missing: $ChecksumManifest"
    }

    $root = [IO.Path]::GetFullPath($ArtifactDirectory).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $expected = @{}
    $lineNumber = 0
    foreach ($line in Get-Content -LiteralPath $ChecksumManifest) {
        $lineNumber += 1
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $match = [regex]::Match($line, "^(?<hash>[A-Fa-f0-9]{64})\s+\*?(?<path>.+)$")
        if (-not $match.Success) {
            throw "Malformed checksum entry at line $($lineNumber): $line"
        }
        $relative = $match.Groups["path"].Value
        if ($relative.StartsWith("./")) {
            $relative = $relative.Substring(2)
        }
        $relative = $relative -replace "\\", "/"
        if ([string]::IsNullOrWhiteSpace($relative) -or
            [IO.Path]::IsPathRooted($relative) -or
            $relative -match "(^|[\\/])\.\.([\\/]|$)") {
            throw "Unsafe artifact path in checksum manifest: $relative"
        }
        if ($expected.ContainsKey($relative)) {
            throw "Duplicate artifact path in checksum manifest: $relative"
        }
        $localPath = [IO.Path]::GetFullPath(
            (Join-Path $root ($relative -replace "/", [IO.Path]::DirectorySeparatorChar))
        )
        if (-not $localPath.StartsWith(
                "$root$([IO.Path]::DirectorySeparatorChar)",
                [StringComparison]::OrdinalIgnoreCase
            )) {
            throw "Checksum path escaped the artifact directory: $relative"
        }
        if (-not (Test-Path -LiteralPath $localPath -PathType Leaf)) {
            throw "Checksum artifact is missing after download: $relative"
        }
        $actualHash = (Get-FileHash -LiteralPath $localPath -Algorithm SHA256).Hash
        if (-not $actualHash.Equals($match.Groups["hash"].Value, [StringComparison]::OrdinalIgnoreCase)) {
            throw "SHA256 mismatch for artifact: $relative"
        }
        $expected[$relative] = $true
    }
    if ($expected.Count -eq 0) {
        throw "Remote checksum manifest did not contain any artifacts."
    }

    $downloaded = @(
        Get-ChildItem -LiteralPath $root -File -Recurse | Where-Object {
            $_.Name -ne "artifacts.sha256"
        } | ForEach-Object {
            $_.FullName.Substring($root.Length).TrimStart(
                [IO.Path]::DirectorySeparatorChar,
                [IO.Path]::AltDirectorySeparatorChar
            ) -replace "\\", "/"
        }
    )
    if ($downloaded.Count -ne $expected.Count) {
        throw "Downloaded artifact count does not match the remote checksum manifest."
    }
    foreach ($relative in $downloaded) {
        if (-not $expected.ContainsKey($relative)) {
            throw "Downloaded artifact was absent from the checksum manifest: $relative"
        }
    }
}

Assert-SafeRemoteValue -Value $RemoteHost -Name "RemoteHost" -Pattern "^[A-Za-z0-9._-]+$"
Assert-SafeRemoteValue -Value $RemoteProjectDirectory -Name "RemoteProjectDirectory" -Pattern "^/[A-Za-z0-9._/-]+$"
Assert-SafeRemoteValue -Value $RemotePython -Name "RemotePython" -Pattern "^/[A-Za-z0-9._/-]+$"
Assert-SafeRemoteValue -Value $RemoteRunName -Name "RemoteRunName" -Pattern "^[A-Za-z0-9._-]+$"

$tarCommand = Get-Command tar.exe -ErrorAction SilentlyContinue
if ($null -eq $tarCommand) {
    throw "tar.exe is required to package the worktree."
}
if ($null -eq (Get-Command ssh.exe -ErrorAction SilentlyContinue)) {
    throw "ssh.exe is required and must be available on PATH."
}
if ($null -eq (Get-Command scp.exe -ErrorAction SilentlyContinue)) {
    throw "scp.exe is required and must be available on PATH."
}

$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$runnerPath = Join-Path $repositoryRoot "test\ct_fdk_simulation\run_fdk_simulated_smoke.py"
if (-not (Test-Path -LiteralPath $runnerPath -PathType Leaf)) {
    throw "The simulated FDK runner is missing: $runnerPath"
}
$remotePackagePath = "/tmp/$RemoteRunName.tar.gz"
$remoteArtifactDirectory = "$RemoteProjectDirectory/artifacts/$RemoteRunName"
$remoteChecksumManifest = "$remoteArtifactDirectory/artifacts.sha256"
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repositoryRoot ("artifacts\{0}_remote" -f $RemoteRunName)
}
$localOutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $localOutputDirectory) {
    throw "Refusing to overwrite existing local output directory: $localOutputDirectory"
}

$temporaryDirectory = Join-Path ([IO.Path]::GetTempPath()) $RemoteRunName
$packagePath = Join-Path $temporaryDirectory "worktree.tar.gz"

try {
    New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null
    $tarArguments = @(
        "--create", "--gzip", "--file=$packagePath",
        "--exclude=.git", "--exclude=./.git", "--exclude=./.git/*",
        "--exclude=artifacts", "--exclude=./artifacts", "--exclude=./artifacts/*",
        "--exclude=.pytest_cache", "--exclude=./.pytest_cache", "--exclude=./.pytest_cache/*",
        "--exclude=__pycache__", "--exclude=*/__pycache__", "--exclude=*/__pycache__/*",
        "--exclude=.mypy_cache", "--exclude=.ruff_cache", "--exclude=.tox", "--exclude=.nox",
        "--exclude=htmlcov", "--exclude=build", "--exclude=dist",
        "--exclude=.venv", "--exclude=venv", "--exclude=*.pyc", "--exclude=*.pyo",
        "-C", $repositoryRoot, "."
    )
    Write-Host "[package worktree]"
    & $tarCommand.Source @tarArguments
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe failed while packaging the worktree."
    }

    $entries = @(& $tarCommand.Source "-tzf" $packagePath)
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe failed while validating the worktree archive."
    }
    foreach ($required in @(
        "./test/ct_fdk_simulation/run_fdk_simulated_smoke.py",
        "./tools/run_remote_fdk_simulated_smoke.ps1"
    )) {
        if (-not ($entries -contains $required)) {
            throw "Worktree archive does not include $required"
        }
    }
    $localPackageHash = (Get-FileHash -LiteralPath $packagePath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "Local package SHA256: $localPackageHash"

    $verifyProject = @'
set -euo pipefail
test -d '__PROJECT_DIRECTORY__'
test -f '__PROJECT_DIRECTORY__/inv_framework/__init__.py'
if [ -e '__ARTIFACT_DIRECTORY__' ]; then
    printf 'Remote artifact directory already exists: %s\n' '__ARTIFACT_DIRECTORY__' >&2
    exit 1
fi
'@.Replace("__PROJECT_DIRECTORY__", $RemoteProjectDirectory).
        Replace("__ARTIFACT_DIRECTORY__", $remoteArtifactDirectory)
    Invoke-RemoteBash -Step "verify remote ct_agent project" -ScriptText $verifyProject | Out-Null

    Invoke-Scp -Step "upload worktree archive" -Arguments @(
        "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=$ConnectionTimeoutSeconds",
        $packagePath, "$($RemoteHost):$remotePackagePath"
    )

    $syncProject = @'
set -euo pipefail
expected='__EXPECTED_HASH__'
actual=$(sha256sum '__PACKAGE_PATH__' | awk '{print $1}')
if [ "$actual" != "$expected" ]; then
    printf 'Uploaded package SHA256 mismatch: expected %s, got %s\n' "$expected" "$actual" >&2
    exit 1
fi
tar -xzf '__PACKAGE_PATH__' --overwrite -C '__PROJECT_DIRECTORY__'
rm -f '__PACKAGE_PATH__'
test -f '__PROJECT_DIRECTORY__/test/ct_fdk_simulation/run_fdk_simulated_smoke.py'
test -f '__PROJECT_DIRECTORY__/tools/run_remote_fdk_simulated_smoke.ps1'
mkdir -p '__ARTIFACT_DIRECTORY__'
printf '%s\n' "$actual"
'@.Replace("__EXPECTED_HASH__", $localPackageHash).
        Replace("__PACKAGE_PATH__", $remotePackagePath).
        Replace("__PROJECT_DIRECTORY__", $RemoteProjectDirectory).
        Replace("__ARTIFACT_DIRECTORY__", $remoteArtifactDirectory)
    $remotePackageHash = (Invoke-RemoteBash -Step "synchronize remote ct_agent project" -ScriptText $syncProject).Trim().ToLowerInvariant()
    if ($remotePackageHash -ne $localPackageHash) {
        throw "Remote package SHA256 did not match the local archive."
    }

    $preflight = @'
set -euo pipefail
cd '__PROJECT_DIRECTORY__'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
'__REMOTE_PYTHON__' - <<'PY' > '__ARTIFACT_DIRECTORY__/preflight.json'
import importlib
import json
import platform
import sys

modules = {name: importlib.import_module(name) for name in ("torch", "astra", "matplotlib", "pytest")}
torch = modules["torch"]
astra = modules["astra"]
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch CUDA is unavailable; FDK smoke requires CUDA.")
if not astra.use_cuda():
    raise RuntimeError("ASTRA CUDA is unavailable; FDK smoke requires ASTRA CUDA.")
print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_name": torch.cuda.get_device_name(0),
    "astra": getattr(astra, "__version__", "unknown"),
    "astra_use_cuda": bool(astra.use_cuda()),
    "matplotlib": modules["matplotlib"].__version__,
    "pytest": modules["pytest"].__version__,
}, indent=2, sort_keys=True))
PY
'@.Replace("__PROJECT_DIRECTORY__", $RemoteProjectDirectory).
        Replace("__ARTIFACT_DIRECTORY__", $remoteArtifactDirectory).
        Replace("__REMOTE_PYTHON__", $RemotePython)
    Invoke-RemoteBash -Step "remote CUDA and ASTRA preflight" -ScriptText $preflight | Out-Null

    $runTests = @'
set -euo pipefail
cd '__PROJECT_DIRECTORY__'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.
'__REMOTE_PYTHON__' -m pytest -q test/test_fdk_backend_adapters.py test/ct_fdk_simulation 2>&1 | tee '__ARTIFACT_DIRECTORY__/pytest.log'
'@.Replace("__PROJECT_DIRECTORY__", $RemoteProjectDirectory).
        Replace("__ARTIFACT_DIRECTORY__", $remoteArtifactDirectory).
        Replace("__REMOTE_PYTHON__", $RemotePython)
    Invoke-RemoteBash -Step "remote FDK tests" -ScriptText $runTests | Out-Null

    $runSmoke = @'
set -euo pipefail
cd '__PROJECT_DIRECTORY__'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.
'__REMOTE_PYTHON__' test/ct_fdk_simulation/run_fdk_simulated_smoke.py --device '__DEVICE__' --output-dir '__ARTIFACT_DIRECTORY__/result' 2>&1 | tee '__ARTIFACT_DIRECTORY__/runner.log'
for required in result/geometry.json result/metrics.json result/manifest.json result/truth_axial.npy result/reconstruction_axial.npy result/comparison.png; do
    test -s '__ARTIFACT_DIRECTORY__'/$required
done
'__REMOTE_PYTHON__' - <<'PY'
import json
import math
from pathlib import Path

artifact_dir = Path('__ARTIFACT_DIRECTORY__') / 'result'
metrics = json.loads((artifact_dir / 'metrics.json').read_text(encoding='utf-8'))
if metrics.get('status') != 'success':
    raise RuntimeError(f"unexpected FDK status: {metrics.get('status')!r}")
if metrics.get('truth_shape') != [1, 64, 64, 64]:
    raise RuntimeError(f"unexpected truth shape: {metrics.get('truth_shape')!r}")
if metrics.get('measurement_shape') != [1, 128, 180, 128]:
    raise RuntimeError(f"unexpected measurement shape: {metrics.get('measurement_shape')!r}")
if metrics.get('reconstruction_shape') != [1, 64, 64, 64]:
    raise RuntimeError(f"unexpected reconstruction shape: {metrics.get('reconstruction_shape')!r}")
if metrics.get('dtype') != 'float32' or not str(metrics.get('device', '')).startswith('cuda'):
    raise RuntimeError(f"unexpected output dtype/device: {metrics.get('dtype')!r}/{metrics.get('device')!r}")
for key in ('measurement_min', 'measurement_max', 'reconstruction_max_abs', 'relative_error', 'data_residual', 'rmse', 'psnr', 'runtime_seconds'):
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"non-finite metric {key}: {value!r}")
if metrics['measurement_max'] <= 0.0 or metrics['reconstruction_max_abs'] <= 0.0:
    raise RuntimeError('simulated projection or FDK reconstruction is zero')
PY
cd '__ARTIFACT_DIRECTORY__'
find . -type f ! -name artifacts.sha256 -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > artifacts.sha256
test -s artifacts.sha256
'@.Replace("__PROJECT_DIRECTORY__", $RemoteProjectDirectory).
        Replace("__ARTIFACT_DIRECTORY__", $remoteArtifactDirectory).
        Replace("__REMOTE_PYTHON__", $RemotePython).
        Replace("__DEVICE__", $Device)
    Invoke-RemoteBash -Step "remote simulated FDK smoke" -ScriptText $runSmoke | Out-Null

    New-Item -ItemType Directory -Path $localOutputDirectory | Out-Null
    Invoke-Scp -Step "download remote FDK artifacts" -Arguments @(
        "-q", "-r", "-o", "BatchMode=yes", "-o", "ConnectTimeout=$ConnectionTimeoutSeconds",
        "$($RemoteHost):$remoteArtifactDirectory", $localOutputDirectory
    )
    $localArtifactDirectory = Join-Path $localOutputDirectory $RemoteRunName
    $localChecksumManifest = Join-Path $localArtifactDirectory "artifacts.sha256"
    Assert-ArtifactChecksums -ArtifactDirectory $localArtifactDirectory -ChecksumManifest $localChecksumManifest

    [ordered]@{
        schema_version = 1
        remote_host = $RemoteHost
        remote_project_directory = $RemoteProjectDirectory
        remote_python = $RemotePython
        remote_artifact_directory = $remoteArtifactDirectory
        device = $Device
        worktree_sha256 = $localPackageHash
        artifacts_sha256_verified = $true
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (
        Join-Path $localOutputDirectory "remote_orchestration.json"
    ) -Encoding UTF8
    Write-Host "Remote simulated FDK smoke completed."
    Write-Host "Synchronized project: $RemoteProjectDirectory"
    Write-Host "Verified local artifacts: $localArtifactDirectory"
}
finally {
    if (Test-Path -LiteralPath $temporaryDirectory) {
        Remove-Item -LiteralPath $temporaryDirectory -Recurse -Force
    }
}
