#requires -Version 5.1
<#!
.SYNOPSIS
Runs the all-solver CT benchmark in a fresh remote worktree and verifies the
returned artifacts.

.DESCRIPTION
The script intentionally archives the current worktree instead of using
`git archive`, so uncommitted benchmark code is included.  It excludes Git
metadata, caches, virtual environments, and all pre-existing artifacts.
The configured SSH host name is `202.121.181.119`; its user, port, and key
are resolved from the local SSH config.
#>

[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$RemoteHost = "202.121.181.119",

    [ValidateNotNullOrEmpty()]
    [string]$RemoteBaseDirectory = "/home/gaodongxu/autodl-tmp",

    [ValidateNotNullOrEmpty()]
    [string]$RemotePython = "/home/gaodongxu/miniconda3/envs/fno/bin/python",

    [ValidateNotNullOrEmpty()]
    [string]$RemoteRunName = "ct_all_solvers_20260804_run1",

    [string]$OutputDirectory,

    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [ValidateRange(5, 300)]
    [int]$ConnectionTimeoutSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-SafeRemoteValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value,

        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Pattern
    )

    if ($Value -notmatch $Pattern) {
        throw "$Name contains characters that are unsafe for the remote shell: $Value"
    }
}

function Invoke-RemoteBash {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Step,

        [Parameter(Mandatory = $true)]
        [string]$ScriptText
    )

    $encodedScript = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($ScriptText)
    )
    $remoteCommand = "printf '%s' '$encodedScript' | base64 --decode | bash"
    $output = @(
        & ssh.exe `
            "-o" "BatchMode=yes" `
            "-o" "ConnectTimeout=$ConnectionTimeoutSeconds" `
            $RemoteHost $remoteCommand 2>&1
    )
    $exitCode = $LASTEXITCODE
    $outputText = ($output | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
    if ($outputText) {
        Write-Host "[$Step]"
        Write-Host $outputText
    }
    if ($exitCode -ne 0) {
        throw "$Step failed with ssh exit code $exitCode."
    }
    return $outputText
}

function Invoke-Scp {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Step,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host "[$Step]"
    & scp.exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with scp exit code $LASTEXITCODE."
    }
}

function Assert-ArtifactChecksums {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ArtifactDirectory,

        [Parameter(Mandatory = $true)]
        [string]$ChecksumManifest
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
    $expectedPaths = @{}
    $lineNumber = 0

    foreach ($line in Get-Content -LiteralPath $ChecksumManifest) {
        $lineNumber += 1
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $match = [regex]::Match(
            $line,
            "^(?<hash>[A-Fa-f0-9]{64})\s+\*?(?<path>.+)$"
        )
        if (-not $match.Success) {
            throw "Malformed checksum entry at line ${lineNumber}: $line"
        }

        $relativePath = $match.Groups["path"].Value
        if ($relativePath.StartsWith("./")) {
            $relativePath = $relativePath.Substring(2)
        }
        if ([string]::IsNullOrWhiteSpace($relativePath) -or
            [IO.Path]::IsPathRooted($relativePath) -or
            $relativePath -match "(^|[\\/])\.\.([\\/]|$)") {
            throw "Unsafe artifact path in checksum manifest: $relativePath"
        }
        if ($expectedPaths.ContainsKey($relativePath)) {
            throw "Duplicate checksum entry for artifact: $relativePath"
        }

        $localPath = [IO.Path]::GetFullPath(
            (Join-Path $root ($relativePath -replace "/", [IO.Path]::DirectorySeparatorChar))
        )
        if (-not $localPath.StartsWith(
                "$root$([IO.Path]::DirectorySeparatorChar)",
                [StringComparison]::OrdinalIgnoreCase
            )) {
            throw "Checksum path escaped the artifact directory: $relativePath"
        }
        if (-not (Test-Path -LiteralPath $localPath -PathType Leaf)) {
            throw "Checksum artifact is missing after download: $relativePath"
        }

        $actualHash = (Get-FileHash -LiteralPath $localPath -Algorithm SHA256).Hash
        if (-not $actualHash.Equals($match.Groups["hash"].Value, [StringComparison]::OrdinalIgnoreCase)) {
            throw "SHA256 mismatch for artifact: $relativePath"
        }
        $expectedPaths[$relativePath] = $true
    }

    if ($expectedPaths.Count -eq 0) {
        throw "Remote checksum manifest did not contain any artifacts."
    }
    $downloadedFiles = @(
        Get-ChildItem -LiteralPath $root -File -Recurse | ForEach-Object {
            $relative = $_.FullName.Substring($root.Length).TrimStart(
                [IO.Path]::DirectorySeparatorChar,
                [IO.Path]::AltDirectorySeparatorChar
            )
            $relative -replace "\\", "/"
        }
    )
    if ($downloadedFiles.Count -ne $expectedPaths.Count) {
        throw (
            "Downloaded artifact count ($($downloadedFiles.Count)) does not match " +
            "the remote checksum manifest ($($expectedPaths.Count))."
        )
    }
    foreach ($relativePath in $downloadedFiles) {
        if (-not $expectedPaths.ContainsKey($relativePath)) {
            throw "Downloaded artifact was absent from the checksum manifest: $relativePath"
        }
    }
}

Assert-SafeRemoteValue -Value $RemoteHost -Name "RemoteHost" -Pattern "^[A-Za-z0-9._-]+$"
Assert-SafeRemoteValue -Value $RemoteBaseDirectory -Name "RemoteBaseDirectory" -Pattern "^/[A-Za-z0-9._/-]+$"
Assert-SafeRemoteValue -Value $RemotePython -Name "RemotePython" -Pattern "^/[A-Za-z0-9._/-]+$"
Assert-SafeRemoteValue -Value $RemoteRunName -Name "RemoteRunName" -Pattern "^[A-Za-z0-9._-]+$"
if ($Device -ne "cuda") {
    throw "This orchestration requires -Device cuda because the remote benchmark must run FDK."
}

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
$driverPath = Join-Path $repositoryRoot "test\ct_all_solvers\run_all_solvers.py"
if (-not (Test-Path -LiteralPath $driverPath -PathType Leaf)) {
    throw "The all-solver benchmark driver is missing: $driverPath"
}

$machineName = $env:COMPUTERNAME
if ([string]::IsNullOrWhiteSpace($machineName)) {
    $machineName = "local"
}
$machineName = $machineName -replace "[^A-Za-z0-9_-]", "_"
$runId = $RemoteRunName
$remoteBase = $RemoteBaseDirectory.TrimEnd("/")
$remoteRunDirectory = "$remoteBase/$runId"
$remoteSourceDirectory = "$remoteRunDirectory/source"
$remotePackagePath = "$remoteRunDirectory/worktree.tar.gz"
$remoteArtifactDirectory = "$remoteRunDirectory/artifacts"
$remoteChecksumManifest = "$remoteRunDirectory/artifacts.sha256"

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repositoryRoot "artifacts\${runId}_remote"
}
$localOutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $localOutputDirectory) {
    throw "Refusing to overwrite existing output directory: $localOutputDirectory"
}

$temporaryDirectory = Join-Path ([IO.Path]::GetTempPath()) $runId
$packagePath = Join-Path $temporaryDirectory "worktree.tar.gz"

try {
    New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null

    $tarArguments = @(
        "--create",
        "--gzip",
        "--file=$packagePath",
        "--exclude=.git",
        "--exclude=./.git",
        "--exclude=./.git/*",
        "--exclude=artifacts",
        "--exclude=./artifacts",
        "--exclude=./artifacts/*",
        "--exclude=.pytest_cache",
        "--exclude=./.pytest_cache",
        "--exclude=./.pytest_cache/*",
        "--exclude=__pycache__",
        "--exclude=*/__pycache__",
        "--exclude=*/__pycache__/*",
        "--exclude=.mypy_cache",
        "--exclude=.ruff_cache",
        "--exclude=.tox",
        "--exclude=.nox",
        "--exclude=htmlcov",
        "--exclude=build",
        "--exclude=dist",
        "--exclude=.venv",
        "--exclude=venv",
        "--exclude=*.pyc",
        "--exclude=*.pyo",
        "-C",
        $repositoryRoot,
        "."
    )
    Write-Host "[package worktree]"
    & $tarCommand.Source @tarArguments
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe failed while packaging the worktree."
    }

    $archiveEntries = @(& $tarCommand.Source "-tzf" $packagePath)
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe failed while validating the worktree archive."
    }
    $disallowedArchiveEntry = $archiveEntries | Where-Object {
        $_ -match "(^|/)(\.git|artifacts|\.pytest_cache|__pycache__|\.mypy_cache|\.ruff_cache|\.tox|\.nox|htmlcov|build|dist|\.venv|venv)(/|$)" -or
        $_ -match "\.(pyc|pyo)$"
    } | Select-Object -First 1
    if ($null -ne $disallowedArchiveEntry) {
        throw "Worktree archive contains an excluded entry: $disallowedArchiveEntry"
    }
    if (-not ($archiveEntries -contains "./test/ct_all_solvers/run_all_solvers.py")) {
        throw "Worktree archive does not include test/ct_all_solvers/run_all_solvers.py."
    }

    $localPackageHash = (Get-FileHash -LiteralPath $packagePath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "Local package SHA256: $localPackageHash"

    $createRemoteDirectory = @'
set -euo pipefail
if [ -e '__RUN_DIRECTORY__' ]; then
    printf 'Remote run directory already exists: %s\n' '__RUN_DIRECTORY__' >&2
    exit 1
fi
mkdir -p '__SOURCE_DIRECTORY__' '__ARTIFACT_DIRECTORY__'
'@.Replace("__RUN_DIRECTORY__", $remoteRunDirectory).
        Replace("__SOURCE_DIRECTORY__", $remoteSourceDirectory).
        Replace("__ARTIFACT_DIRECTORY__", $remoteArtifactDirectory)
    Invoke-RemoteBash -Step "create remote run directory" -ScriptText $createRemoteDirectory | Out-Null

    Invoke-Scp -Step "upload worktree archive" -Arguments @(
        "-q",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=$ConnectionTimeoutSeconds",
        $packagePath,
        "${RemoteHost}:$remotePackagePath"
    )

    $verifyAndExtract = @'
set -euo pipefail
expected='__EXPECTED_HASH__'
actual=$(sha256sum '__PACKAGE_PATH__' | awk '{print $1}')
if [ "$actual" != "$expected" ]; then
    printf 'Uploaded package SHA256 mismatch: expected %s, got %s\n' "$expected" "$actual" >&2
    exit 1
fi
tar -xzf '__PACKAGE_PATH__' -C '__SOURCE_DIRECTORY__'
test -f '__SOURCE_DIRECTORY__/test/ct_all_solvers/run_all_solvers.py'
printf '%s\n' "$actual"
'@.Replace("__EXPECTED_HASH__", $localPackageHash).
        Replace("__PACKAGE_PATH__", $remotePackagePath).
        Replace("__SOURCE_DIRECTORY__", $remoteSourceDirectory)
    $remotePackageHash = (Invoke-RemoteBash -Step "verify and extract worktree" -ScriptText $verifyAndExtract).Trim().ToLowerInvariant()
    if ($remotePackageHash -ne $localPackageHash) {
        throw "Remote package SHA256 did not match the local archive."
    }

    $remotePreflight = @'
set -euo pipefail
cd '__SOURCE_DIRECTORY__'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
'__REMOTE_PYTHON__' - <<'PY' | tee '__RUN_DIRECTORY__/preflight.json'
import importlib
import json
import platform
import sys

modules = {name: importlib.import_module(name) for name in ("torch", "astra", "matplotlib", "h5py", "pytest")}
torch = modules["torch"]
astra = modules["astra"]
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch CUDA is unavailable; FDK is required for this benchmark.")
if not astra.use_cuda():
    raise RuntimeError("ASTRA CUDA is unavailable; FDK is required for this benchmark.")

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
    "h5py": modules["h5py"].__version__,
    "pytest": modules["pytest"].__version__,
}, indent=2, sort_keys=True))
PY
'@.Replace("__SOURCE_DIRECTORY__", $remoteSourceDirectory).
        Replace("__RUN_DIRECTORY__", $remoteRunDirectory).
        Replace("__REMOTE_PYTHON__", $RemotePython)
    Invoke-RemoteBash -Step "remote preflight" -ScriptText $remotePreflight | Out-Null

    $runTests = @'
set -euo pipefail
cd '__SOURCE_DIRECTORY__'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.
'__REMOTE_PYTHON__' -m pytest -q test tests 2>&1 | tee '__RUN_DIRECTORY__/pytest.log'
'@.Replace("__SOURCE_DIRECTORY__", $remoteSourceDirectory).
        Replace("__RUN_DIRECTORY__", $remoteRunDirectory).
        Replace("__REMOTE_PYTHON__", $RemotePython)
    Invoke-RemoteBash -Step "remote regression tests" -ScriptText $runTests | Out-Null

    $runBenchmark = @'
set -euo pipefail
cd '__SOURCE_DIRECTORY__'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.
'__REMOTE_PYTHON__' test/ct_all_solvers/run_all_solvers.py --calibrate --device cuda --output-dir '__ARTIFACT_DIRECTORY__' --require-fdk 2>&1 | tee '__RUN_DIRECTORY__/benchmark.log'

for required in metrics.csv metrics.json calibration_profile.json manifest.json reconstructions.pt; do
    test -s '__ARTIFACT_DIRECTORY__'/$required
done
png_count=$(find '__ARTIFACT_DIRECTORY__' -type f -name '*.png' | wc -l)
if [ "$png_count" -lt 4 ]; then
    printf 'Expected at least four result images, found %s.\n' "$png_count" >&2
    exit 1
fi
if ! grep -qi 'fdk' '__ARTIFACT_DIRECTORY__/metrics.json'; then
    printf 'metrics.json does not contain an FDK result.\n' >&2
    exit 1
fi
'__REMOTE_PYTHON__' - <<'PY'
import json
import math
from pathlib import Path

artifact_dir = Path('__ARTIFACT_DIRECTORY__')
metrics = json.loads((artifact_dir / 'metrics.json').read_text(encoding='utf-8'))
expected_algorithms = (
    'fbp', 'sirt', 'landweber', 'cgls', 'lsqr', 'sart', 'os_sart',
    'mlem', 'osem', 'tikhonov', 'tv_fista', 'fdk',
)
two_d_cases = (
    'parallel_2d/tissue_breast_dense_clean_128',
    'parallel_2d/tissue_breast_sparse_poisson_128',
    'parallel_2d/tissue_breast_limited_angle_128',
)
fdk_case = 'cone_3d/spheres_astra_12'
if metrics.get('algorithm_ids') != list(expected_algorithms):
    raise RuntimeError('metrics.json does not declare exactly the requested 12 algorithms.')
records = metrics.get('records')
if not isinstance(records, list):
    raise RuntimeError('metrics.json records must be a list.')
expected_pairs = {
    (algorithm, case_id)
    for algorithm in expected_algorithms[:-1]
    for case_id in two_d_cases
}
expected_pairs.add(('fdk', fdk_case))
actual_pairs = [(record.get('algorithm'), record.get('case_id')) for record in records]
if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
    raise RuntimeError(
        'metrics.json must contain the complete 34-row 11x3 2D matrix plus FDK.'
    )
if len(set(actual_pairs)) != len(actual_pairs):
    raise RuntimeError('metrics.json contains duplicate algorithm/case records.')
successful_algorithm_ids = {
    record.get('algorithm') for record in records if record.get('status') == 'success'
}
if successful_algorithm_ids != set(expected_algorithms):
    raise RuntimeError(
        'Expected exactly 12 successful algorithm IDs, including every requested solver.'
    )
for record in records:
    if record.get('status') != 'success':
        raise RuntimeError(f"Benchmark record is not successful: {record!r}")
    for metric_name in ('psnr', 'ssim'):
        value = record.get(metric_name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"Invalid {metric_name} for {record!r}")
PY
cd '__ARTIFACT_DIRECTORY__'
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > '__CHECKSUM_MANIFEST__'
test -s '__CHECKSUM_MANIFEST__'
'@.Replace("__SOURCE_DIRECTORY__", $remoteSourceDirectory).
        Replace("__RUN_DIRECTORY__", $remoteRunDirectory).
        Replace("__REMOTE_PYTHON__", $RemotePython).
        Replace("__ARTIFACT_DIRECTORY__", $remoteArtifactDirectory).
        Replace("__CHECKSUM_MANIFEST__", $remoteChecksumManifest)
    Invoke-RemoteBash -Step "remote all-solver benchmark" -ScriptText $runBenchmark | Out-Null

    New-Item -ItemType Directory -Path $localOutputDirectory | Out-Null
    Invoke-Scp -Step "download benchmark artifacts" -Arguments @(
        "-q",
        "-r",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=$ConnectionTimeoutSeconds",
        "${RemoteHost}:$remoteArtifactDirectory",
        $localOutputDirectory
    )
    Invoke-Scp -Step "download remote verification files" -Arguments @(
        "-q",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=$ConnectionTimeoutSeconds",
        "${RemoteHost}:$remoteChecksumManifest",
        "${RemoteHost}:$remoteRunDirectory/preflight.json",
        "${RemoteHost}:$remoteRunDirectory/pytest.log",
        "${RemoteHost}:$remoteRunDirectory/benchmark.log",
        $localOutputDirectory
    )

    $localArtifactDirectory = Join-Path $localOutputDirectory "artifacts"
    $localChecksumManifest = Join-Path $localOutputDirectory "artifacts.sha256"
    Assert-ArtifactChecksums -ArtifactDirectory $localArtifactDirectory -ChecksumManifest $localChecksumManifest

    $orchestrationMetadata = [ordered]@{
        schema_version = 1
        remote_host = $RemoteHost
        remote_directory = $remoteRunDirectory
        remote_python = $RemotePython
        device = $Device
        thread_limits = [ordered]@{
            OMP_NUM_THREADS = 1
            MKL_NUM_THREADS = 1
            OPENBLAS_NUM_THREADS = 1
        }
        worktree_sha256 = $localPackageHash
        artifacts_sha256_verified = $true
    }
    $orchestrationMetadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (
        Join-Path $localOutputDirectory "remote_orchestration.json"
    ) -Encoding UTF8

    Write-Host "Remote all-solver benchmark completed."
    Write-Host "Remote worktree: $remoteRunDirectory"
    Write-Host "Verified local artifacts: $localArtifactDirectory"
}
finally {
    if (Test-Path -LiteralPath $temporaryDirectory) {
        Remove-Item -LiteralPath $temporaryDirectory -Recurse -Force
    }
}
