# Stage the Windows-on-ARM desktop app resources.
#
#   pwsh -File port/packaging/stage-app.ps1 -NodeDir <arm64 node> -PythonDir <arm64 python> -SitePackages <wheels dir>
#
# Everything this script copies from *this repository* is already here: the engine component, Control's Python sources,
# the app. What it cannot take from the repository are three ARM64 runtimes that do not belong in git — a Node 24
# runtime, a CPython 3.15 interpreter and Control's installed wheels. They are parameters, and the script refuses to
# stage an app without them rather than shipping one that fails at the user's first start.
#
# The Tauri shell expects, under app/desktop/src-tauri/resources:
#   node/            the private Node runtime (ARM64)
#   control/         Strata Control: terminal/strata.py, terminal/strata_control/, terminal/strata_cli/, terminal/serve/, terminal/tools/, terminal/data/, python/
#   engine-bundle/   the engine component the app installs on first start (built by port/tools/package_component.py)
#   harness/         the DeepSeek Harness bundle (a placeholder here; the agent features need the real one)
#
# It stages only; the build itself stays `npm run tauri build` in app/desktop.

param(
    [string]$Strata = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$NodeDir = "",
    [string]$PythonDir = "",
    [string]$SitePackages = "",
    [string]$EngineBundle = ""
)

$ErrorActionPreference = "Stop"
$res = Join-Path $Strata "app\desktop\src-tauri\resources"
Write-Host "staging into $res"
New-Item -ItemType Directory -Force -Path $res | Out-Null

# The machine field of a PE file; 0xAA64 is ARM64. Everything staged here runs natively on Windows on ARM, and a runtime
# taken from an x64 development machine runs only under emulation (or, for a DLL the engine loads, not at all), which shows
# up much later as "the engine never loads". So a runtime of the wrong architecture is refused here, at staging time.
function Get-PeMachine([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $head = New-Object 'byte[]' 64
        if ($stream.Read($head, 0, 64) -ne 64) { return 0 }
        $stream.Position = [BitConverter]::ToInt32($head, 0x3C) + 4
        $machine = New-Object 'byte[]' 2
        if ($stream.Read($machine, 0, 2) -ne 2) { return 0 }
        return [BitConverter]::ToUInt16($machine, 0)
    } finally {
        $stream.Dispose()
    }
}

function Assert-Arm64([string]$Path, [string]$What) {
    $machine = Get-PeMachine $Path
    if ($machine -ne 0xAA64) {
        throw ("{0}: {1} is not an ARM64 binary (PE machine 0x{2:X4}); this app ships for Windows on ARM" -f $What, $Path, $machine)
    }
}

# --- node (ARM64) -------------------------------------------------------------------------------------------------
if (-not $NodeDir -or -not (Test-Path (Join-Path $NodeDir "node.exe"))) {
    throw "node: pass -NodeDir pointing at an ARM64 Node 24 runtime (nodejs.org, win-arm64 zip). It is not in this repository."
}
Assert-Arm64 (Join-Path $NodeDir "node.exe") "node"
$nodeDst = Join-Path $res "node"
if (Test-Path $nodeDst) { Remove-Item $nodeDst -Recurse -Force }
Copy-Item $NodeDir $nodeDst -Recurse
Write-Host "node: staged"

# --- control ------------------------------------------------------------------------------------------------------
if (-not $PythonDir -or -not (Test-Path (Join-Path $PythonDir "python.exe"))) {
    throw "control: pass -PythonDir pointing at an ARM64 CPython 3.15 tree. It is not in this repository."
}
Assert-Arm64 (Join-Path $PythonDir "python.exe") "control"
if (-not $SitePackages -or -not (Test-Path $SitePackages)) {
    throw "control: pass -SitePackages pointing at the directory holding Control's installed wheels."
}
$ctl = Join-Path $res "control"
if (Test-Path $ctl) { Remove-Item $ctl -Recurse -Force }
New-Item -ItemType Directory -Force -Path $ctl | Out-Null
# `setup.py` stays: the Terminal and the MCP server read the model catalogue out of it at run time.
# Every entry below is required: a missing one used to be skipped in silence, which shipped an app whose
# Control was missing a module and failed only at the user's first install.
$term = Join-Path $Strata "terminal"
foreach ($item in @("strata.py", "strata_control", "strata_cli", "serve", "tools", "data")) {
    $p = Join-Path $term $item
    if (-not (Test-Path $p)) { throw "control: $item is missing from $Strata - the app would ship without it" }
    Copy-Item $p (Join-Path $ctl $item) -Recurse -Force
}
foreach ($item in @("setup.py", "requirements.txt", "requirements-monitor.txt")) {
    $p = Join-Path $term $item
    if (Test-Path $p) { Copy-Item $p (Join-Path $ctl $item) -Force }
}
# the bundled interpreter: a whole CPython ARM64 tree plus the Control wheels
$pyDst = Join-Path $ctl "python"
Copy-Item $PythonDir $pyDst -Recurse -Force
$sp = Join-Path $pyDst "Lib\site-packages"
New-Item -ItemType Directory -Force -Path $sp | Out-Null
Copy-Item (Join-Path $SitePackages "*") $sp -Recurse -Force
# the app records which interpreter and lock it shipped; the version is the product version, which lives in package.json
$pkg = Get-Content (Join-Path $Strata "app\host\package.json") -Raw | ConvertFrom-Json
# a native command's exit code is not an error in PowerShell, so a broken interpreter would otherwise be recorded as a null
# version and shipped
$pythonVersion = (& (Join-Path $pyDst "python.exe") -c "import platform; print(platform.python_version())")
if ($LASTEXITCODE -ne 0 -or -not $pythonVersion) {
    throw "control: the bundled interpreter at $pyDst could not say its version (exit $LASTEXITCODE) - the app would not start"
}
$bundle = @{
    version   = $pkg.version
    python    = "$pythonVersion".Trim()
    target    = "windows-arm64"
    staged_at = (Get-Date).ToString("o")
} | ConvertTo-Json
Set-Content (Join-Path $ctl "BUNDLE.json") $bundle -Encoding utf8
Write-Host "control: staged"

# --- engine bundle ------------------------------------------------------------------------------------------------
# Built from this repository by `python port/tools/package_component.py`, which writes port/dist/<version>-<platform>-<backend>.
if (-not $EngineBundle) {
    $version = $pkg.version
    $EngineBundle = Join-Path $Strata "port\dist\$version-windows-arm64-vulkan"
}
$engineDst = Join-Path $res "engine-bundle"
if (-not (Test-Path (Join-Path $EngineBundle "engine-component.json"))) {
    throw "engine-bundle: no component at $EngineBundle - run 'python port/tools/package_component.py --verify' first"
}
if (Test-Path $engineDst) { Remove-Item $engineDst -Recurse -Force }
New-Item -ItemType Directory -Force -Path $engineDst | Out-Null
Copy-Item (Join-Path $EngineBundle "*") $engineDst -Recurse -Force
Write-Host "engine-bundle: staged"

# --- harness placeholder ------------------------------------------------------------------------------------------
$harness = Join-Path $res "harness"
New-Item -ItemType Directory -Force -Path $harness | Out-Null
if (-not (Test-Path (Join-Path $harness "package.json"))) {
    @{ name = "strata-harness-placeholder"; private = $true; note = "DeepSeek Harness is not part of this ARM64 port" } |
        ConvertTo-Json | Set-Content (Join-Path $harness "package.json") -Encoding utf8
}
Write-Host "harness: placeholder"
