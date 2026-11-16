# Fetches the Microsoft-signed Hexagon HTP skels and stages them beside the engine.
#
# Why this exists: the DSP refuses to load an unsigned ops library.  Our own build produces `libggml-htp-v*.so` but signs
# it only with a self-signed test certificate (and only when HEXAGON_HTP_CERT is set), while Qualcomm publishes a
# Microsoft-signed catalog for the very same skel interface in the GenieX Windows SDK.  That catalog is what makes the
# NPU usable on this machine without test-signing, and the skels it covers contain the vision-encoder ops (conv2d,
# conv2d_1d, im2col_3d) that the DSP-side source builds but our older installed copy did not have.
#
#   pwsh -File port/tools/fetch-signed-htp.ps1                 # download, verify, stage into build\bin
#   pwsh -File port/tools/fetch-signed-htp.ps1 -Component       # also stage into the engine component directory
#
# Verified on this machine: the signed skels from SDK v0.11.0 load with our ggml-hexagon.dll (a tiny model runs with
# --npu dense), and the fork's tiled repack path now fails cleanly with `dsp-rsp NO-SUPPORT` instead of hanging, which is
# the ops surface the current DSP-side source builds (repack removed, conv2d added).
[CmdletBinding()]
param(
    [string] $Version = "v0.11.0",
    [string] $Cache = (Join-Path $env:USERPROFILE "C-Res\geniex-sdk"),
    [switch] $Component
)

$ErrorActionPreference = "Stop"
$repo = "https://github.com/qualcomm/GenieX/releases/download/$Version"
$zipName = "geniex-sdk-windows-arm64-$Version.zip"
$shaName = "$zipName.sha256"
$zip = Join-Path $Cache $zipName
New-Item -ItemType Directory -Force -Path $Cache | Out-Null

if (-not (Test-Path $zip)) {
    Write-Host "downloading $zipName ..."
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 900 -Uri "$repo/$zipName" -OutFile $zip
}

# Verify against the release's own sha256 asset; a wrong or truncated download must never reach build\bin.
$expected = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 60 -Uri "$repo/$shaName").Content
if ($expected -is [byte[]]) { $expected = [System.Text.Encoding]::UTF8.GetString($expected) }
$expectedHash = ($expected -split "\s+")[0].Trim().ToLowerInvariant()
$actualHash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) {
    throw "sha256 mismatch for $zipName`: expected $expectedHash, got $actualHash"
}
Write-Host "sha256 ok: $actualHash"

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($zip)
try {
    $prefix = "sdk-windows-arm64/lib/llama_cpp/"
    $wanted = @("libggml-htp.cat", "libggml-htp.inf",
                "libggml-htp-v73.so", "libggml-htp-v75.so", "libggml-htp-v79.so",
                "libggml-htp-v81.so", "libggml-htp-v85.so")
    $targets = @((Join-Path $PSScriptRoot "..\..\build\bin"))
    if ($Component) {
        $targets += (Join-Path $env:LOCALAPPDATA "Strata\components\engine\0.2.13-windows-arm64-vulkan")
    }
    foreach ($entry in $archive.Entries) {
        if (-not $entry.FullName.StartsWith($prefix)) { continue }
        $name = $entry.Name
        if ($wanted -notcontains $name) { continue }
        foreach ($dir in $targets) {
            if (-not (Test-Path $dir)) { continue }
            $out = Join-Path $dir $name
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $out, $true)
            Write-Host ("staged {0} -> {1}" -f $name, $dir)
        }
    }
} finally {
    $archive.Dispose()
}

foreach ($dir in $targets) {
    if (-not (Test-Path $dir)) { continue }
    Write-Host "--- $dir"
    Get-ChildItem $dir -Filter "libggml-htp*" | ForEach-Object {
        $h = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.Substring(0, 16).ToLowerInvariant()
        Write-Host ("    {0,-22} {1,9} bytes  sha256 {2}..." -f $_.Name, $_.Length, $h)
    }
}
