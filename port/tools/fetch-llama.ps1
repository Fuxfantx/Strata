# Fetch the llama.cpp tree the engine is built from.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/fetch-llama.ps1
#
# Upstream llama.cpp is not vendored here: four thousand files of somebody else's source, 99% of them untouched, do not
# belong in a tree this small. What is kept is the part that is ours - `port/llama-overlay-vanilla/`, the files this port
# changes or adds, at their real paths - and this script, which downloads the pinned vanilla commit and copies the overlay
# over it. The result is `third_party/llama.cpp`, the tree `BUILD-WOA.md` builds from.
#
# The pin below is the commit the tree is verified on (tag b12722).  The measurements that used to be quoted here were
# taken on the previous pin (a68c6136); they are kept in `port/docs/PERFORMANCE-LOG.md` with the commit they belong to,
# and are not a claim about this pin. `port/docs/MAINTAINER.md` has the composition and the pitfalls.
#
# `-Verify` runs the feature checker and a CMake configure so a broken tree says so instead of printing "ready".

param(
    [string]$Pin = "e6443bd7eb970d940b0669cd9ea0d82d91d620a1",   # tag b12722
    [string]$Repo = "ggml-org/llama.cpp",
    [string]$Dest = "",
    [string]$Overlay = "port\llama-overlay-vanilla",
    [switch]$Verify,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Dest) { $Dest = Join-Path $root "third_party\llama.cpp" }

$overlay = Join-Path $root $Overlay
if (-not (Test-Path $overlay)) { throw "missing $overlay" }
if (Test-Path (Join-Path $Dest "ggml\CMakeLists.txt")) {
    Write-Host "already fetched: $Dest (delete it to fetch again)"
    exit 0
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("llama-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$tar = Join-Path $tmp "llama.tar.gz"
$url = "https://codeload.github.com/$Repo/tar.gz/$Pin"

Write-Host "downloading $Repo at $Pin"
# curl first: Windows PowerShell's Invoke-WebRequest goes through schannel and returns an empty stream from codeload
# often enough that it cannot be the only path.  The fallback forces TLS 1.2, which is the other half of that failure.
$downloaded = $false
if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
    # -sS: no progress meter on stderr. With $ErrorActionPreference = Stop a native command's stderr becomes a terminating
    # error when the caller pipes the script's output, which aborted the download two seconds in.
    & curl.exe -fsSL --retry 3 --retry-delay 2 -o $tar $url
    $downloaded = ($LASTEXITCODE -eq 0) -and (Test-Path $tar) -and ((Get-Item $tar).Length -gt 1MB)
}
if (-not $downloaded) {
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $tar -UseBasicParsing
        $downloaded = (Test-Path $tar) -and ((Get-Item $tar).Length -gt 1MB)
    } catch {
        throw "download failed ($url): $($_.Exception.Message)"
    }
}
if (-not $downloaded) { throw "download produced no usable archive from $url" }

Write-Host "extracting"
& tar -xzf $tar -C $tmp
$src = Get-ChildItem $tmp -Directory | Where-Object { Test-Path (Join-Path $_.FullName "CMakeLists.txt") } | Select-Object -First 1
if (-not $src) { throw "the archive did not contain a llama.cpp tree" }

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Copy-Item (Join-Path $src.FullName "*") $Dest -Recurse -Force

Write-Host "applying $Overlay"
Copy-Item (Join-Path $overlay "*") $Dest -Recurse -Force

Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "ready: $Dest"

if ($Verify) {
    Write-Host "checking the four feature groups survived the overlay"
    & $Python (Join-Path $root "port\tools\check-fork-features.py") $Dest
    if ($LASTEXITCODE -ne 0) { throw "the overlay is missing a feature marker (see above)" }

    $probe = Join-Path ([IO.Path]::GetTempPath()) ("llama-cfg-" + [guid]::NewGuid().ToString("N"))
    Write-Host "configuring $probe"
    & cmake -S $Dest -B $probe -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl `
        -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF -DLLAMA_BUILD_TESTS=OFF `
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "the fetched tree does not configure" }
    Write-Host "configures; the full build is BUILD-WOA.md"
    Remove-Item $probe -Recurse -Force -ErrorAction SilentlyContinue
}
