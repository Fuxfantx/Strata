# Build the strata-vk engine from third_party/llama.cpp (fetch it first with fetch-llama.ps1).
#
#   powershell -ExecutionPolicy Bypass -File port/tools/build-engine.ps1 [-Build build] [-Target strata-vk] [-Extra "..."]
#
# The build directory is reused, so an incremental rebuild after an engine edit is seconds; the first configure is the
# expensive one because it compiles every Vulkan shader variant.
#
# The Hexagon NPU backend is built when a Hexagon SDK is found (looked up in the usual places, or given with -HexagonSdk).
# Only its host side is built here: the DSP skels are shipped prebuilt and signed (`libggml-htp-v8x.so` + `libggml-htp.cat`
# beside the exe), so nothing in this build needs a signing certificate.
param(
    [string]$Build = "build",
    [string]$Target = "strata-vk",
    [string]$Extra = "",
    [string]$Generator = "Ninja",
    [string]$HexagonSdk = "",
    [switch]$NoHexagon,
    [switch]$Reconfigure
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$src  = Join-Path $root "third_party\llama.cpp"
$bld  = Join-Path $root $Build
if (-not (Test-Path (Join-Path $src "CMakeLists.txt"))) { throw "missing $src - run fetch-llama.ps1 first" }

$vs = "C:\Program Files\Microsoft Visual Studio\18\Community"
$env:Path = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$vs\VC\Tools\Llvm\ARM64\bin;" + $env:Path
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.363.1"

# Find a Hexagon SDK: the layout is <sdk>/incs, <sdk>/ipc, <sdk>/tools/HEXAGON_Tools/<version>.
function Find-HexagonSdk {
    $candidates = @()
    if ($HexagonSdk) { $candidates += $HexagonSdk }
    if ($env:HEXAGON_SDK_ROOT) { $candidates += $env:HEXAGON_SDK_ROOT }
    $candidates += @(
        "C:\Users\Fuxfantx\C-Res\hexagon-sdk\6.7.0.0",
        "C:\Qualcomm\Hexagon_SDK\6.4.0.0",
        "C:\Qualcomm\Hexagon_SDK\6.2.0.0"
    )
    foreach ($c in $candidates) {
        if (-not $c -or -not (Test-Path (Join-Path $c "incs"))) { continue }
        $tools = Get-ChildItem (Join-Path $c "tools\HEXAGON_Tools") -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($tools) { return @{ Root = (Resolve-Path $c).Path; Tools = $tools.FullName } }
    }
    return $null
}

if ($Reconfigure -or -not (Test-Path (Join-Path $bld "CMakeCache.txt"))) {
    $vklib = "C:/VulkanSDK/1.4.363.1/Lib-ARM64/vulkan-1.lib"
    $cfg = @(
        "-S", $src, "-B", $bld, "-G", $Generator,
        "-DCMAKE_C_COMPILER=clang-cl", "-DCMAKE_CXX_COMPILER=clang-cl",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_VULKAN=ON", "-DGGML_OPENMP=OFF",
        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=ON", "-DLLAMA_CURL=OFF",
        "-DSTRATA_VK_DIR=$root/port/engine",
        "-DGGML_BACKEND_DL=ON",
        "-DVulkan_LIBRARY=$vklib",
        "-DCMAKE_C_FLAGS_RELEASE=-O2", "-DCMAKE_CXX_FLAGS_RELEASE=-O2"
    )

    $hex = if ($NoHexagon) { $null } else { Find-HexagonSdk }
    if ($hex) {
        $sdkPath   = ($hex.Root  -replace '\\', '/')
        $toolsPath = ($hex.Tools -replace '\\', '/')
        $cfg += @("-DGGML_HEXAGON=ON", "-DHEXAGON_SDK_ROOT=$sdkPath", "-DHEXAGON_TOOLS_ROOT=$toolsPath",
                  "-DPREBUILT_LIB_DIR=toolv21_v81")
        Write-Host "hexagon: building the host backend against $sdkPath"
        Write-Host "hexagon: the DSP skels are shipped prebuilt and signed; this build signs nothing"
    } else {
        Write-Host "hexagon: no SDK found, building without the NPU backend (-HexagonSdk to point at one)"
    }

    if ($Extra) { $cfg += $Extra.Split(" ", [char]0x20, [System.StringSplitOptions]::RemoveEmptyEntries) }
    & cmake @cfg
    if ($LASTEXITCODE -ne 0) { throw "configure failed" }
}

if ((Get-Content (Join-Path $bld "CMakeCache.txt") -Raw) -match "GGML_HEXAGON:BOOL=ON") {
    & cmake --build $bld --target ggml-hexagon
    if ($LASTEXITCODE -ne 0) { throw "hexagon backend failed to build" }
}

& cmake --build $bld --target $Target
if ($LASTEXITCODE -ne 0) { throw "build failed" }

# The engine loads its backends at run time, so the ARM64 loader has to sit beside the exe.  System32's vulkan-1.dll on
# this machine is x64 (the shell runs under emulation), and an x64 loader next to an arm64 process is not found.
$bin = Join-Path $bld "bin"
$loader = Join-Path $env:LOCALAPPDATA "Strata\components\engine\0.2.13-windows-arm64-vulkan\vulkan-1.dll"
if ((Test-Path $bin) -and (Test-Path $loader) -and -not (Test-Path (Join-Path $bin "vulkan-1.dll"))) {
    Copy-Item $loader (Join-Path $bin "vulkan-1.dll")
    Write-Host "staged ARM64 vulkan-1.dll into $bin"
}

# The signed DSP skels come with the engine component; the host backend only needs them beside the exe at run time.
# Stage them by content, not by presence: a stale catalog is what made the NPU look like it had no vision-encoder ops
# (the 11/5 skels carried the fork's repack ops and no conv2d, while the current DSP-side source is the other way round).
# `port/tools/fetch-signed-htp.ps1` refreshes the component with Qualcomm's Microsoft-signed set.
$component = Join-Path $env:LOCALAPPDATA "Strata\components\engine\0.2.13-windows-arm64-vulkan"
if ((Test-Path $bin) -and (Test-Path $component)) {
    Get-ChildItem $component -Filter "libggml-htp*" -ErrorAction SilentlyContinue | ForEach-Object {
        $dst = Join-Path $bin $_.Name
        $same = (Test-Path $dst) -and
                ((Get-FileHash -Algorithm SHA256 $_.FullName).Hash -eq (Get-FileHash -Algorithm SHA256 $dst).Hash)
        if (-not $same) {
            Copy-Item $_.FullName $bin -Force
            Write-Host "staged $($_.Name) (content differs)"
        }
    }
}
Write-Host "built: $bld\bin"
