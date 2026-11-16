# Produce the optional dense-layer companion pack for a Gyro-S style model.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/make-compat.ps1 -Pack <in.gguf> [-Out <out.gguf>] [-Type TQK6]
#
# The companion holds only the dense tensors (everything except the routed experts and the per-layer embedding table),
# requantised to a trellis type.  On this machine TQK6 is the fastest matvec type and needs 2.125 bits per weight, so the
# dense side of a token shrinks from about 3.7 GiB to 1.35 GiB.  It is lossy on the dense layers, which is why the pack is
# opt-in: the engine runs the original file unless it is told to use this one.
#
# Gyro-S reference run: 1224 tensors, 3701 MiB in, 1370 MiB out, about 17 minutes on this machine.
param(
    [Parameter(Mandatory = $true)][string]$Pack,
    [string]$Out = "",
    [string]$Type = "TQK6",
    [string]$Quantize = "build\bin\llama-quantize.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Pack)) { throw "missing $Pack" }
if (-not (Test-Path $Quantize)) { throw "missing $Quantize - build the tools first" }
if (-not $Out) { $Out = [IO.Path]::ChangeExtension($Pack, ".compat.gguf") }

# everything except the routed experts and the per-layer embedding table
$only = '^(?!.*(_exps|per_layer_token_embd)).*$'

Write-Host "reading   $Pack"
Write-Host "writing   $Out"
Write-Host "selection $only -> $Type (Q8_0 for tensors whose shape cannot hold a 128-wide block)"

& $Quantize --only-tensors $only --tensor-type ".*=$Type" --allow-requantize $Pack $Out Q8_0
if ($LASTEXITCODE -ne 0) { throw "llama-quantize failed with $LASTEXITCODE" }

$in  = (Get-Item $Pack).Length
$out = (Get-Item $Out).Length
Write-Host ("done: {0:N0} MiB -> {1:N0} MiB ({2:N2}x smaller)" -f ($in / 1MB), ($out / 1MB), ($in / $out))
