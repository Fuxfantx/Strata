# bench-suite.ps1 - the five-pack TPS table in one command.
#
# Every round of the port's TPS work started by running the same five measurements by hand; this is that list. It prints a
# table and, with -Out, writes the same numbers as JSON so two runs can be diffed.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/bench-suite.ps1
#   ... -PerfCap 80        # spend only 80% of wall time working (see --perf-cap in the engine)
#   ... -Out build\bench.json
#
# The two oversized packs (Q2, Gyro-S) page while they run - they need more RAM than the machine has - so run them when the
# machine is otherwise idle; -SkipBig leaves them out. Gyro-M was replaced by Gyro-S (the smaller TQ1_0 pack) for all tests that
# do not need the M-specific format.
param(
    [string]$Exe = "build\bin\strata-vk.exe",
    [int]$PerfCap = 100,
    [switch]$SkipBig,
    [string]$Out = ""
)

# Deliberately no $ErrorActionPreference = Stop: the engine writes its log to stderr, and a terminating preference turns that
# into an error record.
$packs = @(
    @{ label = "tiny";    path = "C:\Users\Fuxfantx\C-Res\Qwen\Tiny-Qwen.gguf";                                    pp = 512; tg = 64; big = $false },
    @{ label = "27b";     path = "C:\Users\Fuxfantx\C-Res\Qwen\3.8-27B-Q4\Qwen3.8-27B-Q4_0.gguf";                  pp = 512; tg = 16; big = $false },
    @{ label = "moe";     path = "C:\Users\Fuxfantx\C-Res\Qwen\3.6-35A3\Qwen3.6-35B-A3B-Q4_0.gguf";                pp = 512; tg = 32; big = $false },
    @{ label = "q2_0";    path = "C:\Users\Fuxfantx\C-Res\Qwen\3.8-Flash-Next-Q2\Qwen3.8-Flash-Next-Q2_0-00001-of-00002.gguf"; pp = 64; tg = 32; big = $true },
    @{ label = "gyro_s";  path = "C:\Users\Fuxfantx\C-Res\Qwen\3.8-Flash-Next-GS\Gyro-S-TQ1_0.gguf";                pp = 430; tg = 24; big = $true }
)

$errFile = Join-Path $env:TEMP "bench-suite.err"
$outFile = Join-Path $env:TEMP "bench-suite.out"

function Run([string[]]$argv) {
    & $Exe @argv 1> $outFile 2> $errFile | Out-Null
    return (Get-Content $outFile -ErrorAction SilentlyContinue)
}

# -like first: a second -match would overwrite $matches with the tag match, which has no capture groups.
function Rate($lines, [string]$tag) {
    foreach ($l in $lines) {
        if (($l -like "$tag*") -and ($l -match "= ([0-9.]+) tok/s")) { return [double]$matches[1] }
    }
    return 0.0
}

$rows = @()
foreach ($p in $packs) {
    if ($SkipBig -and $p.big) { continue }
    if (-not (Test-Path $p.path)) { Write-Host ("missing: {0}" -f $p.label); continue }
    $common = @("--native", $p.path, "--perf-cap", "$PerfCap")
    $ppLines = Run ($common + @("--bench-pp", "$($p.pp)"))
    $tgLines = Run ($common + @("--bench-tg", "$($p.tg)"))
    $pp = Rate $ppLines "pp "
    $tg = Rate $tgLines "tg "
    $rows += [pscustomobject][ordered]@{ label = $p.label; pp_tokens = $p.pp; tg_tokens = $p.tg; pp_tok_s = $pp; tg_tok_s = $tg }
    Write-Host ("{0,-7} pp{1,-4} {2,9:N1} tok/s   tg{3,-3} {4,7:N2} tok/s" -f $p.label, $p.pp, $pp, $p.tg, $tg)
}

if ($Out -ne "") {
    $dir = Split-Path -Parent $Out
    if ($dir -and (-not (Test-Path $dir))) { New-Item -ItemType Directory -Path $dir | Out-Null }
    $rows | ConvertTo-Json | Set-Content -Path $Out -Encoding utf8
    Write-Host ("wrote {0}" -f $Out)
}
