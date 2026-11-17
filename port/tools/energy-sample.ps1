# Sample the platform power counters once per second and write them with an epoch timestamp.
#
#   powershell -File port/tools/energy-sample.ps1 -Seconds 300 -Out build\energy.csv
#
# The counter paths come from Get-Counter -ListSet 'Energy Meter' rather than being written by hand: the hand-written form
# does not match what the provider publishes and silently returns an empty sample set (PERFORMANCE.md section 20.18).
# The `_Total` instance is 0 on this platform, so it is excluded and the remaining instances are summed per sample.
[CmdletBinding()]
param(
    [int] $Seconds = 300,
    [Parameter(Mandatory = $true)][string] $Out,
    [int] $IntervalMs = 1000
)

$ErrorActionPreference = 'Continue'
$set = Get-Counter -ListSet 'Energy Meter' -ErrorAction SilentlyContinue
if (-not $set) { "no Energy Meter counter set on this machine" | Set-Content -LiteralPath $Out -Encoding ASCII; return }
$paths = @($set.PathsWithInstances | Where-Object { $_ -match '\\Power$' } | Where-Object { $_ -notmatch '\\_Total\\' })
"epoch,pwr_sum,samples" | Set-Content -LiteralPath $Out -Encoding ASCII
$t0 = Get-Date
$deadline = $t0.AddSeconds($Seconds)
while ((Get-Date) -lt $deadline) {
    $sum = 0.0
    $n = 0
    try {
        $c = Get-Counter -Counter $paths -ErrorAction Stop
        foreach ($s in $c.CounterSamples) { $sum += [double]$s.CookedValue; $n++ }
    } catch { }
    if ($n -gt 0) {
        $epoch = [math]::Round(([datetimeoffset](Get-Date)).ToUnixTimeMilliseconds() / 1000.0, 3)
        "$epoch,$sum,$n" | Add-Content -LiteralPath $Out -Encoding ASCII
    }
    Start-Sleep -Milliseconds $IntervalMs
}
