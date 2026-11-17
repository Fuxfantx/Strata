# Sample the machine counters that decide whether a second engine is starved by page-cache eviction or by CPU contention.
#
#   powershell -File port/tools/sample-counters.ps1 -Seconds 240 -Out build\counters.csv
#
# Writes one CSV row per sample: seconds since start, disk read bytes/s, available MiB, hard faults/s.
[CmdletBinding()]
param(
    [int] $Seconds = 240,
    [Parameter(Mandatory = $true)][string] $Out,
    [int] $IntervalMs = 500
)

$ErrorActionPreference = 'Continue'
$paths = @(
    '\LogicalDisk(_Total)\Disk Read Bytes/sec',
    '\Memory\Available MBytes',
    '\Memory\Pages/sec'
)
"t,disk_read_bps,avail_mib,pages_sec" | Set-Content -LiteralPath $Out -Encoding ASCII
$t0 = Get-Date
$deadline = $t0.AddSeconds($Seconds)
while ((Get-Date) -lt $deadline) {
    $row = @{ disk_read_bps = ''; avail_mib = ''; pages_sec = '' }
    try {
        $c = Get-Counter -Counter $paths -ErrorAction Stop
        foreach ($s in $c.CounterSamples) {
            switch -Wildcard ($s.Path) {
                '*disk read bytes*' { $row.disk_read_bps = [string][long]$s.CookedValue }
                '*available mbytes*' { $row.avail_mib = [string][long]$s.CookedValue }
                '*pages/sec*' { $row.pages_sec = [string][long]$s.CookedValue }
            }
        }
    } catch { }
    $t = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    "$t,$($row.disk_read_bps),$($row.avail_mib),$($row.pages_sec)" | Add-Content -LiteralPath $Out -Encoding ASCII
    Start-Sleep -Milliseconds $IntervalMs
}
