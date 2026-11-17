<#
.SYNOPSIS
Register (or remove) HTP/DSP skeleton libraries in the Windows FastRPC module allow-list.

.DESCRIPTION
The Windows NSP-MCDM FastRPC driver (qcnspmcdm) does not implement
FASTRPC_MANAGE_DSP_LIBRARY_PATH (returns AEE_EUNSUPPORTED = 0x14) and ignores
ADSP_LIBRARY_PATH. Instead it only loads a DSP module that appears in a per-device
allow-list:

    <device>\Device Parameters\Mappings\<DOMAIN>\<Set>\<key>
        Local  = host path of the .so
        remote = device path, e.g. \cdsp\libname.so

The <key> is sha256 of the *device* path in UTF-8, lowercase hex. Verified against
four shipped entries (version.so, libUbwcD.so, libbenchmark_skel.so,
libsysmonquery_skel.so): 4/4 match. Without an entry the driver returns
AEE_ERPC (0x200) from remote_handle64_open.

Writing under HKLM requires an elevated prompt.

.PARAMETER Library
One or more .so files. Their file names are used for the device path.

.PARAMETER Domain
CDSP (default) or HTP. ggml-hexagon opens sessions with _dom=cdsp.

.PARAMETER Set
Allow-list set name. The driver reads Mappings\<DOMAIN>\<Set>; shipped values use
"Default". A device with a PathMappingKeyName value uses that name instead.

.PARAMETER Remove
Delete the entries instead of adding them.

.PARAMETER Verify
Only report whether each entry is present. Needs no elevation.

.EXAMPLE
.\htp-register-module.ps1 -Library ..\..\build\bin\libggml-htp-v81.so -Verify
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string[]] $Library,
    [ValidateSet('CDSP', 'HTP')][string] $Domain = 'CDSP',
    [string] $Set = 'Default',
    [switch] $Remove,
    [switch] $Verify
)

$ErrorActionPreference = 'Stop'

# The NSP-MCDM device owns the CDSP and HTP allow-lists on this platform.
$deviceKeys = @(
    'HKLM:\SYSTEM\CurrentControlSet\Enum\ACPI\VEN_QCOM&DEV_0FF0&SUBSYS_CRD08480&REV_0056\2&daba3ff&0'
)
$deviceKeys = $deviceKeys | Where-Object { Test-Path $_ }
if (-not $deviceKeys) {
    throw 'NSP-MCDM device key not found; this platform is not Snapdragon X with qcnspmcdm.'
}

function Get-DevicePathKey([string] $devicePath) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($devicePath)
        return (($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString('x2') }) -join '')
    } finally {
        $sha.Dispose()
    }
}

foreach ($lib in $Library) {
    $resolved = (Resolve-Path -LiteralPath $lib).ProviderPath
    $name = [System.IO.Path]::GetFileName($resolved)
    # Device paths are lower case, as the shipped entries are.
    $devicePath = '\' + $Domain.ToLowerInvariant() + '\' + $name.ToLowerInvariant()
    $key = Get-DevicePathKey $devicePath

    foreach ($device in $deviceKeys) {
        $path = Join-Path $device ('Device Parameters\Mappings\' + $Domain + '\' + $Set + '\' + $key)
        if ($Verify) {
            if (Test-Path $path) {
                $local = (Get-ItemProperty -LiteralPath $path -Name 'Local' -ErrorAction SilentlyContinue).Local
                $remote = (Get-ItemProperty -LiteralPath $path -Name 'remote' -ErrorAction SilentlyContinue).remote
                Write-Output ("present {0}`n        Local  = {1}`n        remote = {2}" -f $name, $local, $remote)
            } else {
                Write-Output ("MISSING {0}`n        key    = {1}`n        want   = Local -> {2}" -f $name, $key, $resolved)
            }
            continue
        }
        if ($Remove) {
            if (Test-Path $path) {
                Remove-Item -LiteralPath $path -Recurse -Force
                Write-Output "removed $path"
            } else {
                Write-Output "absent  $path"
            }
            continue
        }

        if (-not (Test-Path $path)) {
            New-Item -Path $path -Force | Out-Null
        }
        New-ItemProperty -LiteralPath $path -Name 'Local' -Value $resolved -PropertyType String -Force | Out-Null
        New-ItemProperty -LiteralPath $path -Name 'remote' -Value $devicePath -PropertyType String -Force | Out-Null
        Write-Output ("added   {0}`n        Local  = {1}`n        remote = {2}" -f $path, $resolved, $devicePath)
    }
}
