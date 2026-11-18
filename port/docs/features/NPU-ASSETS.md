# NPU and GPU-capture assets on this machine

Asset inventory for a Windows-on-ARM Snapdragon X2 Elite Extreme laptop (Windows ARM64,
`C:\Users\Fuxfantx`). Written 2026-11-18.

**Method.** Every path, byte count and version below was read from this machine. PE
architecture was read from the file itself: `e_lfanew` at `0x3C`, then the `Machine` field
at `e_lfanew + 4`. `0x8664` is x64, `0x014C` is x86, `0xAA64` is ARM64. ELF files were read
from their header instead (magic, class, `e_machine`). No build was run, and no Vulkan
program or API trace was run, so anything about extension lists is marked unverified.

**Static analysis.** Findings in the "code reading" section come from reading source. They
are unconfirmed static analysis, not measurements.

---

## 1. The x86 RenderDoc

### Where it is

RenderDoc 1.47 exists twice, both trees extracted and byte-identical in size (202.53 MB
each). The x86 build is the `x86\` subdirectory **inside** each tree, not a separate install.

| Tree | Path |
| --- | --- |
| Working copy | `C:\Users\Fuxfantx\A-Works\DSH Works\Strata\tools\rdoc\RenderDoc_1.47_64` |
| Second copy | `C:\strata-nospace\tools\rdoc\RenderDoc_1.47_64` |

A full-drive scan (`dir C:\*renderdoc* /s /b`) plus the three Uninstall registry keys
(`HKLM`, `HKLM\WOW6432Node`, `HKCU` under `...\CurrentVersion\Uninstall`) found no other
RenderDoc. There is no RenderDoc entry in Add/Remove Programs, nothing under
`C:\Program Files` or `C:\Program Files (x86)`, and nothing in `Downloads`, `C:\VulkanSDK`,
`C:\ProgramData`, `C:\Tools`, `C:\dev` or `C:\SDK`.

### Files and architectures

Root of the tree is the **x64** build:

| File | Bytes | PE machine | Version |
| --- | --- | --- | --- |
| `qrenderdoc.exe` | 13,700,896 | x64 (0x8664) | 1.47.0.0 / v1.47 |
| `renderdoccmd.exe` | 577,312 | x64 (0x8664) | 1.47.0.0 / v1.47 |
| `renderdoc.dll` | 27,706,656 | x64 (0x8664) | |
| `renderdoc.json` | 1,578 | text | layer manifest |
| `renderdocshim64.dll` | 15,136 | x64 | |
| `renderdocui.exe` | 306,464 | | |
| `renderdoc.chm` | 6,681,722 | | |

The `x86\` subdirectory is the **x86** payload:

| File | Bytes | PE machine | Version |
| --- | --- | --- | --- |
| `renderdoccmd.exe` | 516,896 | x86 (0x014C) | 1.47.0.0 / v1.47 |
| `renderdoc.dll` | 22,398,752 | x86 (0x014C) | |
| `renderdoc.json` | 1,578 | text | layer manifest |
| `renderdocshim32.dll` | 14,112 | x86 | |
| `d3dcompiler_47.dll` | 3,714,016 | | |
| `dbghelp.dll` | 1,138,120 | | |
| `symsrv.dll` | 114,120 | | |
| `symsrv.yes` | 1 | | |

**`qrenderdoc.exe` does not exist for x86.** There is no `x86\qrenderdoc.exe` in either
tree. So the owner's claim is correct but narrower than "an x86 RenderDoc install": what
exists is the 32-bit command-line tool and the 32-bit capture layer that ship inside the
x64 1.47 release. The GUI is x64 only.

### Layer JSON and `library_path`

Both `renderdoc.json` files are identical in content (1,578 bytes):

```json
"name": "VK_LAYER_RENDERDOC_Capture",
"type": "GLOBAL",
"library_path": ".\\renderdoc.dll",
"api_version": "1.4.324",
"implementation_version": "47",
"enable_environment":  { "ENABLE_VULKAN_RENDERDOC_CAPTURE": "1" },
"disable_environment": { "DISABLE_VULKAN_RENDERDOC_CAPTURE_1_47": "1" }
```

`library_path` is **relative** (`.\\renderdoc.dll`). The loader resolves it against the
directory holding the JSON. So:

- `...\RenderDoc_1.47_64\renderdoc.json` loads `...\RenderDoc_1.47_64\renderdoc.dll` (x64).
- `...\RenderDoc_1.47_64\x86\renderdoc.json` loads `...\RenderDoc_1.47_64\x86\renderdoc.dll` (x86).

The layer also declares `VK_EXT_debug_utils` as an instance extension and
`VK_EXT_debug_marker` plus `VK_EXT_tooling_info` as device extensions.

`renderdoc.dll` and the JSON are both present. So is the Vulkan layer JSON. Both trees also
carry `RenderDoc_latest_64.msi` and `RenderDoc_latest_64.zip`, plus a full RenderDoc source
tree at `tools\rd\` with an x64 Release build (`tools\rd\x64\Release\renderdoc.dll`,
25,776,128 bytes, x64) and an x64 build at `build-x64\bin\renderdoc.dll` (27,706,656 bytes,
x64).

### Can an x86 RenderDoc capture a native ARM64 Vulkan process?

**No.**

The mechanism, as RenderDoc uses it: the Vulkan capture is a **layer**, not an external
tool. The target process loads its own Vulkan loader, which reads `VK_LAYER_RENDERDOC_Capture`
from `renderdoc.json` and then loads `library_path` **into the target process's address
space** with `LoadLibrary`. That makes `renderdoc.dll` a PE image inside the target, so it
must match the target's architecture. An x86 DLL cannot be mapped into a native ARM64
process - a 64-bit ARM64 process cannot load a 32-bit x86 image at all. The same rule is
why the release ships `renderdocshim32.dll` next to `renderdocshim64.dll`, and why the two
`renderdoc.json` files sit in two directories with two different `renderdoc.dll` builds.

So the x86 layer can only attach to an x86 process (a WOW64/emulated x86 program), and the
x64 layer only to an x64 process. Nothing in either tree is ARM64, and neither tree has an
`arm64` directory or an ARM64 `renderdoc.json`.

### The one ARM64 RenderDoc binary on the machine

JetBrains Rider ships one:

```
C:\Users\Fuxfantx\AppData\Local\Programs\Rider\plugins\rider-plugins-renderdoc\runtime\windows-aarch64\
    renderdoc.dll      23,449,472 bytes   ARM64 (0xAA64)
    RenderDocHost.exe   1,355,656 bytes   ARM64 (0xAA64)
```

That directory holds only those two files - no `renderdoc.json`, no `VkLayer_*.json`. As
shipped it is a JetBrains runtime for Rider's own graphics debugging, not a registered
Vulkan layer. Making it one would need a layer manifest written by hand. This is the only
ARM64 RenderDoc image found anywhere on the machine.

RenderDoc's Android payloads are also present
(`plugins\android\org.renderdoc.renderdoccmd.arm64.apk`, 8,676,400 bytes, and the arm32
APK, 7,924,127 bytes), but those target Android devices over adb, not Windows ARM64
processes.

### What IS available for GPU capture of an ARM64 process

**Vulkan SDK layers.** `C:\VulkanSDK\1.4.363.1` is the only SDK version installed. It has
`Bin`, `Bin-ARM64`, `Lib`, `Lib-ARM64` and **no `Bin32` and no `Lib32`** - the 32-bit layer
DLLs are absent entirely.

`Bin` (x64) has the full set:

| Layer DLL | Bytes | PE machine |
| --- | --- | --- |
| `VkLayer_khronos_validation.dll` | 21,867,448 | x64 (0x8664) |
| `VkLayer_api_dump.dll` | 8,425,912 | x64 (0x8664) |
| `VkLayer_gfxreconstruct.dll` | 6,723,512 | x64 |
| `VkLayer_crash_diagnostic.dll` | 3,057,592 | x64 |
| `VkLayer_khronos_profiles.dll` | 1,520,056 | x64 |
| `VkLayer_khronos_shader_object.dll` | 418,232 | x64 |
| `VkLayer_khronos_synchronization2.dll` | 378,808 | x64 |
| `VkLayer_monitor.dll` | 155,576 | x64 |
| `VkLayer_screenshot.dll` | 646,584 | x64 |

`Bin-ARM64` has **exactly two** layers, and 42 files in total:

| Layer DLL | Bytes | PE machine | Layer name | Version |
| --- | --- | --- | --- | --- |
| `VkLayer_gfxreconstruct.dll` | 12,635,576 | ARM64 (0xAA64) | `VK_LAYER_LUNARG_gfxreconstruct` | api 1.4.363, impl 4194309 |
| `VkLayer_crash_diagnostic.dll` | 6,087,096 | ARM64 (0xAA64) | `VK_LAYER_LUNARG_crash_diagnostic` | |

So on ARM64 the only capture-capable layer present is **gfxreconstruct**
(`VK_LAYER_LUNARG_gfxreconstruct`, `library_path` `.\\VkLayer_gfxreconstruct.dll`), driven
by `gfxrecon-capture-vulkan.py`, `gfxrecon-capture.py` and `gfxrecon-replay.exe`, all of
which exist in `Bin-ARM64`. That is a working ARM64 GPU-capture path for an ARM64 Vulkan
process.

Absent for ARM64 (x64-only in this install):

- `VK_LAYER_KHRONOS_validation` - no `Bin-ARM64\VkLayer_khronos_validation.dll`. Absent.
- `VK_LAYER_LUNARG_api_dump` - no `Bin-ARM64\VkLayer_api_dump.dll`. Absent.
- `khronos_profiles`, `khronos_shader_object`, `khronos_synchronization2`, `monitor`,
  `screenshot` - all absent for ARM64.

**No layer is registered system-wide.** These keys do not exist, on either hive or the
WOW6432Node variant:

```
HKLM\SOFTWARE\Khronos\Vulkan\Layers          ABSENT
HKLM\SOFTWARE\Khronos\Vulkan\ImplicitLayers  ABSENT
HKCU\SOFTWARE\Khronos\Vulkan\Layers          ABSENT
HKCU\SOFTWARE\Khronos\Vulkan\ImplicitLayers  ABSENT
HKLM\SOFTWARE\Khronos\Vulkan\Drivers         ABSENT
HKLM\SOFTWARE\WOW6432Node\Khronos\Vulkan\Drivers  ABSENT
```

`C:\ProgramData\Vulkan` does not exist. Every layer on this machine is therefore only
reachable by setting `VK_LAYER_PATH` (and `VK_INSTANCE_LAYERS`) by hand to
`C:\VulkanSDK\1.4.363.1\Bin-ARM64` or `...\Bin`.

**Adreno driver.** The Vulkan/OpenCL user-mode driver is
`qcdx8480.inf_arm64_260f5c119a1a0cbb` under
`C:\Windows\System32\DriverStore\FileRepository\`. It ships several UMD flavours:
`qcdx12arm64xum.dll` (18,550,072), `qcdx11arm64xum.dll` (20,728,144),
`qcdx12chpeum.dll` (7,548,816), `qcdx11chpeum.dll` (11,886,360), `qcdx12x86um.dll`
(11,536,968), `qcdx11x86um.dll` (9,588,712), plus `qcdxdml*`/`qcdxsd*` equivalents. The
x86 and CHPE UMDs mean an x86 Vulkan program can use the Adreno GPU under emulation - but
that is a different process architecture from the ARM64 process we want to capture.

**No Adreno-specific Vulkan layer was found.** Searches for `*vulkan*.json`,
`VkLayer*.json` and `*icd*.json` under the driver package, `C:\ProgramData\Qualcomm` and
`C:\Windows\System32` returned no layer manifest and no ICD manifest. `C:\Windows\System32`
contains only `vulkan-1.dll` (1,729,464 bytes, PE machine 0x8664) and
`vulkan-1-999-0-0-0.dll` (same size), plus `vulkaninfo.exe`. The Strata engine also ships
its own loader: `...\components\engine\0.2.13-windows-arm64-vulkan\vulkan-1.dll`,
2,499,544 bytes, ARM64.

**`VK_EXT_debug_utils`.** RenderDoc's layer JSON declares it as a layer-provided instance
extension, and gfxreconstruct uses the same mechanism. Whether the Adreno ICD itself
advertises the driver-side `VK_EXT_debug_utils` instance extension was **not verified** -
that needs `vulkaninfo`, which was not run.

**Summary of ARM64 capture options here:** gfxreconstruct (ARM64 layer + capture/replay
tools in `Bin-ARM64`), RenderDoc's API as an in-process layer via the ARM64
`renderdoc.dll` from the Rider plugin (needs a hand-written manifest), and `VK_EXT_debug_utils`
/ `VK_EXT_debug_marker` markers if the ICD exposes them. Validation layers and api_dump for
ARM64 are absent from this machine.

---

## 2. Qualcomm SDK assets

### `C:\Users\Fuxfantx\C-Res\geniex` - GenieX source checkout, v0.11.0

3.45 MB total. `git describe` = **v0.11.0**, HEAD `ae375185109277a2a019b37c037f65bd6e47578c`,
2026-11-13, "Merge pull request #1667 from qualcomm/chore/bump-llama-cpp-master". Shallow
clone, sparse checkout.

```
geniex\
  .github\certs\hexagon\   ggml-htp-v1.cer (786 B), README.md (1,553 B)
  notes\                   AI.md, bench.md (11,852), build.md (13,016),
                           README.md, release.md (16,719), run.md (18,404)
  sdk\                     CMakeLists.txt (11,005), CMakePresets.json (6,566), README.md
    include\               geniex.h (46,792), logging.h (16,386), image_utils.h (8,600),
                           registry.h (4,166), utils.h (2,923), profile.h (912),
                           power_mode_alias.h (2,312), build_config.h (358)
      external\            date.h, httplib.h, json.hpp, fmt\ (16 headers)
      plugin\              ILlm.h, IVlm.h, Plugin.h
    plugins\
      llama_cpp\           include\ decision.h, htp_session.h, liquid-decision.h, llm.h,
                                   params.h, profiler.h, threadpool.h, vlm.h
                           src\     llm.cpp (44,546), vlm.cpp (39,126),
                                   params.cpp (23,622), liquid-decision.cpp (19,842),
                                   decision.cpp (18,053), plugin.cpp (7,150),
                                   profiler.cpp (3,028), threadpool.cpp (2,486),
                                   htp_session.cpp (2,373)
      qairt\               include\ qnn_runtime_utils.h (10,083), sampler_config_utils.h,
                                   power_mode_utils.h, path_utils.h, llm.h, vlm.h,
                                   metadata_utils.h, chat_message_utils.h
                           src\     llm.cpp (17,304), vlm.cpp (15,618), plugin.cpp (1,916),
                                   log_filter.cpp (1,482)
  BUILD.bazel, MODULE.bazel, MODULE.bazel.lock, README.md, README_zh-CN.md,
  CLAUDE.md, CONTRIBUTING.md, repolint.json
```

What is **not** there: no `libGenie*`, no `QnnHtp*.dll`, no `.so`, no `.bin` context
binaries, no sample apps, no prebuilt binaries at all. This is headers, one header-only SDK
surface and two plugin source trees. Docs are `notes/*.md` plus `README.md`.

Key file of interest: `sdk\plugins\llama_cpp\include\htp_session.h` and its
`src\htp_session.cpp` (2,373 B) - the HTP session wrapper for the llama.cpp path.

### `C:\Users\Fuxfantx\C-Res\geniex-sdk` - GenieX prebuilt SDK, v0.11.0, windows-arm64

200.19 MB total, most of it two identical archives.

```
geniex-sdk\
  geniex-sdk-windows-arm64-v0.11.0.zip   93,270,061 B (88.95 MB)
  sdk.zip                                93,270,061 B (same size - duplicate)
  llama_cpp\                             extracted, 22.30 MB
    geniex_plugin.dll          485,376
    ggml.dll                    72,192
    ggml-base.dll              697,856
    ggml-cpu.dll               877,568
    ggml-hexagon.dll           459,776
    ggml-opencl.dll          4,166,656
    libggml-htp-v73.so         965,480
    libggml-htp-v75.so         965,480
    libggml-htp-v79.so         986,056
    libggml-htp-v81.so       1,002,504
    libggml-htp-v85.so       1,002,504
    libggml-htp.cat           13,538
    libggml-htp.inf              825
    libomp140.aarch64.dll      796,728
    llama-common.dll         5,572,096
    llama.dll                3,406,848
    mtmd.dll                 1,906,688
```

The `geniex-sdk-windows-arm64-v0.11.0.zip` has 72 entries under `sdk-windows-arm64/`, laid
out as `lib\qairt\` and `lib\llama_cpp\`. Besides the files above it carries the QNN runtime
(below) plus import libraries (`QnnHtp.lib`, `QnnIr.lib`, `QnnModelDlc.lib`,
`QnnChrometraceProfilingReader.lib`) and the `.cat` catalogues `libqnnhtpv73.cat`,
`libqnnhtpv81.cat`, `libsnpehtpv73.cat`, `libsnpehtpv81.cat`.

So `geniex-sdk` is the one place on this machine that has both halves: the llama.cpp HTP
skels and the QNN runtime, side by side, all windows-arm64.

### `C:\Users\Fuxfantx\C-Res\hexagon-sdk` - Hexagon SDK 6.7.0.1

3,585.47 MB total, one version directory named `6.7.0.0`:

| Top dir | Size |
| --- | --- |
| `tools` | 3,060.7 MB |
| `libs` | 355.9 MB |
| `utils` | 64.4 MB |
| `rtos` | 60.9 MB |
| `ipc` | 34.5 MB |
| `build` | 8.5 MB |
| `incs` | 0.6 MB |

**Version.** `hexagon_sdk.json` says `"version": "6.7.0.1"`, `"build": "6.7.0.1"`.
`incs\version.h` says `VERSION_MAJOR 6`, `MINOR 7`, `MAINT 0`, `BUILD 1`, and
`VERSION_STRING "HAP SDK 6.7.0.1 (srvr=qtcp406;br=main;cl=1242374)"`. The directory name
`6.7.0.0` is one build behind the manifest.

**HEXAGON_Tools versions: only `21.0.04`.** `tools\HEXAGON_Tools\` contains just `21.0.04`.
`hexagon_sdk.json` also lists Android NDK `android_ndk_27d`, Halide `2.6.03`, QEMU `14.0.1`
and CMake `3.28.3` - **all four are absent**; `tools\` holds only `debug`,
`HEXAGON_Tools`, `qhcg` and `utils`.

**`ipc`** (34.5 MB):

```
ipc\fastrpc\incs     fastrpc headers
ipc\fastrpc\qaic     QAIC
ipc\fastrpc\remote   remote handles
ipc\fastrpc\rpcmem   rpcmem (the allocator ggml-hexagon uses)
ipc\fastrpc\rtld     runtime loader
ipc\synx\ship        synx synchronisation
```

**`incs`** (0.6 MB) - the HAP DSP-side API: `HAP_compute_res.h` (71,874),
`HAP_power.h` (46,521), `HAP_mem.h` (17,997), `HAP_farf.h` (6,591), `HAP_vtcm_mgr.h`
(1,948), `HAP_dcvs.h`, `HAP_perf.h`, `HAP_process.h`, `HAP_user_pmu.h`, plus the FastRPC
headers `remote.h` (67,004, and a `remote.h.orig` of 67,002), `rpcmem.h` (7,613),
`remote64.h`, `dspqueue.h` (37,079), `verify.h` (4,607), `version.h`, and a `stddef\`
directory.

**`rtos\qurt`** has eight DSP revisions, each with `include\posix`, `include\qurt`, `lib`,
`cosims`, `sdksim_bin`, `debugger`: **computev68, v69, v73, v75, v79, v81, v83, v85**. Note
`v83` is here but the engine ships skels only for v73/v75/v79/v81/v85.

**Is the DSP toolchain present?** Yes. `tools\HEXAGON_Tools\21.0.04\Tools\bin\` contains:

| Tool | Bytes |
| --- | --- |
| `hexagon-clang.exe` | 99,057,152 |
| `hexagon-clang++.exe` | 99,057,152 |
| `hexagon-clang-cpp.exe` | 99,057,152 |
| `hexagon-clang-cl.exe` | 99,057,152 |
| `hexagon-link.exe` | present |
| `hexagon-ar.exe` | present |
| `hexagon-sim.exe` | present |
| `hexagon-llvm-objdump.exe` | present |
| `hexagon-llvm-readelf.exe` | 8,525,824 |
| `hexagon-readelf.exe` | 8,525,824 |

Clang 21 is under `Tools\lib\clang\21`, with a bundled `python311`. The build system is
present too: `build\make.d` (with `UsersGuide.txt`), `build\make.d.ext`, `build\cmake`,
`build\cdef`, `build\defines.min`, `build\setup_LE_Toolchain_env.cmd`.

**Can it build a skel?** Statically, the pieces a skel needs are there: `hexagon-clang`,
`hexagon-link`, the `qurt` headers and libraries for each computev, `incs` (HAP), `ipc`
(fastrpc/rpcmem) and the make/cmake build rules. What is missing is everything that would
show you how: **`addons\` is absent** (`hexagon_sdk.json` points at `./addons/compute`),
**`docs\` is absent** (so no `docs/index.html`), and so are `examples\` and `test\`. There is
no sample skel in this SDK. `readme.txt` also says to run
`utils/scripts/SDKDependencyDownloader --sdk_root <path>`; that binary exists
(`utils\scripts\SDKDependencyDownloader.exe`, 63,211,905 bytes) and was not run.

Note from GenieX's own build notes (`C:\Users\Fuxfantx\C-Res\geniex\notes\build.md`): the
Hexagon toolchain has a **250-character path limit**, so a build path must be shortened
with `subst` first.

**Signing tooling and certificates.** No code-signing binary ships in the toolchain:
`Tools\bin` has no `elfsign`, no attestation tool, no `signtool` - only `readelf` and
`objdump`. What exists is a script:

```
C:\Users\Fuxfantx\C-Res\hexagon-sdk\6.7.0.0\utils\scripts\signer.py   13,773 B
C:\Users\Fuxfantx\C-Res\hexagon-sdk\6.7.0.0\utils\scripts\adb_functions.py  19,184 B
```

`signer.py` generates a TestSig or signs a `.so`, and it works **over adb against a
connected device** (`-s` serial, `-d 0|3` for ADSP/CDSP, `getserial` action). It is the
Android workflow. It is not a Windows driver-signing path.

**No certificates are in the Hexagon SDK** - a search of the whole 3.5 GB tree for
`*.pem`, `*.p12`, `*.cer`, `*.crt`, `*.der` and names matching `sign`, `attest`, `qcsign`
found only unrelated sources (`qhmath_copysign*.c`, `qurt_signal*.h`) plus one pre-signed
APK (`tools\utils\sysmon\sysMon_DSP_Profiler_V2_signed.apk`, 2,310,130 bytes).

The only Hexagon HTP signing certificate on the machine is in the GenieX checkout:

```
C:\Users\Fuxfantx\C-Res\geniex\.github\certs\hexagon\ggml-htp-v1.cer   786 B
C:\Users\Fuxfantx\C-Res\geniex\.github\certs\hexagon\README.md       1,553 B
```

Its README says it is the self-signed test certificate used in CI to sign
`libggml-htp-*.so` into `libggml-htp.cat`, shipped only when Release CI cannot find a
Microsoft-signed bundle. **The private key is not checked in** - it lives in the
`HEXAGON_HTP_CERT_PFX` GitHub Actions secret. The README's install path for Windows on
Snapdragon is `bcdedit /set TESTSIGNING ON` (needs a reboot, may need Secure Boot off) and
importing the `.cer` into both Trusted Root Certification Authorities and Trusted
Publishers.

### `C:\Users\Fuxfantx\C-Res\Hexagon SDK.tar.xz`

```
900,461,020 bytes (858.75 MB), last write 2026-11-13 18:02
```

Not extracted and not inspected - listing an `.xz` needs full decompression. It sits next to
`hexagon-sdk\6.7.0.0\` (3,585.47 MB extracted), which is consistent with being the archive
that tree came from, but that was not verified.

### QNN runtime on the machine

**Two copies, plus the driver's own.**

`C:\Users\Fuxfantx\AppData\Local\GenieX CLI\qairt\htp-files\` (the GenieX CLI install):

| File | Bytes |
| --- | --- |
| `QnnHtp.dll` | 3,427,536 |
| `QnnSystem.dll` | 3,570,896 |
| `QnnHtpPrepare.dll` | 89,803,984 |
| `QnnHtpV73Stub.dll` | 574,160 |
| `QnnHtpV81Stub.dll` | 573,648 |
| `QnnHtpV73CalculatorStub.dll` | 35,536 |
| `QnnHtpV81CalculatorStub.dll` | 35,536 |
| `QnnNetRunDirectV81Stub.dll` | 718,032 |
| `QnnHtpNetRunExtensions.dll` | 946,384 |
| `QnnIr.dll` | 1,654,480 |
| `QnnModelDlc.dll` | 2,718,416 |
| `QnnSaver.dll` | 600,272 |
| `QnnHtpProfilingReader.dll` | 315,600 |
| `QnnHtpOptraceProfilingReader.dll` | 3,411,664 |
| `QnnJsonProfilingReader.dll` | 372,944 |
| `QnnChrometraceProfilingReader.dll` | 325,840 |
| `PlatformValidatorShared.dll` | 14,855,888 |
| `calculator.dll` / `calculator_htp.dll` | 35,536 each |
| `libQnnHtpV73.so` | 12,780,796 |
| `libQnnHtpV73Skel.so` | 12,379,344 |
| `libQnnHtpV73QemuDriver.so` | 12,387,480 |
| `libQnnHtpV81.so` | 13,944,024 |
| `libQnnHtpV81Skel.so` | 13,542,412 |
| `libQnnNetRunDirectV81Skel.so` | 3,058,416 |
| `libQnnSystem.so` | 217,044 |
| `libQnnSaver.so` | 310,264 |
| `libSnpeHtpV73Skel.so` | 12,842,236 |
| `libSnpeHtpV81Skel.so` | 14,016,448 |
| `libCalculator_skel.so` | 5,656 |
| `libqnnhtpv73.cat` | 12,216 |
| `libqnnhtpv81.cat` | 12,216 |

`C:\Users\Fuxfantx\C-Res\geniex-sdk\llama_cpp\` next to it is the llama.cpp route, and the
SDK zip's `lib\qairt\htp-files\` is the same QNN set.

**Driver-side QNN** is in
`C:\ProgramData\Qualcomm\Drivers\mcdm_drivers_8480_ARM64_AE0C2EE7-13A5-4FD2-8E42-9C692397E69B\qcnspmcdm8480\HTP\`:

| File | Bytes |
| --- | --- |
| `QnnHtpPrepareDrv.dll` | 179,263,048 |
| `libQnnHtpV81SkelDrv.so` | 11,797,164 |
| `QnnHtpV81StubDrv.dll` | 1,059,400 |
| `QnnSystemInfo.dll` | 485,440 |

**Where it is not.** `QnnHtp.dll`, `QnnSystem.dll` and `libQnnHtp*` are **absent** from
`C:\Program Files`, `C:\Program Files (x86)`, `C:\VulkanSDK`, `C:\strata-nospace`, and from
`C:\Users\Fuxfantx\C-Res` outside the SDK zip. No `.bin` QNN context binaries were found in
any `qairt` directory - only `.so`, `.dll` and `.cat`.

The GenieX CLI itself is at `C:\Users\Fuxfantx\AppData\Local\GenieX CLI\`:
`geniex.exe` (41,709,568), `geniex.dll` (5,011,968), `qairt\geniex_core.dll` (1,129,472),
`geniex_plugin.dll` (428,032), `geniex-proc.dll` (4,545,536), `geniex-proc-vision.dll`
(548,352), `geniex_vlm.dll` (385,024), and `llama_cpp\` with its own
`libggml-htp-v73/v75/v79/v81/v85.so` (1,003,824 / 1,003,792 / 1,024,368 / 1,040,848 /
1,040,848 - slightly larger than the `geniex-sdk` copies) plus `libggml-htp.cat`
(13,537 B), `ggml-hexagon.dll` (408,576) and `libomp140.aarch64.dll` (753,248). Its icon
file is named `geniex-0.10.1-alpha.3-lenovo.ico`, which suggests the installed CLI build is
0.10.1-alpha.3 - the version was not read from a manifest.

### Strata engine component `0.2.13-windows-arm64-vulkan`

```
C:\Users\Fuxfantx\AppData\Local\Strata\components\engine\0.2.13-windows-arm64-vulkan\
```

Full listing, with PE architecture read from each file:

| File | Bytes | Arch |
| --- | --- | --- |
| `engine-component.json` | 1,756 | text |
| `strata.exe` | 101,376 | ARM64 |
| `ggml.dll` | 73,216 | ARM64 |
| `ggml-base.dll` | 795,136 | ARM64 |
| `ggml-cpu.dll` | 733,696 | ARM64 |
| `ggml-vulkan.dll` | 48,331,776 | ARM64 |
| `llama.dll` | 2,736,640 | ARM64 |
| `llama-common.dll` | 4,637,696 | ARM64 |
| `mtmd.dll` | 1,467,904 | ARM64 |
| `libomp140.aarch64.dll` | 753,248 | ARM64 |
| `vulkan-1.dll` | 2,499,544 | ARM64 |
| `libggml-htp-v73.so` | 965,480 | ELF32, EM_HEXAGON |
| `libggml-htp-v75.so` | 965,480 | ELF32, EM_HEXAGON |
| `libggml-htp-v79.so` | 986,056 | ELF32, EM_HEXAGON |
| `libggml-htp-v81.so` | 1,002,504 | ELF32, EM_HEXAGON |
| `libggml-htp-v85.so` | 1,002,504 | ELF32, EM_HEXAGON |
| `libggml-htp.cat` | 13,538 | binary catalogue |
| `libggml-htp.inf` | 825 | text |

**The `libggml-htp` set is complete.** All five `libggml-htp-v*.so` (v73, v75, v79, v81,
v85) and `libggml-htp.cat` are present. Each `.so` was confirmed as a Hexagon DSP object:
magic `7F 45 4C 46`, ELF class 1 (32-bit), `e_type` `0x0003` (`ET_DYN`, shared object),
`e_machine` `0x00A4` (`EM_HEXAGON`). They are not Windows DLLs despite the `.so` suffix
inside a Windows component.

`libggml-htp.cat` SHA-256:
`CBFF7CBFA391F0A79C78432572EF926BBBC07C9D6889DDD21E2A4E48FFA3A8E9`.

`libggml-htp.inf` declares `Class = ComputeAccelerator`,
`ClassGuid = {F01A9D53-3FF6-48D2-9F97-C8A7004BE10C}`, `DriverVer = 01/01/2026,1.0.0.0`,
`CatalogFile = libggml-htp.cat`, `PnpLockDown = 1`, and a `[DefaultInstall.NTarm64]` that
copies the five `.so` files to `Drivers_Dir = 13` with `COPYFLG_NO_OVERWRITE` (`0x10`).

`engine-component.json` says `"version": "0.2.13"`, `"platform": "windows-arm64"`,
`"backend": "vulkan"`, `"archs": ["arm64"]`, `"build_id": "strata-port-vulkan-arm64"`,
`"entrypoint": "strata.exe"`, `"signature": null`, `"variant": "port"`. Its `files[]` array
lists **10** entries with SHA-256 - the DLLs, the loader and `strata.exe` - and **does not
include** the five `.so` files, the `.cat` or the `.inf`. Those seven are laid down by the
INF/driver path, not tracked by the component manifest.

Also present elsewhere in `LocalAppData\Strata`: a rollback copy at
`backup-engine-20261113-201811\` with the same DLL set but no HTP skels (so that backup
predates or omits the NPU payload), and `components\{active,cache,partial,previous}`.

---

## 3. Code reading: the port's NPU path

Files read:

- `C:\Users\Fuxfantx\A-Works\DSH Works\Strata\port\engine\vk-devices.cpp` (25,103 B, 517 lines)
- `C:\Users\Fuxfantx\A-Works\DSH Works\Strata\port\engine\vk-devices.h` (5,411 B, 105 lines)
- `C:\Users\Fuxfantx\A-Works\DSH Works\Strata\port\engine\vk-options.h` / `.cpp`
- `C:\Users\Fuxfantx\A-Works\DSH Works\Strata\third_party\llama.cpp\ggml\src\ggml-hexagon\`
  (130 files, 2.90 MB) - `ggml-hexagon.cpp` (154,845 B, 3,904 lines), `htp-kparams.cpp`
  (236,429 B), `htp-repack.cpp` / `.h`, `htp-fusion.cpp`, `htp/` kernels

### The exact guard that refuses NPU placement by default

**File and symbol:** `port\engine\vk-devices.cpp`, function
`NpuPlan strata::plan_npu(NpuMode mode, const ModelFacts & facts, int layers)`, defined at
line **397**. The refusal is at line **435**:

```cpp
    // Placement is refused by default.  On this build the HTP session opens and the first decode never returns with
    // either pack here (a q6_K head: 417 MB on the 35B-A3B, 1.0 GB on the 27B), so an explicit placement hangs the app
    // instead of being slow - docs/NPU-FEASIBILITY.md records it.  STRATA_VK_NPU_FORCE=1 tries anyway, on a machine
    // where the placement is known to complete.
    if (std::getenv("STRATA_VK_NPU_FORCE") == nullptr) {
        logf("[strata-vk] NPU: placement is off: ...");
        return plan;
    }
```

**Environment variable:** `STRATA_VK_NPU_FORCE`. If it is unset, `plan_npu` returns a plan
whose `dev` is the found HTP device but whose `overrides` vector is empty, so
`NpuPlan::active()` is false and `llama_model_params::tensor_buft_overrides` carries
nothing.

There is a second, earlier gate. The CLI default is `NpuMode::Auto`
(`vk-options.h:79`, `NpuMode npu = NpuMode::Auto;`), and `plan_npu` returns at lines
424-429 for `Auto` with the message "found and reported; the dense tensors stay on the GPU
(measured slower on this machine - docs/ADRENO.md)". So by default nothing is placed, and
even `--npu head|dense|all|split` still stops at `STRATA_VK_NPU_FORCE`.

A third, unconditional setting sits in `configure_npu_budget(NpuMode mode)`
(`vk-devices.cpp:384`), called from `vk-main.cpp:171`:

```cpp
    set_env_once("GGML_HEXAGON_RT_REPACK", "0");
```

with the comment that runtime repacking hands the DSP a tiled weight layout, the host layout
round-trips exactly (`GGML_HEXAGON_REPACK_CHECK` reports 0 of 144,643,072 bytes differing on
the tiny head), but the DSP never answers the first tiled batch. The same function maps
`STRATA_VK_NPU_VMEM` to `GGML_HEXAGON_VMEM` and `STRATA_VK_NPU_MBUF` to
`GGML_HEXAGON_MBUF`, only when those are set explicitly.

**Detection.** `is_npu_name()` (`vk-devices.cpp:72`) treats a device name starting with
`HTP` as the NPU ("ggml-hexagon names its devices HTP0, HTP1, ...; nothing else does").
`looks_like_npu()` (`:77`) also matches a description containing `Hexagon` or `HTP`.

**Placement patterns** (null-terminated, `vk-devices.cpp:85-102`):

| Table | Patterns |
| --- | --- |
| `kAllPatterns` | `.*` |
| `kHeadPatterns` | `^output\.` |
| `kDensePatterns` | `(^\|\.)output\.`, `\.ffn_(up\|down\|gate)\.`, `\.ffn_(up\|down\|gate)_shexp`, `\.ffn_gate_inp\.`, `\.attn_(q\|k\|v\|qkv\|output\|out\|gate)\.` |

The comment above `kDensePatterns` (line 83-84) states the experts are deliberately left on
the GPU "because TQ2_T is not an HTP type".

**CLI surface** (`vk-options.cpp:57-65`, `:129-137`): `--npu off|auto|head|dense|all|split`,
`--npu-layers N`, `--npu-mtp`, `--npu-probe`. `NpuMode` has six values - `Off, Auto, Head,
Dense, All, Split` (`vk-options.h:24-31`).

**Skel resolution.** `vk-devices.h:40-43` documents that FastRPC on Windows resolves the HTP
skel (`libggml-htp-v<N>.so`) as a bare name **against the process's current directory**; an
identical URI fails with `0x80000406` without that cwd, and an absolute path in the URI is
not accepted. `SkelDirGuard` (`vk-devices.h:44`) therefore changes into the skel directory
for the load and restores it afterwards.

Other call sites: `vk-main.cpp:80`, `:181`, `:212` call `plan_npu`; `vk-main.cpp:171` calls
`configure_npu_budget`.

### Which ggml ops the HTP backend claims to support

`supports_op` is `ggml_backend_hexagon_device_supports_op`
(`ggml-hexagon.cpp:3141`). It rejects on an op filter regex, then on `max_vmem` allocation
size, then calls `htp_kparams_check(sess, op)` (`:3161`), which is
`htp_kparams_solve(sess, op, nullptr)` (`htp-kparams.h:74`). The actual op claim is
`htp_op_remap()` (`htp-kparams.cpp:19`), which maps `GGML_OP_*` to `HTP_OP_*`; anything it
does not map returns `HTP_OP_INVALID` and is unsupported. Shape and type limits are then
enforced per op by the `htp_kparams_solve_*` functions. The file also warns
(`ggml-hexagon.cpp:3138-3139`) that `supports_op` must never look at buffer usage, buffer
properties or repack state - only static tensor properties.

**59 `GGML_OP_*` symbols appear in `htp-kparams.cpp`** (plus the UNARY and GLU sub-ops):

`ADD`, `ADD_ID`, `ARGMAX`, `ARGSORT`, `CLAMP`, `CONCAT`, `CONT`, `CONV_2D`,
`CONV_2D_DW`, `CONV_3D`, `CPY`, `CUMSUM`, `DIAG`, `DIV`, `DSV4_HC_POST`,
`DSV4_HC_PRE`, `DUP`, `FILL`, `FLASH_ATTN_EXT`, `GATED_DELTA_NET`, `GET_ROWS`,
`GLU`, `IM2COL`, `IM2COL_3D`, `L2_NORM`, `LEAKY_RELU`, `LIGHTNING_INDEXER`, `LOG`,
`MUL`, `MUL_MAT`, `MUL_MAT_ID`, `NORM`, `PAD`, `POOL_1D`, `POOL_2D`, `POOL_AVG`,
`POOL_MAX`, `REPEAT`, `RMS_NORM`, `ROLL`, `ROPE`, `SCALE`, `SET_ROWS`, `SOFT_MAX`,
`SOLVE_TRI`, `SQR`, `SQRT`, `SSM_CONV`, `SSM_SCAN`, `SUB`, `SUM`, `SUM_ROWS`,
`TIMESTEP_EMBEDDING`, `TOP_K`, `TOP_K_MASK`, `TRI`, `UNARY`, `UPSCALE`

Sub-ops:

- `GGML_OP_UNARY`: `SILU`, `GELU`, `GELU_QUICK`, `GELU_ERF`, `SIGMOID`, `NEG`, `EXP`,
  `SOFTPLUS`, `TANH`, `ABS`, `RELU`, `STEP`, `CEIL`, `SGN`
- `GGML_OP_GLU`: `REGLU`, `SWIGLU`, `SWIGLU_OAI`, `SWIGLU_CLAMP`, `GEGLU`,
  `GEGLU_QUICK`, `GEGLU_ERF`

One special case: `GGML_OP_MUL_MAT` maps to `HTP_OP_HADAMARD` when `htp_is_hadamard(t)` is
true, otherwise `HTP_OP_MUL_MAT`.

The backend therefore has a native kernel for the transformer hot path - `MUL_MAT`,
`MUL_MAT_ID`, `RMS_NORM`, `ROPE`, `FLASH_ATTN_EXT`, `SOFT_MAX`, GLU, the binary/`UNARY` ops,
`GET_ROWS`/`SET_ROWS`, `SSM_SCAN`/`SSM_CONV`, `GATED_DELTA_NET` - and nothing for the
trellis types.

### Do the trellis types appear in it?

**No - absent from the whole backend, not just the op table.**

A scan of all 130 files under `ggml\src\ggml-hexagon\` (2.90 MB) for
`tqk6|tqk7|tq2_t|TQ2_T|TQK6|TQK7|ptq1_0|PTQ1_0` returned **zero matches**. The only apparent
hit for the ids in `ggml-hexagon.cpp` is a false positive: line 1374 is
`uint64_t hash = 1469598103934665603ull;` - the FNV-1a 64-bit offset basis, not a type id.

The four types are defined only in the port's overlay header,
`port\llama-overlay-vanilla\ggml\include\ggml.h:264-268`:

```c
GGML_TYPE_PTQ1_0  = 143,
GGML_TYPE_TQ2_T   = 144,
GGML_TYPE_TQK6    = 145, // trellis K=6, bit-packed sibling of TQ2_T
GGML_TYPE_TQK7    = 146, // trellis K=7
GGML_TYPE_COUNT   = 147,
```

with file types `GGML_FTYPE_MOSTLY_PTQ1_0 = 129`, `MOSTLY_TQ2_T = 130`, `MOSTLY_TQK6 = 131`,
`MOSTLY_TQK7 = 132` at lines 325-328. Reference data for them sits in
`port\llama-overlay-vanilla\tests\data\{tq2t_ref.bin,tqk6_ref.bin,tqk7_ref.bin}` and
`tests\test-tq2t.cpp`.

**The repack path cannot tile them either.** `htp_is_tiled_weight_tensor`
(`htp-repack.h:23`) requires `htp_is_tiled_weight_type(t->type)`, contiguous, not a view,
`ne[0] >= 32`, `ne[0] % 32 == 0`, `ne[3] == 1`. `htp_tiled_row_size` (`htp-repack.h:41`)
handles `F16`/`F32`/`BF16`, `Q6_K`, `Q2_0`, `Q4_K`. The types `htp-repack.cpp` names at all
are: `Q1_0`, `Q2_0`, `Q2_K`, `Q3_K`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_0`, `Q5_1`, `Q5_K`, `Q6_K`,
`Q8_0`, `IQ2_XXS`, `IQ4_NL`, `MXFP4`, `NVFP4`, `F16`, `F32`, `BF16`. No trellis type appears.
So a trellis tensor never enters the tiled layout, never reaches an HTP kernel, and - as
`vk-devices.cpp:84` says - stays on the GPU.

**The repack path itself.** `ggml_hexagon_tensor_repack` (`ggml-hexagon.cpp:2504`) skips a
tensor when its data is null, already tiled, or not tileable (`:2505`), then calls
`htp_repack_tensor_tiled` (`:2511`). `ggml_hexagon_graph_repack` (`:2534`) walks the graph
and repacks `n->src[0]` of each node. Both run from `ggml_hexagon_graph_compute`
(`ggml_hexagon_graph_repack(gf, sess)` at `:2794`). Gates: `opt_rt_repack` defaults to **1**
(`:154`), read from `GGML_HEXAGON_RT_REPACK` (`:3557`, `:3618`). The round-trip check is
`GGML_HEXAGON_REPACK_CHECK` (`:2516-2524`), which unpacks the tiled copy and compares bytes.
The buffer-size helper honours the tiled row size at `:953-959`, and host-side unpacking is
`htp_unpack_tensor_tiled` at `:720` and `:791`. The engine forces
`GGML_HEXAGON_RT_REPACK=0` (see the guard section), so in this port the tiled path is off.

### How tensors are transferred

**Host to DSP: rpcmem plus a FastRPC mapping, zero copy. No bulk memcpy of weights.**

- `ggml_hexagon_rpcmem_block` (`ggml-hexagon.cpp:435`) allocates with
  `rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size)` (`:443`) and takes the
  DMA-BUF fd with `rpcmem_to_fd(base)` (`:447`). It frees with `rpcmem_free` (`:457`).
- `ggml_hexagon_shared_buffer::mmap()` (`:474`) maps that buffer into the DSP domain with
  `fastrpc_mmap(sess->domain_id, fd(), (void *) base(), 0, size(), flags)` (`:482`). Flags:
  `FASTRPC_MAP_FD` when pinned, else `FASTRPC_MAP_FD_DELAYED`, or
  `FASTRPC_MAP_FD_DELAYED_EXTENDED` when extended (`:480`). A failure retries once after
  100 ms, then rethrows `"ggml-hex: fastrpc_mmap failed"` (`:489-498`), with a hint about
  IOVA space and `--no-host`.
- Buffer sizing (`:552-559`): data size rounded to 4 KiB plus a 4 KiB guard page; when
  `opt_dma64` is on (and it defaults on for arch > 79, `:3607`, env `GGML_HEXAGON_DMA64`
  `:3578`), non-pinned buffers round up to a 2 MiB boundary.
- Pinned buffers map at allocation time (`:541`). Unmap is `fastrpc_munmap` (`:516`).
- Cross-session sharing uses the clone constructor (`:562-570`), which shares the same
  `ggml_hexagon_rpcmem_block` by `shared_ptr`, so a weight set is allocated once and mapped
  into each session.

**VTCM is skel-side only, with a static layout.** `htp\htp-vtcm.h` provides
`vtcm_seq_alloc()` and the `VTCM_LAYOUT_ALLOC` / `VTCM_LAYOUT_ALLOC_OPTIONAL` /
`VTCM_LAYOUT_PTR` macros - a bump allocator over a fixed offset layout, not a host API. The
host only learns the size: `htp_iface_hwinfo(handle, &n_threads, &n_hvx, &n_hmx, &vtcm_size)`
(`:2278`), overridable by `GGML_HEXAGON_VTCM_SIZE` (`:3563`, applied `:2285-2293`, warning
if the override exceeds hardware), with a fallback of `8 * 1024 * 1024` when hwinfo fails
(`:2300`).

**DMA is DSP-internal, and only for DDR-to-DDR copies.** `htp\dma-queue.h` defines
`dma_descriptor_1d` / `dma_descriptor_2d` and notes "the ones in HexSDK are a bit out of
date". `htp\dma-copy.h` describes itself as "DDR<->DDR DMA copies of same-type, same-shape
tensors with arbitrary strides. Used by CPY for the copy itself and by CONCAT", chunking to
`DMA_SAFE_CHUNK_SIZE` and flushing the descriptor queue when data is needed. So the DMA
engine moves data **inside** DSP-visible DDR for `CPY`/`CONCAT`; it is not the host-to-DSP
transport. Trace events distinguish the two (`HTP_TRACE_EVT_DMA` at `:193`), and session
errors distinguish VTCM pressure (`HTP_STATUS_VTCM_TOO_SMALL` at `:166-167`).

**Summary.** Weights and activations live in rpcmem (DMA-BUF) buffers that are mapped into
the DSP once and read in place. VTCM is used as kernel scratch inside the skel. The HVX DMA
engine is used for in-DDR reshuffles that a `CPY` or `CONCAT` needs. There is no
copy-the-weights-into-DSP-DDR step in this backend.

---

## What this enables

This section is inference from the facts above, not measurement.

- **An x86 RenderDoc cannot help with this port.** The port's engine is ARM64 only
  (`strata.exe`, `ggml-vulkan.dll`, `llama.dll` are all `0xAA64`; `engine-component.json`
  declares `archs: ["arm64"]`). An x86 layer cannot load into it, and there is no
  `qrenderdoc.exe` for x86 in any case. The realistic capture paths are the ARM64
  gfxreconstruct layer in `C:\VulkanSDK\1.4.363.1\Bin-ARM64`, or RenderDoc's ARM64
  `renderdoc.dll` from the Rider plugin with a layer manifest written by hand. Neither was
  exercised here.
- **No ARM64 validation layer is installed**, so `VK_LAYER_KHRONOS_validation` - the usual
  way to catch a driver-level API misuse - is not available on this machine in an ARM64
  build. Only gfxreconstruct and crash_diagnostic are. Installing the ARM64 Vulkan SDK
  components, or building the validation layer for ARM64, is what would change that.
- **The HTP stack is complete enough to reproduce the port's NPU experiment.** All five
  skels (v73/v75/v79/v81/v85) plus `libggml-htp.cat` are staged in the engine directory and
  described by an INF that installs to `Drivers_Dir = 13`. The Hexagon SDK has the compiler,
  linker, qurt headers and libs for computev68 through v85. What is missing is a sample skel,
  the SDK docs, the addons, and any working signing path for Windows - the only signing
  material is a self-signed public `.cer` whose private key is not on this machine, and the
  Hexagon SDK's own `signer.py` targets adb-connected Android devices.
  **Superseded in part**: `features/NPU.md` records that the session now opens with the locally built skel and
  that the refusal moved one layer down, to `dspqueue_create` (0x14); the signing question in this bullet was
  written while `0x200` was still the wall. Read `features/NPU.md` first for the current state.
- **QNN is on this machine but the port does not use it.** `QnnHtp.dll`, `QnnSystem.dll`,
  `libQnnHtpV73/V81*.so`, `QnnHtpPrepare.dll` and the driver-side
  `QnnHtpPrepareDrv.dll`/`libQnnHtpV81SkelDrv.so` are all present, but the port's NPU path
  goes through ggml-hexagon's own FastRPC skel, not QNN. `geniex`'s `sdk\plugins\qairt`
  plugin is the QNN consumer, and it is source-only here.
- **The two independent findings agree.** The trellis types are absent from the HTP backend,
  and the engine's dense-placement pattern table deliberately excludes `ffn_*_exps`. Both
  point the same way: on this port a trellis-quantised expert stack is GPU-only by
  construction, and only the non-trellis tensors (output head, dense FFN, attention
  projections) are candidates for HTP placement - which is exactly what
  `STRATA_VK_NPU_FORCE` gates and what `docs/NPU-FEASIBILITY.md` records as hanging on the
  first decode.

---

## Where I looked and found nothing

| Looked for | Searched | Result |
| --- | --- | --- |
| RenderDoc install | `C:\` full recursive `dir /s /b`, Program Files, Program Files (x86), LocalAppData, AppData\Local\Programs, AppData\Roaming, Downloads, ProgramData, `C-Res`, `A-Works`, `C:\VulkanSDK`, `C:\Tools`, `C:\dev`, `C:\SDK` | Only the two `RenderDoc_1.47_64` trees, the `tools\rd` source tree, and Rider's ARM64 runtime |
| RenderDoc in Add/Remove Programs | `HKLM`, `HKLM\WOW6432Node`, `HKCU` `...\CurrentVersion\Uninstall\*` | No `DisplayName` matching RenderDoc. Absent |
| `qrenderdoc.exe` for x86 | Both RenderDoc trees | Absent (x64 only) |
| ARM64 RenderDoc layer manifest | Rider `windows-aarch64` runtime dir | Absent (only `renderdoc.dll` and `RenderDocHost.exe`) |
| `Bin32` / `Lib32` | `C:\VulkanSDK\1.4.363.1` | Absent |
| ARM64 validation layer | `C:\VulkanSDK\1.4.363.1\Bin-ARM64` | Absent |
| ARM64 api_dump layer | `C:\VulkanSDK\1.4.363.1\Bin-ARM64` | Absent |
| Any registered Vulkan layer | `HKLM`/`HKCU`/`WOW6432Node` `Khronos\Vulkan\Layers` + `ImplicitLayers`, `C:\ProgramData\Vulkan` | All absent |
| Vulkan ICD registration | `Khronos\Vulkan\Drivers` on three hives; `*icd*.json` in System32 | Keys absent; no ICD JSON found |
| Adreno-specific Vulkan layer | DriverStore `qcdx*` packages, `C:\ProgramData\Qualcomm` | No `VkLayer*.json`, no `*vulkan*.json` |
| `libGenie*` | `C-Res`, Program Files, AppData\Local, AppData\Roaming, ProgramData | Absent everywhere. Only `geniex.h`, `geniex_plugin.dll`, `geniex*.dll` |
| `QnnHtp.dll` / `QnnSystem.dll` outside GenieX | Program Files, Program Files (x86), `C-Res`, `C:\VulkanSDK`, `C:\strata-nospace`, AppData | Absent. Only under `GenieX CLI\qairt\htp-files`, the SDK zip, and the ProgramData driver |
| QNN `.bin` context binaries | `GenieX CLI\qairt` (recursive) | None. Only `.so`, `.dll`, `.cat` |
| Hexagon SDK `addons`, `docs`, `examples`, `test` | `hexagon-sdk\6.7.0.0` | All four absent |
| Hexagon NDK / Halide / QEMU / CMake tools | `hexagon-sdk\6.7.0.0\tools` | Absent (only `debug`, `HEXAGON_Tools`, `qhcg`, `utils`) |
| Hexagon certificate or signing binary | Whole `hexagon-sdk` tree (3.5 GB), `Tools\bin` | No `.pem`/`.p12`/`.cer`/`.crt`/`.der`; no `elfsign`/`attest`/`signtool`. Only `utils\scripts\signer.py` (adb-based) |
| Trellis types in the HTP backend | All 130 files under `ggml\src\ggml-hexagon\` | Zero matches |

Not done, by instruction or because it would be a long job: no build, no
`SDKDependencyDownloader` run, no extraction of `Hexagon SDK.tar.xz`, no `vulkaninfo`, no
gfxreconstruct or RenderDoc capture, no signing.
