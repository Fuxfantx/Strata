# Building the Windows-on-ARM desktop app with a built-in engine

Everything in this repository is enough to build the app and its engine. Three ARM64 runtimes are not in git — they are
release artifacts, listed below with where to get them — and the script that stages them refuses to continue without
them rather than shipping an app that fails at a user's first start.

What the repository already contains:

| Path | What it is |
| --- | --- |
| `port/llama-overlay-vanilla/` | the 36 files this port changes or adds to vanilla llama.cpp, at their real paths: the trellis types and kernels, the MTP graft, the shard fallbacks, the engine target |
| `port/tools/fetch-llama.ps1` | downloads the pinned upstream commit and copies the overlay over it, producing `third_party/llama.cpp` |
| `port/engine` | the Strata component itself: `--serve` line protocol, model load, prefix reuse, Adreno tuning, NPU plan |
| `port/tools/package_component.py` | turns a built engine into the component the app installs, and verifies it with the app's own checker |
| `port/packaging/stage-app.ps1` | stages the four resource trees Tauri bundles |
| `apps/desktop`, `apps/host`, `strata_control`, `strata_cli` | the app, its host server and Control |

## Prerequisites

* **Visual Studio 18** with the ARM64 C++ toolset (`clang-cl`, `lld-link`, `ninja`, CMake ≥ 3.20).
* **Vulkan SDK** with the ARM64 import library (`Lib-ARM64\vulkan-1.lib`). The x64 one is picked by default and fails to
  link with "machine type x64 conflicts with arm64".
* **Node 24, win-arm64** —<https://nodejs.org/dist/v24.21.0/> (`node-v24.21.0-win-arm64.zip`).
* **CPython 3.15, ARM64** — the interpreter Control runs on.
* **Control's wheels** installed into a directory, e.g.
  `python -m pip install --target strata_control\wheels -r strata_control\requirements.txt`.

Pass the last three to the staging step. Paths in the commands below assume you unpacked them to `C:\arm64\`.

### The shell's own toolchain (Rust)

`apps/desktop/src-tauri` is a Rust crate: building the installer needs **Rust stable for `aarch64-pc-windows-msvc`** on
top of the Arm64 C++ toolset above. The host's `rustup-init.exe` is
`https://static.rust-lang.org/rustup/dist/aarch64-pc-windows-msvc/rustup-init.exe`; measured here it downloads at
0.05-0.25 MB/s, so point rustup at a mirror:

```powershell
$env:RUSTUP_DIST_SERVER = "https://mirrors.aliyun.com/rustup"     # ~1.2 MB/s here; tuna and ustc answer 403
.\rustup-init.exe -y --default-toolchain stable --profile minimal --no-modify-path
cargo --version                                                   # 1.96.0 (aarch64-pc-windows-msvc)
```

One blocker that is not the code: `cargo check` stops in `ring v0.17.14`'s C/assembly build with
`clang: error: unable to make temporary file: permission denied`, while the same clang command run by hand from the same
directory compiles that `.S` file. It reproduces from a path with a space and from a junction path without one, with a
wrapper compiler that sets a private `TMP`, and from a plain `cmd.exe` session, so it is how `cc-rs` invokes clang on this
host rather than the crate or the sources. Until it is worked around, the installer build has not been exercised on this
machine.

## 0. Fetch llama.cpp

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/fetch-llama.ps1      # -> third_party/llama.cpp
```

Upstream source is not vendored here; the script downloads the pinned commit (`ggml-org/llama.cpp` at `32bdc3a8`) and copies `port/llama-overlay-vanilla/` over it.

## 0b. Build the engine (one command)

```powershell
powershell -ExecutionPolicy Bypass -File port/tools/build-engine.ps1
```

It configures `build\` with the flags below, builds `strata-vk` and stages the ARM64 `vulkan-1.dll` next to the exe (see
the trap at the end of this section). `-Build` takes another build directory and `-Target` another target -
`mm-probe` is the kernel instrument, `test-backend-ops` the correctness oracle.

### The OpenCL backend (optional, and how it was evaluated)

`ggml-opencl` needs the Khronos headers and an import library, neither of which is on a stock machine, so
`third_party/opencl` holds `OpenCL-Headers` plus `opencl.lib` generated from the system DLL's own export table:

```powershell
# once, to regenerate the import library from the installed driver
llvm-dlltool -m arm64 -d third_party/opencl/opencl.def -l third_party/opencl/opencl.lib
cmake -S third_party/llama.cpp -B build -DGGML_OPENCL=ON ^
  -DOpenCL_INCLUDE_DIR="$PWD/third_party/opencl" -DOpenCL_LIBRARY="$PWD/third_party/opencl/opencl.lib"
```

The engine then selects one backend at run time, never both (`llama.cpp` would split layers between two drivers for the
same GPU): `--gpu vulkan|opencl|cpu`, or `STRATA_VK_GPU`. `port/docs/ADRENO.md` has the measurements and why Vulkan
stays the default.

## 1. Build the engine

```powershell
$vs = "C:\Program Files\Microsoft Visual Studio\18\Community"
$env:Path = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$vs\VC\Tools\Llvm\ARM64\bin;" + $env:Path

cmake -S third_party/llama.cpp -B build -G Ninja `
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Release `
  -DGGML_VULKAN=ON -DGGML_OPENMP=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF `
  -DSTRATA_VK_DIR="$PWD/port/engine" -DGGML_BACKEND_DL=ON `
  -DVulkan_LIBRARY="C:/VulkanSDK/1.4.363.1/Lib-ARM64/vulkan-1.lib"

cmake --build build --target strata-vk
```

`GGML_BACKEND_DL=ON` is not optional: the engine loads `ggml-vulkan.dll` (and `ggml-hexagon.dll` when the NPU path is
enabled) at run time, which is also why the package carries the DLLs beside `strata.exe`.

**Copy the ARM64 `vulkan-1.dll` next to the built exe.** Without it `ggml-vulkan.dll` fails to load, the engine silently
registers the CPU backend alone, and the output is both wrong and slow - a real trap: the failure is a `load_backend`
line in stderr, not an error.

Take it from a known-ARM64 source, never "whichever one Windows has": on this machine the shell runs under x64 emulation,
so `C:\Windows\System32\vulkan-1.dll` is the **x64** loader, and loading it from an ARM64 process fails with
`ERROR_BAD_EXE_FORMAT` (193). The earliest build in this repository has a good copy; alternatively take the DLL that
ships with an ARM64 application.

```powershell
copy "build\bin\vulkan-1.dll" build\bin        # from an earlier build of this tree
```

**Verify what you copied** - the filename does not say:

```powershell
python -c "import struct,sys;d=open(sys.argv[1],'rb').read(0x1000);o=struct.unpack_from('<I',d,0x3c)[0];print('ARM64' if struct.unpack_from('<H',d,o+4)[0]==0xAA64 else 'NOT ARM64')" build\bin\vulkan-1.dll
```

Then confirm the backend actually registered - this is the one-command check, and it should precede any performance run:

```powershell
build\bin\strata-vk.exe --geometry      # must list Vulkan0, not only CPU
```

## 1b. The Hexagon NPU backend (optional)

The engine builds without it - the GPU path is unaffected - but `--npu` needs `ggml-hexagon.dll`, and that needs a Hexagon
SDK on the build machine. A 6.7.0.0 SDK is enough; **no signing is involved**, because the DSP-side artefacts are shipped
prebuilt and already signed.

```powershell
powershell -ExecutionPolicy Bypass -File port/tools/build-engine.ps1
```

`build-engine.ps1` looks for the SDK itself (`C:\Users\Fuxfantx\C-Res\hexagon-sdk\6.7.0.0`, then
`$env:HEXAGON_SDK_ROOT`, then the Qualcomm install paths), passes the three definitions the SDK needs, builds
`ggml-hexagon` before the engine, and stages the signed skels. `-HexagonSdk` takes an explicit SDK path instead, and
`-NoHexagon` builds without the NPU backend.

Two details that are not obvious and cost an hour each when missed:

* **`PREBUILT_LIB_DIR` has to be defined at the top level.** The SDK's `hexagon_fun.cmake` runs
  `string(FIND ${PREBUILT_LIB_DIR} ...)` as soon as it is included, and that variable is normally supplied by the skel
  sub-build. Configure fails inside the SDK's own cmake until it is set (`-DPREBUILT_LIB_DIR=toolv21_v81`).
* **Build only the `ggml-hexagon` target.** The skels (`htp-v73` ... `htp-v85`) are `ExternalProject`s that need the DSP
  toolchain plus a signing certificate. They are not dependencies of the host library, so `--target ggml-hexagon` avoids
  them entirely - and the signed `.so` files plus `libggml-htp.cat` are already beside the exe.

Check it after building:

```powershell
build\bin\strata-vk.exe --geometry      # Vulkan0, GPUOpenCL, HTP0, CPU
```

The backend logs `Hexagon Arch version v81` and `device 0: HTP0 (phys=0, virt=0, domain=cdsp:3)` when it loads. FastRPC
resolves the skel by bare name against the process's current directory, which is why the engine changes directory for the
duration of an NPU load; `port/docs/NPU-FEASIBILITY.md` has the measurements and that story.

## 2. Package the engine component

The app installs the engine as a component and runs the program its manifest names. Build the package and let the app's
own checker verify it — run this with Control's interpreter, because the app derives the platform from the *process*
architecture and an emulated x64 interpreter reports `windows-x86_64`, for which Vulkan is not a supported combination.

```powershell
C:\arm64\python\python.exe port\tools\package_component.py --verify
# packaged 10 files, 59.3 MiB -> port/dist/0.2.13-windows-arm64-vulkan
# ok  read_manifest: strata-engine 0.2.13 windows-arm64/vulkan
# ok  verify(full=True): every file matches its manifest hash
```

## 3. Stage the app resources

```powershell
powershell -File port\packaging\stage-app.ps1 `
  -NodeDir   C:\arm64\node-v24.21.0-win-arm64 `
  -PythonDir C:\arm64\python `
  -SitePackages C:\arm64\wheels
```

This writes `apps/desktop/src-tauri/resources/{node,control,engine-bundle,harness}`. The engine bundle comes from
`port/dist` by default; `-EngineBundle` overrides it.

## 4. Build the app

```powershell
npm --prefix apps/host run build          # the UI bundle
npm --prefix apps/desktop run build       # tauri build, ARM64
```

The installer and the app now carry the engine: the first start installs the bundled component and runs it through the
`--serve` line protocol.

## Verifying a build without opening the app

```powershell
# the protocol the app speaks: READY, INFO, one GEN per token, the 15-field DONE, STOP, QUIT
node port\tests\serve_contract.mjs build\bin\strata-vk.exe C:\Users\Fuxfantx\C-Res\Qwen\Tiny-Qwen.gguf

# throughput, interleaved so thermal drift hits every configuration equally
python port\tools\bench.py --model C:\Users\Fuxfantx\C-Res\Qwen\Tiny-Qwen.gguf --rounds 3 --pp 512 --config "release||"

# one kernel, one shape, one backend (add GGML_VK_DISABLE_F16=1 to measure the fp32-shader configuration)
build\bin\mm-probe.exe -b Vulkan0 -t q4_K -m 896 -n 512 -k 4864

# the Hexagon path: what the driver answers at each step, then what the session does
python port\tools\fastrpc_probe.py
python port\tools\htp_open_probe.py
```

`htp_open_probe.py` is the one that matters if the NPU "does not work": FastRPC resolves the HTP skel by bare name against
the process's current directory, so the same URI opens a session from the skel's folder and fails with `0x80000406` from
anywhere else. The engine now changes directory for the duration of an NPU load (see `port/engine/vk-devices.cpp`), and
the probe is how that conclusion was reached.

## Notes

* `port/docs/ADRENO.md` records the Adreno tuning and what was measured to *not* help.
* `port/docs/NPU-FEASIBILITY.md` covers the Hexagon path, which is off by default.
* The engine's llama.cpp-side change is `port/llama-overlay-vanilla/`; `fetch-llama.ps1` copies it over the pinned upstream
  commit. The overlay is the change itself, not a diff, so it cannot rot against context drift.




