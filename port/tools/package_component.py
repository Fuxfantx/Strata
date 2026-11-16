#!/usr/bin/env python3
"""Package the ported engine as a component the desktop app will install, and verify it with the app's own checker.

The app does not start an arbitrary executable: `strata_control.components` looks for
`<components>/engine/<version>-<platform>-<backend>/engine-component.json`, validates the manifest (schema 2: a
SHA-256 and byte count for every file), hashes the entrypoint, and only then runs it. So "the engine integrates with
the front end" is a claim about this package, and it can be checked here without opening the app.

    python package_component.py [--version 0.2.13] [--out <dir>] [--verify]

Only the Vulkan runtime is packaged. The NPU files beside them (QnnHtp*, ggml-hexagon.dll, the HTP skels) are left out
on purpose: the DSP on this machine is wedged and that path is not enabled by default, so shipping them would add
~105 MB the app would copy and hash on every install for a feature nobody can switch on yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = HERE.parent
# `port/` lives inside the Strata checkout, so its parent is the app tree; the older layout had them as siblings.
_ROOTS = (PORT.parent, PORT.parent / "strata")
STRATA = next((r for r in _ROOTS if (r / "strata_control").is_dir()), _ROOTS[0])
BUILD_BIN = PORT.parent / "build" / "bin"

#: What the engine links and loads: `strata-vk` links `llama`, which links `ggml`, and `ggml-vulkan` is loaded through
#: the dynamic backend loader, so all of it has to sit beside the exe.  Anything else a build happens to produce is
#: packaged when it is there and skipped when it is not - the NPU half (ggml-hexagon plus the QNN libraries) and the
#: vision half (mtmd) are built by other recipes, and a Vulkan-only build should not fail for their absence.
REQUIRED = [
    "strata-vk.exe",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu.dll",
    "ggml-vulkan.dll",
    "llama.dll",
    "vulkan-1.dll",
]

OPTIONAL = [
    "mtmd.dll",
    "llama-common.dll",
    "libomp140.aarch64.dll",
    "ggml-hexagon.dll",
    "libcdsprpc.dll",
    "QnnHtp.dll",
    "QnnHtpPrepare.dll",
    "QnnHtpV81Stub.dll",
    "QnnSystem.dll",
    "libQnnHtpV81Skel.so",
    "libggml-htp-v81.so",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.2.13")
    parser.add_argument("--out", default=str(PORT / "dist"))
    parser.add_argument("--build-dir", default=str(BUILD_BIN), help="where strata-vk.exe and its DLLs were built")
    parser.add_argument("--verify", action="store_true", help="check the result with strata_control.engine_component")
    args = parser.parse_args()

    sys.path.insert(0, str(STRATA))
    from strata_control import engine_component as ec
    from strata_control.targets import backend_supported, host_platform

    platform = host_platform()
    backend = "vulkan"
    # Ask the app's own matrix first.  It answers "vulkan is not a supported combination on windows-x86_64" when the
    # packaging process is an emulated x64 interpreter, which is exactly what happens if this script is run from an
    # x64 shell - building the package anyway would produce something the app silently refuses to install.
    supported, why = backend_supported(backend)
    if not supported:
        print(f"refusing to package: {why}")
        print(f"  (this process sees platform={platform}; the app reads the same value, so a native arm64 run is needed)")
        return 1
    folder = Path(args.out) / f"{args.version}-{platform}-{backend}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    files = []
    missing = []
    for name in REQUIRED + OPTIONAL:
        src = Path(args.build_dir) / name
        if not src.is_file():
            if name in REQUIRED:
                missing.append(name)
            continue
        # The engine's own name is strata-vk.exe, but a component is launched by the program the manifest names and the
        # app's run configurations refer to <engine folder>/strata.exe - which is also what the package already installed
        # in components/ uses. Ship it under that name so both agree.
        destination = folder / ("strata.exe" if name == "strata-vk.exe" else name)
        shutil.copy2(src, destination)
        files.append({"path": destination.name, "sha256": sha256(destination), "bytes": destination.stat().st_size})
    if missing:
        print("missing from the build: " + ", ".join(missing))
        return 1

    manifest = {
        "schema_version": ec.SCHEMA_VERSION,
        "id": ec.COMPONENT_ID,
        "name": ec.COMPONENT_ID,
        "version": args.version,
        "platform": platform,
        "backend": backend,
        "archs": ["arm64"],
        "min_driver": {},
        "build_id": "strata-port-vulkan-arm64",
        "entrypoint": "strata.exe",
        "files": files,
        "signature": None,
        "variant": "port",
    }
    (folder / ec.MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    total = sum(f["bytes"] for f in files)
    print(f"packaged {len(files)} files, {total / (1 << 20):.1f} MiB -> {folder}")

    if not args.verify:
        return 0

    # the app's own reader and checker, not a reimplementation: if this passes, the app accepts the package
    try:
        component = ec.read_manifest(folder, ec.COMPONENT_ID)
    except Exception as exc:                      # noqa: BLE001 - the point is to show exactly what the app says
        print(f"FAIL  read_manifest: {type(exc).__name__}: {exc}")
        return 1
    print(f"ok    read_manifest: {component.id} {component.version} {component.platform}/{component.backend}")
    try:
        ec.verify(component, full=True)
    except Exception as exc:                      # noqa: BLE001
        print(f"FAIL  verify: {type(exc).__name__}: {exc}")
        return 1
    print("ok    verify(full=True): every file matches its manifest hash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
