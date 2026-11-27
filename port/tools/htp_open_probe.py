#!/usr/bin/env python3
"""Call the exact FastRPC open that `ggml-hexagon` calls, and vary the URI until it opens.

`ggml-hexagon.dll` exports the generic FastRPC stubs (`remote_handle64_open`, `remote_session_control`,
`htpdrv_init`), and `htp_iface_open(uri, &handle)` is `remote_handle64_open(uri, &handle)` with the skel-invoke
trampoline the DLL already contains.  So the failing call can be driven from here - no rebuild, no C++ - and the URI,
the working directory and the skel file can be varied freely.

    python port/tools/htp_open_probe.py [skel_dir]

Prints one line per attempt with the return code, so the gate is either "this URI works" or "the driver refuses all of
them", and the difference between those two answers is the whole NPU question.
"""
from __future__ import annotations

import ctypes
import os
import sys

DLL = os.environ.get("HTP_PROBE_DLL") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "build", "bin", "ggml-hexagon.dll")

SUFFIX = "?htp_iface_skel_handle_invoke&_modver=1.0"


def decode(rc: int) -> str:
    if rc == 0:
        return "AEE_SUCCESS"
    return f"0x{rc:08x}"


def main(argv: list[str]) -> int:
    skel_dir = os.path.abspath(argv[0]) if argv else os.path.dirname(DLL)
    if not os.path.isfile(DLL):
        print(f"missing {DLL}")
        return 2

    # Point dependent-DLL resolution at the library's own directory, so the vendor
    # ggml-hexagon.dll can be probed beside its own ggml-base.dll.
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.dirname(DLL))

    lib = ctypes.CDLL(DLL)
    lib.htpdrv_init.restype = ctypes.c_int
    lib.remote_session_control.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
    lib.remote_session_control.restype = ctypes.c_int
    lib.remote_handle64_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.remote_handle64_open.restype = ctypes.c_int
    lib.remote_handle64_close.argtypes = [ctypes.c_uint64]
    lib.remote_handle64_close.restype = ctypes.c_int

    rc = lib.htpdrv_init()
    print(f"htpdrv_init            -> {decode(rc)}")

    class Unsigned(ctypes.Structure):
        _fields_ = [("domain", ctypes.c_int), ("enable", ctypes.c_int)]

    u = Unsigned(domain=-1, enable=1)
    rc = lib.remote_session_control(2, ctypes.cast(ctypes.byref(u), ctypes.c_void_p), ctypes.sizeof(u))
    print(f"unsigned PD            -> {decode(rc)}")

    print(f"skel dir               {skel_dir}")
    present = sorted(n for n in os.listdir(skel_dir) if n.startswith("libggml-htp"))
    print(f"skel files there       {present}")

    def attempt(label: str, uri: str) -> bool:
        h = ctypes.c_uint64(0)
        rc = lib.remote_handle64_open(uri.encode(), ctypes.byref(h))
        ok = rc == 0
        print(f"{label:<34}-> {decode(rc)}" + (f"  handle 0x{h.value:x}" if ok else ""))
        if ok:
            lib.remote_handle64_close(h.value)
        return ok

    arch = "v81"
    # The URI ggml-hexagon builds, verbatim.
    attempt("as ggml builds it", f"file:///libggml-htp-{arch}.so{SUFFIX}&_dom=cdsp&_session=0")
    attempt("without dom/session", f"file:///libggml-htp-{arch}.so{SUFFIX}")

    # With the host path spelled out, which is what FastRPC needs when it does not search.
    fwd = skel_dir.replace("\\", "/")
    attempt("absolute host path", f"file:///{fwd}/libggml-htp-{arch}.so{SUFFIX}&_dom=cdsp&_session=0")
    attempt("absolute, no scheme triple", f"file:/{fwd}/libggml-htp-{arch}.so{SUFFIX}&_dom=cdsp&_session=0")

    # Sometimes the loader wants the module directory as the working directory instead of a path in the URI.
    saved = os.getcwd()
    try:
        os.chdir(skel_dir)
        attempt("relative, cwd = skel dir", f"libggml-htp-{arch}.so{SUFFIX}&_dom=cdsp&_session=0")
        attempt("file:/// under cwd", f"file:///libggml-htp-{arch}.so{SUFFIX}&_dom=cdsp&_session=0")
    finally:
        os.chdir(saved)

    # Other skels in the same directory, in case the archived one for this HTP is not v81.
    for other in ("v73", "v75", "v79", "v85"):
        if f"libggml-htp-{other}.so" in present:
            attempt(f"arch {other}", f"file:///{fwd}/libggml-htp-{other}.so{SUFFIX}&_dom=cdsp&_session=0")

    # Modules the driver already allow-lists, mapped to \cdsp\<name> in
    # HKLM\SYSTEM\CurrentControlSet\Enum\ACPI\VEN_QCOM&DEV_0FF0...\Device Parameters\Mappings\CDSP\Default.
    # If these open and ours do not, the allow-list is the only gap.
    for known in ("version.so", "libbenchmark_skel.so", "example_image.so"):
        attempt(f"allow-listed {known}", f"file:///{known}{SUFFIX}&_dom=cdsp&_session=0")

    # A name that is in no allow-list at all. If this returns the same code as a mapped
    # module, the code does not describe the module and the failure is earlier.
    attempt("unmapped name", f"file:///no-such-module.so{SUFFIX}&_dom=cdsp&_session=0")

    # The sweep below can block inside a driver call, where no Python timeout can reach it,
    # so it only runs on request. One variant per run, in its own process:
    #   HTP_PROBE_EXTRA=1 python port/tools/htp_open_probe.py build\bin
    if not os.environ.get("HTP_PROBE_EXTRA"):
        print("(skipping domain/session sweep; set HTP_PROBE_EXTRA=1 to run it)")
        return 0

    # The adsp device carries ~200 shipped mappings under \adsp. If a load in that domain
    # answers differently from every cdsp load, the cdsp process domain is what fails.
    for dom, known in (("adsp", "libsns_device_mode_skel.so"),
                       ("adsp", "libsysmonquery_skel.so"),
                       ("cdsp", "version.so")):
        attempt(f"_dom={dom} {known}", f"file:///{known}{SUFFIX}&_dom={dom}&_session=0")

    # Session ids past 0, in case the first session slot is already taken on this machine.
    for sid in (1, 2, 3):
        attempt(f"session {sid}", f"file:///libggml-htp-{arch}.so{SUFFIX}&_dom=cdsp&_session={sid}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
