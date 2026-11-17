#!/usr/bin/env python3
"""Ask each installed libcdsprpc.dll the same questions, and try to open our skel on it.

This machine carries two FastRPC user libraries:

  qcnspmcdm8480...\\libcdsprpc.dll   the NSP-MCDM one the engine currently loads (630856 bytes)
  qcadsprpc8480...\\libcdsprpc.dll   the adsp one (380488 bytes)

The first answers DSPRPC_GET_DSP_INFO and DSPRPC_CONTROL_UNSIGNED_MODULE but reports
FASTRPC_GET_DOMAINS as AEE_EUNSUPPORTEDAPI (0x6c) and FASTRPC_MANAGE_DSP_LIBRARY_PATH as
AEE_EUNSUPPORTED (0x14), and then every remote_handle64_open fails with AEE_ERPC (0x200).
If the second library implements those requests, pointing the engine at it is a host-only
change, so this decides whether the NPU needs an elevated registration or a driver swap.

    python port/tools/htp-driver-variant-probe.py [skel_dir]
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys

SUFFIX = "?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0"

# remote.h request ids
DSPRPC_CONTROL_UNSIGNED_MODULE = 2
FASTRPC_GET_DOMAINS = 5
FASTRPC_MANAGE_DSP_LIBRARY_PATH = 6
DSPRPC_GET_DSP_INFO = 16

DRIVERS = sorted(
    glob.glob(r"C:\WINDOWS\System32\DriverStore\FileRepository\qc*nsp*.inf_arm64*\libcdsprpc.dll")
    + glob.glob(r"C:\WINDOWS\System32\DriverStore\FileRepository\qc*dsp*rpc*.inf_arm64*\libcdsprpc.dll")
)


def decode(rc: int) -> str:
    """Name the small AEE codes that matter here; print anything else as hex."""
    table = {
        0: "AEE_SUCCESS",
        0x0E: "AEE_EBADPARM",
        0x14: "AEE_EUNSUPPORTED",
        0x6C: "AEE_EUNSUPPORTEDAPI",
        0x200: "AEE_ERPC",
        0x80000406: "AEE_EUNABLETOLOAD",
    }
    if rc in table:
        return table[rc]
    if (rc & 0xFFFFFFFF) in table:
        return table[rc & 0xFFFFFFFF]
    return f"0x{rc & 0xFFFFFFFF:08x}"


class UnsignedParm(ctypes.Structure):
    _fields_ = [("domain", ctypes.c_int), ("enable", ctypes.c_int)]


class DspCap(ctypes.Structure):
    _fields_ = [("domain", ctypes.c_uint32), ("attribute_ID", ctypes.c_uint32),
                ("capability", ctypes.c_uint32)]


def probe(path: str, skel_dir: str) -> None:
    print(f"\n=== {path}")
    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:
        print(f"    load failed: {exc}")
        return

    for sym in ("remote_session_control", "remote_handle64_open", "remote_handle64_close"):
        if not hasattr(lib, sym):
            print(f"    missing export {sym}")
            return
    lib.remote_session_control.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
    lib.remote_session_control.restype = ctypes.c_int
    lib.remote_handle64_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.remote_handle64_open.restype = ctypes.c_int
    lib.remote_handle64_close.argtypes = [ctypes.c_uint64]
    lib.remote_handle64_close.restype = ctypes.c_int

    cap = DspCap(domain=3, attribute_ID=1, capability=0)  # ARCH_VER
    rc = lib.remote_session_control(DSPRPC_GET_DSP_INFO, ctypes.byref(cap), ctypes.sizeof(cap))
    print(f"    GET_DSP_INFO            -> {decode(rc)}" + (f"  capability 0x{cap.capability & 0xff:x}" if rc == 0 else ""))

    # GET_DOMAINS needs a two-call protocol; a null list still reports whether the id is known.
    class SysReq(ctypes.Structure):
        _fields_ = [("id", ctypes.c_uint32),
                    ("domains", ctypes.c_void_p),
                    ("max_domains", ctypes.c_int),
                    ("num_domains", ctypes.c_int),
                    ("flags", ctypes.c_int)]
    sr = SysReq(id=FASTRPC_GET_DOMAINS, domains=None, max_domains=0, num_domains=0, flags=0)
    rc = lib.remote_session_control(FASTRPC_GET_DOMAINS, ctypes.byref(sr), ctypes.sizeof(sr))
    print(f"    GET_DOMAINS             -> {decode(rc)}")

    parm = UnsignedParm(domain=-1, enable=1)
    rc = lib.remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, ctypes.byref(parm), ctypes.sizeof(parm))
    print(f"    UNSIGNED_MODULE         -> {decode(rc)}")

    for name in ("libggml-htp-v81.so",):
        for label, uri in (
            ("bare + dom", f"file:///{name}{SUFFIX}"),
            ("abs + dom", f"file:///{skel_dir.replace(chr(92), '/')}/{name}{SUFFIX}"),
        ):
            handle = ctypes.c_uint64(0)
            rc = lib.remote_handle64_open(uri.encode(), ctypes.byref(handle))
            print(f"    open {label:<12}    -> {decode(rc)}" + (f"  handle 0x{handle.value:x}" if rc == 0 else ""))
            if rc == 0:
                lib.remote_handle64_close(handle.value)


def main(argv: list[str]) -> int:
    skel_dir = os.path.abspath(argv[0]) if argv else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "build", "bin")
    print(f"skel dir {skel_dir}")
    if not DRIVERS:
        print("no libcdsprpc.dll found in the driver store")
        return 2
    for path in DRIVERS:
        probe(path, skel_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
