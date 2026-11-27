#!/usr/bin/env python3
"""Probe the FastRPC link directly, one call at a time, to find out why the HTP session will not open.

`ggml-hexagon` reports `htp_iface_open` failing with 0x80000406 and prints nothing about the calls before it.  This
reproduces the same sequence through `libcdsprpc.dll` and prints every return code, so the failing gate is visible:

    unsigned PD -> DSP search path (get/set) -> FASTRPC_GET_URI -> the URI a session would be opened with

Run it from anywhere:

    python port/tools/fastrpc_probe.py [skel_dir]

`skel_dir` (default: build\\bin) is added to the DSP library search path, which is where FastRPC looks for
`libggml-htp-v<arch>.so` when it turns a host module path into a device URI.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, byref, c_char_p, c_int, c_uint32, c_void_p, Structure

DRIVER = r"C:\Windows\System32\DriverStore\FileRepository\qcnspmcdm8480.inf_arm64_cde10e01d3a2d2e1\libcdsprpc.dll"

REQ_THREAD_PARAMS = 1
REQ_UNSIGNED_MODULE = 2
REQ_GET_DOMAINS = 13          # placeholder; the real one is a domain-info call, not used here
REQ_GET_EFFECTIVE_DOMAIN_ID = 14
REQ_GET_URI = 15
REQ_MANAGE_DSP_LIB_PATH = 20


class UnsignedModule(Structure):
    _fields_ = [("domain", c_int), ("enable", c_int)]


class EffectiveDomainId(Structure):
    _fields_ = [("domain_name", c_char_p), ("domain_name_len", c_uint32),
                ("session_id", c_uint32), ("effective_domain_id", c_int)]


class GetUri(Structure):
    _fields_ = [("domain_name", c_char_p), ("domain_name_len", c_uint32), ("session_id", c_uint32),
                ("module_uri", c_char_p), ("module_uri_len", c_uint32),
                ("uri", c_void_p), ("uri_len", c_uint32)]


class ManageLibPath(Structure):
    _fields_ = [("effec_domain_id", c_int), ("type", c_int), ("path", c_char_p), ("pathlen", c_uint32)]


def decode_error(rc: int) -> str:
    """FastRPC reports AEE_* codes as 0x80000000 | n; n is the enum position in AEEStdErr.h."""
    if rc == 0:
        return "AEE_SUCCESS"
    low = rc & 0x7FFFFFFF
    return f"0x{rc:08x} (low {low})"


def main(argv: list[str]) -> int:
    skel_dir = argv[0] if argv else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "build", "bin")
    if not os.path.isdir(skel_dir):
        print(f"not a directory: {skel_dir}")
        return 2
    if not os.path.isfile(DRIVER):
        print(f"missing FastRPC driver: {DRIVER}")
        return 2

    lib = ctypes.CDLL(DRIVER)
    lib.remote_session_control.argtypes = [c_uint32, c_void_p, c_uint32]
    lib.remote_session_control.restype = c_int

    def control(req: int, payload, size: int) -> int:
        return lib.remote_session_control(req, ctypes.cast(byref(payload), c_void_p), size)

    print(f"driver  {DRIVER}")
    print(f"skel dir {skel_dir}")

    # 1. Unsigned PD, for every domain (-1), the way ggml-hexagon does it.
    um = UnsignedModule(domain=-1, enable=1)
    rc = control(REQ_UNSIGNED_MODULE, um, ctypes.sizeof(um))
    print(f"unsigned PD            -> {decode_error(rc)}")
    unsigned_ok = rc == 0

    # 2. Whatever the DSP is currently told to search for modules.
    path_buf = ctypes.create_string_buffer(4096)
    mlp = ManageLibPath(effec_domain_id=-1, type=0, path=ctypes.cast(path_buf, c_char_p), pathlen=4096)
    rc = control(REQ_MANAGE_DSP_LIB_PATH, mlp, ctypes.sizeof(mlp))
    print(f"get DSP search path    -> {decode_error(rc)}")
    if rc == 0:
        print(f"  current: {path_buf.value.decode('mbcs', 'replace')!r}")
    else:
        print(f"  (this is the first call that can explain GET_URI failing)")

    # 3. Prepend our directory, which is where the skel and its signed catalog live.
    want = skel_dir.encode("mbcs")
    mlp_set = ManageLibPath(effec_domain_id=-1, type=1, path=want, pathlen=len(want))
    rc = control(REQ_MANAGE_DSP_LIB_PATH, mlp_set, ctypes.sizeof(mlp_set))
    print(f"set DSP search path    -> {decode_error(rc)}  ({skel_dir})")
    if rc == 0:
        path_buf2 = ctypes.create_string_buffer(4096)
        mlp_get = ManageLibPath(effec_domain_id=-1, type=0, path=ctypes.cast(path_buf2, c_char_p), pathlen=4096)
        control(REQ_MANAGE_DSP_LIB_PATH, mlp_get, ctypes.sizeof(mlp_get))
        print(f"  now: {path_buf2.value.decode('mbcs', 'replace')!r}")

    # 4. The URI ggml-hexagon asks for, verbatim.
    for arch in ("v81", "v85", "v79", "v75", "v73"):
        module = f"file:///libggml-htp-{arch}.so?htp_iface_skel_handle_invoke&_modver=1.0".encode()
        uri_buf = ctypes.create_string_buffer(512)
        gu = GetUri(domain_name=b"cdsp", domain_name_len=4, session_id=0,
                    module_uri=module, module_uri_len=len(module),
                    uri=ctypes.cast(uri_buf, c_void_p), uri_len=512)
        rc = control(REQ_GET_URI, gu, ctypes.sizeof(gu))
        got = uri_buf.value.decode("mbcs", "replace")
        print(f"get URI {arch}          -> {decode_error(rc)}  {got!r}" if rc else f"get URI {arch}          -> ok  {got!r}")

    # 5. The effective domain id, which the session URI carries.
    ed = EffectiveDomainId(domain_name=b"cdsp", domain_name_len=4, session_id=0, effective_domain_id=-1)
    rc = control(REQ_GET_EFFECTIVE_DOMAIN_ID, ed, ctypes.sizeof(ed))
    print(f"effective domain id    -> {decode_error(rc)}  {ed.effective_domain_id}")

    print()
    print("verdict: " + ("unsigned PD is available" if unsigned_ok else "the driver refuses an unsigned PD, which no "
                       "placement work can work around"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
