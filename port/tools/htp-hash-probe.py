"""Reverse the Windows FastRPC mapping key rule from the installed NSP-MCDM driver.

The driver keys its module allow-list by a 64-hex string under
Mappings\\CDSP\\Default\\<key>, holding Local (host path) and remote (device path).
Find which hash rule reproduces those keys.
"""
import hashlib
import struct
import sys
from pathlib import Path

CDSP = Path(r"C:\WINDOWS\System32\DriverStore\FileRepository"
            r"\qcnspmcdm8480.inf_arm64_cde10e01d3a2d2e1\CDSP")

# key -> file name, taken from the registry dump
SAMPLES = [
    ("cec094c6fdde90f323146d17505743495e7076fd1a1cb7a28011ca1b7ddbbbf9", "version.so", r"\cdsp\version.so"),
    ("c09f35ad969d2a3f527e777448bfc9bbf3c5eb24508bfb43d06bc8c42e0d54bf", "libUbwcD.so", r"\cdsp\libubwcd.so"),
    ("3535ded43202eede0d204355dd3e0d160517d5c7747e94107d4e9736935e0c70", "libbenchmark_skel.so", r"\cdsp\libbenchmark_skel.so"),
    ("b6668938c8ee9c6046be8e4dccc0912eff69c9339c61ec421f3b0cbb22f08aee", "libsysmonquery_skel.so", r"\cdsp\libsysmonquery_skel.so"),
]


def phdrs(blob):
    """Return (type, offset, filesz) for a 64-bit little-endian ELF, else []."""
    if blob[:4] != b"\x7fELF" or blob[4] != 2:
        return []
    e_phoff = struct.unpack_from("<Q", blob, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", blob, 0x36)[0]
    e_phnum = struct.unpack_from("<H", blob, 0x38)[0]
    out = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type = struct.unpack_from("<I", blob, off)[0]
        p_offset = struct.unpack_from("<Q", blob, off + 0x08)[0]
        p_filesz = struct.unpack_from("<Q", blob, off + 0x20)[0]
        out.append((p_type, p_offset, p_filesz))
    return out


def sha(b):
    return hashlib.sha256(b).hexdigest()


def candidates(name, blob, remote):
    """Yield (label, digest)."""
    yield "full", sha(blob)
    yield "utf8-remote", sha(remote.encode())
    yield "utf8-remote-lower", sha(remote.lower().encode())
    yield "utf16-remote", sha(remote.encode("utf-16-le"))
    yield "utf8-name", sha(name.encode())

    ph = phdrs(blob)
    if not ph:
        return
    # Loadable content ends at the highest p_offset + p_filesz of a PT_LOAD.
    loads = [(o, s) for t, o, s in ph if t == 1 and s]
    if loads:
        end = max(o + s for o, s in loads)
        if end < len(blob):
            yield f"loads-end(0x{end:x})", sha(blob[:end])
            yield f"loads-end+phdr", sha(blob)
    # Qualcomm hash segment: a segment starting past the loadable content.
    for t, o, s in ph:
        yield f"seg-type{t}-off0x{o:x}", sha(blob[:o])
    # Plain truncations on a page grid.
    for pages in range(1, 8):
        cut = len(blob) - 4096 * pages
        if cut > 0:
            yield f"trunc-{pages * 4096}", sha(blob[:cut])
    for pages in range(1, 5):
        cut = len(blob) - 1024 * pages
        if cut > 0:
            yield f"trunc-{pages * 1024}", sha(blob[:cut])


def main():
    hits = 0
    for key, fname, remote in SAMPLES:
        path = CDSP / fname
        if not path.exists():
            print(f"missing {fname}")
            continue
        blob = path.read_bytes()
        ph = phdrs(blob)
        print(f"=== {fname} size={len(blob)} key={key[:16]}...")
        print(f"    phdrs: {[(t, hex(o), hex(s)) for t, o, s in ph]}")
        for label, digest in candidates(fname, blob, remote):
            if digest == key:
                print(f"    HIT  {label}")
                hits += 1
                break
        else:
            print("    no candidate matched")
    print(f"\nhits={hits}/{len(SAMPLES)}")
    return 0 if hits == len(SAMPLES) else 1


if __name__ == "__main__":
    sys.exit(main())
