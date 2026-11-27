"""Smoke test: the saved-backend write goes through the atomic writer and round-trips.

Run from the repository root:  python port/tests/check_save_backend.py
"""
import tempfile
from pathlib import Path

from strata_control import engine_component as ec

d = Path(tempfile.mkdtemp())
env = {
    "APPDATA": str(d),
    "LOCALAPPDATA": str(d),
    "USERPROFILE": str(d),
    "XDG_CONFIG_HOME": str(d / "xdg"),
    "STRATA_SETTINGS_FILE": str(d / "settings.json"),
}

p = ec.save_backend("cuda", env)
assert p is not None and p.is_file(), "save_backend returned no file"
assert '"cuda"' in p.read_text(encoding="utf-8"), p.read_text(encoding="utf-8")
assert not list(p.parent.glob("*.tmp")), "the atomic writer left a temp file behind"

p2 = ec.save_backend("", env)
assert "cuda" not in p2.read_text(encoding="utf-8"), "the key was not cleared"
assert ec.saved_backend(env) is None, ec.saved_backend(env)

# An unknown family is not a choice: nothing is written.
p3 = ec.save_backend("vulkan", env)
assert "vulkan" not in p3.read_text(encoding="utf-8")
print("save_backend ok: write, no temp litter, clear, read-back, unknown name ignored")
