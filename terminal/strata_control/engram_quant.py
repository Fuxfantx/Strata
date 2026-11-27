"""Quantising an Edge0 pack's Engram table - the protocol, with the scheme left open on purpose.

The pack's per-layer n-gram table (`ple-fp8.gguf`) is the largest single file in the download: 320,001,536 rows of 160
signed bytes, 47.68 GiB, read by row through a mapping and never resident.  Compressing it further is worth doing and is
**not decided**: the candidates below were measured on a sample from the released file (48 rows, at 73% of the download)
and are recorded with that caveat, because the choice is a quality decision that a 48-row sample cannot make.

So this module is the seam, not a quantiser:

- `plan()` says what a scheme would do to this pack - source, target name, bytes per row, the size of the result, and the
  environment entry the engine needs to be pointed at the new file - and works for every scheme, including the one that
  changes nothing;
- `apply()` runs a scheme.  Only `keep` is implemented; every other scheme raises `NotImplementedError`, which the
  installer treats as "the stage exists and the scheme is not chosen yet" rather than as a failure.

Two facts a future implementation has to carry, both found in the engine rather than guessed:

- the table is named to the loader by `STRATA_PLE_FILE` (`src/models/qwen4exp.cpp`), so a different file needs that entry
  in the run config's `env`;
- the engine's memory plan reads `ple-fp8.gguf` by name when it sizes the start (`port/engine/vk-engine-load.cpp`), so a
  renamed table also has to be taught to that size calculation or the plan will report the old file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: The table as the pack ships it: 320,001,536 rows of 160 int8 values (one signed byte each), 47.68 GiB.
SOURCE_NAME = "ple-fp8.gguf"
SOURCE_BYTES_PER_ROW = 160
SOURCE_ROWS = 320_001_536


@dataclass(frozen=True)
class Scheme:
    """One candidate for the table's storage, with what was measured about it and whether it is decided."""

    id: str
    label: str
    bytes_per_row: int | None          # None: the same as the source
    status: str                        # "keep" | "open" | "decided"
    note: str
    measured: dict = field(default_factory=dict)     # relative L2 and 1-cos, from the 48-row sample

    def size_of(self, rows: int) -> int | None:
        per_row = SOURCE_BYTES_PER_ROW if self.bytes_per_row is None else self.bytes_per_row
        return rows * per_row


SCHEMES = {
    "keep": Scheme("keep", "the int8 table as it is", None, "keep",
                   "nothing is changed: the file the pack ships is the file the engine reads"),
    "q8_0": Scheme("q8_0", "Q8_0", 170, "open",
                   "larger than the file as it is (170 bytes a row), so not a candidate for saving space",
                   {"relative_l2": 0.0034, "one_minus_cos": 0.00001}),
    "q4_0": Scheme("q4_0", "Q4_0", 90, "open",
                   "about half the file; the error the sample showed is the same as a per-32 GSQ-4",
                   {"relative_l2": 0.0621, "one_minus_cos": 0.00191}),
    "gsq4": Scheme("gsq4", "GSQ-4, one scale per 32 values", 90, "open",
                   "the same size and the same sample error as Q4_0: the two differ only in code handling",
                   {"relative_l2": 0.0621, "one_minus_cos": 0.00191}),
    "affine4": Scheme("affine4", "affine 4-bit, one scale and one zero per row", 84, "open",
                      "slightly better than Q4_0 at fewer bytes on the sample",
                      {"relative_l2": 0.0534, "one_minus_cos": 0.00142}),
    "kmeans4": Scheme("kmeans4", "4-bit k-means, 16 centroids per row", 112, "open",
                      "the accurate end of the 4-bit options, not the cheap end",
                      {"relative_l2": 0.0341, "one_minus_cos": 0.00059}),
}

DEFAULT = "keep"


def scheme(name: str | None) -> Scheme:
    """The scheme a name means; the default when nothing was asked; a clear error when the name is not one."""
    key = (name or DEFAULT).strip().lower()
    if key not in SCHEMES:
        raise KeyError(f"there is no Engram quantisation scheme called {name!r}; "
                       f"the candidates are: {', '.join(SCHEMES)}")
    return SCHEMES[key]


@dataclass(frozen=True)
class Plan:
    """What a scheme would do to one pack, before anything is written."""

    source: Path
    target: Path
    scheme: Scheme
    rows: int
    bytes_per_row: int
    file_bytes: int | None            # what the result would weigh; None when the scheme cannot say yet
    env: dict                          # what the run config must carry for the engine to read the result
    notes: list

    @property
    def changes_anything(self) -> bool:
        return self.target != self.source


def plan(pack_dir: str | Path, name: str | None = None, rows: int = SOURCE_ROWS) -> Plan:
    """What the named scheme would do to this pack.  Reads nothing and writes nothing.

    A scheme that keeps the file points source and target at the same path and needs no environment entry, which is what
    makes "nothing was chosen" behave exactly like a pack that was never touched."""
    s = scheme(name)
    root = Path(pack_dir)
    source = root / SOURCE_NAME
    if s.id == DEFAULT:
        return Plan(source=source, target=source, scheme=s, rows=rows, bytes_per_row=SOURCE_BYTES_PER_ROW,
                    file_bytes=s.size_of(rows), env={},
                    notes=["the pack's own int8 table is used as it is"])
    target = root / f"ple-{s.id}.gguf"
    notes = [
        "the engine is pointed at it with STRATA_PLE_FILE in the run config's \"env\"",
        "the engine sizes a start from ple-fp8.gguf by name today: a renamed table has to be taught to that calculation",
    ]
    if s.status != "decided":
        notes.append("this scheme is not implemented yet: apply() refuses it, and the pack stays as it is")
    return Plan(source=source, target=target, scheme=s, rows=rows, bytes_per_row=s.bytes_per_row or SOURCE_BYTES_PER_ROW,
                file_bytes=s.size_of(rows), env={"STRATA_PLE_FILE": f"./{target.name}"}, notes=notes)


def apply(p: Plan) -> Plan:
    """Run the scheme.  Only "keep" is implemented: the others are candidates whose choice is not made yet.

    Raising rather than pretending is deliberate.  The installer catches `NotImplementedError` and reports the stage as
    open, so the pipeline is complete and honest at the same time: nothing silently quantises a table nobody chose, and
    nothing has to be restructured when a scheme is decided."""
    if p.scheme.id == DEFAULT:
        return p
    raise NotImplementedError(
        f"the Engram quantisation scheme {p.scheme.id!r} ({p.scheme.label}) is not implemented: it is a candidate, not a "
        "decision.  The table the pack ships is used as it is.")
