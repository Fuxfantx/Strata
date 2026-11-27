"""The Engram quantisation stage: a protocol, with the scheme deliberately not chosen.

What is tested is that the seam is real and honest: a plan for every candidate (size, target name, the environment entry
the engine needs), the one scheme that changes nothing behaving exactly like an untouched pack, every other scheme
refusing instead of pretending, and the Edge0 installer reporting the stage without failing when it refuses.

No pack is quantised, nothing is downloaded and no model is run.

    python -m unittest tools.test_setup_engram_quant
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from strata_control import engram_quant as q  # noqa: E402


class Registry(unittest.TestCase):
    def test_the_shipped_table_is_measured_against_itself(self):
        keep = q.SCHEMES["keep"]
        self.assertEqual(keep.status, "keep")
        self.assertIsNone(keep.bytes_per_row)              # the file's own 160
        self.assertEqual(keep.size_of(q.SOURCE_ROWS), 51_200_245_760)

    def test_the_published_size_matches_the_row_arithmetic(self):
        # 320,001,536 rows of 160 bytes, plus a GGUF header, is the file the pack ships (51,200,246,112 bytes)
        self.assertEqual(q.SCHEMES["keep"].size_of(q.SOURCE_ROWS) + 352, 51_200_246_112)

    def test_every_candidate_says_what_was_measured_and_that_it_is_open(self):
        for name, s in q.SCHEMES.items():
            if name == "keep":
                continue
            self.assertEqual(s.status, "open", name)
            self.assertGreater(s.bytes_per_row, 0, name)
            self.assertIn("relative_l2", s.measured, name)  # the 48-row sample, with its caveat in the module
            self.assertIn("one_minus_cos", s.measured, name)

    def test_a_name_that_is_not_a_scheme_fails_plainly(self):
        with self.assertRaises(KeyError) as caught:
            q.scheme("magic")
        self.assertIn("keep", str(caught.exception))


class Plans(unittest.TestCase):
    def test_keep_points_at_the_file_the_pack_ships(self):
        p = q.plan("/packs/edge0", "keep")
        self.assertEqual(p.target, p.source)
        self.assertFalse(p.changes_anything)
        self.assertEqual(p.env, {})

    def test_a_candidate_names_a_new_file_and_the_environment_entry(self):
        p = q.plan("/packs/edge0", "q4_0")
        self.assertEqual(p.target.name, "ple-q4_0.gguf")
        self.assertTrue(p.changes_anything)
        self.assertEqual(p.env, {"STRATA_PLE_FILE": "./ple-q4_0.gguf"})
        self.assertEqual(p.bytes_per_row, 90)
        self.assertEqual(p.file_bytes, q.SOURCE_ROWS * 90)

    def test_the_note_says_what_the_engine_has_to_learn(self):
        p = q.plan("/packs/edge0", "affine4")
        joined = " ".join(p.notes)
        self.assertIn("STRATA_PLE_FILE", joined)
        self.assertIn("ple-fp8.gguf", joined)              # the memory plan reads that name today
        self.assertIn("not implemented", joined)

    def test_no_plan_lies_about_being_decided(self):
        for name in q.SCHEMES:
            p = q.plan("/packs/edge0", name)
            if name != "keep":
                self.assertEqual(p.scheme.status, "open", name)


class Apply(unittest.TestCase):
    def test_keep_changes_nothing(self):
        p = q.plan("/packs/edge0", "keep")
        self.assertIs(q.apply(p), p)

    def test_a_candidate_refuses_rather_than_pretending(self):
        p = q.plan("/packs/edge0", "kmeans4")
        with self.assertRaises(NotImplementedError) as caught:
            q.apply(p)
        self.assertIn("not implemented", str(caught.exception))
        self.assertIn("kmeans4", str(caught.exception))


class Installer(Base := unittest.TestCase):
    """The Edge0 installer reports the stage and finishes either way."""

    def setUp(self):
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ["STRATA_REGION"] = "intl"
        setup._sources.clear()
        self.addCleanup(setup._sources.clear)
        self.tmp = Path(tempfile.mkdtemp(prefix="engram-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        (self.tmp / "models").mkdir()
        self.said = []
        for name in ("warn", "step"):
            p = mock.patch.object(setup, name, lambda *a, **k: None)
            p.start()
            self.addCleanup(p.stop)
        for name in ("say", "ok"):
            p = mock.patch.object(setup, name, lambda *a, _n=name, **k: self.said.append(" ".join(str(x) for x in a)))
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(setup, "ask", lambda *a, **k: "1")
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "free_gb", lambda *a, **k: 10_000.0)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "download", self._download)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "done", lambda dst: Path(dst).exists())
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "verify_sha256", lambda *a, **k: None)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "engine_step", lambda *a, **k: (None, Path("engine"), {"version": "t"}, [],
                                                                     {"index": 0, "vram_gb": 0.0, "count": 1}, "none"))
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "engine_families", lambda eng: {"qwen4exp"})
        p.start()
        self.addCleanup(p.stop)
        for name in ("pip_install", "run", "start"):
            p = mock.patch.object(setup, name, lambda *a, **k: 0)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(setup, "data_folder_for", lambda a: (self.tmp / "data", []))
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "gguf_py_env", lambda: {})
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "home", lambda: self.tmp)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "write_run_script", lambda *a, **k: Path("S.bat"))
        p.start()
        self.addCleanup(p.stop)
        self.configs = []
        p = mock.patch.object(setup, "write_setup_config",
                              lambda path, cfg, adopted, explicit: self.configs.append((Path(path), cfg)))
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "pack_file_url", lambda size, name: f"https://host/{name}")
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "pack_file_meta", lambda size, name: (1, "h"))
        p.start()
        self.addCleanup(p.stop)

    def _download(self, url, dst, what=None):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"x")

    def install(self, **kw):
        base = dict(model="145B-Q3_K", context=32768, gguf_dir=None, models_dir=str(self.tmp / "models"), host=None,
                    api_key=None, browser=None, gpu=None, no_start=True, vram_reserve_mib=None, engram_quant=None)
        base.update(kw)
        setup.install_pack_family(SimpleNamespace(**base), "edge0", {"index": 0, "vram_gb": 0.0, "count": 1},
                                  False, False, 0, False, 8080, 64, None)

    def test_the_stage_says_the_table_is_used_as_it_is(self):
        self.install()
        self.assertTrue(any("Engram: ple-fp8.gguf as it is" in line for line in self.said), self.said[-6:])

    def test_an_undecided_scheme_is_reported_and_the_install_finishes(self):
        self.install(engram_quant="q4_0")
        joined = " ".join(self.said)
        self.assertIn("Engram:", joined)
        self.assertIn("not implemented", joined)
        self.assertEqual(len(self.configs), 1, "the config is still written")

    def test_the_config_is_untouched_when_nothing_was_quantised(self):
        self.install()
        env = self.configs[0][1]["env"]
        self.assertNotIn("STRATA_PLE_FILE", env)


if __name__ == "__main__":
    unittest.main()
