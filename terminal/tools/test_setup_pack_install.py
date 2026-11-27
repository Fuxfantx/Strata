"""Installing an Edge0 pack: the files land in one folder, each is verified, and the run config names the folder,
the skeleton, the picture reader and the D/U table of that build.

Everything outside this file is stubbed: no download, no engine, no model.  What the tests hold down is the contract -
which files are asked for, that a file whose host published no hash is refused, and the shape of the config the engine
is started with.

    python -m unittest tools.test_setup_pack_install
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

SIZE = setup.PACK_FAMILIES["edge0"]["sizes"]["145B-Q3_K"]
SIZE_120 = setup.PACK_FAMILIES["edge0"]["sizes"]["120B-Q3_K"]
OPTIONAL = set(SIZE.get("optional") or {})
WANTED = {n: v for n, v in SIZE["files"].items() if n not in OPTIONAL}


class Base(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("STRATA_SOURCE", None)
        os.environ["STRATA_REGION"] = "intl"
        setup._sources.clear()
        self.addCleanup(setup._sources.clear)

        self.tmp = Path(tempfile.mkdtemp(prefix="edge0-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.folder = self.tmp / "models" / "edge0-145b-q3_k"
        self.folder.mkdir(parents=True)

        for name in ("say", "ok", "warn", "step"):
            p = mock.patch.object(setup, name, lambda *a, **k: None)
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
        self.verified = []
        p = mock.patch.object(setup, "verify_sha256", lambda dst, *pin, **k: self.verified.append((Path(dst).name, pin)))
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "engine_step", lambda *a, **k: (None, Path("engine"), {"version": "test"},
                                                                     [], {"index": 0, "vram_gb": 0.0, "count": 1},
                                                                     "none"))
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
        p = mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, browser: Path("START-HERE.bat"))
        p.start()
        self.addCleanup(p.stop)
        self.configs = []
        p = mock.patch.object(setup, "write_setup_config",
                              lambda path, cfg, adopted, explicit: self.configs.append((Path(path), cfg)))
        p.start()
        self.addCleanup(p.stop)
        self.asked = []
        p = mock.patch.object(setup, "pack_file_url", lambda size, name: self.asked.append(name) or f"https://host/{name}")
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "pack_file_meta", lambda size, name: (1, "from-host"))
        p.start()
        self.addCleanup(p.stop)

    def _download(self, url, dst, what=None):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"x")

    def write_own(self, names):
        for name in names:
            path = self.folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    def args(self, **kw):
        base = dict(model="145B-Q3_K", context=32768, gguf_dir=None, models_dir=str(self.tmp / "models"),
                    host=None, api_key=None, browser=None, gpu=None, no_start=True, vram_reserve_mib=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def install(self, **kw):
        setup.install_pack_family(self.args(**kw), "edge0", {"index": 0, "vram_gb": 0.0, "count": 1}, False, False, 0,
                                  False, 8080, 64, None)


class Downloads(Base):
    def test_the_files_the_run_needs_are_asked_for(self):
        self.install()
        self.assertEqual(sorted(self.asked), sorted(WANTED))

    def test_the_files_the_run_does_not_need_are_not_downloaded(self):
        self.install()
        for name in OPTIONAL:
            self.assertNotIn(name, self.asked)

    def test_every_file_is_verified_against_this_builds_pin(self):
        self.install()
        by_name = dict(self.verified)
        self.assertEqual(len(by_name), len(WANTED))
        for name, entry in WANTED.items():
            size, sha = entry
            base = name.rsplit("/", 1)[-1]
            if sha:
                self.assertEqual(by_name[base], (size, sha), name)
            else:                                          # no hash in the table: the host's published one is used
                self.assertEqual(by_name[base], (1, "from-host"), name)

    def test_a_file_with_no_hash_anywhere_is_refused(self):
        p = mock.patch.object(setup, "pack_file_meta", lambda size, name: None)
        p.start()
        self.addCleanup(p.stop)
        with mock.patch.object(setup, "fail", side_effect=SystemExit(1)) as failed:
            with self.assertRaises(SystemExit):
                self.install()
        self.assertTrue(failed.called)

    def test_a_folder_of_the_persons_own_is_not_downloaded(self):
        self.write_own(WANTED)
        self.install(gguf_dir=str(self.folder))
        self.assertEqual(self.asked, [])

    def test_a_missing_file_in_the_persons_folder_stops(self):
        (self.folder / "experts.bin").write_bytes(b"x")
        with mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as failed:
            with self.assertRaises(SystemExit):
                self.install(gguf_dir=str(self.folder))
        self.assertTrue(failed.called)


class Config(Base):
    def config(self, **kw):
        self.install(**kw)
        self.assertEqual(len(self.configs), 1)
        return self.configs[0][1]

    def test_the_config_names_the_folder_and_the_skeleton(self):
        cfg = self.config()
        args = cfg["args"]
        self.assertEqual(args[args.index("--pack") + 1], str(self.folder))
        self.assertEqual(args[args.index("--native") + 1], str(self.folder / "model-skeleton.gguf"))

    def test_the_config_names_the_q8_table_for_the_145b(self):
        env = self.config()["env"]
        self.assertEqual(env["STRATA_LATENT_DENSE"], "./latent_dense_q8.bin")
        self.assertEqual(env["STRATA_LATENT_DENSE_FMT"], "8")
        self.assertEqual(env["STRATA_LATENT_A"], "./experts.bin")
        self.assertEqual(env["STRATA_LATENT_LABELS"], "./labels.bin")

    def test_the_120b_config_names_the_q4_table(self):
        env = self.config(model="120B-Q3_K")["env"]
        self.assertEqual(env["STRATA_LATENT_DENSE"], "./latent_dense_q4.bin")
        self.assertEqual(env["STRATA_LATENT_DENSE_FMT"], "4")

    def test_the_memory_settings_are_there(self):
        args = self.config()["args"]
        self.assertEqual(args[args.index("--expert-cache") + 1], "auto")
        self.assertEqual(args[args.index("--ple-io") + 1], "direct")
        self.assertEqual(args[args.index("--max-context") + 1], "32768")

    def test_the_pictures_are_wired_when_the_reader_is_in_the_folder(self):
        (self.folder / "mmproj-BF16.gguf").write_bytes(b"x")
        cfg = self.config()
        self.assertIn("--vision", cfg["args"])
        self.assertEqual(cfg["vision"]["mmproj"], str(self.folder / "mmproj-BF16.gguf"))
        self.assertEqual(cfg["vision"]["model"], str(self.folder / "model-skeleton.gguf"))

    def test_the_reader_the_folder_carries_is_always_wired(self):
        """The published folder carries its picture reader, so a normal install turns pictures on."""
        cfg = self.config()
        self.assertIn("--vision", cfg["args"])
        self.assertEqual(cfg["vision"]["gpu"], False)
        self.assertEqual(cfg["vision"]["max_tokens"], 1024)

    def test_a_pack_the_engine_cannot_read_stops_before_any_download(self):
        with mock.patch.object(setup, "engine_families", lambda eng: {"hy_v4_flash"}):
            with mock.patch.object(setup, "fail", side_effect=SystemExit(3)) as failed:
                with self.assertRaises(SystemExit):
                    self.install()
        self.assertTrue(failed.called)
        self.assertEqual(self.asked, [])


class The120BBuild(Base):
    """Its own files, its own Q4 table, and its own name on the config."""

    def test_it_downloads_the_q4_table_and_not_the_q8_one(self):
        self.install(model="120B-Q3_K")
        self.assertIn("latent_dense_q4.bin", self.asked)
        self.assertNotIn("latent_dense_q8.bin", self.asked)

    def test_it_downloads_its_own_experts_file(self):
        self.install(model="120B-Q3_K")
        self.assertIn("experts.bin", self.asked)
        self.assertEqual(SIZE_120["files"]["experts.bin"][0], 28626124800)


if __name__ == "__main__":
    unittest.main()
