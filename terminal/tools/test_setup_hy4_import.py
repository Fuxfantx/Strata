"""Importing the Hy4-flash-Uncensored-Compact pack: no download, one folder, one ordinary run config.

The family exists only as an import, so what the tests hold down is that there is nothing to download, that the
architecture in the file is checked, that the companion file is required, and that the config written is the same shape
every other family gets - which is what makes the terminal and the app treat it like any other model once it is there.

No model is run.

    python -m unittest tools.test_setup_hy4_import
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

FAM = setup.OWN_PATH_FAMILIES["hy4"]
TRUNK = setup.HY4_TRUNK
ENGRAM = setup.HY4_ENGRAM


class Base(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        self.tmp = Path(tempfile.mkdtemp(prefix="hy4-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.folder = self.tmp / "pack"
        self.folder.mkdir()

        for name in ("say", "ok", "warn", "step"):
            p = mock.patch.object(setup, name, lambda *a, **k: None)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(setup, "ask", lambda *a, **k: "1")
        p.start()
        self.addCleanup(p.stop)
        for name in ("pip_install", "run", "start"):
            p = mock.patch.object(setup, name, lambda *a, **k: 0)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(setup, "engine_step",
                              lambda *a, **k: (None, Path("engine"), {"version": "test"}, [], {"index": 0}, "none"))
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(setup, "engine_families", lambda eng: {"hy_v4_flash"})
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
        p = mock.patch("strata_control.gguf_preflight.inspect_metadata", lambda path: {"architecture": "hy_v4_flash"})
        p.start()
        self.addCleanup(p.stop)

    def args(self, **kw):
        base = dict(model=None, context=32768, gguf_dir=None, models_dir=str(self.tmp / "models"), host=None,
                    api_key=None, browser=None, gpu=None, no_start=True)
        base.update(kw)
        return SimpleNamespace(**base)

    def write_pack(self, trunk=TRUNK, engram=ENGRAM):
        if trunk:
            (self.folder / trunk).write_bytes(b"gguf")
        if engram:
            (self.folder / engram).write_bytes(b"gguf")

    def install(self, **kw):
        return setup.install_imported(self.args(**kw), "hy4", FAM, {"index": 0}, False, False, 0, False, 8080, 64, None)


class NoDownload(Base):
    def test_the_family_names_nothing_to_download(self):
        for key in ("hf", "sha256", "repo", "sources", "revision"):
            self.assertNotIn(key, FAM, key)

    def test_it_is_marked_as_an_import(self):
        self.assertTrue(FAM.get("import_only"))

    def test_the_branding_is_tencents(self):
        self.assertEqual(FAM["title"], "Hy4 flash")
        self.assertEqual(FAM["by"], "Tencent Hy Team")
        self.assertIn("Tencent Hy License", FAM["license"])

    def test_its_own_qualifier_is_one_line(self):
        self.assertIn("Compact repack of an uncensored derivative", FAM["about"])


class Refusals(Base):
    def test_no_folder_is_refused_with_the_flag_to_use(self):
        with mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as failed:
            with self.assertRaises(SystemExit):
                self.install()
        self.assertTrue(failed.called)
        self.assertIn("--gguf-dir", str(failed.call_args))

    def test_a_folder_without_the_pack_is_refused(self):
        with mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as failed:
            with self.assertRaises(SystemExit):
                self.install(gguf_dir=str(self.folder))
        self.assertTrue(failed.called)

    def test_a_missing_companion_is_refused(self):
        self.write_pack(engram=None)
        with mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as failed:
            with self.assertRaises(SystemExit):
                self.install(gguf_dir=str(self.folder))
        self.assertTrue(failed.called)
        self.assertIn("Engram", str(failed.call_args))

    def test_another_architecture_is_refused(self):
        self.write_pack()
        with mock.patch("strata_control.gguf_preflight.inspect_metadata",
                        lambda path: {"architecture": "qwen4exp"}):
            with mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as failed:
                with self.assertRaises(SystemExit):
                    self.install(gguf_dir=str(self.folder))
        self.assertTrue(failed.called)
        self.assertEqual(self.configs, [])

    def test_an_engine_without_the_architecture_is_refused(self):
        self.write_pack()
        with mock.patch.object(setup, "engine_families", lambda eng: {"qwen4exp"}):
            with mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as failed:
                with self.assertRaises(SystemExit):
                    self.install(gguf_dir=str(self.folder))
        self.assertTrue(failed.called)
        self.assertEqual(self.configs, [])


class Config(Base):
    def test_a_valid_pack_writes_an_ordinary_config(self):
        self.write_pack()
        self.install(gguf_dir=str(self.folder))
        self.assertEqual(len(self.configs), 1)
        cfg = self.configs[0][1]
        args = cfg["args"]
        self.assertEqual(args[args.index("--native") + 1], str(self.folder / TRUNK))
        self.assertNotIn("--pack", args, "an imported pack is a file, not a pack folder")
        self.assertEqual(args[args.index("--gpu") + 1], "cpu")
        self.assertEqual(cfg["arch"], "hy_v4_flash")
        self.assertEqual(cfg["model_name"], "hy4-flash-uncensored-compact")

    def test_the_config_carries_what_every_screen_reads(self):
        self.write_pack()
        self.install(gguf_dir=str(self.folder))
        cfg = self.configs[0][1]
        for key in ("exe", "args", "cwd", "tokenizer", "model_name", "arch", "log", "lib_dirs", "port"):
            self.assertIn(key, cfg, key)

    def test_the_context_is_the_one_chosen(self):
        self.write_pack()
        self.install(gguf_dir=str(self.folder), context=8192)
        cfg = self.configs[0][1]
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "8192")

    def test_a_renamed_folder_still_imports(self):
        self.write_pack(trunk="my-hy4-trunk.gguf", engram=ENGRAM)
        found = setup.find_imported_pack(FAM, self.folder)
        self.assertIsNotNone(found, "a renamed trunk is still the pack's trunk")
        self.assertEqual(found.name, "my-hy4-trunk.gguf")


class FamilyList(Base):
    def test_the_terminal_lists_it_with_the_other_families(self):
        self.assertIn("hy4", setup.PREVIEW_FAMILIES)
        self.assertIn("hy4", setup.OWN_PATH_FAMILIES)

    def test_it_is_not_offered_as_a_download_in_the_model_table(self):
        self.assertNotIn("hy4", setup.MODELS)
        self.assertNotIn("hy4", setup.PACK_FAMILIES)

    def test_the_model_id_it_reports_is_the_pack(self):
        self.assertEqual(FAM["file"], TRUNK)
        self.assertEqual(FAM["companion"], ENGRAM)


if __name__ == "__main__":
    unittest.main()
