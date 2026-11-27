"""Custom builds (0.2.23): a person's own .gguf files of a model Strata's engines run, used in place under a name of their own.

Control side: the neutral names, the GGUF layout match with a file that is not a published build, the maker's own name as the
suggested label, the own-path fit, removal that keeps the weights, and the official download that stays strict.
Small synthetic files only; no model, no download, no GPU.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import setup                                                    # noqa: E402
from strata_control import custom_models, models                 # noqa: E402
from strata_control.profiles import ProfileError             # noqa: E402
from tests.terminal.test_models import gguf_fixture          # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def folder(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True)
        return p


class Names(Base):
    def test_a_label_is_one_clean_line_and_a_slug_is_never_a_path(self):
        self.assertEqual(custom_models.clean_label("  My \x1b[31m fine-tune\n v2\x00 "), "My [31m fine-tune v2")
        self.assertEqual(custom_models.clean_label("x" * 500), "x" * custom_models.MAX_LABEL)
        self.assertEqual(custom_models.clean_label("", "Custom build"), "Custom build")
        for hostile in ("../../etc/passwd", "..\\..\\Windows\\x", "C:\\evil", "a/b/../c", "/abs/path", "...", "\u202eevil", "con.json"):
            s = custom_models.slug(hostile)
            self.assertRegex(s, r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
            self.assertLessEqual(len(s), custom_models.MAX_SLUG)
        self.assertEqual(custom_models.slug("../../etc/passwd"), "etc-passwd")
        self.assertEqual(custom_models.slug("???"), "build")
        self.assertEqual(custom_models.slug("Ünïcode Mërge 7B"), "unicode-merge-7b")

    def test_the_run_tag_of_each_family_is_its_own_and_never_an_official_one(self):
        expect = {"qwen": "qwen-custom-mine", "swift": "swift-custom-mine", "coder": "coder-custom-mine",
                  "unsloth": "unsloth-custom-mine", "gyro": "gyro-custom-mine", "qwen36": "qwen36-custom-mine",
                  "ornith": "ornith-1.5-35b-a3b-custom-mine"}
        for family, tag in expect.items():
            self.assertEqual(custom_models.run_tag(family, "Mine"), tag)
            self.assertTrue(models.MODEL_ID.fullmatch(tag))
        with self.assertRaises(ValueError):
            custom_models.run_tag("llama", "x")                 # a family Strata's engines do not run

    def test_the_same_files_keep_their_tag_and_other_files_with_the_same_name_get_another(self):
        used = {"qwen36-custom-mine": str(self.root / "a.gguf")}
        self.assertEqual(custom_models.run_tag("qwen36", "Mine", self.root / "a.gguf", used), "qwen36-custom-mine")
        self.assertEqual(custom_models.run_tag("qwen36", "Mine", self.root / "b.gguf", used), "qwen36-custom-mine-2")

    def test_run_fields_and_summary(self):
        f = custom_models.run_fields("My merge", "qwen36", "UD-Q4_K_XL", 123, "/x/y.gguf")
        self.assertEqual(f, {"custom": True, "display_name": "My merge", "base_family": "qwen36", "base_model": "UD-Q4_K_XL", "custom_bytes": 123, "custom_source": "/x/y.gguf"})
        s = custom_models.summary_fields(f)
        self.assertEqual((s["display_family"], s["badge"]), ("Qwen3.6-35B-A3B", "Custom"))
        self.assertEqual(custom_models.summary_fields({"model_name": "x"}), {})              # an official config has no custom row fields
        self.assertEqual(custom_models.model_alias("ornith", "My Tune"), "ornith-1.5-35b-a3b-custom-my-tune")

    def test_every_supported_family_has_a_display_family(self):
        titles = {custom_models.DISPLAY_FAMILY[f] for f in custom_models.SUPPORTED_FAMILIES}
        self.assertEqual(titles, {"Qwen3.8-Flash-Next", "Qwen3.6-35B-A3B", "Ornith-1.5-35B-A3B"})


class GgufImportPlan(Base):
    def fake_root(self, table: dict) -> Path:
        root = self.root / "fake-root"
        (root / "data").mkdir(parents=True)
        (root / "data" / "gguf_fingerprints.json").write_text(json.dumps(table))
        return root

    def test_a_fine_tuned_qwen38_gguf_with_another_name_size_and_hash_is_accepted_as_the_same_layout(self):
        f = self.folder("q38")
        (f / "My-Flash-Next-Tune-IQ3_S.gguf").write_bytes(gguf_fixture(expert_type=42))
        table = {"qwen/IQ3_S": {"arch": "qwen4exp", "tensors": 1, "bytes": 999_999, "expert_bytes": {"Q2_0": 1, "IQ3_S": 1},
                                "files": ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"], "repo": "x"}}
        with mock.patch.object(models, "ROOT", self.fake_root(table)), mock.patch.object(models, "SUPPORTED_SELECTIONS", frozenset(table)):
            plan = models.plan_import(str(f / "My-Flash-Next-Tune-IQ3_S.gguf"))
        cand = plan["candidate_selection"]
        self.assertEqual(plan["issues"], [])
        self.assertEqual(cand["basis"], "same_layout_other_build")
        self.assertTrue(cand["custom"])
        self.assertEqual((cand["base_family"], cand["base_model"], cand["family_title"]), ("qwen", "IQ3_S", "Qwen3.8-Flash-Next"))
        self.assertEqual(cand["suggested_label"], "My-Flash-Next-Tune-IQ3_S")
        self.assertEqual(plan["file_kind"], "gguf")

    def test_the_makers_general_name_is_the_suggested_label_when_the_file_has_one(self):
        from strata_control import gguf_preflight

        def s(v):
            raw = v.encode()
            return struct.pack("<Q", len(raw)) + raw
        pairs = [("general.architecture", 8, s("qwen4exp")), ("general.name", 8, s("Ink Remix 7")),
                 ("tokenizer.ggml.model", 8, s("gpt2")), ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 1) + s("a")),
                 ("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 1) + s("a b"))]
        head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(pairs)) + b"".join(s(k) + struct.pack("<I", t) + v for k, t, v in pairs)
        head += s("blk.0.ffn_gate_exps.weight") + struct.pack("<IQIQ", 1, 32, 8, 0)
        f = self.folder("named")
        path = f / "file-name-is-different.gguf"
        path.write_bytes(head + b"\0" * (-len(head) % 32) + b"\0" * 34)
        self.assertEqual(gguf_preflight.general_name(path), "Ink Remix 7")
        self.assertIsNone(gguf_preflight.general_name(f / "missing.gguf"))
        payload = models.plan_import(str(path))["metadata"]["payload_bytes_required"]
        table = {"qwen/IQ3_S": {"arch": "qwen4exp", "tensors": 1, "bytes": payload, "expert_bytes": {"Q8_0": 1},
                                "files": ["x.gguf"], "repo": "x"}}
        with mock.patch.object(models, "ROOT", self.fake_root(table)), mock.patch.object(models, "SUPPORTED_SELECTIONS", frozenset(table)):
            plan = models.plan_import(str(path))
        self.assertEqual(plan["candidate_selection"]["suggested_label"], "Ink Remix 7")

    def test_an_exact_header_match_is_the_official_build_not_custom(self):
        f = self.folder("q38exact")
        p = f / "renamed-copy.gguf"
        p.write_bytes(gguf_fixture(expert_type=42))
        first = models.plan_import(str(p))
        table = {"qwen/IQ3_S": {"arch": "qwen4exp", "tensors": 1, "bytes": first["metadata"]["payload_bytes_required"], "expert_bytes": {"Q2_0": 1},
                                "files": ["x.gguf"], "repo": "x"}}
        with mock.patch.object(models, "ROOT", self.fake_root(table)), mock.patch.object(models, "SUPPORTED_SELECTIONS", frozenset(table)):
            plan = models.plan_import(str(p))
        self.assertEqual(plan["candidate_selection"]["basis"], "header_size_and_tensors")
        self.assertFalse(plan["candidate_selection"]["custom"])

    def test_a_qwen36_gguf_of_another_name_is_custom_and_the_engine_loader_rules_still_apply(self):
        f = self.folder("q36")
        (f / "Qwen3.6-community-merge-Q4_K.gguf").write_bytes(gguf_fixture(architecture="qwen35moe", expert_type=12))
        plan = models.plan_import(str(f / "Qwen3.6-community-merge-Q4_K.gguf"))
        cand = plan["candidate_selection"]
        self.assertEqual((plan["issues"], cand["family"], cand["custom"]), ([], "qwen36", True), plan["issues"])
        self.assertEqual(cand["family_title"], "Qwen3.6-35B-A3B")
        (f / "bad.gguf").write_bytes(gguf_fixture(architecture="qwen35moe", expert_type=39))          # a type the loader does not take
        self.assertIn("unsupported_or_missing_architecture", models.plan_import(str(f / "bad.gguf"))["issues"])

    def test_the_import_only_hy4_pack_is_named_from_its_header(self):
        f = self.folder("hy4")
        path = f / "renamed-hy4-pack.gguf"
        path.write_bytes(gguf_fixture(architecture="hy_v4_flash"))
        plan = models.plan_import(str(path))
        cand = plan["candidate_selection"]
        self.assertEqual(plan["issues"], [], plan["issues"])
        self.assertEqual((cand["family"], cand["model"], cand["basis"], cand["custom"]),
                         ("hy4", "Compact", "architecture_of_an_imported_family", False))
        self.assertTrue(any("does not download this family" in m for m in plan["messages"]), plan["messages"])

    def test_an_unrelated_architecture_keeps_the_custom_model_api_guidance(self):
        f = self.folder("llama")
        (f / "x.gguf").write_bytes(gguf_fixture(architecture="llama"))
        plan = models.plan_import(str(f / "x.gguf"))
        self.assertEqual(plan["file_set_status"], "invalid_or_incomplete")
        self.assertIn("llama-server", models.UNSUPPORTED_MODEL_TYPE)
        self.assertIn(models.UNSUPPORTED_MODEL_TYPE, plan["messages"])


class StrataFilesAreNotImported(Base):
    """`.strata` packs are the engine's own format: the GGUF importer refuses them in words rather than misreading one."""

    def write(self, name: str, data: bytes) -> Path:
        p = self.root / name
        p.write_bytes(data)
        return p

    def test_a_strata_file_is_refused_in_words(self):
        p = self.write("my.strata", b"STRATA\x00\x01" + b"\0" * 5000)
        with self.assertRaises(ProfileError) as raised:
            models.plan_import(str(p))
        self.assertIn("needs a GGUF file", str(raised.exception))

    def test_a_damaged_gguf_is_refused_and_named(self):
        p = self.write("cut.gguf", b"GGUF" + b"\0" * 5000)
        plan = models.plan_import(str(p))
        self.assertEqual(plan["file_set_status"], "invalid_or_incomplete")
        self.assertTrue(any("cut.gguf" in issue for issue in plan["issues"]), plan["issues"])


class OwnPathFit(Base):
    def cfg(self, family, base, nbytes):
        return {"custom": True, "base_family": family, "base_model": base, "custom_bytes": nbytes, "display_name": "x"}

    def test_qwen36_context_rule_uses_the_closest_size_and_a_family_without_a_size_rule_says_none(self):
        got = custom_models.own_path_inputs(self.cfg("qwen36", "UD-Q4_K_XL", 20 * 10 ** 9))
        self.assertIsNotNone(got)
        self.assertIsNone(custom_models.own_path_inputs(self.cfg("qwen", "IQ3_S", 1)))


def namespace(**over):
    base = dict(drafter=None, model="UD-Q4_K_XL", family="qwen36", context=32768, vision="no", yes=True, no_start=True, gguf_dir=None, models_dir=None,
                data_dir=None, backend=None, gpu=None, host=None, api_key=None, browser=None, port=8080, custom_label=None, custom_source=None,
                rope_scaling=None, rope_scale=None, kv=None, resident_budget_gib=None, low_ram=None, feed=None, parallel=None)
    base.update(over)
    return argparse.Namespace(**base)


class SetupRunConfig(Base):
    def setUp(self):
        super().setUp()
        self.install_dir = self.folder("install")
        self.data_dir = self.folder("data")
        self.addCleanup(setup._CUSTOM.clear)
        setup._CUSTOM.clear()

    def test_run_tag_without_a_label_is_the_official_tag(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        self.assertEqual(setup.run_tag(namespace(gguf_dir=str(self.root)), "qwen36", fam, "UD-Q4_K_XL"), "qwen36-UD-Q4_K_XL")
        self.assertEqual(setup.run_tag(namespace(), "qwen36", fam, "UD-Q4_K_XL"), "qwen36-UD-Q4_K_XL")
        self.assertFalse(setup._CUSTOM)

    def test_a_label_without_the_persons_own_files_does_nothing(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        self.assertFalse(setup.custom_requested(namespace(custom_label="x")))                 # a download never becomes a custom build
        self.assertEqual(setup.run_tag(namespace(custom_label="x"), "qwen36", fam, "UD-Q4_K_XL"), "qwen36-UD-Q4_K_XL")

    def test_a_label_with_the_persons_own_files_gets_a_custom_tag(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        mine = self.root / "Mine-Q4_K_XL.gguf"
        mine.write_bytes(b"x")
        a = namespace(gguf_dir=str(mine), custom_label="Mine", custom_source=str(mine))
        self.assertTrue(setup.custom_requested(a))
        self.assertEqual(setup.run_tag(a, "qwen36", fam, "UD-Q4_K_XL"), "qwen36-custom-mine")
        self.assertEqual(setup._CUSTOM["label"], "Mine")
        self.assertEqual(setup._CUSTOM["family"], "qwen36")


class SetupQwenFamilies(Base):
    """The two GGUF families: a custom Qwen3.6 file through the real own-path steps, and the Flash-Next file set pick."""

    def setUp(self):
        super().setUp()
        setup._CUSTOM.clear()
        self.addCleanup(setup._CUSTOM.clear)

    def test_a_custom_qwen36_gguf_is_used_in_place_its_picture_reader_is_the_official_one_checked_strictly(self):
        install, data, models_dir = self.folder("install"), self.folder("data"), self.folder("models")
        mine = self.folder("mine")
        weights = mine / "Qwen3.6-community-merge.gguf"
        weights.write_bytes(gguf_fixture(architecture="qwen35moe", expert_type=12))
        official_gguf = mine / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"                     # the official file sits in the same folder: it must not be picked
        official_gguf.write_bytes(b"official")
        official_dir = models_dir / "qwen36-UD-Q4_K_XL"
        official_dir.mkdir()
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        reader = official_dir / fam["mmproj"]
        reader.write_bytes(b"reader")
        setup.mark(reader)
        eng = self.folder("eng")
        (eng / setup.VEXE).write_text("x")
        checks, cfg, calls = [], {}, []
        gpu = {"index": 0, "count": 1, "vram_gb": 16.0, "name": "Test GPU"}
        a = namespace(family="qwen36", model="UD-Q4_K_XL", gguf_dir=str(mine), custom_label="Community merge", custom_source=str(weights), vision="cpu",
                      models_dir=str(models_dir), context=32768, vram_reserve_mib=None, check=False, low_ram="auto")
        patches = [mock.patch.object(setup, "_OWN_FIRST", weights), mock.patch.object(setup, "home", return_value=install),
                   mock.patch.object(setup, "verify_sha256", lambda p, size, sha, own=False, **k: checks.append((p.name, own))),
                   mock.patch.object(setup, "download", lambda *x, **k: calls.append(("download",) + x)),
                   mock.patch.object(setup, "free_gb", lambda p: 500.0), mock.patch.object(setup, "pip_install", lambda *x, **k: None),
                   mock.patch.object(setup, "engine_step", lambda a_, g, hip, amd, ck, v: (self.root / "llama", eng, {"version": "0.2.12"}, [], g, v)),
                   mock.patch.object(setup, "engine_families", lambda e: {"qwen4exp", "qwen35moe"}),
                   mock.patch.object(setup, "run", lambda cmd, **k: calls.append(("run", cmd))),
                   mock.patch.object(setup, "data_folder_for", return_value=(data, None)),
                   mock.patch.object(setup, "write_run_script", lambda tag, p, port, o=True: self.root / f"run-{tag}.sh"),
                   mock.patch.object(setup, "recommend_pool_workers", lambda args: args)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with mock.patch.object(setup, "say"), mock.patch.object(setup, "ok"), mock.patch.object(setup, "warn"), mock.patch.object(setup, "step"):
            rc = setup.install_own_path(a, "qwen36", gpu, False, [], 13, [], 8080, 64.0, None)
        self.assertEqual(rc, 0)
        written = install / "strata-qwen36-custom-community-merge.json"
        self.assertEqual([p.name for p in install.glob("strata-*.json")], [written.name])
        cfg.update(json.loads(written.read_text(encoding="utf-8")), path=written.name)
        self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1], str(weights))      # the picked file, not the official one beside it
        self.assertIn((weights.name, True), checks)                                          # the person's file: own, no pinned size or hash
        self.assertIn((reader.name, False), checks)                                          # the official picture reader: strict
        self.assertEqual(cfg["args"][cfg["args"].index("--mmproj") + 1] if "--mmproj" in cfg["args"] else str(reader), str(reader))
        self.assertFalse([c for c in calls if c and c[0] == "download"])                      # the reader of an official install is reused, nothing is downloaded
        self.assertEqual(cfg["path"], "strata-qwen36-custom-community-merge.json")
        self.assertEqual((cfg["custom"], cfg["base_family"], cfg["base_model"]), (True, "qwen36", "UD-Q4_K_XL"))
        self.assertEqual(cfg["model_name"], "qwen3.6-35b-a3b-custom-community-merge")
        self.assertEqual(cfg["custom_bytes"], weights.stat().st_size)
        self.assertEqual(official_gguf.read_bytes(), b"official")                              # the official file beside it is untouched

    def test_the_flash_next_file_set_is_the_one_the_person_picked_not_the_published_name_beside_it(self):
        fam = setup.FAMILIES["qwen"]
        folder = self.folder("sets")
        official = folder / setup.model_file(fam, "IQ3_S", 1)
        official.write_bytes(b"o")
        mine = folder / "Tuned-Flash-Next-00001-of-00002.gguf"
        mine.write_bytes(b"1")
        (folder / "Tuned-Flash-Next-00002-of-00002.gguf").write_bytes(b"22")
        with mock.patch.object(setup, "_OWN_FIRST", None):
            self.assertEqual(setup.gguf_dir_shards(folder, fam, "IQ3_S")[0], official)
        with mock.patch.object(setup, "_OWN_FIRST", mine):
            got = setup.gguf_dir_shards(folder, fam, "IQ3_S")
        self.assertEqual([p.name for p in got], ["Tuned-Flash-Next-00001-of-00002.gguf", "Tuned-Flash-Next-00002-of-00002.gguf"])
        self.assertEqual(setup.custom_weight_bytes(mine), 3)
        with mock.patch.object(setup, "_CUSTOM", {"label": "x"}):                             # a custom file is not refused by what its name says
            self.assertIsNone(setup.gguf_dir_problem(folder, mine, fam, "IQ3_S"))
        odd = folder / "Tuned-Q4_K_M-00001-of-00002.gguf"
        odd.write_bytes(b"1")
        self.assertIsNotNone(setup.gguf_dir_problem(folder, odd, fam, "IQ3_S"))               # an official-path import still is

    def test_label_flags_are_checked_before_anything_is_done(self):
        import io
        for argv, text in ((["--custom-label", "x"], "--gguf-dir"), (["--custom-label", "x", "--gguf-dir", "/tmp/a.gguf"], "--family and --model")):
            err = io.StringIO()
            with mock.patch.object(sys, "argv", ["setup.py", *argv]), mock.patch("sys.stderr", err), mock.patch.object(setup, "move_over_old_layout"):
                with self.assertRaises(SystemExit) as raised:
                    setup.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(text, err.getvalue())


class OfficialStaysStrict(Base):
    """Only the person's own file skips the pinned size and hash; a download with a wrong hash or size is still refused (and kept as .bad, never trusted)."""

    def make(self, content: bytes) -> Path:
        p = self.root / "official.gguf"
        p.write_bytes(content)
        return p

    def test_a_download_with_the_wrong_hash_is_rejected(self):
        p = self.make(b"x" * 4096)
        with mock.patch.object(setup, "fail", side_effect=SystemExit(1)) as failed, mock.patch.object(setup, "say"):
            with self.assertRaises(SystemExit):
                setup.verify_sha256(p, 4096, "0" * 64, own=False)
        self.assertIn("does not match the pinned SHA-256", failed.call_args[0][0])
        self.assertFalse(p.exists())                                                     # kept as .bad, not trusted as the official file
        self.assertTrue(p.with_name(p.name + ".bad").exists())

    def test_a_download_with_the_wrong_size_is_rejected(self):
        p = self.make(b"x" * 100)
        with mock.patch.object(setup, "fail", side_effect=SystemExit(1)), mock.patch.object(setup, "say"):
            with self.assertRaises(SystemExit):
                setup.verify_sha256(p, 4096, "0" * 64, own=False)

    def test_the_persons_own_whole_gguf_with_another_size_and_hash_is_used_and_kept(self):
        p = self.root / "mine.gguf"
        p.write_bytes(gguf_fixture())
        before = p.read_bytes()
        with mock.patch.object(setup, "warn") as warned, mock.patch.object(setup, "fail", side_effect=SystemExit(1)):
            setup.verify_sha256(p, 123456789, "0" * 64, own=True)
        self.assertEqual(p.read_bytes(), before)
        self.assertTrue(warned.called)


class RemoveKeepsTheWeights(Base):
    def build(self, family: str, weights_name: str, in_strata_folder: bool, tag: str | None = None):
        from tools import strata_mcp
        install = self.folder("strata-" + (tag or family))
        weights_dir = install / "models" if in_strata_folder else self.folder("userfiles-" + (tag or family))
        weights_dir.mkdir(exist_ok=True)
        weights = weights_dir / weights_name
        weights.write_bytes(b"W" * 1000)
        shard = weights_dir / "mmproj-x.gguf"
        shard.write_bytes(b"P")
        cfg = {"exe": "component:strata-engine", "tokenizer": str(install / "tok"), "args": ["--native", str(weights), "--max-context", "32768"],
               "model_name": "alias", **custom_models.run_fields("Mine", family, "X", 1000, str(weights))}
        config = install / f"strata-{custom_models.tag_prefix(family)}custom-mine.json"
        config.write_text(json.dumps(cfg), encoding="utf-8")
        (install / config.name.replace(".json", ".log")).write_text("log")
        return strata_mcp, install, config, weights

    def plan_for(self, mcp, install, config):
        return mcp.Strata(install).removal_plan(config)

    def test_removal_plan_lists_only_the_run_config_and_strata_own_leftovers_for_every_family(self):
        for family, name in (("qwen", "Tune-IQ3_S-00001-of-00002.gguf"), ("qwen36", "Q36-tune.gguf"), ("ornith", "ornith-tune.gguf")):
            for inside in (False, True):
                mcp, install, config, weights = self.build(family, name, inside, tag=f"{family}-{int(inside)}-{name[-8:]}")
                plan = self.plan_for(mcp, install, config)
                removed = {Path(i["path"]).name for i in plan["remove"]}
                self.assertEqual(removed, {config.name, config.name.replace(".json", ".log")}, (family, name, inside, plan))
                self.assertTrue(any(k["path"] == str(weights) for k in plan["keep"]), plan["keep"])

    def test_the_cli_remove_deletes_only_the_run_config(self):
        from strata_cli.launch import custom
        mcp, install, config, weights = self.build("qwen36", "t.gguf", True)
        before = weights.read_bytes()
        custom.remove_reference(config)
        self.assertFalse(config.exists())
        self.assertEqual(weights.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
