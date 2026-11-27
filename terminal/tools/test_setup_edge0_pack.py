"""An Edge0 pack: what a family names, where each file comes from, and the sha256 it is checked against.

The pack is a directory of files with a different repository name on each host.  Both builds are pinned from their own
repository's file listing, so what is tested is: both hosts are named for both builds, each build names the D/U table it
reads and leaves the other one out, every file carries a size (and a hash where the host publishes one), and the picture
reader is the one Qwen3.8-Flash-Next also publishes.

No network (the two host APIs are answered by a stub) and no model is run.

    python -m unittest tools.test_setup_edge0_pack
"""
from __future__ import annotations

import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# the files the engine opens by name: without any of these the pack cannot be read at all
NEEDED = ("model-skeleton.gguf", "experts.bin", "labels.bin", "latent_dense_q8.bin", "latent_dense_q4.bin",
          "ple-fp8.gguf", "native_experts.txt", "mmproj-BF16.gguf", "dense.bin", "index.txt", "profile.bin",
          "scales/l0.f32")
QWEN_MMPROJ = (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0")


class Base(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for k in ("STRATA_REGION", "STRATA_SOURCE", "HF_ENDPOINT", "MODELSCOPE_ENDPOINT"):
            os.environ.pop(k, None)
        setup._sources.clear()
        setup._hf_files.clear()
        setup._ms_files.clear()
        self.addCleanup(setup._sources.clear)
        self.addCleanup(setup._hf_files.clear)
        self.addCleanup(setup._ms_files.clear)

    def family(self):
        return setup.PACK_FAMILIES["edge0"]

    def size(self, name="145B-Q3_K"):
        return setup.pack_size(self.family(), name)[1]


class Table(Base):
    def test_both_builds_name_both_hosts(self):
        for name, size in self.family()["sizes"].items():
            sources = size.get("sources") or {}
            self.assertTrue(sources.get("huggingface"), name)
            self.assertTrue(sources.get("modelscope"), name)

    def test_the_repositories_are_the_published_ones(self):
        fam = self.family()
        self.assertEqual(fam["sizes"]["145B-Q3_K"]["sources"]["huggingface"], "Edge0/Edge0-145B-q3k")
        self.assertEqual(fam["sizes"]["120B-Q3_K"]["sources"]["huggingface"], "Edge0/Edge0-120B-q3k")
        self.assertEqual(fam["sizes"]["145B-Q3_K"]["sources"]["modelscope"], "Edge0/Edge0-145B-q3k")

    def test_every_file_the_engine_needs_is_named(self):
        for name, size in self.family()["sizes"].items():
            for needed in NEEDED:
                self.assertIn(needed, size["files"], f"{name} is missing {needed}")

    def test_every_file_carries_a_size(self):
        for name, size in self.family()["sizes"].items():
            for path, entry in size["files"].items():
                got = entry[0] if isinstance(entry, tuple) else entry
                self.assertIsInstance(got, int, f"{name} {path}")
                self.assertGreater(got, 0, f"{name} {path}")

    def test_the_weight_files_carry_the_published_hashes(self):
        big = self.family()["sizes"]["145B-Q3_K"]["files"]
        self.assertEqual(big["model-skeleton.gguf"][1],
                         "ab49fbe7590cdcda45d7a2bcae4be0bd42c6a70b951d594a00eb158d7df0b72f")
        self.assertEqual(big["experts.bin"][1],
                         "947d15f3d69ecdb45d602a62bc84803451da779a22294e1db5b3568290b42879")
        self.assertEqual(big["ple-fp8.gguf"][1],
                         "c7615c59b060277ddf6f84729fa8f7469cfc82c7fb983d6e7eab5a2f7446f170")

    def test_the_measured_sizes_are_the_ones_published(self):
        files = self.family()["sizes"]["145B-Q3_K"]["files"]
        self.assertEqual(files["model-skeleton.gguf"][0], 5991569248)
        self.assertEqual(files["experts.bin"][0], 40076574720)
        self.assertEqual(files["labels.bin"][0], 98304)

    def test_the_two_builds_differ_only_in_the_files_they_say(self):
        fam = self.family()
        self.assertEqual(fam["sizes"]["145B-Q3_K"]["latent"], 1792)
        self.assertEqual(fam["sizes"]["120B-Q3_K"]["latent"], 1280)
        self.assertNotEqual(fam["sizes"]["145B-Q3_K"]["files"]["experts.bin"][0],
                            fam["sizes"]["120B-Q3_K"]["files"]["experts.bin"][0])
        # the picture reader is the same file in both, and the same bytes as Qwen's projector
        for name in fam["sizes"]:
            self.assertEqual(fam["sizes"][name]["files"]["mmproj-BF16.gguf"], QWEN_MMPROJ, name)

    def test_the_picture_reader_is_the_one_qwen_publishes(self):
        """The projector Edge0 ships is byte-identical to Qwen3.8-Flash-Next's, which is the one this port reads."""
        pinned = setup.ISTA_FILES["mmproj-Qwen3.8-Flash-Next-BF16.gguf"]
        self.assertEqual(pinned, QWEN_MMPROJ)

    def test_the_scales_are_per_layer_and_differ_between_builds(self):
        big = self.family()["sizes"]["145B-Q3_K"]["files"]
        small = self.family()["sizes"]["120B-Q3_K"]["files"]
        self.assertEqual(len([p for p in big if p.startswith("scales/")]), 48)
        self.assertEqual(big["scales/l0.f32"][0], 7168)
        self.assertEqual(small["scales/l0.f32"][0], 5120)

    def test_the_draft_head_is_optional(self):
        for name, size in self.family()["sizes"].items():
            optional = size.get("optional") or {}
            for path in ("mtp/dense.bin", "mtp/experts.bin"):
                self.assertIn(path, optional, f"{name} {path}")

    def test_the_branding_is_the_models_own(self):
        fam = self.family()
        self.assertEqual(fam["title"], "Edge0")
        self.assertIn("latent sparse MoE", fam["about"])
        self.assertIn("Edge0 Community Model License", fam["license"])
        self.assertTrue(fam["vision"])
        self.assertEqual(fam["drafter"], "none")


class Pins(Base):
    def test_a_pin_is_read_as_size_and_hash(self):
        files = self.family()["sizes"]["145B-Q3_K"]["files"]
        self.assertEqual(setup.pack_file_pin(files, "labels.bin"), (98304,
                         "ce9b4a20b92c5c866e65d84a3884c104986522471cf11cc5f797a153affbcc06"))

    def test_a_plain_size_has_no_hash(self):
        files = self.family()["sizes"]["145B-Q3_K"]["files"]
        self.assertEqual(setup.pack_file_pin(files, "index.txt"), (117464, None))

    def test_a_file_the_table_does_not_name_has_no_pin(self):
        self.assertEqual(setup.pack_file_pin({}, "nope.bin"), (None, None))


class Urls(Base):
    def test_the_region_decides_the_host(self):
        size = self.size()
        os.environ["STRATA_REGION"] = "intl"
        self.assertEqual(setup.pack_file_url(size, "experts.bin"),
                         "https://huggingface.co/Edge0/Edge0-145B-q3k/resolve/main/experts.bin")
        os.environ["STRATA_REGION"] = "cn"
        self.assertEqual(setup.pack_file_url(size, "experts.bin"),
                         "https://www.modelscope.cn/models/Edge0/Edge0-145B-q3k/resolve/master/experts.bin")

    def test_an_explicit_source_wins(self):
        size = self.size()
        os.environ["STRATA_REGION"] = "cn"
        os.environ["STRATA_SOURCE"] = "huggingface"
        self.assertTrue(setup.pack_file_url(size, "labels.bin").startswith("https://huggingface.co/"))

    def test_a_mirror_endpoint_is_used_for_the_url(self):
        size = self.size()
        os.environ["STRATA_REGION"] = "intl"
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.example"
        self.assertTrue(setup.pack_file_url(size, "labels.bin").startswith("https://hf-mirror.example/Edge0/"))

    def test_the_two_builds_use_their_own_repositories(self):
        os.environ["STRATA_REGION"] = "intl"
        self.assertIn("Edge0-120B-q3k", setup.pack_file_url(self.size("120B-Q3_K"), "experts.bin"))


class Meta(Base):
    def test_the_hash_comes_from_hugging_face(self):
        tree = [{"path": "experts.bin", "size": 40076574720, "lfs": {"oid": "abc123", "size": 40076574720}},
                {"path": "index.txt", "size": 117464}]
        with mock.patch.object(setup, "_urlopen", return_value=io.BytesIO(json.dumps(tree).encode())):
            os.environ["STRATA_REGION"] = "intl"
            self.assertEqual(setup.pack_file_meta(self.size(), "experts.bin"), (40076574720, "abc123"))

    def test_the_hash_comes_from_modelscope(self):
        payload = {"Data": {"Files": [{"Path": "experts.bin", "Size": 40076574720, "Sha256": "DEF456"}]}}
        with mock.patch.object(setup, "_urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
            os.environ["STRATA_REGION"] = "cn"
            self.assertEqual(setup.pack_file_meta(self.size(), "experts.bin"), (40076574720, "def456"))

    def test_a_host_that_cannot_be_asked_says_nothing(self):
        with mock.patch.object(setup, "_urlopen", side_effect=OSError("no network")):
            self.assertIsNone(setup.pack_file_meta(self.size(), "experts.bin"))


class Unknown(Base):
    def test_an_unknown_build_lists_the_ones_there_are(self):
        seen = []

        def stop(message, next_step=""):
            seen.append((message, next_step))
            raise SystemExit(10)

        with mock.patch.object(setup, "fail", stop):
            with self.assertRaises(SystemExit):
                setup.pack_size(self.family(), "999B-Q3_K")
        self.assertEqual(len(seen), 1)
        self.assertIn("145B-Q3_K", seen[0][1])
        self.assertIn("120B-Q3_K", seen[0][1])


if __name__ == "__main__":
    unittest.main()
