"""Where a model is read from: the region rule and a family that names its sources.

The request this covers is one line: the mainland reads ModelScope, everywhere else reads Hugging Face.  What the tests
pin down is the part that is easy to get wrong: an explicit choice always wins, a family that names one source is read
from that one, and a family that names none keeps the behaviour it had before (Hugging Face unless asked otherwise).

The machine running the tests has a locale and a time zone of its own, so both inputs are controlled here rather than
inherited.

No network, and no model is run.

    python -m unittest tools.test_setup_region
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

BOTH = {"sources": {"modelscope": "Edge0/Edge0-145B-q3k", "huggingface": "Edge0-AI/Edge0-145B-q3k"}}
MS_ONLY = {"sources": {"modelscope": "Edge0/Edge0-145B-q3k"}}
HF_ONLY = {"sources": {"huggingface": "Edge0-AI/Edge0-145B-q3k"}}
LEGACY = {"hf": "unsloth/Qwen3.8-Flash-Next-GGUF"}


class Base(unittest.TestCase):
    hints = []

    def setUp(self):
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for k in ("STRATA_REGION", "STRATA_SOURCE", "TZ", "HF_ENDPOINT", "MODELSCOPE_ENDPOINT", "LANG", "LC_ALL"):
            os.environ.pop(k, None)
        setup._sources.clear()
        self.addCleanup(setup._sources.clear)
        hints = mock.patch.object(setup, "locale_hints", lambda: list(self.hints))
        hints.start()
        self.addCleanup(hints.stop)
        tz = mock.patch("time.tzname", ("UTC", "UTC"))
        tz.start()
        self.addCleanup(tz.stop)


class Region(Base):
    def test_the_explicit_variable_wins(self):
        for value, want in (("cn", "cn"), ("CN", "cn"), ("cn ", "cn"), ("intl", "intl"), ("INTL", "intl")):
            os.environ["STRATA_REGION"] = value
            self.assertEqual(setup.region(), want, f"STRATA_REGION={value!r}")

    def test_a_mainland_time_zone_is_the_mainland(self):
        for tz in ("Asia/Shanghai", "Asia/Urumqi", "Asia/Chongqing", "Asia/Harbin", "Asia/Kashgar"):
            os.environ["TZ"] = tz
            self.assertEqual(setup.region(), "cn", tz)

    def test_another_time_zone_is_not(self):
        for tz in ("Europe/Berlin", "America/New_York", "UTC", "Asia/Tokyo"):
            os.environ["TZ"] = tz
            self.assertNotEqual(setup.region(), "cn", tz)

    def test_the_c_library_time_zone_name_counts_too(self):
        with mock.patch("time.tzname", ("China Standard Time", "China Daylight Time")):
            self.assertEqual(setup.region(), "cn")

    def test_a_mainland_locale_is_the_mainland(self):
        for lang in ("zh_CN.UTF-8", "zh_CN", "zh-Hans-CN", "zh_Hans_CN"):
            with mock.patch.object(setup, "locale_hints", lambda lang=lang: [lang]):
                self.assertEqual(setup.region(), "cn", lang)

    def test_a_non_mainland_chinese_locale_is_not_the_mainland(self):
        for lang in ("zh_TW.UTF-8", "zh_HK", "zh_SG.utf8"):
            with mock.patch.object(setup, "locale_hints", lambda lang=lang: [lang.split()[0]]):
                self.assertNotEqual(setup.region(), "cn", lang)

    def test_an_unknown_machine_is_the_rest_of_the_world(self):
        self.assertEqual(setup.region(), "intl")

    def test_region_decides_nothing_by_itself_for_a_legacy_family(self):
        """#908 stays true: the families that were here before do not change host because of a region."""
        os.environ["STRATA_REGION"] = "cn"
        self.assertEqual(setup.family_source(LEGACY), "huggingface")


class FamilySource(Base):
    def test_an_explicit_choice_beats_the_region(self):
        os.environ["STRATA_REGION"] = "cn"
        os.environ["STRATA_SOURCE"] = "huggingface"
        self.assertEqual(setup.family_source(BOTH), "huggingface")
        os.environ["STRATA_SOURCE"] = "modelscope"
        os.environ["STRATA_REGION"] = "intl"
        self.assertEqual(setup.family_source(BOTH), "modelscope")

    def test_both_sources_follow_the_region(self):
        os.environ["STRATA_REGION"] = "cn"
        self.assertEqual(setup.family_source(BOTH), "modelscope")
        os.environ["STRATA_REGION"] = "intl"
        self.assertEqual(setup.family_source(BOTH), "huggingface")

    def test_one_source_is_read_whatever_the_region_says(self):
        os.environ["STRATA_REGION"] = "intl"
        self.assertEqual(setup.family_source(MS_ONLY), "modelscope")
        os.environ["STRATA_REGION"] = "cn"
        self.assertEqual(setup.family_source(HF_ONLY), "huggingface")

    def test_a_family_without_sources_keeps_the_old_answer(self):
        self.assertEqual(setup.family_source(LEGACY), "huggingface")
        self.assertEqual(setup.family_source({}), "huggingface")
        self.assertEqual(setup.family_source(None), "huggingface")
        os.environ["STRATA_SOURCE"] = "modelscope"
        self.assertEqual(setup.family_source(None), "modelscope")

    def test_an_empty_repository_is_not_a_source(self):
        os.environ["STRATA_REGION"] = "cn"
        self.assertEqual(setup.family_source({"sources": {"modelscope": "", "huggingface": ""}}), "huggingface")

    def test_the_names_a_person_reads(self):
        self.assertEqual(setup.source_name("modelscope"), "ModelScope")
        self.assertEqual(setup.source_name("huggingface"), "Hugging Face")

    def test_the_host_follows_the_endpoint_variables(self):
        os.environ["MODELSCOPE_ENDPOINT"] = "https://ms.example/"
        os.environ["HF_ENDPOINT"] = "https://hf.example"
        self.assertEqual(setup.source_host("modelscope"), "https://ms.example")
        self.assertEqual(setup.source_host("huggingface"), "https://hf.example")


if __name__ == "__main__":
    unittest.main()
