"""Regression tests for Pass 2 TUI/CLI, setup.py, and strata_control fixes:
- #6469: --prompt-cache-spread alignment in strata launch derive_config & BOOL_FLAGS
- #6695: strata_control/drafting.py:gguf_facts() skips string arrays (general.tags, general.languages)
- #6662: setup.py:gpus() records free_gb alongside vram_gb and respects free memory for tier/vision sizing
"""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import setup  # noqa: E402
from strata_cli.launch import runner as R, settings as S  # noqa: E402
from strata_control import drafting, tune_knobs  # noqa: E402
from tests.terminal.test_drafting_file_fit import _s as gguf_str  # noqa: E402


class PromptCacheSpread6469Tests(unittest.TestCase):
    BASE = {
        "exe": "component:strata-engine",
        "args": ["--native", "x.gguf", "--max-context", "131072", "--kv", "int8"],
        "port": 8080,
        "model_name": "qwen3.8-flash-next-iq3_s",
    }

    def test_prompt_cache_spread_enabled_by_default_and_toggleable(self):
        self.assertIn("--prompt-cache-spread", tune_knobs.BOOL_FLAGS)
        cfg = R.derive_config(self.BASE, S.LaunchSettings(), bool_flags=tune_knobs.BOOL_FLAGS)
        self.assertIn("--prompt-cache-spread", cfg["args"])

        # Explicit --set prompt-cache-spread=off removes it via arg_layers
        cfg_off = R.derive_config(
            self.BASE,
            S.LaunchSettings(extra={"--prompt-cache-spread": "off"}),
            bool_flags=tune_knobs.BOOL_FLAGS,
        )
        self.assertNotIn("--prompt-cache-spread", cfg_off["args"])

        # Config flag prompt_cache_spread=False removes it
        cfg_flag_off = R.derive_config(
            {**self.BASE, "prompt_cache_spread": False},
            S.LaunchSettings(),
            bool_flags=tune_knobs.BOOL_FLAGS,
        )
        self.assertNotIn("--prompt-cache-spread", cfg_flag_off["args"])


class GgufFactsStringArray6695Tests(unittest.TestCase):
    def _gguf_with_tags(self, arch="qwen4exp", layers=48, hidden=2560, vocab=248320) -> bytes:
        tags = ["qwen", "moe", "text-generation", "llama.cpp"]
        langs = ["en", "zh"]
        tags_val = struct.pack("<IIQ", 9, 8, len(tags)) + b"".join(gguf_str(t) for t in tags)
        langs_val = struct.pack("<IIQ", 9, 8, len(langs)) + b"".join(gguf_str(x) for x in langs)
        kvs = [
            (gguf_str("general.architecture"), struct.pack("<I", 8) + gguf_str(arch)),
            (gguf_str("general.tags"), tags_val),
            (gguf_str("general.languages"), langs_val),
            (gguf_str(f"{arch}.block_count"), struct.pack("<II", 4, layers)),
            (gguf_str(f"{arch}.embedding_length"), struct.pack("<II", 4, hidden)),
            (gguf_str("tokenizer.ggml.tokens"), struct.pack("<IIQ", 9, 8, vocab)),
        ]
        return b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(k + v for k, v in kvs)

    def test_gguf_facts_reads_past_general_tags_and_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "renamed-flash-next.gguf"
            p.write_bytes(self._gguf_with_tags())
            self.assertEqual(
                drafting.gguf_facts(p),
                {"arch": "qwen4exp", "layers": 48, "hidden": 2560, "vocab": 248320},
            )
            self.assertEqual(drafting.file_fit(["--native", str(p)]), "fits")

            # Corrupt absurd string-array count never hangs or raises
            bad = (
                b"GGUF"
                + struct.pack("<IQQ", 3, 0, 2)
                + gguf_str("general.architecture") + struct.pack("<I", 8) + gguf_str("qwen4exp")
                + gguf_str("general.tags") + struct.pack("<IIQ", 9, 8, 1 << 60)
            )
            p.write_bytes(bad)
            self.assertEqual(drafting.gguf_facts(p), {"arch": "qwen4exp", "layers": None, "hidden": None, "vocab": None})


class GpuFreeVram6662Tests(unittest.TestCase):
    def test_gpus_records_free_gb_and_tolerates_legacy_5_column_output(self):
        csv_6 = "0, NVIDIA GeForce RTX 4090, 24576, 8.9, 565.57, 8192\n"
        with mock.patch.object(setup, "out", return_value=csv_6):
            found = setup.gpus()
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]["vram_gb"], 24.0, places=2)
        self.assertAlmostEqual(found[0]["free_gb"], 8.0, places=2)
        self.assertAlmostEqual(setup.effective_vram_gb(found[0]), 8.0, places=2)
        # Busy 24 GB card with only 8 GB free uses CPU vision encoder and 8 GB for low_ram_vram / tier_pick
        self.assertEqual(setup.flash_vision_encoder("yes", found[0]), "cpu")
        self.assertAlmostEqual(setup.low_ram_vram(found[0]), 8.0, places=2)
        self.assertEqual(setup.tier_pick(40.0, setup.low_ram_vram(found[0])), ("swift", "IQ2_XS"))

        # Legacy 5-column mock keeps working cleanly with vram_gb fallback
        csv_5 = "0, NVIDIA GeForce RTX 4090, 24576, 8.9, 565.57\n"
        with mock.patch.object(setup, "out", return_value=csv_5):
            found5 = setup.gpus()
        self.assertEqual(len(found5), 1)
        self.assertNotIn("free_gb", found5[0])
        self.assertAlmostEqual(setup.effective_vram_gb(found5[0]), 24.0, places=2)
        self.assertEqual(setup.flash_vision_encoder("yes", found5[0]), "gpu")


if __name__ == "__main__":
    unittest.main()
