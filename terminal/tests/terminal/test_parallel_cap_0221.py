"""0221 APPTUI: `strata launch --parallel N` takes up to 64 (the GRID lane's 64-client server target) and says plainly when a card cannot hold N.

Limits, from the engine: the Strata engine's batch window holds 16 rows in the CUDA and HIP builds and 8 in the others (a lone card cannot run
more). The family does not raise or lower that window. The default stays what it was (fit to the card). Nothing is started."""

from __future__ import annotations

import argparse
import unittest

from strata_cli.launch import cmd
from strata_cli.launch import settings as S


def levels(s, **kw):
    return [(n.level, n.text) for n in S.check_parallel(s, **kw)]


class ParseAndHelp(unittest.TestCase):
    def test_the_flag_takes_1_to_64(self):
        self.assertEqual(cmd._parallel("64"), 64)
        self.assertEqual(cmd._parallel("1"), 1)
        for bad in ("0", "65", "x", "-3"):
            with self.assertRaises(argparse.ArgumentTypeError):
                cmd._parallel(bad)

    def test_the_help_does_not_say_8(self):
        parser = argparse.ArgumentParser()
        sp = parser.add_subparsers()
        cmd.add_parsers(sp)
        text = sp.choices["launch"].format_help()
        self.assertIn("1 to 64", text)
        self.assertNotIn("1 to 8,", text)


class BatchWindow(unittest.TestCase):
    def test_one_request_is_never_a_notice(self):
        for fam, model in (("qwen", "IQ3_S"), ("qwen36", "UD-Q4_K_XL"), ("hy4-flash", "SQ-64")):
            s = S.LaunchSettings(family=fam, model=model, parallel=1)
            self.assertEqual(levels(s, vram_gb=8.0, ram_gb=16.0, vendor="nvidia"), [], fam)

    def test_the_engine_keeps_what_its_batch_window_holds_and_says_so(self):
        s = S.LaunchSettings(family="qwen", model="IQ3_S", parallel=64)
        for vendor, cap in (("nvidia", 16), ("amd", 16), ("intel", 8), (None, 8)):
            got = levels(s, vram_gb=32.0, vendor=vendor)
            self.assertEqual([lv for lv, _ in got], ["warn"], vendor)
            self.assertIn(f"it keeps {cap}", got[0][1])
        self.assertEqual(S.max_parallel("qwen", "nvidia"), 16)
        self.assertEqual(S.max_parallel("qwen", "intel"), 8)
        # the family does not change the window: the engine's batch is the limit
        self.assertEqual(S.max_parallel("qwen36", "nvidia"), 16)
        self.assertEqual(S.max_parallel("hy4-flash", "nvidia"), 16)

    def test_up_to_the_window_is_the_old_card_check(self):
        s = S.LaunchSettings(family="qwen", model="IQ3_S", parallel=12)
        got = levels(s, vram_gb=12.0, vendor="nvidia")
        self.assertEqual([lv for lv, _ in got], ["warn"])
        self.assertIn("usually holds", got[0][1])

    def test_more_than_64_is_an_error_whatever_the_family(self):
        s = S.LaunchSettings(family="qwen36", model="UD-Q4_K_XL", parallel=65)
        self.assertEqual([lv for lv, _ in levels(s, vram_gb=32.0, ram_gb=64.0)], ["error"])


class SetupGetsNoSlots(unittest.TestCase):
    def test_setup_args_never_carry_parallel(self):
        from strata_cli.launch.runner import Runner
        from tests.terminal.launch_support import HW_12, Home, catalog
        with Home():
            for fam, model in (("qwen", "IQ3_S"), ("qwen36", "UD-Q4_K_XL")):
                args = Runner(S.LaunchSettings(family=fam, model=model, context=32768, parallel=8), catalog=catalog(HW_12)).setup_args()
                self.assertNotIn("--parallel", args, fam)


if __name__ == "__main__":
    unittest.main()
