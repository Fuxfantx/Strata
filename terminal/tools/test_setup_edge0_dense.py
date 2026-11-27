"""Each Edge0 build downloads and names exactly one D/U table: the 145B the Q8 one, the 120B the Q4 one.

No network, no model run.

    python -m unittest tools.test_setup_edge0_dense
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Tables(unittest.TestCase):
    def setUp(self):
        import setup
        self.setup = setup
        self.fam = setup.PACK_FAMILIES["edge0"]

    def test_the_145b_build_reads_the_q8_table(self):
        size = self.fam["sizes"]["145B-Q3_K"]
        self.assertEqual(size["dense"], "q8")
        self.assertIn("latent_dense_q8.bin", size["files"])
        self.assertIn("latent_dense_q4.bin", size.get("optional") or {})

    def test_the_120b_build_reads_the_q4_table(self):
        size = self.fam["sizes"]["120B-Q3_K"]
        self.assertEqual(size["dense"], "q4")
        self.assertIn("latent_dense_q4.bin", size["files"])
        self.assertIn("latent_dense_q8.bin", size.get("optional") or {})

    def test_no_build_downloads_both_tables(self):
        for name, size in self.fam["sizes"].items():
            optional = size.get("optional") or {}
            left = set(size["files"]) - set(optional)
            self.assertEqual(len({n for n in left if n.startswith("latent_dense_")}), 1, name)

    def test_a_build_without_a_dense_word_is_a_table_the_installer_cannot_pick(self):
        """Every build must name its table: the installer uses the word to choose the file and the format."""
        for name, size in self.fam["sizes"].items():
            self.assertIn(size.get("dense"), ("q8", "q4"), name)


class Installer(unittest.TestCase):
    """The installer's own code, read rather than run: it must take the file and the format from the build."""

    def setUp(self):
        self.text = (ROOT / "setup.py").read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(self.text)
        self.src = ast.get_source_segment(self.text, next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "install_pack_family"))

    def test_the_environment_names_the_file_of_this_build(self):
        self.assertIn('f"./latent_dense_{dense}.bin"', self.src)

    def test_the_format_follows_the_build(self):
        self.assertIn('"8" if dense == "q8" else "4"', self.src)

    def test_the_download_list_leaves_the_other_table_out(self):
        self.assertIn("wanted = [name for name in size[\"files\"] if name not in optional]", self.src)


if __name__ == "__main__":
    unittest.main()
