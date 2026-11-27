"""Regression tests for Pass 2 serve fixes:
- #6686: /v1/models and /v1/models/<id> include OpenAI-required `created` (int) and `owned_by` ("strata")
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[1]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (  # noqa: E402
    ByteTokenizer,
    MockEngine,
    Service,
    serve,
)


class TestModelsOpenAISchema6686(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(
            MockEngine(tok, "</think>\n\nok", max_context=8192),
            tok,
            ChatTemplate(TERMINAL / "serve" / "chat_template.jinja"),
            model_name="qwen3.8-flash-next-iq3_xxs",
        )
        cls.svc.set_aliases(["qwen"])
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))

    def test_models_list_and_retrieve_include_created_and_owned_by(self):
        for path in ("/v1/models", "/models"):
            body = self._get(path)
            self.assertEqual(body["object"], "list")
            self.assertEqual(len(body["data"]), 2)
            for item in body["data"]:
                self.assertIsInstance(item["created"], int)
                self.assertGreater(item["created"], 1_700_000_000)
                self.assertEqual(item["owned_by"], "strata")
                self.assertEqual(item["object"], "model")
                self.assertEqual(item["status"]["value"], "loaded")
                self.assertEqual(item["max_model_len"], 8192)
                self.assertIn("architecture", item)

        for path in ("/v1/models/qwen3.8-flash-next-iq3_xxs", "/v1/models/qwen", "/models/qwen"):
            item = self._get(path)
            self.assertIsInstance(item["created"], int)
            self.assertEqual(item["owned_by"], "strata")


if __name__ == "__main__":
    unittest.main()
