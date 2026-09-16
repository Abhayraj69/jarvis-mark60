"""Unit tests for core/backend_router.py — per-task-kind backend selection,
circuit breaker, and failover. Uses fake adapters (no network, no real
Ollama/Claude/Gemini config needed)."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import backend_router as router  # noqa: E402
from core.backend_router import TaskKind  # noqa: E402


def _ok(name):
    def _adapter(messages, tools, images, timeout):
        return {"content": f"ok from {name}", "tool_calls": [], "usage": {}, "backend": name}
    return _adapter


def _fail(name, exc=RuntimeError("boom")):
    def _adapter(messages, tools, images, timeout):
        raise exc
    return _adapter


class TestBackendRouter(unittest.TestCase):
    def setUp(self):
        router.reset_breakers()
        self._orig_adapters = dict(router._ADAPTERS)
        self._configured_patch = patch.object(router, "_is_configured", return_value=True)
        self._configured_patch.start()
        # These tests are about ordering / breaker / failover mechanics, so
        # the local-model gate is held open (a "big" local model). The gate
        # itself is covered by TestLocalModelGate below.
        self._capable_patch = patch.object(router, "ollama_capability",
                                           return_value=(True, "llama3.1:8b", "8B"))
        self._capable_patch.start()

    def tearDown(self):
        router._ADAPTERS.clear()
        router._ADAPTERS.update(self._orig_adapters)
        router.reset_breakers()
        self._configured_patch.stop()
        self._capable_patch.stop()

    def test_policy_order_respected(self):
        router._ADAPTERS["claude"] = _ok("claude")
        router._ADAPTERS["ollama"] = _ok("ollama")
        router._ADAPTERS["gemini"] = _ok("gemini")

        result = router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])
        self.assertEqual(result["backend"], "claude")   # first in DEFAULT_POLICY[CODE_GEN]

        result = router.complete(TaskKind.INTENT, [{"role": "user", "content": "hi"}])
        self.assertEqual(result["backend"], "ollama")   # first in DEFAULT_POLICY[INTENT]

    def test_falls_over_to_next_backend_on_failure(self):
        router._ADAPTERS["claude"] = _fail("claude")
        router._ADAPTERS["ollama"] = _ok("ollama")
        router._ADAPTERS["gemini"] = _ok("gemini")

        result = router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])
        self.assertEqual(result["backend"], "gemini")   # second in DEFAULT_POLICY[CODE_GEN]

    def test_breaker_opens_after_failure_and_skips_on_next_call(self):
        calls = {"claude": 0}

        def _flaky(messages, tools, images, timeout):
            calls["claude"] += 1
            raise RuntimeError("down")

        router._ADAPTERS["claude"] = _flaky
        router._ADAPTERS["ollama"] = _ok("ollama")
        router._ADAPTERS["gemini"] = _ok("gemini")

        router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])
        router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi again"}])

        self.assertEqual(calls["claude"], 1)   # second call skipped claude — breaker open

    def test_breaker_closes_after_cooldown_window(self):
        router._ADAPTERS["claude"] = _fail("claude")
        router._ADAPTERS["ollama"] = _ok("ollama")
        router._ADAPTERS["gemini"] = _ok("gemini")

        router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])
        self.assertTrue(router._breaker_open("claude"))

        with patch("time.monotonic", return_value=__import__("time").monotonic() + 61):
            self.assertFalse(router._breaker_open("claude"))

    def test_unconfigured_backend_is_skipped(self):
        def _configured(name):
            return name != "claude"

        with patch.object(router, "_is_configured", side_effect=_configured):
            router._ADAPTERS["claude"] = _ok("claude")   # would succeed if tried
            router._ADAPTERS["ollama"] = _ok("ollama")
            router._ADAPTERS["gemini"] = _ok("gemini")

            result = router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])
            self.assertEqual(result["backend"], "gemini")

    def test_all_backends_failing_raises(self):
        router._ADAPTERS["claude"] = _fail("claude")
        router._ADAPTERS["ollama"] = _fail("ollama")
        router._ADAPTERS["gemini"] = _fail("gemini")

        with self.assertRaises(RuntimeError):
            router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])

    def test_all_backends_unconfigured_raises(self):
        with patch.object(router, "_is_configured", return_value=False):
            with self.assertRaises(RuntimeError):
                router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])

    def test_usage_and_content_pass_through(self):
        def _adapter(messages, tools, images, timeout):
            return {"content": "hello", "tool_calls": [{"id": "1"}], "usage": {"tokens": 5}, "backend": "claude"}
        router._ADAPTERS["claude"] = _adapter

        result = router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])
        self.assertEqual(result["content"], "hello")
        self.assertEqual(result["usage"], {"tokens": 5})
        self.assertEqual(result["tool_calls"], [{"id": "1"}])

    def test_get_text_model_wraps_complete(self):
        router._ADAPTERS["claude"] = _ok("claude")
        model = router.get_text_model(TaskKind.CODE_GEN)
        result = model.generate_content("write a haiku")
        self.assertEqual(result.text, "ok from claude")

    def test_custom_policy_overrides_default(self):
        router._ADAPTERS["ollama"] = _ok("ollama")
        router._ADAPTERS["claude"] = _ok("claude")
        custom_policy = {TaskKind.CODE_GEN: ["ollama", "claude"]}

        result = router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}],
                                  policy=custom_policy)
        self.assertEqual(result["backend"], "ollama")


class TestLocalModelGate(unittest.TestCase):
    """resolve_order(): a local model below OLLAMA_MIN_PARAMS_B never answers
    before a configured cloud backend, whatever the saved policy says."""

    def setUp(self):
        router.reset_breakers()

    def test_param_parsing(self):
        cases = {
            "qwen3:1.7b": 1.7, "llama3.1:8b-instruct-q4_K_M": 8.0, "mistral:7b": 7.0,
            "gemma3:27b": 27.0, "phi4:14B": 14.0, "deepseek-r1:70b": 70.0,
            "mistral": None, "gemma:latest": None, "": None,
        }
        for tag, expected in cases.items():
            self.assertEqual(router.model_param_billions(tag), expected, tag)

    def test_small_model_is_demoted_to_last(self):
        policy = {TaskKind.CHAT: ["ollama", "gemini", "claude"]}
        with patch.object(router, "ollama_capability", return_value=(False, "qwen3:1.7b", "small")), \
             patch.object(router, "_is_configured", return_value=True):
            self.assertEqual(router.resolve_order(TaskKind.CHAT, policy), ["gemini", "claude", "ollama"])

    def test_small_model_demoted_even_when_not_literally_first(self):
        # Saved "claude, ollama, gemini" with no Claude key used to send every
        # code task to the tiny local model.
        policy = {TaskKind.CODE_GEN: ["claude", "ollama", "gemini"]}
        with patch.object(router, "ollama_capability", return_value=(False, "qwen3:1.7b", "small")), \
             patch.object(router, "_is_configured", side_effect=lambda n: n == "gemini"):
            self.assertEqual(router.resolve_order(TaskKind.CODE_GEN, policy), ["claude", "gemini", "ollama"])

    def test_small_model_kept_when_nothing_else_is_configured(self):
        policy = {TaskKind.CHAT: ["ollama", "gemini"]}
        with patch.object(router, "ollama_capability", return_value=(False, "qwen3:1.7b", "small")), \
             patch.object(router, "_is_configured", side_effect=lambda n: n == "ollama"):
            self.assertEqual(router.resolve_order(TaskKind.CHAT, policy), ["ollama", "gemini"])

    def test_big_model_keeps_its_place(self):
        policy = {TaskKind.CHAT: ["ollama", "gemini"]}
        with patch.object(router, "ollama_capability", return_value=(True, "llama3.1:8b", "8B")), \
             patch.object(router, "_is_configured", return_value=True):
            self.assertEqual(router.resolve_order(TaskKind.CHAT, policy), ["ollama", "gemini"])

    def test_capability_reads_configured_model(self):
        with patch("core.llm_client.get_llm_settings", return_value=("http://x", "qwen3:1.7b")):
            capable, model, reason = router.ollama_capability()
            self.assertFalse(capable); self.assertEqual(model, "qwen3:1.7b"); self.assertIn("1.7B", reason)
        with patch("core.llm_client.get_llm_settings", return_value=("http://x", "qwen2.5:14b")):
            self.assertTrue(router.ollama_capability()[0])

    def test_defaults_never_lead_with_ollama_except_intent(self):
        for kind, order in router.DEFAULT_POLICY.items():
            if kind is TaskKind.INTENT:
                continue
            self.assertNotEqual(order[0], "ollama", kind)

    def test_gemini_lite_is_a_backend(self):
        self.assertIn("gemini_lite", router._ADAPTERS)
        self.assertEqual(router.DEFAULT_POLICY[TaskKind.VISION], ["gemini", "gemini_lite"])
        policy = router.load_policy_from_config({"vision": "gemini_lite, gemini"})
        self.assertEqual(policy[TaskKind.VISION], ["gemini_lite", "gemini"])

    def test_describe_routing_mentions_demotion(self):
        policy = {k: ["gemini"] for k in TaskKind}
        policy[TaskKind.CHAT] = ["ollama", "gemini"]
        with patch.object(router, "ollama_capability", return_value=(False, "qwen3:1.7b", "'qwen3:1.7b' is 1.7B")), \
             patch.object(router, "_is_configured", side_effect=lambda n: n != "claude"):
            text = router.describe_routing(policy)
        self.assertIn("chat", text)
        self.assertIn("demoted", text)
        self.assertIn("not configured: claude", text)


class TestTransientRetry(unittest.TestCase):
    """A 503/429 gets ONE quick retry before the breaker trips — a single
    Gemini 'high demand' blip must not route the next minute to a fallback."""

    def setUp(self):
        router.reset_breakers()
        self._orig_adapters = dict(router._ADAPTERS)
        self._patches = [
            patch.object(router, "_is_configured", return_value=True),
            patch.object(router, "ollama_capability", return_value=(True, "big:8b", "8B")),
            patch.object(router, "TRANSIENT_RETRY_DELAY_S", 0.0),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        router._ADAPTERS.clear(); router._ADAPTERS.update(self._orig_adapters)
        router.reset_breakers()
        for p in self._patches:
            p.stop()

    def _flaky(self, name, fail_times, exc):
        calls = {"n": 0}
        def _adapter(messages, tools, images, timeout):
            calls["n"] += 1
            if calls["n"] <= fail_times:
                raise exc
            return {"content": "ok", "tool_calls": [], "usage": {}, "backend": name}
        return _adapter, calls

    def test_503_is_retried_once_and_succeeds(self):
        adapter, calls = self._flaky("gemini", 1, RuntimeError("503 UNAVAILABLE high demand"))
        router._ADAPTERS["gemini"] = adapter
        router._ADAPTERS["claude"] = _ok("claude")
        out = router.complete(TaskKind.CHAT, [{"role": "user", "content": "hi"}],
                              policy={TaskKind.CHAT: ["gemini", "claude"]})
        self.assertEqual(out["backend"], "gemini")
        self.assertEqual(calls["n"], 2)
        self.assertFalse(router._breaker_open("gemini"))

    def test_persistent_503_trips_breaker_after_retry(self):
        adapter, calls = self._flaky("gemini", 5, RuntimeError("503 UNAVAILABLE"))
        router._ADAPTERS["gemini"] = adapter
        router._ADAPTERS["claude"] = _ok("claude")
        out = router.complete(TaskKind.CHAT, [{"role": "user", "content": "hi"}],
                              policy={TaskKind.CHAT: ["gemini", "claude"]})
        self.assertEqual(out["backend"], "claude")
        self.assertEqual(calls["n"], 2)          # exactly one retry
        self.assertTrue(router._breaker_open("gemini"))

    def test_non_transient_error_is_not_retried(self):
        adapter, calls = self._flaky("gemini", 5, RuntimeError("400 INVALID_ARGUMENT"))
        router._ADAPTERS["gemini"] = adapter
        router._ADAPTERS["claude"] = _ok("claude")
        router.complete(TaskKind.CHAT, [{"role": "user", "content": "hi"}],
                        policy={TaskKind.CHAT: ["gemini", "claude"]})
        self.assertEqual(calls["n"], 1)


class TestGeminiAdapters(unittest.TestCase):
    """Both Gemini adapters share _gemini_generate; images go in as Parts
    ahead of the prompt and the model name follows the backend."""

    def test_models_and_image_order(self):
        seen = {}

        class _Resp:
            text = " hi "
            usage_metadata = None

        class _Models:
            def generate_content(self, model, contents):
                seen["model"] = model; seen["contents"] = contents
                return _Resp()

        class _Client:
            def __init__(self, api_key): self.models = _Models()

        from google import genai as real_genai
        with patch.object(real_genai, "Client", _Client),              patch.object(router, "_get_api_config", return_value={"gemini_api_key": "k"}):
            out = router._call_gemini_lite([{"role": "user", "content": "q"}], None, [(b"img", "image/png")], 10)
        self.assertEqual(out["backend"], "gemini_lite")
        self.assertEqual(seen["model"], router.GEMINI_LITE_MODEL)
        self.assertEqual(out["content"], "hi")
        self.assertEqual(len(seen["contents"]), 2)
        self.assertEqual(seen["contents"][-1], "q")


if __name__ == "__main__":
    unittest.main()
