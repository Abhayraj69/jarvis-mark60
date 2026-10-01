"""core/think.py (the reasoning core), core/backend_router.complete_stream,
and main.py's `think` delivery into the Live session."""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import think  # noqa: E402
from core import backend_router as router  # noqa: E402
from core.backend_router import TaskKind  # noqa: E402


def _fake_stream(deltas, backend="gemini", usage=None):
    def _s(kind, messages, images=None, timeout=60, policy=None):
        _s.calls.append({"kind": kind, "messages": messages, "images": images, "policy": policy})
        for d in deltas:
            yield {"delta": d}
        yield {"done": {"backend": backend, "usage": usage or {}}}
    _s.calls = []
    return _s


class BuildMessagesTests(unittest.TestCase):

    def test_structure_and_blocks(self):
        msgs = think.build_messages(
            "why is the sky blue",
            session_log=["User: hi", "JARVIS: hello"],
            memory_block="[ABOUT THE USER]\nName: Raj",
            context_block="[USER PREFERENCES]\n- coffee: black",
            now="Monday 10:00",
        )
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        sysm = msgs[0]["content"]
        self.assertTrue(sysm.startswith(think.REASONING_SYSTEM))
        self.assertIn("[NOW] Monday 10:00", sysm)
        self.assertIn("Name: Raj", sysm)
        self.assertIn("coffee: black", sysm)
        self.assertIn("[RECENT TURNS]\nUser: hi\nJARVIS: hello", sysm)
        self.assertEqual(msgs[1]["content"], "why is the sky blue")

    def test_recent_turns_not_duplicated_when_context_has_them(self):
        msgs = think.build_messages("q", session_log=["User: a"], memory_block="",
                                    context_block="[RECENT CONVERSATION]\nUser: a")
        self.assertNotIn("[RECENT TURNS]", msgs[0]["content"])

    def test_screen_note_and_query_cap(self):
        msgs = think.build_messages("x" * 5000, memory_block="", context_block="", screen_attached=True)
        self.assertIn("[SCREEN]", msgs[0]["content"])
        self.assertEqual(len(msgs[1]["content"]), think.MAX_QUERY_CHARS)


class RunTests(unittest.TestCase):

    def setUp(self):
        self._p1 = patch.object(think, "_memory_block", return_value="")
        self._p2 = patch.object(think, "_context_block", return_value="")
        self._p1.start(); self._p2.start()

    def tearDown(self):
        self._p1.stop(); self._p2.stop()

    def test_sentences_stream_in_order(self):
        got = []
        stream = _fake_stream(["FSRS is the better", " choice. It adapts", " to you. Shall I set it up?"],
                              backend="claude", usage={"tokens_in": 10})
        r = think.run("sm2 or fsrs", session_log=["User: hi"],
                      on_sentence=lambda s, i: got.append((i, s)), stream=stream)
        self.assertEqual([s for _, s in got], ["FSRS is the better choice.", "It adapts to you.", "Shall I set it up?"])
        self.assertEqual([i for i, _ in got], [0, 1, 2])
        self.assertEqual(r.sentences, [s for _, s in got])
        self.assertEqual(r.backend, "claude")
        self.assertEqual(r.usage, {"tokens_in": 10})
        self.assertEqual(r.text, "FSRS is the better choice. It adapts to you. Shall I set it up?")
        self.assertIs(stream.calls[0]["kind"], TaskKind.CHAT)
        self.assertIsNone(stream.calls[0]["images"])

    def test_screen_forces_gemini_policy(self):
        stream = _fake_stream(["ok."])
        r = think.run("what's this error", include_screen=True,
                      capture=lambda: (b"png", "image/png"), stream=stream,
                      policy={TaskKind.CHAT: ["claude", "gemini", "ollama"]})
        self.assertTrue(r.included_screen)
        self.assertEqual(stream.calls[0]["images"], [(b"png", "image/png")])
        self.assertEqual(stream.calls[0]["policy"], {TaskKind.CHAT: ["gemini"]})

    def test_capture_failure_is_tolerated(self):
        def _boom():
            raise RuntimeError("no mss")
        stream = _fake_stream(["fine."])
        r = think.run("q", include_screen=True, capture=_boom, stream=stream)
        self.assertFalse(r.included_screen)
        self.assertIsNone(stream.calls[0]["images"])

    def test_on_sentence_exception_does_not_abort(self):
        def _bad(s, i):
            raise ValueError("delivery broke")
        r = think.run("q", on_sentence=_bad, stream=_fake_stream(["One. Two."]))
        self.assertEqual(r.sentences, ["One.", "Two."])

    def test_backend_failure_propagates(self):
        def _s(kind, messages, images=None, timeout=60, policy=None):
            raise RuntimeError("all failed")
            yield  # pragma: no cover
        with self.assertRaises(RuntimeError):
            think.run("q", stream=_s)


class CompleteStreamTests(unittest.TestCase):

    def setUp(self):
        router.reset_breakers()
        self._orig = dict(router._STREAMERS)
        self._patches = [
            patch.object(router, "_is_configured", return_value=True),
            patch.object(router, "ollama_capability", return_value=(True, "big:8b", "8B")),
            patch.object(router, "TRANSIENT_RETRY_DELAY_S", 0.0),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        router._STREAMERS.clear(); router._STREAMERS.update(self._orig)
        router.reset_breakers()
        for p in self._patches:
            p.stop()

    @staticmethod
    def _ok(name, deltas):
        def _s(messages, images, timeout):
            for d in deltas:
                yield {"delta": d}
            yield {"done": {"backend": name, "usage": {}}}
        return _s

    def test_yields_deltas_then_done(self):
        router._STREAMERS["gemini"] = self._ok("gemini", ["a", "b"])
        evs = list(router.complete_stream(TaskKind.CHAT, [{"role": "user", "content": "x"}],
                                          policy={TaskKind.CHAT: ["gemini"]}))
        self.assertEqual(evs, [{"delta": "a"}, {"delta": "b"}, {"done": {"backend": "gemini", "usage": {}}}])

    def test_fails_over_before_first_delta(self):
        def _dead(messages, images, timeout):
            raise RuntimeError("400 bad")
            yield  # pragma: no cover
        router._STREAMERS["gemini"] = _dead
        router._STREAMERS["claude"] = self._ok("claude", ["hi"])
        evs = list(router.complete_stream(TaskKind.CHAT, [{"role": "user", "content": "x"}],
                                          policy={TaskKind.CHAT: ["gemini", "claude"]}))
        self.assertEqual(evs[-1]["done"]["backend"], "claude")
        self.assertTrue(router._breaker_open("gemini"))

    def test_no_failover_after_first_delta(self):
        def _half(messages, images, timeout):
            yield {"delta": "part"}
            raise RuntimeError("connection reset")
        router._STREAMERS["gemini"] = _half
        router._STREAMERS["claude"] = self._ok("claude", ["hi"])
        gen = router.complete_stream(TaskKind.CHAT, [{"role": "user", "content": "x"}],
                                     policy={TaskKind.CHAT: ["gemini", "claude"]})
        self.assertEqual(next(gen), {"delta": "part"})
        with self.assertRaises(RuntimeError):
            next(gen)

    def test_transient_retry_before_first_delta(self):
        n = {"calls": 0}
        def _flaky(messages, images, timeout):
            n["calls"] += 1
            if n["calls"] == 1:
                raise RuntimeError("503 UNAVAILABLE")
            yield {"delta": "ok"}
            yield {"done": {"backend": "gemini", "usage": {}}}
        router._STREAMERS["gemini"] = _flaky
        evs = list(router.complete_stream(TaskKind.CHAT, [{"role": "user", "content": "x"}],
                                          policy={TaskKind.CHAT: ["gemini"]}))
        self.assertEqual(n["calls"], 2)
        self.assertEqual(evs[0], {"delta": "ok"})
        self.assertFalse(router._breaker_open("gemini"))

    def test_all_unconfigured_raises(self):
        with patch.object(router, "_is_configured", return_value=False):
            with self.assertRaises(RuntimeError):
                list(router.complete_stream(TaskKind.CHAT, [{"role": "user", "content": "x"}]))


class _FakeSession:
    def __init__(self):
        self.tool_responses = []
        self.client_content = []

    async def send_tool_response(self, function_responses):
        self.tool_responses.append(function_responses)

    async def send_client_content(self, turns, turn_complete=True):
        self.client_content.append(turns)


class ThinkDeliveryTests(unittest.TestCase):
    """main.py: short answers arrive whole in one WHEN_IDLE FunctionResponse;
    long answers arrive as first-part response + one follow-up text turn
    after JARVIS stops speaking; failures come back as ok=false."""

    def _bare(self):
        import main
        import threading
        j = object.__new__(main.JarvisLive)
        j.ui = MagicMock()
        j.session = _FakeSession()
        j._session_log = ["User: hi"]
        j._current_turn = MagicMock()
        j._false_success = __import__("core.result_contract", fromlist=["x"]).FalseSuccessTracker()
        j._think_tasks = set()
        j._speaking_lock = threading.Lock()
        j._is_speaking = False
        j._THINK_FOLLOWUP_WAIT_S = 0.5
        j._loop = None
        return j

    def _fc(self):
        fc = MagicMock(); fc.id = "call-1"; fc.name = "think"
        return fc

    def _run(self, j, fake_result_sentences, backend="gemini", raise_exc=None):
        import main

        def fake_run(query, session_log, include_screen, on_sentence, policy, **kw):
            if raise_exc:
                raise raise_exc
            for i, s in enumerate(fake_result_sentences):
                on_sentence(s, i)
            return think.ThinkResult(text=" ".join(fake_result_sentences), backend=backend,
                                     sentences=list(fake_result_sentences), elapsed_ms=5.0)

        real_sleep = asyncio.sleep

        async def quick_sleep(*_a, **_k):
            await real_sleep(0)

        async def go():
            with patch.object(main.think_core, "run", fake_run), \
                 patch.object(main, "get_plugin_config", return_value={}), \
                 patch.object(main.asyncio, "sleep", quick_sleep):
                await j._run_think(self._fc(), {"query": "why", "include_screen": False})
        asyncio.run(go())

    def test_short_answer_single_response(self):
        j = self._bare()
        self._run(j, ["Because Rayleigh scattering favours blue."])
        self.assertEqual(len(j.session.tool_responses), 1)
        fr = j.session.tool_responses[0][0]
        self.assertEqual(fr.id, "call-1"); self.assertEqual(fr.name, "think")
        self.assertTrue(fr.response["ok"])
        self.assertEqual(fr.response["detail"], "Because Rayleigh scattering favours blue.")
        self.assertEqual(str(fr.scheduling), "FunctionResponseScheduling.WHEN_IDLE")
        self.assertEqual(j.session.client_content, [])
        j._current_turn.add_tool_span.assert_called_once()
        self.assertEqual(j._current_turn.add_tool_span.call_args[0][0], "think:gemini")

    def test_long_answer_two_parts(self):
        j = self._bare()
        sents = ["First point, stated plainly.", "Second point with a bit more detail attached.",
                 "Third point.", "Fourth and final point, sir."]
        self._run(j, sents)
        self.assertEqual(len(j.session.tool_responses), 1)
        first = j.session.tool_responses[0][0].response["detail"]
        self.assertTrue(first.startswith("First point"))
        self.assertIn("Second point", first)
        self.assertEqual(len(j.session.client_content), 1)
        follow = j.session.client_content[0]["parts"][0]["text"]
        self.assertTrue(follow.startswith("[THINK, continued]"))
        self.assertIn("Third point.", follow)
        self.assertIn("Fourth and final point", follow)
        self.assertNotIn("First point", follow)

    def test_failure_is_ok_false_and_registered(self):
        j = self._bare()
        self._run(j, [], raise_exc=RuntimeError("All backends failed"))
        fr = j.session.tool_responses[0][0]
        self.assertFalse(fr.response["ok"])
        self.assertIn("All backends failed", fr.response["detail"])
        self.assertTrue(j._false_success.pending)

    def test_empty_query(self):
        import main
        j = self._bare()
        asyncio.run(j._run_think(self._fc(), {"query": "   "}))
        self.assertFalse(j.session.tool_responses[0][0].response["ok"])

    def test_execute_tool_returns_none_for_think(self):
        import main
        j = self._bare()
        started = {}
        j._start_think = lambda fc, args: started.update(args=args)
        fc = self._fc(); fc.args = {"query": "q"}
        out = asyncio.run(j._execute_tool(fc))
        self.assertIsNone(out)
        self.assertEqual(started["args"], {"query": "q"})

    def test_declaration_is_non_blocking(self):
        import main
        decl = next(d for d in main.TOOL_DECLARATIONS if d["name"] == "think")
        self.assertEqual(decl["behavior"], "NON_BLOCKING")
        self.assertEqual(decl["parameters"]["required"], ["query"])


if __name__ == "__main__":
    unittest.main()
