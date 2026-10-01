"""The Live session must survive the server's ~30 s idle drop (1008): a
keepalive whenever nothing has been sent for a while, in any state, and a
quiet immediate reconnect when a working session is dropped anyway."""
import asyncio
import time
import unittest
from unittest.mock import MagicMock


class IdleDropTest(unittest.TestCase):
    def test_1008_on_working_session_is_idle_drop(self):
        import main
        err = "unhandled errors in a TaskGroup | 1008 None. The operation was aborted."
        self.assertTrue(main._is_idle_drop(err, uptime=150.0))

    def test_1008_right_after_connect_is_not(self):
        import main
        self.assertFalse(main._is_idle_drop("1008 None. The operation was aborted.", uptime=2.0))

    def test_other_errors_are_not(self):
        import main
        self.assertFalse(main._is_idle_drop("1011 Internal error", uptime=150.0))


class KeepaliveTest(unittest.TestCase):
    def _jarvis(self, *, awake: bool, idle_for: float):
        import main
        j = object.__new__(main.JarvisLive)
        j.session = object()
        j.out_queue = asyncio.Queue()
        j._last_realtime_send = time.monotonic() - idle_for
        j._wake_enabled = True
        j._awake = awake
        j._speaking_lock = MagicMock()
        j._is_speaking = True          # a long reply playing: mic not streamed
        j._tools_running = 0
        j._tail_active = lambda: False
        j._ensure_voice_gate = lambda: None
        return j

    def _one_tick(self, j):
        import main

        async def run():
            real_sleep = asyncio.sleep
            calls = 0

            async def fake_sleep(_):
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise asyncio.CancelledError
                await real_sleep(0)

            main.asyncio.sleep = fake_sleep
            try:
                await j._run_sleep_watch()
            except asyncio.CancelledError:
                pass
            finally:
                main.asyncio.sleep = real_sleep

        asyncio.run(run())

    def test_keepalive_sent_while_awake_and_idle(self):
        import main
        j = self._jarvis(awake=True, idle_for=main.KEEPALIVE_IDLE_SECONDS + 1)
        self._one_tick(j)
        self.assertEqual(j.out_queue.qsize(), 1)
        self.assertEqual(j.out_queue.get_nowait()["data"], main._KEEPALIVE_SILENCE)

    def test_keepalive_sent_while_asleep_and_idle(self):
        import main
        j = self._jarvis(awake=False, idle_for=main.KEEPALIVE_IDLE_SECONDS + 1)
        self._one_tick(j)
        self.assertEqual(j.out_queue.qsize(), 1)

    def test_no_keepalive_when_audio_is_flowing(self):
        j = self._jarvis(awake=True, idle_for=0.5)
        self._one_tick(j)
        self.assertEqual(j.out_queue.qsize(), 0)


if __name__ == "__main__":
    unittest.main()
