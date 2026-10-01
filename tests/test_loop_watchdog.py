"""Stall detector (core/loop_watchdog.py) and the mic queue's drop-oldest
behaviour (JarvisLive._enqueue_audio)."""
import asyncio
import time
import unittest

from core import loop_watchdog
from tests.live_harness import make_jarvis


def block_the_loop_for_a_while():
    time.sleep(1.8)


class WatchdogTest(unittest.TestCase):
    def test_names_the_blocking_line(self):
        logs = []

        async def main():
            dog = loop_watchdog.LoopWatchdog(asyncio.get_running_loop(), log=logs.append)
            dog.start()
            await asyncio.sleep(0.6)
            block_the_loop_for_a_while()          # synchronous: freezes the loop
            await asyncio.sleep(1.2)
            dog.stop()

        asyncio.run(main())
        stuck = [l for l in logs if "stuck here" in l]
        self.assertEqual(len(stuck), 1, logs)
        self.assertIn("block_the_loop_for_a_while", stuck[0])
        self.assertTrue(any("free again" in l for l in logs), logs)

    def test_quiet_when_nothing_blocks(self):
        logs = []

        async def main():
            dog = loop_watchdog.LoopWatchdog(asyncio.get_running_loop(), log=logs.append)
            dog.start()
            await asyncio.sleep(1.5)
            dog.stop()

        asyncio.run(main())
        self.assertEqual(logs, [])


class EnqueueTest(unittest.TestCase):
    def test_full_queue_drops_oldest_quietly(self):
        j = make_jarvis()
        j.out_queue = asyncio.Queue(maxsize=3)
        for i in range(5):
            j._enqueue_audio({"data": bytes([i])})
        kept = [j.out_queue.get_nowait()["data"][0] for _ in range(3)]
        self.assertEqual(kept, [2, 3, 4], "the newest audio survives")
        self.assertEqual(j._audio_dropped, 2)
        j._enqueue_audio({"data": b"x"})
        self.assertEqual(j._audio_dropped, 0, "reset once the backlog clears")


if __name__ == "__main__":
    unittest.main()
