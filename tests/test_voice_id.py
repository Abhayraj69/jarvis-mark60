"""Voice ID gate (core/voice_id.py), with a stand-in embedder.

Two "voices" are sine tones: the owner at 200 Hz, someone else at 700 Hz.
The fake embedder reports which band holds the most energy, so the gate's
own logic — holding, releasing, silencing, timing — is what is under test,
not the neural model.
"""
import unittest

import numpy as np

from core import voice_id
from core.voice_id import SAMPLE_RATE, VoiceGate

BLOCK = 1024                      # what the mic callback delivers
OWNER_HZ, OTHER_HZ = 200, 700


def tone(hz: float, seconds: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * 32767 * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.int16)


def fake_embed(audio: np.ndarray) -> np.ndarray:
    a = voice_id._to_float(audio)
    spec = np.abs(np.fft.rfft(a))
    freqs = np.fft.rfftfreq(len(a), 1 / SAMPLE_RATE)
    low = spec[(freqs > 100) & (freqs < 400)].sum()
    high = spec[(freqs > 500) & (freqs < 900)].sum()
    return np.array([low, high], dtype=np.float32)


OWNER_PROFILE = np.array([1.0, 0.0], dtype=np.float32)


def run_gate(gate: VoiceGate, audio: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, len(audio), BLOCK):
        for b in gate.process(audio[i:i + BLOCK]):
            out.append(np.frombuffer(b, dtype=np.int16))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)


def make_gate(decisions=None):
    return VoiceGate(fake_embed, OWNER_PROFILE, threshold=0.6,
                     on_decision=(lambda ok, s: decisions.append(ok)) if decisions is not None else None)


class GateTest(unittest.TestCase):
    def test_owner_speech_passes_through_intact(self):
        audio = np.concatenate([silence(0.5), tone(OWNER_HZ, 2.0), silence(1.0)])
        out = run_gate(make_gate(), audio)
        self.assertEqual(len(out), len(audio), "no audio lost or added")
        self.assertTrue(np.array_equal(out, audio))

    def test_other_voice_is_silenced_but_keeps_its_length(self):
        audio = np.concatenate([silence(0.5), tone(OTHER_HZ, 2.0), silence(1.0)])
        out = run_gate(make_gate(), audio)
        self.assertEqual(len(out), len(audio))
        self.assertEqual(int(np.abs(out).max()), 0)

    def test_owner_after_stranger_is_heard(self):
        decisions = []
        audio = np.concatenate([tone(OTHER_HZ, 1.5), silence(1.0), tone(OWNER_HZ, 1.5), silence(1.0)])
        out = run_gate(make_gate(decisions), audio)
        self.assertEqual(decisions, [False, True])
        owner_part = out[int(2.5 * SAMPLE_RATE) - BLOCK:]
        self.assertGreater(int(np.abs(owner_part).max()), 1000)

    def test_one_decision_per_utterance(self):
        decisions = []
        run_gate(make_gate(decisions), np.concatenate([tone(OWNER_HZ, 4.0), silence(1.0)]))
        self.assertEqual(decisions, [True])

    def test_short_word_is_still_checked(self):
        # "stop" is shorter than VERIFY_SECONDS; it is decided when it ends.
        decisions = []
        out = run_gate(make_gate(decisions), np.concatenate([tone(OTHER_HZ, 0.4), silence(1.2)]))
        self.assertEqual(decisions, [False])
        self.assertEqual(int(np.abs(out).max()), 0)

    def test_background_hum_is_not_speech(self):
        decisions = []
        hum = tone(OTHER_HZ, 3.0, amp=0.004)
        out = run_gate(make_gate(decisions), hum)
        self.assertEqual(decisions, [])
        self.assertTrue(np.array_equal(out, hum))


class WakeCheckTest(unittest.TestCase):
    def _feed(self, gate, audio):
        for i in range(0, len(audio), BLOCK):
            gate.note_recent(audio[i:i + BLOCK])

    def test_owner_wake_phrase(self):
        g = make_gate()
        self._feed(g, np.concatenate([silence(3.0), tone(OWNER_HZ, 1.0)]))
        self.assertTrue(g.recent_is_owner())

    def test_stranger_wake_phrase(self):
        g = make_gate()
        self._feed(g, np.concatenate([tone(OWNER_HZ, 3.0), tone(OTHER_HZ, 1.5)]))
        self.assertFalse(g.recent_is_owner(), "only the last moments count")


class EnrollTest(unittest.TestCase):
    def test_profile_from_speech(self):
        prof = voice_id.build_profile(fake_embed, tone(OWNER_HZ, 8.0))
        self.assertGreater(voice_id.cosine(prof, OWNER_PROFILE), 0.99)

    def test_silence_is_rejected(self):
        with self.assertRaises(ValueError):
            voice_id.build_profile(fake_embed, silence(12.0))


class WakeIntegrationTest(unittest.TestCase):
    def test_stranger_cannot_wake_jarvis(self):
        from tests.live_harness import make_jarvis
        j = make_jarvis(awake=False)
        g = make_gate()
        WakeCheckTest._feed(self, g, tone(OTHER_HZ, 1.5))
        j._voice_gate = g
        j._on_wake_detected()
        self.assertFalse(j._awake)

    def test_owner_wakes_jarvis(self):
        from tests.live_harness import make_jarvis
        j = make_jarvis(awake=False)
        g = make_gate()
        WakeCheckTest._feed(self, g, tone(OWNER_HZ, 1.5))
        j._voice_gate = g
        j._on_wake_detected()
        self.assertTrue(j._awake)


if __name__ == "__main__":
    unittest.main()
