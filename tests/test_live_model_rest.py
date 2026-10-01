"""A Live model resting after failures stays resting across a restart."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from core import gemini


class LiveRestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "live_model_rest.json"
        p = patch.object(gemini, "LIVE_REST_FILE", self.tmp)
        p.start()
        self.addCleanup(p.stop)
        self.saved = dict(gemini._cooldown)
        self.addCleanup(lambda: (gemini._cooldown.clear(), gemini._cooldown.update(self.saved)))
        gemini._cooldown.clear()

    def test_failing_model_is_remembered_after_restart(self):
        model = gemini.LIVE_MODELS[0]
        for _ in range(gemini._INTERNAL_STRIKES):
            gemini.note_live_failure(model, "1011 None. Internal error encountered.")
        self.assertTrue(self.tmp.exists())
        gemini._cooldown.clear()                    # a restart forgets everything…
        gemini._load_live_rest()                    # …until the file is read back
        self.assertTrue(gemini._cooling(model))
        self.assertNotEqual(gemini.live_model(), model)

    def test_expired_rest_is_ignored(self):
        self.tmp.write_text(json.dumps({gemini.LIVE_MODELS[0]: time.time() - 5}))
        gemini._load_live_rest()
        self.assertFalse(gemini._cooling(gemini.LIVE_MODELS[0]))

    def test_unreadable_file_is_harmless(self):
        self.tmp.write_text("not json")
        gemini._load_live_rest()
        self.assertEqual(gemini.live_model(), gemini.LIVE_MODELS[0])
