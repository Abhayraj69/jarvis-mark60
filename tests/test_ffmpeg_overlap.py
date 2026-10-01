"""opencv-python and PyAV (pulled in by faster-whisper) each bundle FFmpeg, and
on macOS both copies of libavdevice define the ObjC classes AVFFrameReceiver
and AVFAudioReceiver. The runtime warns about it at launch. The warning is
harmless only while nothing opens FFmpeg's avfoundation capture device, the
one place those classes are used. This test fails if code that could open it
is added; route camera/mic capture through OpenCV's own backend or
sounddevice instead."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", "venv", "tests", "__pycache__", ".git", "build", "dist"}

FORBIDDEN = [
    (re.compile(r"\bCAP_FFMPEG\b"), "OpenCV's FFmpeg backend"),
    (re.compile(r"""["']avfoundation["']""", re.I), "FFmpeg's avfoundation device"),
    (re.compile(r"^\s*(import av\b|from av\b)", re.M), "PyAV used directly"),
    (re.compile(r"\bav\.open\("), "PyAV used directly"),
]


def _sources():
    for path in ROOT.rglob("*.py"):
        if SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
            continue
        yield path


class FfmpegOverlapTest(unittest.TestCase):
    def test_nothing_opens_ffmpeg_capture_devices(self):
        hits = []
        for path in _sources():
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern, why in FORBIDDEN:
                for m in pattern.finditer(text):
                    line = text.count("\n", 0, m.start()) + 1
                    hits.append(f"{path.relative_to(ROOT)}:{line} ({why})")
        self.assertEqual(hits, [], "see this test's docstring:\n" + "\n".join(hits))


if __name__ == "__main__":
    unittest.main()
