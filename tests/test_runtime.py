import tempfile
import threading
import time
import unittest
from pathlib import Path

from grasph_runtime import wait_if_paused


class RuntimeControlTests(unittest.TestCase):
    def test_pause_sentinel_blocks_until_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            pause_file = Path(directory) / "run.pause"
            pause_file.touch()
            timer = threading.Timer(0.15, pause_file.unlink)
            timer.start()
            started = time.monotonic()
            wait_if_paused(pause_file, poll_interval=0.01)
            timer.join()
            self.assertGreaterEqual(time.monotonic() - started, 0.12)
            self.assertFalse(pause_file.exists())


if __name__ == "__main__":
    unittest.main()
