"""Frame bus hold-last rule and RTSP mount selection; no camera, GStreamer or robot."""
import itertools
import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CameraConfig, load
from src.devices import apply_overrides, parse_overrides
from src.frame_bus import FrameBusPublisher, FrameBusStalled, HoldLastReader

_ids = itertools.count()


def bus_name():
    return f"test_hold_{os.getpid()}_{next(_ids)}"


class HoldLastReaderTests(unittest.TestCase):
    def setUp(self):
        self.name = bus_name()
        self.pub = FrameBusPublisher(self.name, height=4, width=6, channels=3)
        self.reader = HoldLastReader(self.name, max_repeats=3)

    def tearDown(self):
        self.reader.close()
        self.pub.close()

    def publish(self, value, ts):
        self.pub.publish(np.full((4, 6, 3), value, dtype=np.uint8), ts)

    def repeat(self, n):
        for _ in range(n):
            self.reader.read()

    def test_nothing_before_the_first_frame_does_not_count(self):
        for _ in range(10):
            self.assertEqual(self.reader.read(), (None, None))
        self.assertEqual(self.reader.repeats, 0)

    def test_last_frame_is_reused_while_no_new_one_arrives(self):
        self.publish(7, 1.0)
        ts, frame = self.reader.read()
        self.assertEqual((ts, int(frame[0, 0, 0])), (1.0, 7))
        for n in (1, 2, 3):
            ts, again = self.reader.read()
            self.assertEqual(ts, 1.0)
            self.assertIs(again, frame)
            self.assertEqual(self.reader.repeats, n)

    def test_stalls_after_max_repeats_naming_camera_and_bus(self):
        self.publish(7, 1.0)
        self.reader.read()
        self.repeat(3)
        with self.assertRaises(FrameBusStalled) as ctx:
            self.reader.read()
        self.assertIn(repr(self.name), str(ctx.exception))
        self.assertIn(f"lab_{self.name}", str(ctx.exception))
        with self.assertRaises(FrameBusStalled):
            self.reader.read()

    def test_fresh_frame_resets_the_count(self):
        self.publish(7, 1.0)
        self.reader.read()
        self.repeat(2)
        self.publish(8, 2.0)
        ts, frame = self.reader.read()
        self.assertEqual((ts, int(frame[0, 0, 0]), self.reader.repeats), (2.0, 8, 0))
        self.repeat(3)
        with self.assertRaises(FrameBusStalled):
            self.reader.read()

    def test_recovers_when_a_fresh_frame_arrives_after_a_stall(self):
        self.publish(7, 1.0)
        self.reader.read()
        self.repeat(3)
        with self.assertRaises(FrameBusStalled):
            self.reader.read()
        self.publish(9, 3.0)
        ts, frame = self.reader.read()
        self.assertEqual((ts, int(frame[0, 0, 0]), self.reader.repeats), (3.0, 9, 0))

    def test_writer_going_away_counts_as_missing_frames(self):
        self.publish(7, 1.0)
        self.reader.read()
        self.pub.close()
        for _ in range(3):
            self.assertEqual(self.reader.read()[0], 1.0)
        with self.assertRaises(FrameBusStalled):
            self.reader.read()


class StreamMountTests(unittest.TestCase):
    def cfg(self, *names, **stream):
        cfg = load("so101")
        cfg.cameras = [CameraConfig(name=n, source=i) for i, n in enumerate(names)]
        cfg.stream.enabled = True
        for key, value in stream.items():
            setattr(cfg.stream, key, value)
        return cfg

    @staticmethod
    def names(mounts):
        return {mount: cam.name for mount, cam in mounts.items()}

    def test_first_camera_on_ai_others_by_name(self):
        self.assertEqual(self.names(self.cfg("wrist", "eye").stream_mounts()), {"ai": "wrist", "eye": "eye"})

    def test_stream_camera_selects_the_ai_mount(self):
        mounts = self.cfg("wrist", "eye", camera="eye").stream_mounts()
        self.assertEqual(self.names(mounts), {"ai": "eye", "wrist": "wrist"})

    def test_setups_that_cannot_stream_raise(self):
        for cfg in (self.cfg(), self.cfg("wrist", camera="eye"), self.cfg("wrist", "ai")):
            with self.assertRaises(ValueError):
                cfg.stream_mounts()
        cfg = self.cfg("wrist")
        cfg.cameras[0].publish_frames = False
        with self.assertRaises(ValueError):
            cfg.stream_mounts()

    def test_cli(self):
        cfg = apply_overrides(load("so101"), parse_overrides(
            ["--cameras.wrist.source=0", "--cameras.eye.source=2", "--stream.enabled=true", "--stream.camera=eye"]))
        self.assertEqual(self.names(cfg.stream_mounts()), {"ai": "eye", "wrist": "wrist"})
        with self.assertRaises(SystemExit):
            apply_overrides(load("so101"), parse_overrides(["--stream.enabled=true"]))


if __name__ == "__main__":
    unittest.main()
