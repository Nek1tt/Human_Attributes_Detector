from __future__ import annotations

import unittest

from silhouette_detector.detection import Detection
from silhouette_detector.tracking import SFSORTTracker


class TrackerTests(unittest.TestCase):
    def test_same_detection_keeps_track_id(self) -> None:
        tracker = SFSORTTracker(640, 480, 10)
        first = tracker.update([Detection((10, 10, 100, 200), 0, 0.9)])
        second = tracker.update([Detection((12, 10, 102, 200), 0, 0.9)])
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0].track_id, second[0].track_id)


if __name__ == "__main__":
    unittest.main()
