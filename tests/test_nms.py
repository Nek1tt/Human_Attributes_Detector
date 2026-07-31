from __future__ import annotations

import unittest

import numpy as np

from silhouette_detector.nms import class_aware_nms


class NmsTests(unittest.TestCase):
    def test_overlapping_different_classes_are_kept(self) -> None:
        boxes = np.array([[0, 0, 100, 100], [1, 1, 99, 99]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        classes = np.array([0, 1])
        kept = class_aware_nms(boxes, scores, classes, 0.5)
        self.assertEqual(kept.tolist(), [0, 1])

    def test_overlapping_same_class_is_suppressed(self) -> None:
        boxes = np.array([[0, 0, 100, 100], [1, 1, 99, 99]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        classes = np.array([0, 0])
        kept = class_aware_nms(boxes, scores, classes, 0.5)
        self.assertEqual(kept.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
