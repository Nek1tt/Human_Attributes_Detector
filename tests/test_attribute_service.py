from __future__ import annotations

import time
import unittest

from PIL import Image

from silhouette_detector.attribute_service import AttributeService
from silhouette_detector.attributes.base import AttributeBackend


class FakeBackend(AttributeBackend):
    calls = 0

    @property
    def name(self) -> str:
        return "fake"

    def predict(self, image: Image.Image) -> dict[str, str]:
        del image
        self.calls += 1
        return {"gender": "мужчина"}


class AttributeServiceTests(unittest.TestCase):
    def test_one_track_is_submitted_only_once(self) -> None:
        backend = FakeBackend()
        service = AttributeService(backend)
        image = Image.new("RGB", (32, 64))
        try:
            self.assertTrue(service.submit("job", 7, image))
            self.assertFalse(service.submit("job", 7, image))
            result = None
            for _ in range(100):
                result = service.get("job", 7)
                if result is not None:
                    break
                time.sleep(0.005)
            self.assertIsNotNone(result)
            self.assertEqual(result["gender"], "мужчина")
            self.assertEqual(backend.calls, 1)
        finally:
            service.close()


if __name__ == "__main__":
    unittest.main()
