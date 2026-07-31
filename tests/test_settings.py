from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from silhouette_detector.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_invalid_backend_is_rejected(self) -> None:
        with patch.dict(os.environ, {"HAD_ATTRIBUTE_BACKEND": "mystery"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_short_api_key_is_rejected(self) -> None:
        with patch.dict(os.environ, {"HAD_API_KEY": "short"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_valid_cpu_settings(self) -> None:
        with patch.dict(
            os.environ,
            {"HAD_DEVICE": "cpu", "HAD_ATTRIBUTE_BACKEND": "none"},
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.attribute_backend, "none")


if __name__ == "__main__":
    unittest.main()
