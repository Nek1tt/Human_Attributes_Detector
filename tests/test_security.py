from __future__ import annotations

import unittest

from silhouette_detector.security import is_authorized, validated_video_suffix


class SecurityTests(unittest.TestCase):
    def test_filename_is_reduced_to_an_allowed_suffix(self) -> None:
        self.assertEqual(validated_video_suffix("../../private/movie.MP4"), ".mp4")

    def test_disallowed_extension_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validated_video_suffix("payload.py")

    def test_key_is_required_when_configured(self) -> None:
        self.assertFalse(is_authorized("a" * 24, None, "127.0.0.1", True))
        self.assertTrue(is_authorized("a" * 24, "a" * 24, "203.0.113.1", True))

    def test_keyless_mode_is_loopback_only(self) -> None:
        self.assertTrue(is_authorized(None, None, "127.0.0.1", True))
        self.assertTrue(is_authorized(None, None, "::1", True))
        self.assertFalse(is_authorized(None, None, "192.168.1.10", True))


if __name__ == "__main__":
    unittest.main()
