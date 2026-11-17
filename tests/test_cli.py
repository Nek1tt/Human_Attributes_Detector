"""Keep the old model download command compatible after moving to Click."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from silhouette_detector.model_store import parse_args


class CliTests(unittest.TestCase):
    def test_help_exits_successfully(self):
        with patch.object(sys, "argv", ["download", "--help"]):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    parse_args()
        self.assertEqual(result.exception.code, 0)

    def test_existing_snapshot_flags_and_path_are_preserved(self):
        with patch.object(
            sys, "argv", ["download", "--existing-snapshot", "--destination", "models/test"]
        ):
            options = parse_args()
        self.assertTrue(options.existing_snapshot)
        self.assertEqual(options.destination, Path("models/test"))


if __name__ == "__main__":
    unittest.main()
