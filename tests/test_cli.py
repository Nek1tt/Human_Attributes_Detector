"""Check the installed project's main CLI entry point."""

import unittest

from click.testing import CliRunner

from silhouette_detector.commands import main


class CliTests(unittest.TestCase):
    def test_help_lists_course_commands(self):
        result = CliRunner().invoke(main, ["--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        for operation in ("train", "infer", "evaluate", "original-download", "export", "package"):
            self.assertIn(operation, result.output)
