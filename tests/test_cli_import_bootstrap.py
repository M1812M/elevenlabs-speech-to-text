import os
import subprocess
import sys
import unittest

from cli_support import ROOT


class CliImportBootstrapTests(unittest.TestCase):
    def test_module_cli_runs_without_pythonpath(self) -> None:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)

        result = subprocess.run(
            [sys.executable, "-m", "elevenlabs_toolkit.cli.transform", "--help"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )

        self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

    def test_transcribe_help_does_not_require_elevenlabs_sdk(self) -> None:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)

        result = subprocess.run(
            [sys.executable, "-m", "elevenlabs_toolkit.cli.transcribe", "--help"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )

        self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")


if __name__ == "__main__":
    unittest.main()
