"""Run-file contracts checked without downloading or training a model."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class A1RunFileTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        for folder in (
            "bins/train",
            "bins/experiments",
            "bins/ci_tasks",
            ".venv/bin",
            "datasets/prepared",
        ):
            (self.root / folder).mkdir(parents=True)
        for relative in ("bins/train/encoder_smoke.sh", "bins/experiments/run_a1.sh"):
            shutil.copy2(ROOT / relative, self.root / relative)
        python = self.root / ".venv/bin/python"
        python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
        python.chmod(0o755)

    def run_script(self, relative: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.root / relative), *args],
            cwd=self.root,
            capture_output=True,
            text=True,
        )

    def test_smoke_uses_five_cpu_steps_and_the_requested_model(self) -> None:
        result = self.run_script(
            "bins/train/encoder_smoke.sh",
            "answerdotai/ModernBERT-base",
            "experiment_runs/new-smoke",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertEqual(args[args.index("--model") + 1], "answerdotai/ModernBERT-base")
        self.assertEqual(args[args.index("--max-steps") + 1], "5")
        self.assertEqual(args[args.index("--batch-size") + 1], "2")
        self.assertIn("--use-cpu", args)

    def test_smoke_refuses_an_existing_run(self) -> None:
        (self.root / "experiment_runs/existing").mkdir(parents=True)
        result = self.run_script(
            "bins/train/encoder_smoke.sh", "a/model", "experiment_runs/existing"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)
        self.assertTrue((self.root / "experiment_runs/existing").exists())

    def test_failed_ci_stops_experiment_before_training(self) -> None:
        (self.root / "bins/ci_tasks/perform_ci.sh").write_text("#!/bin/bash\nexit 23\n")
        result = self.run_script(
            "bins/experiments/run_a1.sh", "1a", "experiment_runs/new-smoke"
        )
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertNotIn("--max-steps", result.stdout)

    def test_ultrabert_requires_the_modernbert_run_argument(self) -> None:
        result = self.run_script(
            "bins/experiments/run_a1.sh", "1b", "experiment_runs/ultrabert"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MODERNBERT_RUN", result.stderr)


if __name__ == "__main__":
    unittest.main()
