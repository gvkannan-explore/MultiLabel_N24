"""Modal run-file contracts with a fake CLI; no network or paid jobs."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ModalRunFileTests(unittest.TestCase):
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
            "fake-bin",
        ):
            (self.root / folder).mkdir(parents=True)
        for relative in (
            "bins/train/encoder_smoke_modal.sh",
            "bins/experiments/run_a1.sh",
            "bins/experiments/run_a1_modal.sh",
        ):
            shutil.copy2(ROOT / relative, self.root / relative)
        (self.root / ".venv/bin/python").symlink_to(sys.executable)
        self.capture = self.root / "commands.jsonl"
        fake = self.root / "fake-bin/uv"
        fake.write_text(
            f"#!{sys.executable}\n"
            + """
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["A1_TEST_COMMAND_LOG"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if "modal" in args and "--name" in args and os.environ.get("A1_TEST_MODAL_FAIL"):
    raise SystemExit(17)
if "volume" in args and "get" in args:
    target = Path(args[-1]) / args[-2]
    target.mkdir(parents=True)
    (target / "metrics.json").write_text(json.dumps({"remote": True}))
"""
        )
        fake.chmod(0o755)

    def run_script(
        self, relative: str, *args: str, fail: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.root / relative), *args],
            cwd=self.root,
            env={
                **os.environ,
                "PATH": str(self.root / "fake-bin") + os.pathsep + os.environ["PATH"],
                "A1_TEST_COMMAND_LOG": str(self.capture),
                "A1_TEST_MODAL_FAIL": "1" if fail else "",
            },
            capture_output=True,
            text=True,
        )

    def commands(self) -> list[list[str]]:
        return [json.loads(line) for line in self.capture.read_text().splitlines()]

    def test_modal_smoke_submits_five_gpu_steps_and_prints_downloaded_metrics(
        self,
    ) -> None:
        result = self.run_script(
            "bins/train/encoder_smoke_modal.sh",
            "answerdotai/ModernBERT-base",
            "a1-modal-test",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        self.assertEqual(len(commands), 2)
        submitted = commands[0]
        self.assertIn("n24.train.modal_train", submitted)
        flags = submitted[submitted.index("--args") + 1].split()
        self.assertEqual(
            flags[flags.index("--model") + 1], "answerdotai/ModernBERT-base"
        )
        self.assertEqual(flags[flags.index("--max-steps") + 1], "5")
        self.assertNotIn("--use-cpu", flags)
        self.assertIn("n24-runs", commands[1])
        self.assertTrue((self.root / "runs_modal/a1-modal-test/metrics.json").exists())
        self.assertIn('"remote": true', result.stdout)

    def test_model_argument_cannot_inject_extra_remote_flags(self) -> None:
        result = self.run_script(
            "bins/train/encoder_smoke_modal.sh", "a/model --use-cpu", "unsafe-model"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_successful_ci_routes_experiment_to_modal_and_downloads(self) -> None:
        (self.root / "bins/ci_tasks/perform_ci.sh").write_text("#!/bin/bash\nexit 0\n")
        result = self.run_script(
            "bins/experiments/run_a1_modal.sh", "1a", "a1-modal-gated"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.commands()), 2)
        self.assertTrue((self.root / "runs_modal/a1-modal-gated/metrics.json").exists())

    def test_remote_failure_stops_before_download(self) -> None:
        result = self.run_script(
            "bins/train/encoder_smoke_modal.sh", "a/model", "a1-modal-fail", fail=True
        )
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(len(self.commands()), 1)

    def test_existing_local_run_is_preserved_without_submission(self) -> None:
        (self.root / "runs_modal/existing").mkdir(parents=True)
        result = self.run_script(
            "bins/train/encoder_smoke_modal.sh", "a/model", "existing"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_invalid_run_name_is_rejected_before_submission(self) -> None:
        result = self.run_script(
            "bins/train/encoder_smoke_modal.sh", "a/model", "../escape"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_modal_experiment_respects_failed_ci_before_submission(self) -> None:
        (self.root / "bins/ci_tasks/perform_ci.sh").write_text("#!/bin/bash\nexit 23\n")
        result = self.run_script(
            "bins/experiments/run_a1_modal.sh", "1a", "a1-modal-blocked"
        )
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
