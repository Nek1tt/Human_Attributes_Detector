from __future__ import annotations

import hashlib
import json
import re
import tempfile
import tomllib
import unittest
from pathlib import Path

from silhouette_detector.model_store import (
    PINNED_MINICPM_REPO,
    PINNED_MINICPM_REVISION,
    write_snapshot_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "reproducibility" / "windows-cuda-baseline.json"


class ReproducibilityConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        cls.pyproject = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )

    def test_minicpm_source_constants_match_baseline(self) -> None:
        self.assertEqual(PINNED_MINICPM_REPO, self.baseline["minicpm"]["repository"])
        self.assertEqual(PINNED_MINICPM_REVISION, self.baseline["minicpm"]["revision"])

    def test_validated_minicpm_packages_match_pyproject_extras(self) -> None:
        optional = self.pyproject["project"]["optional-dependencies"]
        for extra in ("minicpm", "minicpm-int4"):
            pinned = {
                requirement.split("==", 1)[0].lower(): requirement.split("==", 1)[1]
                for requirement in optional[extra]
                if "==" in requirement and "[" not in requirement.split("==", 1)[0]
            }
            for name, version in self.baseline["validated_packages"].items():
                if name == "auto-gptq":
                    continue
                self.assertEqual(
                    pinned.get(name),
                    version,
                    f"{extra} does not pin validated {name}=={version}",
                )
            self.assertEqual(pinned.get("torchaudio"), "2.8.0")

    def test_requirements_files_reference_existing_extras(self) -> None:
        optional = self.pyproject["project"]["optional-dependencies"]
        for path in sorted((PROJECT_ROOT / "requirements").glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            self.assertTrue(text, f"empty requirements file: {path}")
            match = re.fullmatch(r"-e \.\[([a-z0-9,-]+)]", text)
            self.assertIsNotNone(match, f"unexpected requirements entry: {path}: {text!r}")
            for extra in match.group(1).split(","):
                self.assertIn(extra, optional, f"unknown pyproject extra {extra!r} in {path}")

    def test_docker_bases_match_digest_pins(self) -> None:
        expected = {
            "Dockerfile": self.baseline["docker"]["cpu_base"],
            "Dockerfile.gpu": self.baseline["docker"]["gpu_base"],
        }
        for filename, image in expected.items():
            text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
            match = re.search(r"(?m)^FROM\s+([^\s]+)", text)
            self.assertIsNotNone(match)
            self.assertEqual(match.group(1), image)
            self.assertRegex(image, r"@sha256:[0-9a-f]{64}$")
            requirement = (
                "requirements/cpu.txt"
                if filename == "Dockerfile"
                else "requirements/gpu.txt"
            )
            self.assertIn(requirement, text)

    def test_windows_installer_contains_exact_baseline_pins(self) -> None:
        installer = (PROJECT_ROOT / "scripts" / "setup_minicpm_int4_windows.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn('"3.11.9"', installer)
        self.assertIn(self.baseline["autogptq"]["repository"], installer)
        self.assertIn(self.baseline["autogptq"]["commit"], installer)
        self.assertIn('"torchaudio==2.8.0"', installer)
        self.assertIn("release 12\\.8", installer)

    def test_readme_exposes_model_identity_and_capture_command(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(self.baseline["minicpm"]["revision"], readme)
        self.assertIn(self.baseline["yolo"]["checkpoint_sha256"], readme)
        self.assertIn(self.baseline["transformer"]["checkpoint_sha256"], readme)
        self.assertIn("capture_reproducibility.ps1", readme)
        self.assertIn("export_windows_lock.ps1", readme)
        self.assertIn("verify_clean_cpu_install.ps1", readme)
        self.assertIn("run_minicpm_video_test.ps1", readme)

    def test_cpu_and_video_scripts_are_wired_to_real_system_test(self) -> None:
        setup = (PROJECT_ROOT / "scripts" / "setup_cpu_windows.ps1").read_text(
            encoding="utf-8"
        )
        clean = (PROJECT_ROOT / "scripts" / "verify_clean_cpu_install.ps1").read_text(
            encoding="utf-8"
        )
        video = (PROJECT_ROOT / "scripts" / "run_video_test.ps1").read_text(
            encoding="utf-8"
        )
        system_test = (PROJECT_ROOT / "tests" / "test_system.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("[switch]$RequireNew", setup)
        self.assertIn('"torch==2.8.0"', setup)
        self.assertIn('"https://download.pytorch.org/whl/cpu"', setup)
        self.assertIn("setup_cpu_windows.ps1", clean)
        self.assertIn("export_windows_lock.ps1", clean)
        self.assertIn("run_video_test.ps1", clean)
        self.assertIn('[string]$ReuseCleanVenv = ""', clean)
        self.assertIn("seedLockHash", clean)
        self.assertIn("cleanLockHash", clean)
        self.assertIn("tests\\test_system.py", video)
        self.assertIn("YOLO checkpoint SHA256 mismatch", video)
        self.assertIn("checkpoint SHA256 mismatch", video)
        self.assertIn("python_files_sha256_after_patch", video)
        self.assertIn('choices=("resnet", "minicpm", "none")', system_test)
        self.assertIn("HAD_MINICPM_MODEL_DIR", system_test)
        self.assertIn('"HAD_TEST_PROFILE": args.device', system_test)
        self.assertIn('"HAD_REQUIRE_CUDA_TESTS": (', system_test)

        device_tests = (PROJECT_ROOT / "tests" / "test_model_devices.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def test_resnet_checkpoint_runs_on_cpu", device_tests)
        self.assertIn("def test_resnet_checkpoint_runs_on_cuda", device_tests)
        self.assertIn("def test_transformer_checkpoint_runs_on_cpu", device_tests)
        self.assertIn("def test_transformer_checkpoint_runs_on_cuda", device_tests)
        self.assertNotIn("test_transformer_checkpoint_runs_on_cpu_and_cuda", device_tests)

    def test_all_recorded_hashes_are_sha256(self) -> None:
        hashes = list(
            self.baseline["minicpm"]["python_files_sha256_after_patch"].values()
        ) + [
            self.baseline["minicpm"]["config_sha256"],
            self.baseline["yolo"]["checkpoint_sha256"],
            self.baseline["transformer"]["checkpoint_sha256"],
            self.baseline["resnet"]["checkpoint_sha256"],
            self.baseline["autogptq"]["tracked_diff_sha256"],
            self.baseline["locks"]["windows_cuda_py311"]["sha256"],
            self.baseline["locks"]["windows_cpu_py311"]["sha256"],
        ]
        for digest in hashes:
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def _lock_versions(self, lock_name: str) -> dict[str, str]:
        lock = self.baseline["locks"][lock_name]
        lock_path = PROJECT_ROOT / lock["path"]
        payload = lock_path.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), lock["sha256"])

        text = payload.decode("utf-8-sig")
        pins = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
            and not line.startswith("#")
            and not line.startswith("--")
        ]
        self.assertEqual(len(pins), lock["exact_package_pins"])
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s]+", pin) for pin in pins))
        self.assertFalse(any(" @ " in pin or pin.startswith("-e ") for pin in pins))

        return {
            name.lower().replace("_", "-"): version
            for name, version in (pin.split("==", 1) for pin in pins)
        }

    def test_windows_cuda_lock_matches_validated_baseline(self) -> None:
        versions = self._lock_versions("windows_cuda_py311")
        for name, version in self.baseline["validated_packages"].items():
            if name != "auto-gptq":
                self.assertEqual(versions.get(name), version)
        for name in ("torch", "torchvision", "torchaudio"):
            self.assertEqual(versions.get(name), self.baseline["pytorch"][name])
        self.assertIn("onnxruntime-gpu", versions)
        self.assertNotIn("onnxruntime", versions)
        self.assertEqual(
            versions["onnxruntime-gpu"], self.baseline["onnxruntime"]["cuda_version"]
        )

    def test_windows_cpu_lock_matches_validated_baseline(self) -> None:
        versions = self._lock_versions("windows_cpu_py311")
        self.assertEqual(versions.get("torch"), "2.8.0+cpu")
        self.assertEqual(versions.get("torchvision"), "0.23.0+cpu")
        self.assertEqual(
            versions.get("onnxruntime"), self.baseline["onnxruntime"]["cpu_version"]
        )
        self.assertNotIn("onnxruntime-gpu", versions)
        self.assertNotIn("auto-gptq", versions)

    def test_onnxruntime_pins_match_validated_baseline(self) -> None:
        optional = self.pyproject["project"]["optional-dependencies"]
        self.assertIn(
            f"onnxruntime-gpu=={self.baseline['onnxruntime']['cuda_version']}",
            optional["gpu"],
        )
        self.assertIn("onnxruntime>=1.28,<2", optional["cpu"])

        verifier = (PROJECT_ROOT / "scripts" / "verify_minicpm_env.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            f'"onnxruntime-gpu": "{self.baseline["onnxruntime"]["cuda_version"]}"',
            verifier,
        )
        self.assertIn('"CUDAExecutionProvider"', verifier)

    def test_generated_artifacts_are_ignored(self) -> None:
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/test-data/", gitignore)
        self.assertIn("/human-attributes-detector-reproducibility.patch", gitignore)

    def test_snapshot_manifest_patch_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "resampler.py").write_text(
                "from typing import Optional, Tuple\n\nvalue: Optional[int] = None\n",
                encoding="utf-8",
            )
            first = write_snapshot_manifest(
                PINNED_MINICPM_REPO,
                PINNED_MINICPM_REVISION,
                PINNED_MINICPM_REVISION,
                model_dir,
                patch_resampler=True,
            )
            second = write_snapshot_manifest(
                PINNED_MINICPM_REPO,
                PINNED_MINICPM_REVISION,
                PINNED_MINICPM_REVISION,
                model_dir,
                patch_resampler=True,
            )

            self.assertIn(
                "from typing import List, Optional, Tuple",
                (model_dir / "resampler.py").read_text(encoding="utf-8"),
            )
            self.assertTrue(first["patches"][0]["changed_during_this_run"])
            self.assertFalse(second["patches"][0]["changed_during_this_run"])
            self.assertEqual(first["python_files_sha256"], second["python_files_sha256"])


if __name__ == "__main__":
    unittest.main()
