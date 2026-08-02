import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    from silhouette_detector.attributes.minicpm import _is_gptq_snapshot


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class MiniCPMSnapshotTests(unittest.TestCase):
    def test_detects_gptq_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(
                json.dumps({"quantization_config": {"quant_method": "GPTQ"}}),
                encoding="utf-8",
            )

            self.assertTrue(_is_gptq_snapshot(model_dir))

    def test_does_not_use_directory_name_for_quantization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model-int4-name-only"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            self.assertFalse(_is_gptq_snapshot(model_dir))

    def test_rejects_invalid_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text("not-json", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "not valid JSON"):
                _is_gptq_snapshot(model_dir)


if __name__ == "__main__":
    unittest.main()
