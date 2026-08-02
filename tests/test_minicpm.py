import json

import pytest

from silhouette_detector.attributes.minicpm import _is_gptq_snapshot


def test_detects_gptq_from_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "GPTQ"}}),
        encoding="utf-8",
    )

    assert _is_gptq_snapshot(tmp_path) is True


def test_does_not_use_directory_name_for_quantization(tmp_path):
    model_dir = tmp_path / "model-int4-name-only"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    assert _is_gptq_snapshot(model_dir) is False


def test_rejects_invalid_config_json(tmp_path):
    (tmp_path / "config.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not valid JSON"):
        _is_gptq_snapshot(tmp_path)
