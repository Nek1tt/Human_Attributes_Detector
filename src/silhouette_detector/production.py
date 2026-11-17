"""Export and package the trained attribute classifiers."""

import importlib.metadata
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from .market_inference import ensure_checkpoint
from .market_model import load_model


def input_spec(model, preprocess):
    if getattr(model, "architecture", None) == "original_transformer":
        return "embeddings", [1, 64, 3584]
    return "images", [3, preprocess["height"], preprocess["width"]]


def export_onnx(config):
    import onnx
    import onnxruntime

    checkpoint = ensure_checkpoint(config.inference.checkpoint)
    model, preprocess = load_model(checkpoint)
    output = Path(config.production.onnx)
    output.parent.mkdir(parents=True, exist_ok=True)
    input_name, shape = input_spec(model, preprocess)
    sample = torch.zeros(1, *shape)
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        torch.onnx.export(
            model,
            sample,
            str(output),
            input_names=[input_name],
            output_names=["logits"],
            dynamic_axes={input_name: {0: "batch"}, "logits": {0: "batch"}},
            opset_version=config.production.opset,
            dynamo=False,
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    onnx.checker.check_model(str(output))
    session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for batch_size in [1, 2]:
        images = torch.randn(batch_size, *shape)
        with torch.inference_mode():
            expected = model(images).numpy()
        actual = session.run(None, {input_name: images.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    print(f"ONNX: {output}; PyTorch/ONNX predictions match for batches 1 and 2")


def package_mlflow(config):
    import mlflow.pyfunc

    from .serving import AttributeServing

    checkpoint = ensure_checkpoint(config.inference.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model_config = payload.get("model_config", {})
    package_names = ["mlflow", "torch", "torchvision", "pillow", "numpy"]
    if model_config.get("architecture") in {"minicpm_vision", "original_transformer"}:
        package_names.extend(["transformers", "huggingface-hub", "safetensors"])
    dependencies = [f"{name}=={importlib.metadata.version(name)}" for name in package_names]
    if importlib.metadata.version("torch").endswith("+cpu"):
        dependencies.insert(0, "--extra-index-url https://download.pytorch.org/whl/cpu")
    if "+cu128" in importlib.metadata.version("torch"):
        dependencies.insert(0, "--extra-index-url https://download.pytorch.org/whl/cu128")
    with tempfile.TemporaryDirectory(dir="var") as temporary:
        artifacts = {"checkpoint": str(checkpoint.resolve())}
        if model_config.get("architecture") == "original_transformer":
            artifacts["original_visual"] = str(Path(model_config["visual_directory"]).resolve())
        if model_config.get("directory"):
            source = Path(temporary) / "vision-source"
            source.mkdir()
            for filename in ["config.json", "modeling_navit_siglip.py"]:
                shutil.copyfile(Path(model_config["directory"]) / filename, source / filename)
            artifacts["vision_source"] = str(source.resolve())
        mlflow.pyfunc.save_model(
            path=config.production.mlflow_model,
            python_model=AttributeServing(),
            artifacts=artifacts,
            code_paths=["src/silhouette_detector"],
            pip_requirements=dependencies,
        )
    print(f"MLflow model: {config.production.mlflow_model}")


def to_tensorrt(config):
    executable = shutil.which(config.production.trtexec)
    if executable is None:
        raise RuntimeError("Install NVIDIA TensorRT and add trtexec to PATH")
    model, preprocess = load_model(ensure_checkpoint(config.inference.checkpoint))
    input_name, dimensions = input_spec(model, preprocess)
    shape = "x".join(map(str, dimensions))
    subprocess.run(
        [
            executable,
            f"--onnx={config.production.onnx}",
            f"--saveEngine={config.production.trt_engine}",
            f"--minShapes={input_name}:1x{shape}",
            f"--optShapes={input_name}:1x{shape}",
            f"--maxShapes={input_name}:{config.production.max_batch_size}x{shape}",
        ],
        check=True,
    )


def package_triton(config):
    model, preprocess = load_model(ensure_checkpoint(config.inference.checkpoint))
    directory = Path(config.production.triton_repository) / "attributes"
    version = directory / "1"
    version.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.production.onnx, version / "model.onnx")
    dimension = sum(len(labels) for labels in model.labels.values())
    input_name, dimensions = input_spec(model, preprocess)
    shape = ", ".join(map(str, dimensions))
    (directory / "config.pbtxt").write_text(
        'name: "attributes"\nplatform: "onnxruntime_onnx"\n'
        f"max_batch_size: {config.production.max_batch_size}\n"
        f'input [{{ name: "{input_name}" data_type: TYPE_FP32 dims: [{shape}] }}]\n'
        f'output [{{ name: "logits" data_type: TYPE_FP32 dims: [{dimension}] }}]\n'
        "instance_group [{ kind: KIND_CPU }]\n",
        encoding="utf-8",
    )
    print(f"Triton repository: {directory.parent}")
