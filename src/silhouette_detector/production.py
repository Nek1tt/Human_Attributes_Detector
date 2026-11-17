"""Export and package the trained Market-1501 model."""

import importlib.metadata
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .market_inference import ensure_checkpoint
from .market_model import LABELS, load_model


def export_onnx(config):
    import onnx
    import onnxruntime

    checkpoint = ensure_checkpoint(config.inference.checkpoint)
    model, preprocess = load_model(checkpoint)
    output = Path(config.production.onnx)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, preprocess["height"], preprocess["width"])
    torch.onnx.export(
        model,
        sample,
        str(output),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=config.production.opset,
        dynamo=False,
    )
    onnx.checker.check_model(str(output))
    session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for batch_size in [1, 2]:
        images = torch.randn(batch_size, 3, preprocess["height"], preprocess["width"])
        with torch.inference_mode():
            expected = model(images).numpy()
        actual = session.run(None, {"images": images.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    print(f"ONNX: {output}; PyTorch/ONNX predictions match for batches 1 and 2")


def package_mlflow(config):
    import mlflow.pyfunc

    from .serving import AttributeServing

    checkpoint = ensure_checkpoint(config.inference.checkpoint)
    dependencies = [
        f"{name}=={importlib.metadata.version(name)}"
        for name in ["mlflow", "torch", "torchvision", "pillow", "numpy"]
    ]
    if importlib.metadata.version("torch").endswith("+cpu"):
        dependencies.insert(0, "--extra-index-url https://download.pytorch.org/whl/cpu")
    mlflow.pyfunc.save_model(
        path=config.production.mlflow_model,
        python_model=AttributeServing(),
        artifacts={"checkpoint": str(checkpoint.resolve())},
        code_paths=["src/silhouette_detector"],
        pip_requirements=dependencies,
    )
    print(f"MLflow model: {config.production.mlflow_model}")


def to_tensorrt(config):
    executable = shutil.which(config.production.trtexec)
    if executable is None:
        raise RuntimeError("Install NVIDIA TensorRT and add trtexec to PATH")
    _, preprocess = load_model(ensure_checkpoint(config.inference.checkpoint))
    shape = f"3x{preprocess['height']}x{preprocess['width']}"
    subprocess.run(
        [
            executable,
            f"--onnx={config.production.onnx}",
            f"--saveEngine={config.production.trt_engine}",
            f"--minShapes=images:1x{shape}",
            f"--optShapes=images:1x{shape}",
            f"--maxShapes=images:{config.production.max_batch_size}x{shape}",
        ],
        check=True,
    )


def package_triton(config):
    _, preprocess = load_model(ensure_checkpoint(config.inference.checkpoint))
    directory = Path(config.production.triton_repository) / "attributes"
    version = directory / "1"
    version.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.production.onnx, version / "model.onnx")
    dimension = sum(len(labels) for labels in LABELS.values())
    (directory / "config.pbtxt").write_text(
        'name: "attributes"\nplatform: "onnxruntime_onnx"\n'
        f"max_batch_size: {config.production.max_batch_size}\n"
        'input [{ name: "images" data_type: TYPE_FP32 '
        f'dims: [3, {preprocess["height"]}, {preprocess["width"]}] }}]\n'
        f'output [{{ name: "logits" data_type: TYPE_FP32 dims: [{dimension}] }}]\n'
        "instance_group [{ kind: KIND_CPU }]\n",
        encoding="utf-8",
    )
    print(f"Triton repository: {directory.parent}")
