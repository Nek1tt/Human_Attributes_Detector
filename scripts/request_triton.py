"""Send one cropped person image to Triton over its HTTP API."""

import json
import urllib.request
from pathlib import Path

import click
from PIL import Image

from silhouette_detector.market_model import decode, image_transform, load_model


@click.command()
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option("--checkpoint", default="models/peta_attributes.pt", type=click.Path(path_type=Path))
@click.option("--url", default="http://127.0.0.1:8000/v2/models/attributes/infer")
def main(image, checkpoint, url):
    import torch

    model, preprocess = load_model(checkpoint)
    with Image.open(image) as source:
        tensor = image_transform(preprocess)(source.convert("RGB")).unsqueeze(0)
    body = {
        "inputs": [
            {
                "name": "images",
                "shape": list(tensor.shape),
                "datatype": "FP32",
                "data": tensor.flatten().tolist(),
            }
        ]
    }
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        output = json.load(response)["outputs"][0]
    logits = torch.tensor(output["data"]).reshape(output["shape"])
    print(json.dumps(decode(logits, labels=model.labels)[0]))


if __name__ == "__main__":
    main()
