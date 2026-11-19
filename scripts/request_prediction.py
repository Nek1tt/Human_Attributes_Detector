"""Send one cropped person image to MLflow Serving."""

import base64
import json
import urllib.request
from pathlib import Path

import click


@click.command()
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option("--url", default="http://127.0.0.1:5001/invocations")
@click.option("--check-with", type=click.Path(exists=True, path_type=Path))
@click.option("--device", default="cpu", help="Device for the local comparison model")
def main(image, url, check_with, device):
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    body = {"dataframe_split": {"columns": ["image"], "data": [[encoded]]}}
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    if check_with:
        from silhouette_detector.model import predict_image

        expected = predict_image(check_with, image, device=device)
        if result["predictions"] != [expected]:
            raise ValueError("Server and local predictions differ")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
