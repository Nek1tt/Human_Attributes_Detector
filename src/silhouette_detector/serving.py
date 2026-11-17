"""MLflow input: a dataframe with a base64-encoded image in each row."""

import base64
import io

import mlflow.pyfunc
import torch
from PIL import Image

from .market_model import decode, image_transform, load_model


class AttributeServing(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model, preprocess = load_model(
            context.artifacts["checkpoint"], vision_directory=context.artifacts.get("vision_source")
        )
        self.transform = image_transform(preprocess)

    def predict(self, context, model_input, params=None):
        images = []
        for encoded in model_input["image"]:
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                images.append(self.transform(image.convert("RGB")))
        with torch.inference_mode():
            return decode(self.model(torch.stack(images)))
