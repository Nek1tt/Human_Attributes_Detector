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
            context.artifacts["checkpoint"],
            vision_directory=context.artifacts.get("vision_source")
            or context.artifacts.get("original_visual"),
        )
        self.transform = image_transform(preprocess)
        self.original_encoder = None
        if getattr(self.model, "architecture", None) == "original_transformer":
            from .original_visual import OriginalVisualEncoder

            self.original_encoder = OriginalVisualEncoder(self.model.visual_directory, "cpu")

    def predict(self, context, model_input, params=None):
        images = []
        for encoded in model_input["image"]:
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                images.append(
                    image.convert("RGB")
                    if self.original_encoder is not None
                    else self.transform(image.convert("RGB"))
                )
        with torch.inference_mode():
            if self.original_encoder is not None:
                tokens = self.original_encoder.extract(images)
                logits = torch.cat([self.model(tensor.unsqueeze(0)) for tensor in tokens])
                return decode(logits, labels=self.model.labels)
            return decode(self.model(torch.stack(images)), labels=self.model.labels)
