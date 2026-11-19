"""Use the trained PETA classifiers in the video pipeline."""

from threading import Lock

import torch

from ..device import resolve_torch_device
from ..model import decode, image_transform, load_model
from .base import AttributeBackend
from .labels import COLORS_RU


class PetaBackend(AttributeBackend):
    def __init__(self, checkpoint, device="auto"):
        self.device = resolve_torch_device(device)
        self.model, preprocess = load_model(checkpoint, self.device)
        self.transform = image_transform(preprocess)
        self.original_encoder = None
        if getattr(self.model, "architecture", None) == "original_transformer":
            from ..original_visual import OriginalVisualEncoder

            self.original_encoder = OriginalVisualEncoder(self.model.visual_directory, self.device)
        self.lock = Lock()

    @property
    def name(self):
        return "peta"

    def predict(self, image):
        with (
            self.lock,
            torch.inference_mode(),
            torch.autocast(
                torch.device(self.device).type,
                dtype=torch.bfloat16,
                enabled=self.device.startswith("cuda"),
            ),
        ):
            tensor = (
                self.original_encoder.extract([image])[0].unsqueeze(0)
                if self.original_encoder is not None
                else self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            )
            attributes = decode(self.model(tensor).cpu(), labels=self.model.labels)[0]
        translations = {
            **COLORS_RU,
            "male": "мужчина",
            "female": "женщина",
            "less15": "младше 15 лет",
            "16-30": "16-30 лет",
            "31-45": "31-45 лет",
            "46-60": "46-60 лет",
            "over60": "старше 60 лет",
            "jacket": "куртка",
            "sweater": "свитер",
            "suit": "костюм",
            "tshirt": "футболка",
            "other": "другая одежда",
            "trousers": "брюки",
            "shorts": "шорты",
            "skirt": "юбка",
            "capri": "капри",
            "yes": "да",
            "no": "отсутствует",
            "unknown": "не определен",
        }
        return self.normalize({name: translations[value] for name, value in attributes.items()})
