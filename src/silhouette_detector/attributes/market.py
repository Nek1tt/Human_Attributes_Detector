"""Use trained PETA and Market-1501 classifiers in the video pipeline."""

from threading import Lock

import torch

from ..device import resolve_torch_device
from ..market_model import decode, image_transform, load_model
from .base import AttributeBackend

COLORS_RU = {
    "black": "черный",
    "white": "белый",
    "pink": "розовый",
    "purple": "фиолетовый",
    "yellow": "желтый",
    "gray": "серый",
    "grey": "серый",
    "orange": "оранжевый",
    "blue": "синий",
    "green": "зеленый",
    "brown": "коричневый",
    "red": "красный",
}


class MarketBackend(AttributeBackend):
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
        return "peta" if "top_type" in self.model.labels else "market"

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
        if "top_type" in attributes:
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
        bottom_type = (
            "юбка"
            if attributes["lower_type"] == "dress"
            else ("шорты" if attributes["lower_length"] == "short" else "брюки")
        )
        return self.normalize(
            {
                "gender": "мужчина" if attributes["gender"] == "male" else "женщина",
                "top_color": COLORS_RU[attributes["top_color"]],
                "bottom_color": COLORS_RU[attributes["bottom_color"]],
                "bottom_type": bottom_type,
                "headwear": "отсутствует" if attributes["hat"] == "no" else "не определен",
                "backpack": "да" if attributes["backpack"] == "yes" else "отсутствует",
                "handbag": "да" if attributes["handbag"] == "yes" else "отсутствует",
            }
        )
