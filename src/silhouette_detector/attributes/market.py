"""Use the new Market-1501 checkpoint in the existing video pipeline."""

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
        self.lock = Lock()

    @property
    def name(self):
        return "market"

    def predict(self, image):
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.lock, torch.inference_mode():
            attributes = decode(self.model(tensor).cpu())[0]
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
