"""MiniCPM vision encoder with a small Transformer attribute head."""

import importlib.util
import json
import sys
from pathlib import Path

from torch import nn
from torch.nn import functional

from .market_model import LABELS
from .vision_store import verify_source


class EfficientAttention(nn.Module):
    """Use PyTorch SDPA with the same projection weights as MiniCPM."""

    def __init__(self, original):
        super().__init__()
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.out_proj = original.out_proj
        self.num_heads = original.num_heads

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        batch, tokens, dimension = hidden_states.shape
        query, key, value = [
            projection(hidden_states)
            .view(batch, tokens, self.num_heads, dimension // self.num_heads)
            .transpose(1, 2)
            for projection in [self.q_proj, self.k_proj, self.v_proj]
        ]
        attended = functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask
        )
        output = attended.transpose(1, 2).reshape(batch, tokens, dimension)
        return self.out_proj(output), None


def load_vision(directory, pretrained):
    from safetensors.torch import load_file

    directory = Path(directory)
    verify_source(directory)
    spec = importlib.util.spec_from_file_location(
        "minicpm_vision", directory / "modeling_navit_siglip.py"
    )
    source = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = source
    spec.loader.exec_module(source)
    config = json.loads((directory / "config.json").read_text())["vision_config"]
    backbone = source.SiglipVisionTransformer(source.SiglipVisionConfig(**config))
    if pretrained:
        backbone.load_state_dict(load_file(directory / "vision.safetensors"), strict=True)
    for layer in backbone.encoder.layers:
        layer.self_attn = EfficientAttention(layer.self_attn)
    return backbone


class VisionAttributeTransformer(nn.Module):
    def __init__(
        self,
        directory,
        hidden_dim=256,
        layers=2,
        heads=4,
        dropout=0.1,
        unfreeze_layers=0,
        pretrained=True,
    ):
        super().__init__()
        self.backbone = load_vision(directory, pretrained)
        self.unfreeze_layers = unfreeze_layers
        self.backbone.requires_grad_(False)
        if unfreeze_layers:
            for layer in self.backbone.encoder.layers[-unfreeze_layers:]:
                layer.requires_grad_(True)
            self.backbone.post_layernorm.requires_grad_(True)
        self.projection = nn.Linear(self.backbone.config.hidden_size, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, sum(len(labels) for labels in LABELS.values()))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self.unfreeze_layers:
            for layer in self.backbone.encoder.layers[-self.unfreeze_layers :]:
                layer.train(mode)
        return self

    def forward(self, images):
        features = self.backbone(images).last_hidden_state
        tokens = self.projection(features)
        return self.classifier(self.norm(self.transformer(tokens).mean(dim=1)))
