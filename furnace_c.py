"""C variant with a single 8-dimensional inter-group attention head."""
import math

import numpy as np
import torch
from torch import nn

from furnace import GROUPS, SCHEMA, FurnaceClassifier

C_PARAMETERS = 4267


class SingleHeadGroupAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(16, 8)
        self.key = nn.Linear(16, 8)
        self.value = nn.Linear(16, 8)
        self.output = nn.Linear(8, 16)
        self.register_buffer("diagonal_mask", torch.eye(len(GROUPS), dtype=torch.bool), persistent=False)

    def forward(self, z):
        q, k, v = self.query(z), self.key(z), self.value(z)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.query.out_features)
        weights = scores.masked_fill(self.diagonal_mask, float("-inf")).softmax(dim=-1)
        return self.output(weights @ v), weights


class FurnaceClassifierC(FurnaceClassifier):
    def __init__(self):
        # Initialize the same encoder, normalization, and classifier as A/B.
        super().__init__("A")
        self.variant = "C"
        self.attention = SingleHeadGroupAttention()

    def forward(self, x):
        z = self.encode(x)
        message, _ = self.attention(z)
        return self.head(self.norm(z + message).flatten(start_dim=1))


def load_c_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["variant"] != "C" or checkpoint["schema"] != SCHEMA:
        raise ValueError(f"{path}: 需要与当前字段清单匹配的 C 版权重")
    scaler = checkpoint["scaler"]
    mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
    width = len(SCHEMA["features"])
    if (mean.shape != (width,) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError(f"{path}: scaler 无效")
    model = FurnaceClassifierC()
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), mean, scale


@torch.inference_mode()
def verify_c_forward(model, batch):
    model.eval()
    assert sum(param.numel() for param in model.parameters() if param.requires_grad) == C_PARAMETERS
    for sample in (batch[:1], batch):
        z = model.encode(sample)
        message, weights = model.attention(z)
        assert z.shape == message.shape == (len(sample), 6, 16)
        assert weights.shape == (len(sample), 6, 6)
        assert torch.count_nonzero(weights.diagonal(dim1=-2, dim2=-1)).item() == 0
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
        logits = model(sample)
        assert logits.shape == (len(sample), 3) and logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
    torch.testing.assert_close(model(batch[:1])[0], logits[0], rtol=1e-5, atol=1e-6)
    changed = batch.clone()
    changed[1:] += 2
    torch.testing.assert_close(model(changed)[0], logits[0], rtol=1e-5, atol=1e-6)
