"""Shared data contract, model, and checkpoint inference."""
import csv
import json
import math

import numpy as np
import torch
from torch import nn

GROUPS = [
    ["8 冷风流量", "18 冷风压力", "19 冷风压力2", "24 冷风温度",
     "32 鼓风湿度", "21 热风压力", "25 热风温度", "23 实际风速"],
    ["17 富氧压力", "7 富氧流量"],
    ["34 本小时实际喷煤量", "35 上小时实际喷煤量", "12 理论燃烧温度", "10 炉腹煤气量"],
    ["13 顶压", "31 阻力系数"],
    ["3 CO", "4 H2", "5 CO2"],
    ["26 顶温东北", "27 顶温西南", "28 顶温西北", "29 顶温东南", "30 顶温下降管"],
]
FEATURES = [column for group in GROUPS for column in group]
WIDTHS = [len(group) for group in GROUPS]
CLASSES = [0, 1, 2]
SCHEMA = {"features": FEATURES, "groups": GROUPS, "widths": WIDTHS, "classes": CLASSES}
PARAMETERS = {"A": 3715, "B": 4803}


def read_numeric(path, encoding, columns, label_column=None):
    """Read selected fields by name, preserving 1-based data row IDs."""
    records = []
    with path.open(encoding=encoding, newline="") as file:
        reader = csv.reader(file, strict=True)
        header = [value.strip().lstrip("\ufeff").strip() for value in next(reader, [])]
        duplicates = sorted({name for name in header if header.count(name) > 1})
        if duplicates:
            raise ValueError(f"{path}: 重复字段 {duplicates}")
        missing = [name for name in columns if name not in header]
        if missing:
            raise ValueError(f"{path}: 缺少字段 {missing}")
        indices = [header.index(name) for name in columns]
        for row_id, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise ValueError(f"{path}: 数据行 {row_id} 有 {len(row)} 列，表头有 {len(header)} 列")
            values = []
            for name, index in zip(columns, indices):
                raw = row[index]
                try:
                    value = float(raw)
                except ValueError as error:
                    raise ValueError(f"{path}: 数据行 {row_id} 字段 {name!r} 非数值 {raw!r}") from error
                if not math.isfinite(value):
                    raise ValueError(f"{path}: 数据行 {row_id} 字段 {name!r} 非有限值 {raw!r}")
                if name == label_column and value not in CLASSES:
                    raise ValueError(f"{path}: 数据行 {row_id} 字段 {name!r} 需要类别 0/1/2，实际为 {raw!r}")
                values.append(value)
            records.append(values)
    return np.asarray(records, dtype=np.float64).reshape(-1, len(columns)), len(header)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_predictions(path, probabilities, labels=None, folds=None):
    columns = ["row_id", "pred_label", "p0", "p1", "p2"]
    if labels is not None:
        columns.append("true_label")
    if folds is not None:
        columns.append("fold")
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for index, row in enumerate(probabilities):
            record = [index + 1, int(row.argmax()), *row.tolist()]
            if labels is not None:
                record.append(int(labels[index]) if index < len(labels) else "")
            if folds is not None:
                record.append(int(folds[index]))
            writer.writerow(record)


class FurnaceClassifier(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in PARAMETERS:
            raise ValueError(f"Unknown model variant {variant!r}")
        self.variant = variant
        self.encoders = nn.ModuleList(nn.Sequential(nn.Linear(width, 16), nn.GELU()) for width in WIDTHS)
        self.norm = nn.LayerNorm(16)
        self.head = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Dropout(0.10), nn.Linear(32, 3))
        if variant == "B":
            self.attention = nn.MultiheadAttention(16, 2, dropout=0.0, batch_first=True)
            self.register_buffer("diagonal_mask", torch.eye(len(GROUPS), dtype=torch.bool), persistent=False)

    def encode(self, x):
        groups = torch.split(x, WIDTHS, dim=-1)
        return torch.stack([encoder(group) for encoder, group in zip(self.encoders, groups)], dim=1)

    def group_attention(self, z, need_weights=False):
        return self.attention(z, z, z, attn_mask=self.diagonal_mask,
                              need_weights=need_weights, average_attn_weights=False)

    def forward(self, x):
        z = self.encode(x)
        if self.variant == "B":
            message, _ = self.group_attention(z)
            z = z + message
        return self.head(self.norm(z).flatten(start_dim=1))


def standardize(x, mean, scale):
    values = ((x - mean) / scale).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("标准化后出现非有限值")
    return torch.from_numpy(values)


@torch.inference_mode()
def predict(model, x, mean, scale):
    model.eval()
    return model(standardize(x, mean, scale)).softmax(dim=-1).numpy()


def save_checkpoint(path, model, scaler, training):
    torch.save({
        "variant": model.variant,
        "schema": SCHEMA,
        "model_state": model.state_dict(),
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "training": training,
    }, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema"] != SCHEMA:
        raise ValueError(f"{path}: 模型的字段顺序、工艺组或类别与当前实现不一致")
    scaler = checkpoint["scaler"]
    mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
    if (mean.shape != (len(FEATURES),) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError(f"{path}: scaler 无效")
    variant = checkpoint["variant"]
    if variant == "C":
        # C subclasses FurnaceClassifier, so import it after this module is loaded.
        from furnace_c import FurnaceClassifierC

        model = FurnaceClassifierC()
    else:
        model = FurnaceClassifier(variant)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), mean, scale


@torch.inference_mode()
def verify_forward(model, batch):
    """Check output shape, sample isolation and per-head masking."""
    model.eval()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == PARAMETERS[model.variant]
    logits = model(batch)
    single = model(batch[:1])
    assert single.shape == (1, 3) and logits.shape == (len(batch), 3)
    assert logits.dtype == single.dtype == torch.float32
    assert torch.isfinite(logits).all() and torch.isfinite(single).all()
    torch.testing.assert_close(single[0], logits[0], rtol=1e-5, atol=1e-6)
    changed = batch.clone()
    changed[1:] += 2
    torch.testing.assert_close(model(changed)[0], logits[0], rtol=1e-5, atol=1e-6)
    if model.variant == "B":
        _, weights = model.group_attention(model.encode(batch), need_weights=True)
        assert weights.shape == (len(batch), 2, 6, 6)
        assert torch.count_nonzero(weights.diagonal(dim1=-2, dim2=-1)).item() == 0
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    return count
