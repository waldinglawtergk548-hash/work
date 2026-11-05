"""Train paired A/B models and generate the prescribed evaluation artifacts."""
import argparse
import platform
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from furnace import (
    CLASSES, FEATURES, PARAMETERS, SCHEMA, FurnaceClassifier, load_checkpoint,
    predict, read_numeric, save_checkpoint, standardize, verify_forward, write_json,
    write_predictions,
)

CONFIG = {
    "device": "cpu", "torch_threads": 1, "optimizer": "AdamW", "lr": 1e-3,
    "weight_decay": 1e-3, "loss": "CrossEntropyLoss", "batch_size": 32,
    "max_epochs": 100, "patience": 15, "folds": 5, "outer_seed": 42,
    "validation_fraction": 0.2, "validation_seed_base": 1000,
    "training_seed_base": 2026, "final_seed": 2042,
}
METRIC_KEYS = ("macro_f1", "accuracy", "recall_0", "recall_1", "recall_2")


def configure_cpu():
    torch.set_num_threads(CONFIG["torch_threads"])
    torch.use_deterministic_algorithms(True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def paired_models(seed):
    seed_all(seed)
    model_a = FurnaceClassifier("A")
    model_b = FurnaceClassifier("B")
    for name in ("encoders", "norm", "head"):
        getattr(model_b, name).load_state_dict(getattr(model_a, name).state_dict())
    return {"A": model_a, "B": model_b}


def split_data(y):
    outer = StratifiedKFold(n_splits=CONFIG["folds"], shuffle=True, random_state=CONFIG["outer_seed"])
    splits = []
    for fold, (development, held_out) in enumerate(outer.split(y, y), start=1):
        train, validation = train_test_split(
            development, test_size=CONFIG["validation_fraction"], stratify=y[development],
            random_state=CONFIG["validation_seed_base"] + fold,
        )
        for indices, counts in zip((train, validation, held_out), ([128, 96, 96], [32, 24, 24], [40, 30, 30])):
            np.testing.assert_array_equal(np.bincount(y[indices], minlength=3), counts)
        np.testing.assert_array_equal(np.sort(np.concatenate((train, validation, held_out))), np.arange(len(y)))
        splits.append({"fold": fold, "train": train, "validation": validation, "held_out": held_out})
    np.testing.assert_array_equal(np.sort(np.concatenate([split["held_out"] for split in splits])), np.arange(len(y)))
    return splits


def score(y, predictions):
    recalls = recall_score(y, predictions, labels=CLASSES, average=None, zero_division=0)
    return {
        "n": len(y),
        "macro_f1": float(f1_score(y, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, predictions)),
        **{f"recall_{label}": float(recall) for label, recall in zip(CLASSES, recalls)},
        "confusion_matrix": confusion_matrix(y, predictions, labels=CLASSES).tolist(),
    }


def fit(model, x, y, seed, epochs, validation=None, *, stop_at_perfect_train=False):
    seed_all(seed)
    loader = DataLoader(
        TensorDataset(x, torch.from_numpy(y)), batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=0, drop_last=False, generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    best_score, best_epoch, best_state = -1.0, 0, None
    best_train_correct = -1
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"{model.variant}: epoch {epoch} loss 非有限")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        record = {"epoch": epoch, "train_loss": total_loss / len(y)}
        history.append(record)
        if validation is not None:
            x_val, y_val = validation
            model.eval()
            with torch.inference_mode():
                val_logits = model(x_val)
            if not torch.isfinite(val_logits).all().item():
                raise FloatingPointError(f"{model.variant}: epoch {epoch} 验证 logits 非有限")
            val_score = float(f1_score(y_val, val_logits.argmax(dim=1).numpy(),
                                       labels=CLASSES, average="macro", zero_division=0))
            record["val_macro_f1"] = val_score
            if val_score > best_score:
                best_score, best_epoch = val_score, epoch
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            if epoch - best_epoch >= CONFIG["patience"]:
                break
        elif stop_at_perfect_train:
            model.eval()
            with torch.inference_mode():
                train_logits = model(x)
            if not torch.isfinite(train_logits).all().item():
                raise FloatingPointError(f"{model.variant}: epoch {epoch} 训练集 logits 非有限")
            correct = int((train_logits.argmax(dim=1).numpy() == y).sum())
            record["train_correct"] = correct
            if correct > best_train_correct:
                best_train_correct, best_epoch = correct, epoch
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            if correct == len(y):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_epoch": best_epoch or epochs,
            "best_val_macro_f1": best_score if validation is not None else None, "history": history}


def load_external(args):
    paths = {"test_input": args.test_input, "test_labels": args.test_labels}
    missing = [f"{key}={path}" for key, path in paths.items() if path is None or not path.is_file()]
    shapes, warnings = {}, []
    x_test, labels = None, None
    if args.test_input is not None and args.test_input.is_file():
        x_test, columns = read_numeric(args.test_input, args.test_input_encoding, FEATURES)
        shapes["test_input"] = [len(x_test), columns]
        if len(x_test) != 200:
            warnings.append(f"测试输入实际形状 {shapes['test_input']}，预期 200 行，等待确认映射")
    if args.test_labels is not None and args.test_labels.is_file():
        raw, columns = read_numeric(args.test_labels, "utf-8-sig", ["Label"], "Label")
        shapes["test_labels"] = [len(raw), columns]
        labels = raw[:, 0].astype(np.int64)
        counts = np.bincount(labels, minlength=3).tolist()
        if len(labels) != 150 or columns != 1 or counts != [50, 50, 50]:
            warnings.append(f"测试标签实际形状 {shapes['test_labels']}，类别计数 {counts}，等待确认映射")
    status = "scored"
    if warnings:
        status = "mapping_confirmation_required"
    elif missing:
        status = "pending"
    external = {
        "status": status, "missing": missing, "warnings": warnings, "actual_shapes": shapes,
        "alignment": "测试标签第 1–150 行按顺序对应测试输入第 1–150 行，后 50 行仅预测",
    }
    return x_test, labels if status == "scored" else None, external


def differences(a, b):
    return {key: b[key] - a[key] for key in METRIC_KEYS}


def write_report(path, report):
    models, external = report["models"], report["external"]
    lines = [
        "# 高炉工况三分类 A/B 对照", "",
        f"训练输入为 `{report['train_file']}`，原始形状 {report['train_shape']}，"
        f"编码 {report['train_encoding']}。按表头选取并重排 24 列。",
        "",
        "A 版 3,715 个可训练参数。B 版 4,803 个，增加 1,088 个参数。"
        "差值衡量加入整层组间注意力后的变化。",
        "",
        "## 内部评估", "",
        "随机分层五折反映本批数据内的区分能力。训练表的类别与采集日期绑定，"
        "跨日期泛化需用独立日期的数据验证。每折仅用 320 条训练记录拟合 scaler，"
        "80 条验证记录用于选择轮次，100 条留出记录用于计算折内指标。",
        "",
        "| 模型 | 集合 | 样本数 | 最佳 epoch | 宏 F1 | 准确率 | 召回 0 | 召回 1 | 召回 2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def metric_row(variant, name, metric, epoch="—"):
        values = " | ".join(f"{metric[key]:.6f}" for key in METRIC_KEYS)
        return f"| {variant} | {name} | {metric['n']} | {epoch} | {values} |"

    def comparison(title, delta):
        recall_key = max(("recall_0", "recall_1", "recall_2"), key=lambda key: abs(delta[key]))
        f1_delta = delta["macro_f1"]
        direction = "上升" if f1_delta > 0 else "下降" if f1_delta < 0 else "持平"
        recalls = "、".join(f"类别 {label} {delta[f'recall_{label}']:+.6f}" for label in CLASSES)
        biggest = (f"变化最大的是类别 {recall_key[-1]}（{delta[recall_key]:+.6f}）。"
                   if delta[recall_key] else "三类召回率持平。")
        return f"{title} B − A 的宏 F1 {direction}（{f1_delta:+.6f}），召回率差值为{recalls}。{biggest}"

    for variant, result in models.items():
        for fold in result["folds"]:
            lines.append(metric_row(variant, f"第 {fold['fold']} 折", fold["held_out"], fold["best_epoch"]))
        lines.append(metric_row(variant, "500 条折外汇总", result["oof"]))
    lines += ["", "五折宏 F1 均值 ± 样本标准差（ddof=1）", ""]
    for variant, result in models.items():
        lines.append(f"- {variant} 版 {result['fold_macro_f1_mean']:.6f} ± {result['fold_macro_f1_sd']:.6f}，"
                     f"最终重训 {result['final_epochs']} 轮。")
    lines += [
        f"- 五折宏 F1 均值 B − A 为 {report['difference']['fold_macro_f1_mean']:+.6f}。",
        "", comparison("500 条折外汇总", report["difference"]["oof"]),
        "", "折外混淆矩阵按 0、1、2 排列，行是真实类别，列是预测类别。", "",
    ]
    for variant, result in models.items():
        lines.append(f"- {variant} 版 `{result['oof']['confusion_matrix']}`")
    lines += ["", "## 独立测试样例", "", external["alignment"] + "。", ""]
    if external["status"] == "scored":
        lines += [
            "| 模型 | 集合 | 样本数 | 最佳 epoch | 宏 F1 | 准确率 | 召回 0 | 召回 1 | 召回 2 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *(metric_row(variant, "前 150 行", result["test"]) for variant, result in models.items()),
            "", comparison("测试样例", report["difference"]["test"]),
        ]
    else:
        lines += ["外部计分待补。", "", *(f"- 缺少 {item}" for item in external["missing"]),
                  *(f"- {warning}" for warning in external["warnings"])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test-input", type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True, help="新的或空的结果目录")
    parser.add_argument("--train-encoding", default="utf-8-sig")
    parser.add_argument("--test-input-encoding", default="gb18030")
    args = parser.parse_args()
    configure_cpu()
    label = "工况类别"
    raw, columns = read_numeric(args.train, args.train_encoding, [*FEATURES, label], label)
    x, y = raw[:, :-1], raw[:, -1].astype(np.int64)
    counts = np.bincount(y, minlength=3).tolist()
    if len(y) != 500 or counts != [200, 150, 150]:
        raise ValueError(f"{args.train}: 实际形状 {[len(y), columns]}，标签计数 {counts}，"
                         "预期 500 行且标签计数 [200, 150, 150]")
    x_test, labels, external = load_external(args)
    splits = split_data(y)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error(f"结果目录需要为空：{out}")
    versions = {"python": platform.python_version(), "torch": str(torch.__version__),
                "numpy": np.__version__, "scikit-learn": sklearn.__version__}
    command = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    write_json(out / "run.json", {"versions": versions, "config": CONFIG, "command": command,
                                 "working_directory": str(Path.cwd())})
    write_json(out / "feature_schema.json", SCHEMA)
    write_json(out / "splits.json", [
        {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in split.items()}
        for split in splits
    ])

    report = {"train_file": str(args.train.resolve()), "train_shape": [len(y), columns],
              "train_encoding": args.train_encoding, "external": external,
              "models": {variant: {"folds": [], "parameters": count} for variant, count in PARAMETERS.items()}}
    oof = {variant: np.full((len(y), len(CLASSES)), np.nan, dtype=np.float32) for variant in PARAMETERS}
    fold_ids = np.empty(len(y), dtype=np.int64)
    for split in splits:
        fold = split["fold"]
        train, val, held_out = split["train"], split["validation"], split["held_out"]
        scaler = StandardScaler().fit(x[train])
        mean, scale = scaler.mean_, scaler.scale_
        x_train, x_val = standardize(x[train], mean, scale), standardize(x[val], mean, scale)
        fold_ids[held_out] = fold
        seed = CONFIG["training_seed_base"] + fold
        for variant, model in paired_models(seed).items():
            trained = fit(model, x_train, y[train], seed, CONFIG["max_epochs"], (x_val, y[val]))
            probabilities = predict(model, x[held_out], mean, scale)
            oof[variant][held_out] = probabilities
            metrics = score(y[held_out], probabilities.argmax(axis=1))
            report["models"][variant]["folds"].append({
                "fold": fold, "seed": seed, "held_out": metrics,
                "best_epoch": trained["best_epoch"], "best_val_macro_f1": trained["best_val_macro_f1"],
            })
            write_json(out / f"history_{variant}_fold{fold}.json", trained["history"])
            save_checkpoint(out / f"{variant}_fold{fold}.pt", model, scaler,
                            {**CONFIG, "fold": fold, "seed": seed, "epochs": trained["best_epoch"]})
            print(f"fold={fold} model={variant} best_epoch={trained['best_epoch']} "
                  f"held_out_macro_f1={metrics['macro_f1']:.6f}", flush=True)

    seed = CONFIG["final_seed"]
    scaler = StandardScaler().fit(x)
    mean, scale = scaler.mean_, scaler.scale_
    full = standardize(x, mean, scale)
    for variant, model in paired_models(seed).items():
        result = report["models"][variant]
        assert np.isfinite(oof[variant]).all()
        write_predictions(out / f"oof_{variant}.csv", oof[variant], y, fold_ids)
        result["oof"] = score(y, oof[variant].argmax(axis=1))
        fold_f1 = [fold["held_out"]["macro_f1"] for fold in result["folds"]]
        result["fold_macro_f1_mean"] = float(np.mean(fold_f1))
        result["fold_macro_f1_sd"] = float(np.std(fold_f1, ddof=1))
        epochs = int(np.median([fold["best_epoch"] for fold in result["folds"]]))
        result["final_epochs"] = epochs
        verify_forward(model, full[:8])
        trained = fit(model, full, y, seed, epochs)
        write_json(out / f"history_{variant}_final.json", trained["history"])
        checkpoint = out / f"{variant}_final.pt"
        save_checkpoint(checkpoint, model, scaler, {**CONFIG, "seed": seed, "epochs": epochs})
        reloaded, saved_mean, saved_scale = load_checkpoint(checkpoint)
        probe = x if x_test is None or not len(x_test) else x_test
        verify_forward(reloaded, standardize(x[:8], saved_mean, saved_scale))
        probabilities = predict(reloaded, probe, saved_mean, saved_scale)
        np.testing.assert_array_equal(probabilities, predict(model, probe, mean, scale))
        result["reload_prediction_check"] = "exact_match"
        if x_test is not None:
            prediction_rows = probabilities if len(x_test) else np.empty((0, len(CLASSES)))
            write_predictions(out / f"test_{variant}.csv", prediction_rows, labels)
        if labels is not None:
            result["test"] = score(labels, probabilities[:len(labels)].argmax(axis=1))

    a, b = (report["models"][variant] for variant in PARAMETERS)
    report["difference"] = {
        "oof": differences(a["oof"], b["oof"]),
        "fold_macro_f1_mean": b["fold_macro_f1_mean"] - a["fold_macro_f1_mean"],
    }
    if labels is not None:
        report["difference"]["test"] = differences(a["test"], b["test"])
    write_json(out / "metrics.json", report)
    write_report(out / "report.md", report)
    print(f"Report: {out.resolve() / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
