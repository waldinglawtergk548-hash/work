"""Train C from scratch on the saved A/B splits and compare like-for-like."""
import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import torch

from furnace import FEATURES, SCHEMA, FurnaceClassifier, load_checkpoint, predict, read_numeric, save_checkpoint, standardize, write_json, write_predictions
from furnace_c import C_PARAMETERS, FurnaceClassifierC, load_c_checkpoint, verify_c_forward
from train import CONFIG, METRIC_KEYS, configure_cpu, fit, load_external, score, seed_all, split_data


def fresh_model(seed):
    seed_all(seed)
    model = FurnaceClassifierC()
    # The first modules must match the original A/B initialization at this seed.
    seed_all(seed)
    reference = FurnaceClassifier("A")
    for name, weight in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], weight, atol=0, rtol=0)
    return model


def render_report(path, models, external):
    def table(section, key):
        lines = [
            f"## {section}", "",
            "| 版本 | 样本数 | 宏 F1 | 准确率 | 召回 0 | 召回 1 | 召回 2 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for variant, model in models.items():
            metrics = model[key]
            values = " | ".join(f"{metrics[field]:.6f}" for field in METRIC_KEYS)
            lines.append(f"| {variant} | {metrics['n']} | {values} |")
        lines.append("")
        return lines

    lines = [
        "# A/B/C 工况分类对照", "",
        "A 版直接拼接（3,715 参数），B 版原双头（4,803 参数），C 版单个 8 维头（4,267 参数）。",
        "沿用原分折、种子和训练配置。C 版每折独立从头初始化，按验证宏 F1 选权重。", "",
        *table("500 行折外评估", "oof"),
        "| 版本 | 五折宏 F1 均值 | 样本标准差 | 各折最佳轮次 | 全量重训轮次 |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for variant, model in models.items():
        epochs = ", ".join(str(fold["best_epoch"]) for fold in model["folds"])
        lines.append(f"| {variant} | {model['fold_macro_f1_mean']:.6f} | {model['fold_macro_f1_sd']:.6f} | "
                     f"{epochs} | {model['final_epochs']} |")
    lines += ["", "类别与采集日期绑定，折外结果反映这批数据内的区分能力。", ""]
    if external["status"] == "scored":
        lines += table("原标签文件，前 150 行", "test")
        lines += [external["alignment"] + "。", ""]
    else:
        lines += ["原标签文件计分待补。", "", *external["missing"], *external["warnings"], ""]
    if "test_reference_200" in models["C"]:
        lines += table("按对话给出的区间假设，全部 200 行", "test_reference_200")
        lines += [
            "参考标签为 1–100 行类别 0、101–150 行类别 1、151–200 行类别 2。",
            "此表以区间假设为条件。两张测试表均使用各版全量重训权重，权重选择基于训练流程中的验证集。",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-encoding", default="utf-8-sig")
    parser.add_argument("--test-input", type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--test-input-encoding", default="gb18030")
    parser.add_argument("--ab-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-reference-200", action="store_true",
                        help="额外按 1–100 类别 0、101–150 类别 1、151–200 类别 2 的假设计分")
    args = parser.parse_args()
    configure_cpu()
    label = "工况类别"
    raw, _ = read_numeric(args.train, args.train_encoding, [*FEATURES, label], label)
    x, y = raw[:, :-1], raw[:, -1].astype(np.int64)
    if np.bincount(y, minlength=3).tolist() != [200, 150, 150]:
        raise ValueError(f"{args.train}: 需要 500 条原训练数据，类别计数 [200, 150, 150]")

    ab = args.ab_results
    original_run = json.loads((ab / "run.json").read_text(encoding="utf-8"))
    if original_run["config"] != CONFIG:
        raise ValueError("当前训练配置与原 A/B 实验不一致")
    original_splits = json.loads((ab / "splits.json").read_text(encoding="utf-8"))
    splits = split_data(y)
    split_records = [
        {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in split.items()}
        for split in splits
    ]
    if split_records != original_splits:
        raise ValueError("当前划分与原 A/B 保存的划分不一致")
    models = json.loads((ab / "metrics.json").read_text(encoding="utf-8"))["models"]
    # Recompute both baseline test rows on the requested input and label mapping.
    for baseline in models.values():
        baseline.pop("test", None)
    x_test, labels, external = load_external(args)
    reference_200 = None
    if args.score_reference_200:
        if x_test is None or len(x_test) != 200:
            parser.error("--score-reference-200 需要恰好 200 行的测试输入")
        reference_200 = np.repeat([0, 1, 2], [100, 50, 50])

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    versions = {"python": platform.python_version(), "torch": str(torch.__version__),
                "numpy": np.__version__, "scikit-learn": sklearn.__version__}
    command = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    write_json(out / "run.json", {
        "config": CONFIG, "versions": versions, "command": command,
        "working_directory": str(Path.cwd()), "ab_results": str(ab.resolve()),
    })
    write_json(out / "splits.json", original_splits)
    write_json(out / "feature_schema.json", SCHEMA)
    result = {"parameters": C_PARAMETERS, "folds": []}
    models["C"] = result
    oof = np.full((len(y), 3), np.nan, dtype=np.float32)
    fold_ids = np.empty(len(y), dtype=np.int64)
    for split in splits:
        fold = split["fold"]
        train, val, held_out = split["train"], split["validation"], split["held_out"]
        seed = CONFIG["training_seed_base"] + fold
        model = fresh_model(seed)
        scaler = StandardScaler().fit(x[train])
        mean, scale = scaler.mean_, scaler.scale_
        train_batch = standardize(x[train], mean, scale)
        verify_c_forward(model, train_batch[:8])
        trained = fit(model, train_batch, y[train], seed, CONFIG["max_epochs"],
                      (standardize(x[val], mean, scale), y[val]))
        probabilities = predict(model, x[held_out], mean, scale)
        oof[held_out], fold_ids[held_out] = probabilities, fold
        metrics = score(y[held_out], probabilities.argmax(axis=1))
        result["folds"].append({
            "fold": fold, "seed": seed, "best_epoch": trained["best_epoch"],
            "best_val_macro_f1": trained["best_val_macro_f1"], "held_out": metrics,
        })
        save_checkpoint(out / f"C_fold{fold}.pt", model, scaler,
                        {**CONFIG, "fold": fold, "seed": seed, "epochs": trained["best_epoch"]})
        write_json(out / f"history_C_fold{fold}.json", trained["history"])
        print(f"C fold={fold} best_epoch={trained['best_epoch']} held_out_macro_f1={metrics['macro_f1']:.6f}", flush=True)

    assert np.isfinite(oof).all()
    write_predictions(out / "oof_C.csv", oof, y, fold_ids)
    result["oof"] = score(y, oof.argmax(axis=1))
    fold_f1 = [fold["held_out"]["macro_f1"] for fold in result["folds"]]
    result["fold_macro_f1_mean"] = float(np.mean(fold_f1))
    result["fold_macro_f1_sd"] = float(np.std(fold_f1, ddof=1))
    epochs = int(np.median([fold["best_epoch"] for fold in result["folds"]]))
    result["final_epochs"] = epochs
    seed = CONFIG["final_seed"]
    model = fresh_model(seed)
    scaler = StandardScaler().fit(x)
    full_batch = standardize(x, scaler.mean_, scaler.scale_)
    trained = fit(model, full_batch, y, seed, epochs)
    write_json(out / "history_C_final.json", trained["history"])
    save_checkpoint(out / "C_final.pt", model, scaler, {**CONFIG, "seed": seed, "epochs": epochs})
    reloaded, mean, scale = load_c_checkpoint(out / "C_final.pt")
    verify_c_forward(reloaded, full_batch[:8])
    probe = x if x_test is None else x_test
    probabilities = predict(reloaded, probe, mean, scale)
    np.testing.assert_array_equal(probabilities, predict(model, probe, scaler.mean_, scaler.scale_))
    result["reload_prediction_check"] = "exact_match"

    if x_test is not None:
        write_predictions(out / "test_C.csv", probabilities, labels)
        if reference_200 is not None:
            write_predictions(out / "test_C_reference_200.csv", probabilities, reference_200)
        for variant in ("A", "B", "C"):
            if variant == "C":
                predictions = probabilities.argmax(axis=1)
            else:
                baseline, base_mean, base_scale = load_checkpoint(ab / f"{variant}_final.pt")
                predictions = predict(baseline, x_test, base_mean, base_scale).argmax(axis=1)
            if labels is not None:
                models[variant]["test"] = score(labels, predictions[:len(labels)])
            if reference_200 is not None:
                models[variant]["test_reference_200"] = score(reference_200, predictions)

    write_json(out / "metrics.json", {"models": models, "external": external})
    render_report(out / "report.md", models, external)
    print(f"Report: {out.resolve() / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
