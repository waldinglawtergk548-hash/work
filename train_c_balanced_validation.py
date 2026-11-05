"""Train C with 10 validation rows per class and stop at 100% train accuracy."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

from furnace import CLASSES, FEATURES, predict, read_numeric, save_checkpoint, standardize, write_json, write_predictions
from furnace_c import load_c_checkpoint, verify_c_forward
from train import CONFIG, configure_cpu, fit, score
from train_c import fresh_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--train-encoding", default="utf-8-sig")
    parser.add_argument("--test-input", type=Path, help="可选，按此前给出的 200 行参考答案额外计分")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-epochs", type=int, default=300)
    args = parser.parse_args()
    if args.max_epochs < 1:
        parser.error("--max-epochs 必须为正整数")

    configure_cpu()
    label = "工况类别"
    raw, _ = read_numeric(args.train, args.train_encoding, [*FEATURES, label], label)
    x, y = raw[:, :-1], raw[:, -1].astype(np.int64)
    if np.bincount(y, minlength=3).tolist() != [200, 150, 150]:
        raise ValueError(f"{args.train}: 需要类别计数为 [200, 150, 150] 的 500 条原始记录")
    split_seed = CONFIG["outer_seed"]
    rng = np.random.default_rng(split_seed)
    val_rows = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(y == target), size=10, replace=False) for target in CLASSES
    ]))
    train_rows = np.setdiff1d(np.arange(len(y)), val_rows)
    np.testing.assert_array_equal(np.bincount(y[val_rows], minlength=3), [10, 10, 10])
    np.testing.assert_array_equal(np.bincount(y[train_rows], minlength=3), [190, 140, 140])

    x_test = None
    if args.test_input is not None:
        x_test, _ = read_numeric(args.test_input, "gb18030", FEATURES)
        if len(x_test) != 200:
            parser.error("按此前区间计分需要恰好 200 行测试输入")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "split.json", {"split_seed": split_seed, "row_id_base": 1,
                                    "train": (train_rows + 1).tolist(),
                                    "validation": (val_rows + 1).tolist()})
    scaler = StandardScaler().fit(x[train_rows])
    batch = standardize(x[train_rows], scaler.mean_, scaler.scale_)
    seed = CONFIG["final_seed"]
    model = fresh_model(seed)
    verify_c_forward(model, batch[:8])
    trained = fit(model, batch, y[train_rows], seed, args.max_epochs, stop_at_perfect_train=True)
    history = trained["history"]
    best_epoch = trained["best_epoch"]
    hit_target = history[best_epoch - 1]["train_correct"] == len(train_rows)
    if hit_target:
        assert best_epoch == len(history)
        assert all(row["train_correct"] < len(train_rows) for row in history[:-1])

    checkpoint = out / ("C_train100.pt" if hit_target else "C_train_best.pt")
    config = {
        "device": CONFIG["device"], "torch_threads": CONFIG["torch_threads"],
        "optimizer": CONFIG["optimizer"], "loss": CONFIG["loss"], "lr": CONFIG["lr"],
        "weight_decay": CONFIG["weight_decay"], "batch_size": CONFIG["batch_size"],
        "seed": seed, "split_seed": split_seed, "val_per_class": 10,
        "max_epochs": args.max_epochs, "selected_epoch": best_epoch,
        "stop_rule": "first epoch with 470/470 training predictions correct",
    }
    save_checkpoint(checkpoint, model, scaler, config)
    reloaded, mean, scale = load_c_checkpoint(checkpoint)
    verify_c_forward(reloaded, batch[:8])
    probabilities = predict(reloaded, x, mean, scale)
    np.testing.assert_array_equal(probabilities, predict(model, x, scaler.mean_, scaler.scale_))
    predictions = probabilities.argmax(axis=1)
    train_metrics = score(y[train_rows], predictions[train_rows])
    val_metrics = score(y[val_rows], predictions[val_rows])
    assert int((predictions[train_rows] == y[train_rows]).sum()) == history[best_epoch - 1]["train_correct"]

    summary = {
        "config": config, "checkpoint": checkpoint.name, "target_reached": hit_target,
        "epochs_run": len(history), "train": train_metrics, "validation": val_metrics,
        "command": subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
    }
    if x_test is not None:
        reference = np.repeat(CLASSES, [100, 50, 50])
        test_probs = predict(reloaded, x_test, mean, scale)
        summary["test_reference_200"] = score(reference, test_probs.argmax(axis=1))
        write_predictions(out / "test_C_reference_200.csv", test_probs, reference)

    write_json(out / "metrics.json", summary)
    write_json(out / "history.json", history)
    write_predictions(out / "train_and_validation_predictions.csv", probabilities, y)
    print(f"epoch={best_epoch}, train={train_metrics['accuracy']:.4%}, "
          f"validation={val_metrics['accuracy']:.4%}, checkpoint={checkpoint}", flush=True)
    if not hit_target:
        raise SystemExit("达到轮次上限，训练准确率仍低于 100%")


if __name__ == "__main__":
    main()
