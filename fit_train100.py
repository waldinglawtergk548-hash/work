"""Continue A/B training to perfect in-sample accuracy, without using test data."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from furnace import FEATURES, load_checkpoint, predict, read_numeric, save_checkpoint, standardize, write_json, write_predictions
from train import CONFIG, configure_cpu, fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-encoding", default="utf-8-sig")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=300)
    args = parser.parse_args()
    if args.max_epochs < 1:
        parser.error("--max-epochs 必须为正整数")
    configure_cpu()
    label = "工况类别"
    raw, _ = read_numeric(args.train, args.train_encoding, [*FEATURES, label], label)
    x, y = raw[:, :-1], raw[:, -1].astype(np.int64)
    if len(y) != 500 or np.bincount(y, minlength=3).tolist() != [200, 150, 150]:
        raise ValueError("需要原训练集的 500 行数据，三类计数为 [200, 150, 150]")
    scaler = StandardScaler().fit(x)
    batch = standardize(x, scaler.mean_, scaler.scale_)
    _, group_ids = np.unique(batch.numpy(), axis=0, return_inverse=True)
    labels_per_group = {}
    for row_id, (group, target) in enumerate(zip(group_ids, y), start=1):
        if group in labels_per_group and labels_per_group[group][0] != target:
            first = labels_per_group[group][1]
            raise ValueError(f"数据行 {first} 和 {row_id} 的模型输入相同但标签不同")
        labels_per_group[group] = (target, row_id)

    args.out_dir.mkdir(parents=True, exist_ok=False)
    seed = CONFIG["final_seed"] + 1
    summary = {}
    for variant in ("A", "B"):
        source = args.checkpoints / f"{variant}_final.pt"
        model, mean, scale = load_checkpoint(source)
        if model.variant != variant:
            raise ValueError(f"{source}: 预期模型 {variant}，实际模型 {model.variant}")
        np.testing.assert_array_equal(mean, scaler.mean_)
        np.testing.assert_array_equal(scale, scaler.scale_)
        before = int((predict(model, x, mean, scale).argmax(axis=1) == y).sum())
        print(f"{variant}: initial {before}/{len(y)}", flush=True)
        history, selected = [], 0
        if before < len(y):
            result = fit(model, batch, y, seed, args.max_epochs, stop_at_perfect_train=True)
            history, selected = result["history"], result["best_epoch"]
        probabilities = predict(model, x, mean, scale)
        correct = int((probabilities.argmax(axis=1) == y).sum())
        if correct < before:
            model, mean, scale = load_checkpoint(source)
            probabilities = predict(model, x, mean, scale)
            correct, selected = before, 0
        filename = f"{variant}_train100.pt" if correct == len(y) else f"{variant}_train_fit.pt"
        checkpoint = args.out_dir / filename
        summary[variant] = {
            "initial_correct": before, "final_correct": correct, "n": len(y),
            "training_accuracy": correct / len(y), "additional_epochs": len(history),
            "selected_additional_epoch": selected, "source_checkpoint": str(source.resolve()),
            "checkpoint": filename,
        }
        save_checkpoint(checkpoint, model, scaler, {
            **CONFIG, "objective": "in_sample_accuracy", "seed": seed,
            "max_epochs": args.max_epochs, "source": str(source.resolve()), "additional_epochs": selected,
        })
        reloaded, saved_mean, saved_scale = load_checkpoint(checkpoint)
        np.testing.assert_array_equal(probabilities, predict(reloaded, x, saved_mean, saved_scale))
        write_predictions(args.out_dir / f"{variant}_train_predictions.csv", probabilities, y)
        write_json(args.out_dir / f"{variant}_history.json", history)
        print(f"{variant}: {correct}/{len(y)}, additional epochs={len(history)}, checkpoint={checkpoint}", flush=True)
    write_json(args.out_dir / "train_fit.json", summary)
    if any(result["training_accuracy"] < 1.0 for result in summary.values()):
        raise SystemExit("达到轮次上限，至少一版仍未达到 100%")


if __name__ == "__main__":
    main()
