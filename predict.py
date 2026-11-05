"""Load a final checkpoint and predict each CSV row in its original order."""
import argparse
from pathlib import Path

import torch

from furnace import FEATURES, load_checkpoint, predict, read_numeric, write_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--encoding", default="gb18030")
    args = parser.parse_args()
    if args.output.resolve() in (args.input.resolve(), args.checkpoint.resolve()):
        parser.error("--output 必须与输入文件和模型文件不同")
    torch.set_num_threads(1)
    model, mean, scale = load_checkpoint(args.checkpoint)
    x, _ = read_numeric(args.input, args.encoding, FEATURES)
    probabilities = predict(model, x, mean, scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(args.output, probabilities)
    print(f"{model.variant}: {len(x)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
