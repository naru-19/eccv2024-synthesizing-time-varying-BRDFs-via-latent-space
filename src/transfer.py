import argparse
from pathlib import Path

import numpy as np
import torch
from ntm.model import NTM
from ntm.transfer import Transfer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--model", type=Path, required=True, help="model path")
    parser.add_argument("--tgt", type=Path, required=True, help="target latent path")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default="./output",
        help="output directory",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(exist_ok=True, parents=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    tvnet = NTM(
        lr=1,
        in_features=10,
        out_features=10,
        device=device,
    )
    print(args)
    epoch = tvnet.load(args.model)
    print("epoch:", epoch)
    target_latent = np.load(args.tgt)[..., :10]
    print(target_latent.shape)
    if len(target_latent.shape) == 3:
        target_latent = target_latent[:, 0, :]

    rust_transfer = Transfer(tvnet)
    synth_result = rust_transfer.run(target_latent)
    if synth_result.shape[0] == 1:
        # steel case (merl material)
        # tile to 90000
        synth_result = np.tile(synth_result, (90000, 1, 1))

    out_path = args.out_dir / f"transferred.npy"
    print("out_path: ", out_path)
    if device == "cpu":
        np.save(out_path, synth_result)
    else:
        np.save(out_path, synth_result.cpu().numpy())


if __name__ == "__main__":
    main()
