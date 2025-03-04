import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ntm.gen_dataset import build_datasets
from ntm.model import NTM
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ntm_config import get_config

SEED = 2023
torch.manual_seed(SEED)
np.random.seed(SEED)
END_FRAME = 36
l1loss = nn.L1Loss()  # reduction="sum")


def loss_print(loss: float, min_loss: Optional[float], prefix: Optional[str] = None):
    if min_loss >= 10**6:
        print("\033[32m" + f"{prefix} loss improved! inf -> {loss:.6f}" + "\033[0m")
        return loss
    elif loss < min_loss:
        print(
            "\033[32m"
            + f"{prefix} loss improved! {min_loss:.6f} -> {loss:.6f}"
            + "\033[0m"
        )
        return loss
    else:
        print(f"{prefix} loss didn't improved. The min loss is still {min_loss:.4f}")
        return min_loss


def criterion(pred, gt, src):
    mse = nn.MSELoss()
    rgb_pred = pred[..., :3]
    rgb_gt = gt[..., :3]

    # rescale
    for _ in range(4):
        rgb_pred = torch.clamp(torch.exp(rgb_pred) - 1, max=10)
        rgb_gt = torch.clamp(torch.exp(rgb_gt) - 1, max=10)

    return mse(rgb_pred, rgb_gt) + mse(pred[..., 3:], gt[..., 3:])


def run_one_epoch(
    ntm: NTM,
    dataloader: DataLoader,
    device: str,
    isTrain: bool = True,
) -> torch.tensor:
    total_loss = 0
    num_batches = 0

    for src, gt in tqdm(dataloader):
        src = src.to(device)  # bs x 1 x latent_dim,
        num_batches += 1
        pred = ntm.predict(x=src)
        assert pred.shape == gt.shape, f"pred.shape={pred.shape},gt.shape={gt.shape}"
        batch_loss = criterion(pred, gt.to(device), src)
        if isTrain:
            ntm.backward_loss(batch_loss)
        total_loss += batch_loss.detach()
    mean_batch_loss = total_loss / num_batches
    return mean_batch_loss


def train(
    ntm: NTM,
    device: str,
    data_loaders: Dict[str, DataLoader],
    run_name: str,
    model_dir: Path = None,
):
    model_dir.mkdir(parents=True, exist_ok=True)

    min_train_loss, min_valid_loss = 10**10, 10**10
    for epoch in range(1000):
        print("-" * 50 + f"\n{run_name}: epoch-{epoch} start.")
        # train
        train_loss = run_one_epoch(
            ntm, dataloader=data_loaders["train_loader"], device=device
        )
        # save if the loss is min.
        if train_loss < min_train_loss:
            ntm.save(model_dir / "min.pth", epoch=epoch + 1)
        print(f"train loss: {train_loss:.4f}")
        min_train_loss = loss_print(train_loss, min_train_loss, prefix="train")
        ntm.step_scheduler()
        # validation
        with torch.no_grad():
            if data_loaders["valid_loader"] is not None:
                valid_loss = run_one_epoch(
                    ntm,
                    dataloader=data_loaders["valid_loader"],
                    device=device,
                    isTrain=False,
                )
                print(f"valid loss: {valid_loss:.4f}")
                min_valid_loss = loss_print(valid_loss, min_valid_loss, prefix="valid")

        # save model
        if (epoch + 1) % 100 == 0 or epoch == 0:
            ntm.save(model_dir / "latest.pth", epoch=epoch + 1)
            print(f"model saved at {model_dir / 'latest.pth'}")
            if (epoch + 1) % 50 == 0 or epoch == 0:
                ntm.save(model_dir / f"epoch{epoch+1}.pth", epoch=epoch + 1)


def lrfunc(epoch: int) -> float:
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, default="./output")
    parser.add_argument("--run_name", type=str, default="debug")
    parser.add_argument("--resume_path", type=Optional[Path], default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--case", type=str, default="rust")
    args = parser.parse_args()
    print(args)

    cfg = get_config(args.case)

    model_dir = args.out_dir / cfg.suffix
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # data setup
    train_dataset, _ = build_datasets(
        cfg.data_path,
        end_frame=cfg.end_frame,
        debug=args.debug,
        seed=SEED,
        fold=args.fold,
    )
    data_loaders = {
        "train_loader": DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True
        ),
        "valid_loader": None,
    }

    # model setup
    ntm = NTM(
        lr=cfg.lr,
        in_features=train_dataset.dim,
        out_features=cfg.latent_dim,
        device=device,
    )
    ntm.train()
    # training start
    train(
        ntm=ntm,
        device=device,
        data_loaders=data_loaders,
        run_name=args.run_name,
        model_dir=model_dir,
    )


if __name__ == "__main__":
    main()
