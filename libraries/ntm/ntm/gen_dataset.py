import random
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class LatentDataset(Dataset):
    def __init__(
        self,
        src_latent_vectors: Union[torch.tensor, np.ndarray],
        gt_latent_vectors: Union[torch.tensor, np.ndarray],
    ) -> None:
        self.src = torch.tensor(
            src_latent_vectors, dtype=torch.float32
        )  # data length x embed dim
        self.gt = torch.tensor(
            gt_latent_vectors, dtype=torch.float32
        )  # data length x latent dim

        assert len(self.src) == len(self.gt)
        self.dim = self.src.shape[-1]

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.src[index],
            self.gt[index],
        )


def load_latents(
    tv_latent_dirs: List[Union[Path, str]],
    end_frame: int = 36,
) -> List[List[np.ndarray]]:
    """
    latent vectorをloadしてreturnするだけ
    return tv_latent_vectors: px x (時間方向 x latent_dim:TVLatentVector)
    """
    tv_latent_vectors = []
    for tv_latent_dir in tqdm(tv_latent_dirs, desc="loading latent vectors."):
        try:
            tv_latent_vectors.append(
                [
                    np.load(Path(tv_latent_dir) / f"latent_{frame}.npy")
                    for frame in range(end_frame)
                ]
            )
        except KeyboardInterrupt:
            exit()
        except:
            print(f"Can't load {tv_latent_dir}'s data.")
    return tv_latent_vectors


def positional_encoding(input, t):
    """
    cos,sinのpositional encoding
    """
    latent_dim = np.squeeze(input).shape[0]
    dim_counter = np.arange(latent_dim) // 2 * 2
    dim_matrix = np.power(10000.0, dim_counter / latent_dim)
    pos_encoding = np.sin(t / dim_matrix)
    pos_encoding[1::2] = np.cos(t / dim_matrix[1::2])
    pos_encoding = pos_encoding.reshape(input.shape)
    return input + pos_encoding


def build_dataset(tv_latent_vectors: np.ndarray):
    """
    transformerの入力(src,tgt)+GTを作る
    """
    src_latents = tv_latent_vectors[:, 0:1, :]
    gt_latents = tv_latent_vectors[:, 1:, :]
    return LatentDataset(
        src_latents,
        gt_latents,
    )


def read_csv(csv_path: Union[Path, str]):
    """
    csvを読み込んでtrain/validに分ける
    """
    df = pd.read_csv(csv_path)
    train_df = df[df["isTrain"]]["fpath"]
    valid_df = df[~df["isTrain"]]["fpath"]
    return train_df.values, valid_df.values


def build_datasets(
    data_path: Union[Path, str],
    shuffle=True,
    seed: int = 2023,
    debug: bool = False,
    end_frame: int = 36,
    fold: int = 0,
):
    """
    # train,validation,visualization用のdatasetを作る
    # csv_pathを渡す場合，それを元にtrain/validに分ける．
    # csv_pathにファイルがない場合，現在のsplitをcsv_pathに保存する．
    """

    train = np.load(data_path)[..., :10]

    np.random.seed(2023)
    train = train[
        np.random.choice(
            a=train.shape[0],
            size=90000,
            replace=True,
        )
    ]

    # gen train/valid dataset + visalize dataset(可視化用)
    train_dataset = build_dataset(train)
    # valid_dataset = build_dataset(valid[:16])
    return train_dataset, None


def test_build_datasets():
    tv_latent_dir = "/dataset/brdfnps-latent/tvBTF32"
    csv_path = "./output-debug/split.csv"
    _, _, _ = build_datasets(tv_latent_dir, debug=True, csv_path=csv_path)


if __name__ == "__main__":
    test_build_datasets()
