import copy
from typing import Any, Dict, List
import torch
import numpy as np
from TorranceSparrow.torrance_sparrow_model import torspa
from slib.utils import get_merl_angles, angles_to_wiwo,wiwo_to_angles

class TorspaBRDFNPsDataset(torch.utils.data.Dataset):
    def __init__(self, params, batch_size=32, device="cuda:0"):
        self.params = params
        self.params.kd = torch.tensor(self.params.kd, dtype=torch.float32).to(
            device
        )
        self.params.ks = torch.tensor(self.params.ks, dtype=torch.float32).to(
            device
        )
        self.params.sigma = torch.tensor(
            self.params.sigma, dtype=torch.float32
        ).to(device)
        self.batch_size = batch_size
        self.num_baches = len(self.params.kd) // self.batch_size + min(
            1, len(self.params.kd) % self.batch_size
        )
        self.length = len(self.params.kd)
        self.device = self.params.kd.device
        self.datasize = 16200
        self.setup_wiwo()
        print("device", self.device)

    def get_batch(self, batch_idx):
        batch_params = self.get_batch_param(batch_idx)
        bs = batch_params.kd.shape[0]
        params_cat = self.tile_params(batch_params)

        Y = torspa(
            kd=params_cat.kd,
            ks=params_cat.ks,
            sigma=params_cat.sigma,
            wi=torch.tile(self.wi, (bs, 1)).to(self.device),
            wo=torch.tile(self.wo, (bs, 1)).to(self.device),
        ).to(torch.float32)

        Y = self.preprocess_Y(Y)
        return torch.cat([torch.tile(self.X, (bs, 1)), Y], dim=-1)

    def preprocess_Y(self, Y: np.ndarray) -> torch.Tensor:
        for _ in range(4):
            Y = torch.log1p(Y)
        return Y

    def tile_params(self, params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        複数のdisneyパラメータをconcatする
        """
        ret = copy.deepcopy(params)

        ret.kd = torch.tile(
            ret.kd.reshape(-1, 1, 3), (self.datasize, 1)
        ).reshape(-1, 3)
        ret.ks = torch.tile(
            ret.ks.reshape(-1, 1, 3), (self.datasize, 1)
        ).reshape(-1, 3)
        ret.sigma = torch.tile(
            ret.sigma.reshape(-1, 1, 1), (self.datasize, 1)
        ).reshape(-1, 1)
        return ret

    def get_batch_param(self, batch_idx):
        batch_param = {}
        batch_param = self.params[
            batch_idx
            * self.batch_size : min(
                (batch_idx + 1) * self.batch_size, self.length
            )
        ]
        return batch_param

    def setup_wiwo(self):
        wi, wo = angles_to_wiwo(get_merl_angles())
        wiwo = np.concatenate([wi, wo], axis=1)
        visible = wi[:, 2] > 0
        wiwo = wiwo[visible]
        np.random.seed(0)
        wiwo = wiwo[
            np.random.choice(a=wiwo.shape[0], size=self.datasize, replace=True)
        ]
        wi, wo = wiwo[:, :3], wiwo[:, 3:]
        angles = wiwo_to_angles(wi, wo)
        self.wi = torch.tensor(wi)
        self.wo = torch.tensor(wo)
        self.X = torch.tensor(
            np.concatenate(
                [
                    angles[:, :2],
                    np.sin(angles[:, 2] * 2).reshape(-1, 1),
                    np.cos(angles[:, 2] * 2).reshape(-1, 1),
                ],
                axis=1,
            ),
            dtype=torch.float32,
        ).to(self.device)