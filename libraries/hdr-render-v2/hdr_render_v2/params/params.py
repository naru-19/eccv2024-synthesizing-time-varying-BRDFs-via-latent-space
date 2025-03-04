# import epfl_loader
from pathlib import Path
from typing import Union, overload

# import nbrdf
import numpy as np
import torch
from hdr_render_v2.utils import Merl, angles_to_wiwo, get_merl_angles


class BaseRenderParams:
    def to_device(self, device: str = "cpu"):
        raise NotImplementedError


class EpflRenderParams(BaseRenderParams):
    def __init__(self, fpath):
        self.fpath = fpath

    def to_device(self, device):
        pass


class TorranceSparrowRenderParams(BaseRenderParams):
    def __init__(self, kd, ks, sigma):
        self.kd = kd
        self.ks = ks
        self.sigma = sigma

    def to_device(self, device: str = "cpu"):
        self.kd = torch.tensor(self.kd, dtype=torch.float32, device=device)
        self.ks = torch.tensor(self.ks, dtype=torch.float32, device=device)
        self.sigma = torch.tensor(self.sigma, dtype=torch.float32, device=device)


# class NbrdfRenderParams(BaseRenderParams):
#     def __init__(self, fpath):
#         self.model = nbrdf.NBRDF()
#         self.model.load_state_dict(torch.load(fpath, map_location="cpu"))

#     def to_device(self, device):
#         self.model.to(device)


class LayeredRenderParams(BaseRenderParams):
    def __init__(self, latent_path):
        assert latent_path[-3:] == "npy", "latent path must be npy"
        self.latent = np.load(latent_path)
        self.latent = torch.tensor(self.latent, dtype=torch.float32)

    def to_device(self, device):
        self.latent = self.latent.to(device)


class BaseTextureRenderParams(BaseRenderParams):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TorranceSparrowTextureRenderParams(BaseTextureRenderParams):
    def __init__(self, params):
        assert params.shape[1] == 5 or params.shape[1] == 7, f"{params.shape}"
        self.params = params

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        return self.params[idx]

    def to_device(self, device: str = "cpu"):
        self.params = torch.tensor(self.params, dtype=torch.float32, device=device)


class BRDFNPsRenderParams(BaseRenderParams):
    def __init__(self, latent_path, latent_dim=10):
        self.latent = np.load(latent_path)[..., :latent_dim]

    def to_device(self, device):
        self.latent = torch.tensor(self.latent, dtype=torch.float32).to(device)


class BRDFNPsTextureRenderParams(BaseTextureRenderParams):
    @overload
    def __init__(self, fpath: Union[str, Path]): ...

    def __init__(self, fpath: Union[str, Path]):
        self.latent = np.load(fpath)
        assert (
            len(self.latent.shape) == 2
        ), "params must be 2d array (length,latent_dim)"
        self.latent = self.latent[:, :10]

    @overload
    def __init__(self, latent: Union[np.ndarray, torch.Tensor]): ...

    def __init__(self, latent: Union[np.ndarray, torch.Tensor]):
        self.latent = latent

    def __len__(self):
        return self.latent.shape[0]

    def __getitem__(self, idx):
        return self.latent[idx]

    def to_device(self, device):
        if not isinstance(self.latent, torch.Tensor):
            self.latent = torch.tensor(self.latent)
        self.latent = self.latent.to(device)


class DisneyTextureRenderParams(BaseTextureRenderParams):
    def __init__(self, params):
        self.params = params
        self.keys = [
            "base_color",
            "metallic",
            "specular",
            "anisotropic",
            "sheen",
            "sheen_tint",
            "clearcoat",
            "clearcoat_gloss",
            "roughness",
        ]

    def __len__(self):
        return len(self.params[self.keys[0]])

    def __getitem__(self, idx):
        return torch.cat(
            [self.params[key][idx].reshape(-1) for key in self.keys], dim=-1
        )

    def to_device(self, device):
        for key in self.keys:
            self.params[key] = self.params[key].to(device)
        return self


class DisneyRenderParams(BaseRenderParams):
    def __init__(self, parameter):
        import eval_lib as elib

        if isinstance(parameter, dict):
            p = []
            for key in elib.get_disney_keys():
                p.append(parameter[key].reshape(1, -1))
            parameter = np.concatenate(p, axis=-1)

        assert parameter.shape == (11,) or parameter.shape == (1, 11)
        # baseColor=tiled_param[..., :3],
        # metallic=tiled_param[..., 3:4],
        # specular=tiled_param[..., 4:5],
        # anisotropic=tiled_param[..., 5:6],
        # sheen=tiled_param[..., 6:7],
        # sheenTint=tiled_param[..., 7:8],
        # clearcoat=tiled_param[..., 8:9],
        # clearcoatGloss=tiled_param[..., 9:10],
        # roughness=tiled_param[..., 10:11],
        # の順番
        self.parameter = parameter

    def to_device(self, device):
        self.parameter = self.parameter.to(device)
