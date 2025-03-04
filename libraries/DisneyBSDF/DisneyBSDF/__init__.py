import numpy as np
import torch
from render_utils import LocalEnv

from DisneyBSDF.disney_bsdf import disney_bsdf
from DisneyBSDF.disney_bsdf_train import disney_bsdf_train


class DisneyBSDF:
    def __init__(self, env: LocalEnv, device: str = "cpu"):
        n = np.array([0, 0, 1.0]).reshape(1, 3)
        self.n = torch.tensor(np.tile(n, (env.wi.shape[0], 1))).to(device)
        self.l = torch.tensor(env.wi).to(device)
        self.v = torch.tensor(env.wo).to(device)
        self.device = device
        self.env = env

    def get_bsdf(self, params):
        params = params.copy()
        for key in params:
            params[key] = torch.tensor(
                np.tile(
                    np.array(params[key]).reshape(1, -1),
                    (self.env.wi.shape[0], 1),
                )
            ).to(self.device)
        img = disney_bsdf(L=self.l, V=self.v, N=self.n, params=params)
        return img


class DisneyRenderer:
    def __init__(self, env: LocalEnv, device: str = "cpu"):
        n = np.array([0, 0, 1.0]).reshape(1, 3)
        self.n = torch.tensor(np.tile(n, (env.wi.shape[0], 1))).to(device)
        self.l = torch.tensor(env.wi).to(device)
        self.v = torch.tensor(env.wo).to(device)
        self.device = device
        self.env = env

    def render(self, params):
        params = params.copy()
        for key in params:
            params[key] = torch.tensor(
                np.tile(
                    np.array(params[key]).reshape(1, -1),
                    (self.env.wi.shape[0], 1),
                )
            ).to(self.device)
        img = disney_bsdf(L=self.l, V=self.v, N=self.n, params=params)
        return img


class DisneyRenderer2:
    def __init__(self, env: LocalEnv, device: str = "cpu"):
        n = np.array([0, 0, 1.0]).reshape(1, 3)
        self.n = torch.tensor(np.tile(n, (env.wi.shape[0], 1))).to(device)
        self.l = torch.tensor(env.wi).to(device)
        self.v = torch.tensor(env.wo).to(device)
        self.device = device
        self.env = env

    def render(self, params):
        params = params.copy()
        for key in params:
            params[key] = torch.tensor(
                np.tile(
                    np.array(params[key]).reshape(1, -1),
                    (self.env.wi.shape[0], 1),
                )
            ).to(self.device)
        img = disney_bsdf2(L=self.l, V=self.v, N=self.n, params=params)
        return img
