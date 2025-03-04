import copy
from typing import Union

# import epfl_loader
import numpy as np
import torch
from DisneyBSDF import disney_bsdf, disney_bsdf_train
from hdr_render_v2.env import HDREnv
from hdr_render_v2.params import BaseRenderParams, BaseTextureRenderParams
from hdr_render_v2.utils import torspa, torspa3d, wiwo_to_angles_torch


class IRenderer:
    def __init__(self) -> None:
        self.cache = {"wi": {}, "params": {}, "intensity": {}, "ones": {}}

    def render(
        self,
        params: BaseRenderParams,
        wi: torch.Tensor,
        wo: torch.Tensor,
        n: torch.Tensor,
        intensity: HDREnv,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        n: sphere normal map
        """
        raise NotImplementedError

    def clear_cache(self):
        self.cache = {"wi": {}, "params": {}, "intensity": {}, "ones": {}}


class TorranceSparrowRenderer(IRenderer):
    def render(self, params: BaseRenderParams, wi, wo, n, intensity):
        """
        n: sphere normal map
        """
        bs = wo.shape[0]
        assert wo.shape == (bs, 3), f"{wo.shape}!={(bs,3)}"
        assert n.shape == (bs, 3), f"{n.shape}!={(bs,3)}"
        length = wi.shape[0]
        if bs in self.cache["wi"]:
            wi = self.cache["wi"][bs]
            ones = self.cache["ones"][bs]
            params = self.cache["params"][bs]
        else:
            wi = torch.tile(wi, (bs, 1))
            ones = torch.ones(length, bs, 3, dtype=torch.float32, device=wi.device)
            params = copy.deepcopy(params)
            params.kd = torch.tile(params.kd, (bs * length, 1))
            params.ks = torch.tile(params.ks, (bs * length, 1))
            params.sigma = torch.tile(params.sigma, (bs * length, 1))
            self.cache["wi"][bs] = wi
            self.cache["ones"][bs] = ones
            self.cache["params"][bs] = params

        wo = ones * wo
        wo = wo.permute(1, 0, 2).reshape(-1, 3)
        if params.ks.shape[-1] == 3:
            brdf = torspa3d(params.kd, params.ks, params.sigma, wi, wo)
        else:
            brdf = torspa(params.kd, params.ks, params.sigma, wi, wo)
        n = torch.tile(n.reshape(-1, 1, 3), (1, length, 1))
        ln = torch.sum(n * wi.reshape(bs, length, 3)[0], dim=-1).reshape(-1, 1)
        ln[ln <= 0] = 0
        brdf = brdf.reshape(bs, -1, 3)
        ln = ln.reshape(bs, -1, 1)

        brdf = (brdf * intensity * ln).reshape(bs, -1, 3)
        brdf = torch.mean(brdf, dim=1)
        return brdf


class BRDFNPsRenderer(IRenderer):
    def __init__(self, brdfnps):
        super().__init__()
        self.brdfnps = brdfnps
        self.device = brdfnps.device
        self.cache["latent"] = {}

    def render(self, params, wi, wo, n, intensity):
        bs = wo.shape[0]
        assert wo.shape == (bs, 3), f"{wo.shape}!={(bs,3)}"
        assert n.shape == (bs, 3), f"{n.shape}!={(bs,3)}"
        length = wi.shape[0]
        if bs in self.cache["wi"]:
            wi = self.cache["wi"][bs]
            ones = self.cache["ones"][bs]
            latent = self.cache["latent"][bs]
        else:
            wi = torch.tile(wi, (bs, 1))
            latent = torch.tile(params.latent.reshape(1, -1)[:, :10], (length * bs, 1))
            ones = torch.ones((length, bs, 3)).to(wi.device)
            self.cache["wi"][bs] = wi
            self.cache["ones"][bs] = ones
            self.cache["latent"][bs] = latent
        # latent = torch.tile(
        #     params.latent.reshape(1, -1)[:, :10], (length * bs, 1)
        # )

        wo = ones * wo
        wo = wo.permute(1, 0, 2).reshape(-1, 3)
        angles = self.wiwo_to_angles(wi, wo).reshape(-1, 4)
        decoder_input = torch.cat([angles, latent], dim=1).reshape(bs, length, -1)
        brdf = self.brdfnps.decode(decoder_input).reshape(bs, length, 3)
        for _ in range(4):
            brdf = torch.clamp(brdf, 0.0, 1)
            brdf = torch.exp(brdf) - 1

        n = torch.tile(n.reshape(-1, 1, 3), (1, length, 1))
        ln = torch.sum(n * wi.reshape(bs, length, 3)[0], dim=-1).reshape(-1, 1)
        ln[ln <= 0] = 0
        brdf = (brdf * intensity * (ln.reshape(bs, length, 1))).reshape(bs, length, 3)
        brdf = torch.mean(brdf, dim=1)
        return brdf

    def wiwo_to_angles(self, wi, wo):
        angles = wiwo_to_angles_torch(wi, wo)
        return torch.cat(
            [
                angles[:, :2],
                torch.sin(angles[..., 2] * 2).reshape(-1, 1),
                torch.cos(angles[..., 2] * 2).reshape(-1, 1),
            ],
            dim=1,
        )

    def clear_cache(self):
        super().clear_cache()
        self.cache["latent"] = {}


class PointLightRenderer:
    def __init__(self, brdfnps):
        super().__init__()
        self.brdfnps = brdfnps
        self.device = brdfnps.device

    @torch.no_grad()
    def render(self, latent, wi, wo):
        if isinstance(latent, np.ndarray):
            latent = torch.tensor(latent, device=self.device)
        if isinstance(wi, np.ndarray):
            wi = torch.tensor(wi, device=self.device)
        if isinstance(wo, np.ndarray):
            wo = torch.tensor(wo, device=self.device)

        angles = wiwo_to_angles_torch(wi, wo)
        angles = torch.cat(
            [
                angles[:, :2],
                torch.sin(2 * angles[:, 2]).reshape(-1, 1),
                torch.cos(2 * angles[:, 2]).reshape(-1, 1),
            ],
            dim=-1,
        )
        angles = angles.to(torch.float32)
        latent = latent.to(torch.float32)
        decoder_input = torch.cat([angles, latent], dim=-1)
        rec = self.brdfnps.decoder(decoder_input)
        for _ in range(4):
            rec = torch.exp(rec) - 1
        return rec


class DisneyRenderer(IRenderer):
    def render(self, params: BaseRenderParams, wi, wo, n, intensity):
        bs = wo.shape[0]
        length = wi.shape[0]
        length = wi.shape[0]
        if bs in self.cache["wi"]:
            wi = self.cache["wi"][bs]
            ones = self.cache["ones"][bs]
            params = self.cache["params"][bs]
        else:
            wi = torch.tile(wi, (bs, 1))
            ones = torch.ones(length, bs, 3, dtype=torch.float32, device=wi.device)
            params = copy.deepcopy(params.parameter)
            params = torch.tile(params.reshape(1, 11), (bs * length, 1))
            self.cache["wi"][bs] = wi
            self.cache["ones"][bs] = ones
            self.cache["params"][bs] = params
            self.n = torch.tensor([0, 0, 1], dtype=torch.float32).to(wi.device)
        wo = ones * wo
        wo = wo.permute(1, 0, 2).reshape(-1, 3)
        brdf = disney_bsdf_train(
            L=wi,
            V=wo,
            N=self.n,
            baseColor=params[..., :3],
            metallic=params[..., 3:4],
            specular=params[..., 4:5],
            anisotropic=params[..., 5:6],
            sheen=params[..., 6:7],
            sheenTint=params[..., 7:8],
            clearcoat=params[..., 8:9],
            clearcoatGloss=params[..., 9:10],
            roughness=params[..., 10:11],
        )
        n = torch.tile(n.reshape(-1, 1, 3), (1, length, 1))
        ln = torch.sum(n * wi.reshape(bs, length, 3)[0], dim=-1).reshape(-1, 1)
        ln[ln <= 0] = 0
        brdf = brdf.reshape(bs, -1, 3)
        ln = ln.reshape(bs, -1, 1)

        brdf = (brdf * intensity * ln).reshape(bs, -1, 3)
        brdf = torch.mean(brdf, dim=1)
        return brdf
