import torch
from hdr_render_v2.env import HDREnv
from hdr_render_v2.params import BaseTextureRenderParams
from hdr_render_v2.renderer import IRenderer
from hdr_render_v2.utils import torspa
from tqdm import tqdm
from hdr_render_v2.utils import torspa, wiwo_to_angles_torch, torspa3d
from DisneyBSDF import disney_bsdf


class TorranceSparrowTextureRenderer(IRenderer):
    def render(self, params: BaseTextureRenderParams, wi, wo, n, intensity):
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
        else:
            wi = torch.tile(wi, (bs, 1))
            ones = torch.ones(
                length, bs, 3, dtype=torch.float32, device=wi.device
            )
        if params.shape[-1] == 5:
            params = torch.tile(
                params.reshape(bs, 1, 5), (1, length, 1)
            ).reshape(-1, 5)
            wo = ones * wo
            wo = wo.permute(1, 0, 2).reshape(-1, 3)
            brdf = torspa(
                params[:, :3], params[:, 3:4], params[:, 4:5], wi, wo
            )
            n = torch.tile(n.reshape(-1, 1, 3), (1, length, 1))
            ln = torch.sum(n * wi.reshape(bs, length, 3)[0], dim=-1).reshape(
                -1, 1
            )
            ln[ln <= 0] = 0
            brdf = brdf.reshape(bs, -1, 3)
            ln = ln.reshape(bs, -1, 1)

            brdf = (brdf * intensity * ln).reshape(bs, -1, 3)
            brdf = torch.mean(brdf, dim=1)
            return brdf
        elif params.shape[-1] == 7:
            params = torch.tile(
                params.reshape(bs, 1, 7), (1, length, 1)
            ).reshape(-1, 7)
            wo = ones * wo
            wo = wo.permute(1, 0, 2).reshape(-1, 3)
            brdf = torspa3d(
                params[:, :3], params[:, 3:6], params[:, -1:], wi, wo
            )
            n = torch.tile(n.reshape(-1, 1, 3), (1, length, 1))
            ln = torch.sum(n * wi.reshape(bs, length, 3)[0], dim=-1).reshape(
                -1, 1
            )
            ln[ln <= 0] = 0
            brdf = brdf.reshape(bs, -1, 3)
            ln = ln.reshape(bs, -1, 1)

            brdf = (brdf * intensity * ln).reshape(bs, -1, 3)
            brdf = torch.mean(brdf, dim=1)
            return brdf


class BRDFNPsTextureRenderer(IRenderer):
    def __init__(self, brdfnps):
        super().__init__()
        self.brdfnps = brdfnps
        self.device = brdfnps.device

    def render(self, params: BaseTextureRenderParams, wi, wo, n, intensity):
        bs = wo.shape[0]
        assert wo.shape == (bs, 3), f"{wo.shape}!={(bs,3)}"
        assert n.shape == (bs, 3), f"{n.shape}!={(bs,3)}"
        length = wi.shape[0]
        if bs in self.cache["wi"]:
            wi = self.cache["wi"][bs]
            ones = self.cache["ones"][bs]
        else:
            wi = torch.tile(wi, (bs, 1))
            ones = torch.ones((length, bs, 3)).to(wi.device)
            self.cache["wi"][bs] = wi
            self.cache["ones"][bs] = ones
        latent = torch.tile(
            params.reshape(bs, 1, -1), (1, length, 1)
        ).reshape(-1, params.shape[-1])

        wo = ones * wo
        wo = wo.permute(1, 0, 2).reshape(-1, 3)
        angles = self.wiwo_to_angles(wi, wo).reshape(-1, 4)
        decoder_input = torch.cat([angles, latent], dim=1).reshape(
            bs, length, -1
        )
        brdf = self.brdfnps.decode(decoder_input).reshape(bs, length, 3)
        for _ in range(4):
            brdf = torch.clamp(brdf, 0.0, 2)
            brdf = torch.exp(brdf) - 1

        n = torch.tile(n.reshape(-1, 1, 3), (1, length, 1))
        ln = torch.sum(n * wi.reshape(bs, length, 3)[0], dim=-1).reshape(-1, 1)
        ln[ln <= 0] = 0
        brdf = (brdf * intensity * (ln.reshape(bs, length, 1))).reshape(
            bs, length, 3
        )
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


class DisneyTextureRenderer(IRenderer):
    def __init__(self):
        super().__init__()
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
        self.n = torch.tensor([0, 0, 1], dtype=torch.float32)

    def render(self, params, wi, wo, n, intensity):
        bs = wo.shape[0]
        assert wo.shape == (bs, 3), f"{wo.shape}!={(bs,3)}"
        assert n.shape == (bs, 3), f"{n.shape}!={(bs,3)}"
        length = wi.shape[0]
        if bs in self.cache["wi"]:
            wi = self.cache["wi"][bs]
            ones = self.cache["ones"][bs]
        else:
            wi = torch.tile(wi, (bs, 1))
            ones = torch.ones(
                length, bs, 3, dtype=torch.float32, device=wi.device
            )
            self.n = self.n.to(wi.device)
        disney_params = {}
        for i, key in enumerate(self.keys):
            if key == "base_color":
                d = 3
            else:
                d = 1
            disney_params[key] = torch.tile(
                params[:, i : i + d].reshape(bs, 1, -1), (1, length, 1)
            ).reshape(-1, d)

        wo = ones * wo
        wo = wo.permute(1, 0, 2).reshape(-1, 3)
        brdf = disney_bsdf(L=wi, V=wo, N=self.n, params=disney_params)
        n = torch.tile(n.reshape(-1, 1, 3), (1, length, 1))
        ln = torch.sum(n * wi.reshape(bs, length, 3)[0], dim=-1).reshape(-1, 1)
        ln[ln <= 0] = 0
        brdf = brdf.reshape(bs, -1, 3)
        ln = ln.reshape(bs, -1, 1)

        brdf = (brdf * intensity * ln).reshape(bs, -1, 3)
        brdf = torch.mean(brdf, dim=1)
        return brdf
