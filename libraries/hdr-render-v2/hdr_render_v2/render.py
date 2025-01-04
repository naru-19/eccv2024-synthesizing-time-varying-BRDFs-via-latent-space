import numpy as np
import torch
from hdr_render_v2.env.env import HDREnv
from hdr_render_v2.params import BaseRenderParams
from hdr_render_v2.renderer import IRenderer
from render_utils import LocalEnvInfLightPoint
from tqdm import tqdm


class RenderDataset(torch.utils.data.Dataset):
    def __init__(self, norm, device):
        env = LocalEnvInfLightPoint.from_normalmap(norm, [-3, 0, 5])
        self.V = torch.tensor(env.wo, dtype=torch.float32).to(device)
        self.n_raw = torch.tensor(norm, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.n_raw)

    def __getitem__(self, idx):
        return self.V[idx], self.n_raw[idx]


class TextureRenderDataset(torch.utils.data.Dataset):
    def __init__(self, norm, device, params):
        env = LocalEnvInfLightPoint.from_normalmap(norm, [-3, 0, 5])
        self.V = torch.tensor(env.wo, dtype=torch.float32).to(device)
        self.n_raw = torch.tensor(norm, dtype=torch.float32).to(device)
        self.params = params

    def __len__(self):
        return len(self.n_raw)

    def __getitem__(self, idx):
        return self.V[idx], self.n_raw[idx], self.params[idx]


# 実際にレンダリングするクラス
class Renderer:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def render(
        self,
        renderer: IRenderer,
        params: BaseRenderParams,
        env: HDREnv,
        material_norm: torch.Tensor,
        batch_size: int = 64,
        isAngleMode: bool = False,
    ):
        """
        render params using renderer and return rendered image
        """
        # prepare dataset
        dataset = RenderDataset(material_norm, self.device)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
        )

        params.to_device(self.device)
        wi = torch.tensor(
            env.wi, dtype=torch.float32, device=self.device
        ).reshape(-1, 3)
        intensity = torch.tensor(
            env.intensity, dtype=torch.float32, device=self.device
        ).reshape(-1, 3)
        brdf = []
        with torch.no_grad():
            for wo, n in tqdm(dataloader):
                brdf.append(renderer.render(params, wi, wo, n, intensity))
                assert isinstance(brdf[-1], torch.Tensor) or isinstance(
                    brdf[-1], np.ndarray
                )

        renderer.clear_cache()

        if isinstance(brdf[0], torch.Tensor):
            return torch.cat(brdf, dim=0)
        elif isinstance(brdf[0], np.ndarray):
            return np.concatenate(brdf, axis=0)


class TextureRenderer(Renderer):
    def render(
        self,
        renderer: IRenderer,
        params: BaseRenderParams,
        env: HDREnv,
        material_norm: np.ndarray,
        batch_size=64,
    ):
        assert len(params) == len(
            material_norm
        ), f"{len(params)}!={len(material_norm)}"
        dataset = TextureRenderDataset(
            norm=material_norm, device=self.device, params=params
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        params.to_device(self.device)
        wi = torch.tensor(
            env.wi, dtype=torch.float32, device=self.device
        ).reshape(-1, 3)
        intensity = torch.tensor(
            env.intensity, dtype=torch.float32, device=self.device
        ).reshape(-1, 3)

        brdf = []
        with torch.no_grad():
            for wo, n, param in tqdm(dataloader):
                with torch.cuda.amp.autocast():
                    brdf.append(renderer.render(param, wi, wo, n, intensity))
                    assert isinstance(brdf[-1], torch.Tensor) or isinstance(
                        brdf[-1], np.ndarray
                    )

            if isinstance(brdf[0], torch.Tensor):
                return torch.cat(brdf, dim=0)
            elif isinstance(brdf[0], np.ndarray):
                return np.concatenate(brdf, axis=0)
