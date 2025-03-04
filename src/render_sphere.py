import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from brdf_enc_dec.model import BRDFNPs
from hdr_render_v2 import Renderer, TextureRenderer
from hdr_render_v2.env import SphereHDREnv
from hdr_render_v2.params import BRDFNPsTextureRenderParams
from hdr_render_v2.renderer import BRDFNPsTextureRenderer
from PIL import Image
from tqdm import tqdm

from settings import get_settings

settings = get_settings()

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")


def get_pot_norm(size):
    pot = np.load(settings.pot_norm_path)
    pot = Image.fromarray(((pot + 1) / 2 * 255).astype(np.uint8))
    pot = pot.resize((size, size))
    pot = np.array(pot).astype(np.float32)
    return pot * 2 / 255 - 1.0


def pp_exp_pot(img, exposure=0.5, gamma=2.2, norm=True):
    if len(img.shape) == 3 and norm:
        norm3d = get_pot_norm(img.shape[0])
        ret = (1 - np.exp(-2 * exposure * img)) ** (1 / gamma)
        ret[norm3d[..., 2] <= 0] = 1.0
        return ret
    return (1 - np.exp(-2 * exposure * img)) ** (1 / gamma)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--latent_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--begin", type=int, default=0, help="begin frame")
    parser.add_argument("--out_dir", type=Path, default="./output")
    args = parser.parse_args()
    print(args)
    device = f"cuda:{args.gpu}"
    brdfnps = BRDFNPs(device=device, onlyDecoder=True)
    brdfnps.latent_dim = 10

    args.out_dir.mkdir(exist_ok=True, parents=True)

    steps = list(range(args.begin, settings.frames, args.step))
    renderer = TextureRenderer(device)
    trenderer = BRDFNPsTextureRenderer(brdfnps)
    latent = np.load(args.latent_path)
    assert len(latent.shape) == 3
    size = int(np.sqrt(latent.shape[0]))
    print("material norm: ", size)
    material_norm = get_pot_norm(size)

    env = pd.read_pickle(settings.env_path)

    sphere_rendered_imgs = []
    for step in steps:
        print(step)
        save_path = args.out_dir / f"{step:03d}.npy"
        if save_path.exists():
            sphere_rendered_imgs.append(np.load(save_path))
            print(f"skip step (file exists) {step}")
            continue
        # print(f"{cfg.suffix} rendering step {step}")
        img = (
            renderer.render(
                renderer=trenderer,
                params=BRDFNPsTextureRenderParams(latent[:, step, :]),
                env=env,
                material_norm=material_norm.reshape(-1, 3),
                batch_size=args.batch_size,
            )
            .reshape(size, size, 3)
            .cpu()
            .numpy()
        )
        np.save(save_path, img)


if __name__ == "__main__":
    main()
