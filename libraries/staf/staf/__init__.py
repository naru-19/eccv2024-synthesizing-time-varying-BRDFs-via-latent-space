import array
import dataclasses
import glob
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import Imath as im
import numpy as np
import OpenEXR as oe
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

from TorranceSparrow.torrance_sparrow_model import TorranceSparrowParams


# 多項式のデータ(STAF)
@dataclasses.dataclass
class STAF:
    kd_polynomial: List[np.ndarray]
    ks_polynomial: List[np.ndarray]
    sigma_polynomial: List[np.ndarray]
    size: Tuple[int, int]
    length: int

    @classmethod
    def parse_file(cls, fpath: Union[Path, str]):
        exr = oe.InputFile(str(fpath))
        dw = exr.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        pt = im.PixelType(im.PixelType.FLOAT)
        img = np.zeros((*size, 3))
        color = exr.channels("RGB", pt)
        for i, ch in enumerate(color):
            img[..., i] = np.array(array.array("f", ch)).reshape(size)
        return img

    @classmethod
    def parse_files(cls, target_dir: Union[Path, str]):
        target_dir = Path(target_dir)
        if not target_dir.exists():
            raise FileNotFoundError(f"{target_dir} does not exist.")
        diffuse_files = sorted(list(target_dir.glob("Diffuse-*.exr")))
        specular_files = sorted(list(target_dir.glob("Specular-*.exr")))
        roughness_files = sorted(list(target_dir.glob("Roughness-*.exr")))        
        size = [cls.parse_file(fpath) for fpath in diffuse_files][0].shape[:-1]
        return cls(
            kd_polynomial=[cls.parse_file(fpath) for fpath in diffuse_files],
            ks_polynomial=[cls.parse_file(fpath) for fpath in specular_files],
            sigma_polynomial=[cls.parse_file(fpath) for fpath in roughness_files],
            size=size,
            length=size[0] * size[1],
        )
        
    def get_tsp(self, t: float, px: Optional[int] = None, device="cpu") -> TorranceSparrowParams:
        if px is not None:
            H, W = self.size
            h = px // W
            w = px % W

            tsp = TorranceSparrowParams(
                kd=self.calc_polynomial(np.array(self.kd_polynomial)[:, h, w], t, device),
                ks=self.calc_polynomial(np.array(self.ks_polynomial)[:, h, w], t, device),
                sigma=self.calc_polynomial(np.array(self.sigma_polynomial)[:, h, w], t, device),
            )

            return tsp
        else:
            tsp = TorranceSparrowParams(
                kd=self.calc_polynomial(self.kd_polynomial, t, device),
                ks=self.calc_polynomial(self.ks_polynomial, t, device),
                sigma=self.calc_polynomial(self.sigma_polynomial, t, device),
            )

            return tsp
    
    def calc_polynomial(self, polynomial: List[np.ndarray], t: float, device: str) -> np.ndarray:
        value_at_t = 0
        for i, coefficient in enumerate(polynomial):
            value_at_t += coefficient * t**i
        return torch.tensor(value_at_t).to(device)
        