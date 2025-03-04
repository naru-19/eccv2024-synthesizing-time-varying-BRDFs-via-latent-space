import numpy as np

from render_utils.LocalEnv import (
    LocalEnv,
    LocalEnvInfLightPoint,
    get_sphere_normalmap,
    get_plate_wiwo,
)
from render_utils.merl import Merl


def gamma_postprocess(img: np.ndarray, gamma: float = 1.0):
    # if img.max() != 0.0:
    #     img /= img.max()
    return img**gamma
