import ctypes
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


def _to_pp(arr: np.ndarray):
    return (
        arr.__array_interface__["data"][0] + np.arange(arr.shape[0]) * arr.strides[0]
    ).astype(np.uintp)


def to_local(
    normalmap: np.ndarray,
    light_point: Union[List[float], np.ndarray] = np.array([-3, 0, 5]),
    view_point: Union[List[float], np.ndarray] = np.array([0, 0, 1]),
) -> Tuple[np.ndarray, np.ndarray]:
    # preprocess: view_point
    view_point = np.array(view_point)
    view_point = view_point / np.linalg.norm(view_point)
    view = view_point.astype(np.float32)

    lib = np.ctypeslib.load_library("/brdfnps/libs/localenv/localenv/env_api.so", ".")
    # 二次元配列のpointer type
    ctype_vector = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags="C")
    lib.env_from_normalmap.argtypes = [
        ctype_vector,
        ctype_vector,
        ctype_vector,
        ctypes.c_int32,
    ]
    lib.env_from_normalmap.restype = None
    # c++実装に対応した引数へ変換
    normalmap = normalmap.astype(np.float32)
    length = normalmap.shape[0]
    wo = np.zeros((length, 3), dtype=np.float32)
    normalmap = normalmap.copy()
    normalmap_pp = _to_pp(normalmap)
    # c++で実行
    lib.env_from_normalmap(normalmap_pp, _to_pp(wo), _to_pp(view), length)
    # 無限遠光源
    light_point = np.array(light_point)
    light_point = (light_point / np.linalg.norm(light_point)).reshape(1, 3)
    wi = np.tile(light_point, (length, 1))
    wi[normalmap[:, 2] <= 0] = 0
    return wi, wo
