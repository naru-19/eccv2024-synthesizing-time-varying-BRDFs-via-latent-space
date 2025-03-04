import dataclasses
from typing import Optional, Tuple, Union

import numpy as np
from localenv import to_local
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def _expand_dim(v):
    return np.tile(v.reshape((v.shape[0], 1)), (1, 3))


@dataclasses.dataclass
class LocalEnv:
    wi: np.ndarray
    wo: np.ndarray
    h: np.ndarray
    light_point: np.ndarray
    view_point: np.ndarray

    @classmethod
    def from_normalmap(
        cls,
        normalmap,
        light_point,
        view_point: np.ndarray = np.array([0, 0, 1]),
    ):
        # wi, wo = [], []
        light_point = np.array(light_point) / np.linalg.norm(
            np.array(light_point)
        )
        view_point = np.array(view_point) / np.linalg.norm(
            np.array(view_point)
        )
        # wi, wo = to_local(normalmap, light_point, view_point)
        # update
        x = np.arctan2(normalmap[..., 1], normalmap[..., 2], dtype=np.float32)
        y = np.arctan2(
            -normalmap[..., 0],
            np.hypot(normalmap[..., 1], normalmap[..., 2]),
            dtype=np.float32,
        )
        rx = np.zeros((normalmap.shape[0], 3, 3))
        ry = np.zeros((normalmap.shape[0], 3, 3))
        rx[:, 0] = np.array([1, 0, 0])
        rx[:, 1, 1] = np.cos(x)
        rx[:, 1, 2] = -np.sin(x)
        rx[:, 2, 1] = np.sin(x)
        rx[:, 2, 2] = np.cos(x)

        ry[:, 0, 0] = np.cos(y)
        ry[:, 0, 2] = np.sin(y)
        ry[:, 2, 0] = -np.sin(y)
        ry[:, 2, 2] = np.cos(y)
        ry[:, 1] = np.array([0, 1, 0])
        ryrx = np.einsum("ijk,ikl->ijl", ry, rx)
        # ryrxn = np.einsum("ijk,ikl->ijl", ryrx, normalmap.reshape(-1, 3, 1))
        wo_loc = np.tile(wo.reshape(1, -1), (len(normalmap), 1)) - normalmap
        wo_loc = wo_loc / np.linalg.norm(wo_loc, axis=-1, keepdims=True)

        wo_loc = (
            np.tile(view_point.reshape(1, -1), (len(normalmap), 1)) - normalmap
        )
        wo_loc = wo_loc / np.linalg.norm(wo_loc, axis=-1, keepdims=True)

        ryrxv = np.einsum("ijk,ik->ij", ryrx, wo_loc)
        ryrxl = np.einsum("ijk,ikl->ijl", ryrx, light_point.reshape(-1, 3, 1))
        wo = ryrxv.reshape(-1, 3)
        wi = ryrxl.reshape(-1, 3)
        # wi = np.tile(light_point.reshape(1, 3), (normalmap.shape[0], 1))
        # kokomade
        h = wi + wo
        norm = np.linalg.norm(h, axis=1)
        norm[norm == 0] = 1
        h = h / _expand_dim(norm)
        return cls(
            wi=wi, wo=wo, h=h, light_point=light_point, view_point=view_point
        )

    @classmethod
    def get_rx(cls, x):
        ret = np.matrix(
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(x), -np.sin(x)],
                    [0, np.sin(x), np.cos(x)],
                ]
            ),
            dtype=np.float64,
        )
        return ret

    @classmethod
    def get_ry(cls, y):
        ret = np.matrix(
            np.array(
                [
                    [np.cos(y), 0, np.sin(y)],
                    [0, 1, 0],
                    [-np.sin(y), 0, np.cos(y)],
                ]
            ),
            dtype=np.float64,
        )
        return ret


def get_sphere_normalmap(size: Union[Tuple, int]) -> np.ndarray:
    if type(size) == int:
        size = (size, size)
    x = (
        np.array(list(range(-1 * size[0], size[0], 2)), dtype=np.float64) + 1
    ) / size[0]
    y = np.flip(np.repeat(x, size[0]))
    x = np.tile(x, size[0])
    z = 1 - x * x - y * y
    x[z <= 0] = 0
    y[z <= 0] = 0
    z[z <= 0] = 0
    normalmap = np.array([x, y, z**0.5]).T
    return normalmap


def get_plate_wiwo(
    size, view_point=np.array([0, 0, 1]), light_point=np.array([0, 0, 1])
):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    coord = np.stack(np.meshgrid(x, y), axis=-1)
    z = np.zeros_like(coord[..., :1])
    norm = np.concatenate([coord, z], axis=-1)
    view_point = np.tile(view_point, (size, size, 1))
    wi = view_point - norm
    wi = wi / np.linalg.norm(wi, axis=-1, keepdims=True)
    wo = np.tile(light_point, (size, size, 1))
    return wi.reshape(-1, 3), wo.reshape(-1, 3)


@dataclasses.dataclass
class LocalEnvInfLightPoint:
    wi: np.ndarray
    wo: np.ndarray
    h: np.ndarray
    light_point: np.ndarray
    view_point: np.ndarray
    n: np.ndarray

    @classmethod
    def from_normalmap(
        cls,
        normalmap,
        light_point,
        view_point: np.ndarray = np.array([0, 0, 1]),
    ):
        # wi, wo = [], []
        light_point = np.array(light_point) / np.linalg.norm(
            np.array(light_point)
        )
        view_point = np.array(view_point) / np.linalg.norm(
            np.array(view_point)
        )
        # view_point = np.array(view_point)
        # wi, wo = to_local(normalmap, light_point, view_point)
        # update
        x = np.arctan2(normalmap[..., 1], normalmap[..., 2], dtype=np.float32)
        y = np.arctan2(
            -normalmap[..., 0],
            np.hypot(normalmap[..., 1], normalmap[..., 2]),
            dtype=np.float32,
        )
        rx = np.zeros((normalmap.shape[0], 3, 3))
        ry = np.zeros((normalmap.shape[0], 3, 3))
        rx[:, 0] = np.array([1, 0, 0])
        rx[:, 1, 1] = np.cos(x)
        rx[:, 1, 2] = -np.sin(x)
        rx[:, 2, 1] = np.sin(x)
        rx[:, 2, 2] = np.cos(x)

        ry[:, 0, 0] = np.cos(y)
        ry[:, 0, 2] = np.sin(y)
        ry[:, 2, 0] = -np.sin(y)
        ry[:, 2, 2] = np.cos(y)
        ry[:, 1] = np.array([0, 1, 0])
        ryrx = np.einsum("ijk,ikl->ijl", ry, rx)
        ryrxn = np.einsum("ijk,ikl->ijl", ryrx, normalmap.reshape(-1, 3, 1))
        # wo_loc = np.tile(view_point.reshape(1, -1), (len(normalmap), 1)) - normalmap
        # wo_loc = wo_loc / np.linalg.norm(wo_loc, axis=-1, keepdims=True)
        # ryrxv = np.einsum("ijk,ik->ij", ryrx, wo_loc)
        ryrxv = np.einsum("ijk,ikl->ijl", ryrx, view_point.reshape(-1, 3, 1))

        wo = ryrxv.reshape(-1, 3)
        wi = np.tile(light_point.reshape(1, 3), (normalmap.shape[0], 1))
        # kokomade
        h = wi + wo
        norm = np.linalg.norm(h, axis=1)
        norm[norm == 0] = 1
        h = h / _expand_dim(norm)
        return cls(
            wi=wi,
            wo=wo,
            h=h,
            light_point=light_point,
            view_point=view_point,
            n=ryrxn.reshape(-1, 3),
        )

    @classmethod
    def get_rx(cls, x):
        ret = np.matrix(
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(x), -np.sin(x)],
                    [0, np.sin(x), np.cos(x)],
                ]
            ),
            dtype=np.float64,
        )
        return ret

    @classmethod
    def get_ry(cls, y):
        ret = np.matrix(
            np.array(
                [
                    [np.cos(y), 0, np.sin(y)],
                    [0, 1, 0],
                    [-np.sin(y), 0, np.cos(y)],
                ]
            ),
            dtype=np.float64,
        )
        return ret


# @dataclasses.dataclass
# class LocalEnvInfLightPoint:
#     wi: np.ndarray
#     wo: np.ndarray
#     h: np.ndarray
#     light_point: np.ndarray
#     view_point: np.ndarray

#     @classmethod
#     def from_normalmap(
#         cls, normalmap, light_point, view_point: np.ndarray = np.array([0, 0, 1])
#     ):
#         wi, wo = [], []
#         light_point = light_point / np.linalg.norm(light_point)
#         for n in tqdm(normalmap):
#             if np.linalg.norm(n) == 0:
#                 wi.append(np.array([0, 0, 0]))
#                 wo.append(np.array([0, 0, 0]))
#             else:
#                 x = np.arctan2(n[1], n[2], dtype=np.float64)
#                 y = np.arctan2(-n[0], np.hypot(n[1], n[2]), dtype=np.float64)
#                 rx, ry = cls.get_rx(x), cls.get_ry(y)
#                 _wi = light_point
#                 if _wi[2] < 0:
#                     _wi = [0, 0, 0]
#                 _wo = np.ravel(np.array(ry * rx * (np.matrix(view_point).T)))
#                 if _wo[2] < 0:
#                     _wo = [0, 0, 0]
#                 wi.append(_wi)
#                 wo.append(_wo)
#         wi = np.array(wi)
#         wo = np.array(wo)
#         h = wi + wo
#         norm = np.linalg.norm(h, axis=1)
#         norm[norm == 0] = 1
#         h = h / _expand_dim(norm)
#         return cls(wi=wi, wo=wo, h=h, light_point=light_point, view_point=view_point)

#     @classmethod
#     def get_rx(cls, x):
#         ret = np.matrix(
#             np.array(
#                 [[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]
#             ),
#             dtype=np.float64,
#         )
#         return ret

#     @classmethod
#     def get_ry(cls, y):
#         ret = np.matrix(
#             np.array(
#                 [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
#             ),
#             dtype=np.float64,
#         )
#         return ret
