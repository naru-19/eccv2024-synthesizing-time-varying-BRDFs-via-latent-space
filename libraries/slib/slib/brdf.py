import numpy as np
from slib.utils import angles_to_wiwo, wiwo_to_rusinkiewicz


class BRDF:
    def __init__(self, values, angles=None, wi=None, wo=None, description=None):
        self.description = description
        self.values = np.array(values)  # length x 3
        self.length = values.shape[0]
        if angles is None:
            assert (
                wi is not None and wo is not None
            ), "wi and wo must be provided if angles is None"
            self.wi, self.wo = wi, wo
            angles = wiwo_to_rusinkiewicz(wi, wo)
            self.angles = np.array(
                [angles[0], angles[2], angles[3]]
            ).T  # only theta_h,theta_d,phi_d in angles
            assert (
                self.angles.shape == self.wi.shape
            ), f"angles{self.angles.shape} and wi {wi.shape} must have the same shape"
        else:
            self.angles = angles
            assert self.angles.shape == values.shape
            self.wi, self.wo = angles_to_wiwo(angles)
        self.get_X()
        self.values[self.wi[:, 2] <= 0] = 0
        self.all = np.concatenate(
            [self.X, self.values], axis=1
        )  # NPs用にXとvaluesを結合

    def get_X(self):
        self.X = np.concatenate(
            [
                self.angles[:, :2],
                np.sin(self.angles[:, 2] * 2).reshape(-1, 1),
                np.cos(self.angles[:, 2] * 2).reshape(-1, 1),
            ],
            axis=1,
        )

    def sample(self, k: int, seed: int = 2023):
        indexes = list(range(self.length))
        np.random.seed(seed)
        indexes_sampled = np.random.choice(indexes, k, replace=False)
        values = self.values[indexes_sampled]
        angles = self.angles[indexes_sampled]
        return BRDF(values, angles=angles)
