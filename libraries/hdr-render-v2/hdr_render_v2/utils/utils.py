import numpy as np
import torch

from RusCoord import rusinkiewicz_to_wiwo, wiwo_to_rusinkiewicz


def wiwo_to_angles(wi, wo):
    theta_h, _, theta_d, phi_d = wiwo_to_rusinkiewicz(wi, wo)
    angles = np.array([theta_h, theta_d, phi_d]).T
    return angles


def angles_to_wiwo(angles):
    theta_h, theta_d, phi_d = angles[:, 0], angles[:, 1], angles[:, 2]
    phi_h = np.zeros(theta_h.shape)
    wi, wo = rusinkiewicz_to_wiwo(theta_h, phi_h, theta_d, phi_d)
    return wi, wo


def get_merl_angles():
    h = np.arange(90) / 90
    h = h * h * np.pi / 2
    d = np.arange(90) / 90 * np.pi / 2
    phi = np.arange(180) / 180 * np.pi
    H, D, PHI = np.meshgrid(h, d, phi, indexing="ij")
    angles = np.stack((H.flatten(), D.flatten(), PHI.flatten()), axis=-1)
    return angles  # (90*90*180, 3)


def rotate_vector_around_axis(a, axis, theta):
    """
    Parameters
    ----------
    a: (batch_size, 3) tensor
    axis: (batch_size, 3) tensor
    theta: radian
    Returns
    -------
    rotated: rotated tensor
    """
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    axis = torch.tile(axis.reshape(1, 3), (a.shape[0], 1))
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    rotation_matrix = torch.stack(
        [
            cos_theta + axis[:, 0] ** 2 * (1 - cos_theta),
            axis[:, 0] * axis[:, 1] * (1 - cos_theta) - axis[:, 2] * sin_theta,
            axis[:, 0] * axis[:, 2] * (1 - cos_theta) + axis[:, 1] * sin_theta,
            axis[:, 1] * axis[:, 0] * (1 - cos_theta) + axis[:, 2] * sin_theta,
            cos_theta + axis[:, 1] ** 2 * (1 - cos_theta),
            axis[:, 1] * axis[:, 2] * (1 - cos_theta) - axis[:, 0] * sin_theta,
            axis[:, 2] * axis[:, 0] * (1 - cos_theta) - axis[:, 1] * sin_theta,
            axis[:, 2] * axis[:, 1] * (1 - cos_theta) + axis[:, 0] * sin_theta,
            cos_theta + axis[:, 2] ** 2 * (1 - cos_theta),
        ]
    ).T.reshape(-1, 3, 3)
    rotated = torch.einsum("ijk,ik->ij", rotation_matrix, a)
    return rotated


def to_spherical(v):
    """
    Parameters
    ----------
    v: (batch_size, 3) tensor
    Returns
    -------
    theta, phi
    """
    norm = torch.norm(v, dim=1)
    norm[norm == 0] = 1
    theta = torch.acos(v[:, 2] / norm)
    phi = torch.atan2(v[:, 1], v[:, 0])
    return theta, phi


def wiwo_to_angles_torch(wi, wo):
    """
    Parameters
    ----------
    wi: (batch_size, 3) incoming vector
    wo: (batch_size, 3) outgoing vector
    Returns
    -------
    theta_h,theta_d,phi_d
    """
    # half vector
    h = wi + wo
    hnorm = torch.norm(h, dim=1, keepdim=True)
    hnorm[hnorm == 0] = 1
    h = h / hnorm
    theta_h, phi_h = to_spherical(h)
    # difference vector
    bi_normal = torch.tensor([0.0, 1.0, 0.0], device=wi.device)
    normal = torch.tensor([0.0, 0.0, 1.0], device=wi.device)
    tmp = rotate_vector_around_axis(wi, normal, -phi_h)
    d = rotate_vector_around_axis(tmp, bi_normal, -theta_h)
    theta_d, phi_d = to_spherical(d)
    return torch.cat(
        [
            theta_h.unsqueeze(1),
            theta_d.unsqueeze(1),
            phi_d.unsqueeze(1),
        ],
        dim=-1,
    )
