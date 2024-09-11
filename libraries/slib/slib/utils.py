import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def rotate_vector_around_axis(a, axis, theta):
    """
    Parameters
    ----------
    a: (3,) vector
    axis: (3,) vector
    theta: radian
    Returns
    -------
    rotated: rotated vector
    """
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = np.tile(axis.reshape(1, 3), (a.shape[0], 1))
    rotation = Rotation.from_rotvec(
        np.tile(theta.reshape(-1, 1), (1, 3)) * np.array(axis)
    )
    R = rotation.as_matrix()
    rotated = np.matmul(R, a.reshape(*a.shape, 1))
    return rotated.reshape(*a.shape)


def to_spherical(v):
    """
    Convert vector v to spherical coordinates, (theta, phi)
    """
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    return theta, phi


def wiwo_to_rusinkiewicz(wi, wo):
    """
    Parameters
    ----------
    wi: (length, 3) incoming vector
    wo: (length, 3) outgoing vector
    Returns
    -------
    theta_h,phi_h,theta_d,phi_d
    """
    length = wi.shape[0]
    # half vector
    h = wi + wo
    hnorm = np.tile(np.linalg.norm(h, axis=1).reshape(-1, 1), (1, 3))
    hnorm[hnorm == 0] = 1
    h = h / hnorm
    theta_h, phi_h = to_spherical(h)
    # difference vector
    bi_normal = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    d = []
    tmp = rotate_vector_around_axis(wi, normal, -phi_h)
    d = rotate_vector_around_axis(tmp, bi_normal, -theta_h)
    d = np.array(d)
    theta_d, phi_d = to_spherical(d)
    return theta_h, phi_h, theta_d, phi_d


def rusinkiewicz_to_wiwo(theta_h, phi_h, theta_d, phi_d):
    """
    Convert rusinkiewicz parameter to wi, wo
    Parameters
    ----------
    theta_h,phi_h,theta_d,phi_d: (length)

    Returns
    -------
    wi,wo: (length, 3) vector
    """
    length = len(theta_h)
    h = np.array(
        [
            np.sin(theta_h) * np.cos(phi_h),
            np.sin(theta_h) * np.sin(phi_h),
            np.cos(theta_h),
        ]
    ).T  # (length, 3)
    d = np.array(
        [
            np.sin(theta_d) * np.cos(phi_d),
            np.sin(theta_d) * np.sin(phi_d),
            np.cos(theta_d),
        ]
    ).T  # (length, 3)
    assert h.shape == d.shape
    # wiwo_to_rusinkiewiczの逆に回転
    bi_normal = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    wi, wo = [], []
    # for i in range(length):
    tmp = rotate_vector_around_axis(d, bi_normal, theta_h)
    wi = rotate_vector_around_axis(tmp, normal, phi_h)
    # ↓https://github.com/asztr/Neural-BRDF/tree/main/binary_to_nbrdf
    wo = 2 * np.tile(np.sum(wi * h, axis=1).reshape(-1, 1), (1, 3)) * h - wi
    wo = wo / np.tile(np.linalg.norm(wo, axis=1).reshape(-1, 1), (1, 3))
    return wi, wo


def angles_to_wiwo(angles):
    theta_h, theta_d, phi_d = angles[:, 0], angles[:, 1], angles[:, 2]
    phi_h = np.zeros(theta_h.shape)
    wi, wo = rusinkiewicz_to_wiwo(theta_h, phi_h, theta_d, phi_d)
    return wi, wo


def wiwo_to_angles(wi, wo):
    theta_h, _, theta_d, phi_d = wiwo_to_rusinkiewicz(wi, wo)
    angles = np.array([theta_h, theta_d, phi_d]).T
    return angles
