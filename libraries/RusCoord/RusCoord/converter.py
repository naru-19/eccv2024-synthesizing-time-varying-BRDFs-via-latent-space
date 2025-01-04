import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def my_logger(f):
    def _wrapper(*args, **keywords):
        print("\033[32m" + str(f.__name__) + "の実行" + "\033[0m")
        v = f(*args, **keywords)
        return v

    return _wrapper


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


@my_logger
def test_wiwo_to_rusinkiewicz(wi, wo, idx=0):
    if len(wi.shape) == 1:
        wi = wi.reshape(1, 3)
        wo = wo.reshape(1, 3)
    theta_h, phi_h, theta_d, phi_d = wiwo_to_rusinkiewicz(wi, wo)
    print("theta_h", theta_h[idx])
    print("phi_h", phi_h[idx])
    print("theta_d", theta_d[idx])
    print("phi_d", phi_d[idx])
    return theta_h, phi_h, theta_d, phi_d


@my_logger
def test_render(merl, theta_h, theta_d, phi_d, norm):
    img = []
    for h, d, phi in zip(theta_h, theta_d, phi_d):
        img.append(merl.eval_interp(h, d, phi))
    img = np.array(img)
    img[norm[:, 2] <= 0] = 0
    img = img.reshape(100, 100, 3)
    plt.imshow(img * 3)
    plt.savefig("/brdfnps/notebooks/test1.png")


@my_logger
def test_rusinkiewicz_to_wiwo(theta_h, phi_h, theta_d, phi_d, idx=0):
    wi, wo = rusinkiewicz_to_wiwo(theta_h, phi_h, theta_d, phi_d)
    print("wi", wi[idx])
    print("wo", wo[idx])
    return wi, wo


if __name__ == "__main__":
    from merl import Merl
    from render_utils import LocalEnvInfLightPoint, get_sphere_normalmap

    merl = Merl("/dataset/merl_data/gold-paint.binary")
    size = (100, 100, 3)
    norm = get_sphere_normalmap(size[:2])
    env = LocalEnvInfLightPoint.from_normalmap(norm, [-3, 2, 5])
    print("=" * 10, "GT", "=" * 10)
    print("wi", env.wi[5050])
    print("wo", env.wo[5050])
    h, phi_h, d, phi_d = test_wiwo_to_rusinkiewicz(env.wi, env.wo, idx=5050)
    print("=" * 10, "wiwo->rusin->wiwo", "=" * 10)
    phi_h[:] = 0
    wi, wo = test_rusinkiewicz_to_wiwo(h, phi_h, d, phi_d, idx=5050)
    h, phi_h, d, phi_d = test_wiwo_to_rusinkiewicz(wi, wo, idx=5050)
    test_render(merl, h, d, phi_d, norm)
