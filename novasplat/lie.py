import numpy as np


def invert_se3(t):
    """Inverse of a 4x4 SE3 transform (R,t).

    Args:
        t: The 4x4 SE3 transform.

    Returns:
        The inverse of the 4x4 SE3 transform.
    """
    r = t[:3, :3]
    t = t[:3, 3]
    r_inv = r.T
    t_inv = -r_inv @ t
    tinv = np.eye(4)
    tinv[:3, :3] = r_inv
    tinv[:3, 3] = t_inv
    return tinv
