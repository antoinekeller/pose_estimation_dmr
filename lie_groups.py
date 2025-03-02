import torch
from scipy.spatial.transform import Rotation


def exp_map_SO3(tangent_vector: torch.Tensor) -> torch.Tensor:
    """Compute the exponential map of SO(3).

    Args:
        tangent_vector: Tangent vector; an `so(3)` tangent vector.
    Returns:
        R rotation matrices.
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros(
        (log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device
    )
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R


def log_map_SO3(R: torch.Tensor) -> torch.Tensor:
    """Compute the logarithm map of SO(3).

    Args:
        R: Rotation matrices.
    Returns:
        log_rot: An `so(3)` tangent vector.
    """
    # code for SO3 map grabbed from pytorch3d and stripped

    # Compute the rotation angle with trace
    cos_angle = 0.5 * (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = cos_angle.acos()

    # Compute the skew-symmetric matrix
    log_rot = torch.zeros((R.shape[0], 3), dtype=R.dtype, device=R.device)
    mask = angle > 1e-6
    angle_sin = angle.sin()
    diag = torch.ones_like(angle)
    diag[mask] = angle[mask] / (2.0 * angle_sin[mask])
    diag_f = diag.float()
    # what is this diag ?
    log_rot[:, 0] = (R[:, 2, 1] - R[:, 1, 2]) * diag_f
    log_rot[:, 1] = (R[:, 0, 2] - R[:, 2, 0]) * diag_f
    log_rot[:, 2] = (R[:, 1, 0] - R[:, 0, 1]) * diag_f

    return log_rot


if __name__ == "__main__":
    R = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()
    print(R)

    log = log_map_SO3(torch.tensor([R]))
    print(log)

    print(exp_map_SO3(log))

    crap = torch.tensor([[0.1, 0.2, 0.3]])
    print(log_map_SO3(exp_map_SO3(crap)), crap)
