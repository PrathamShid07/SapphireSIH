import numpy as np


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ],
        dtype=np.float64,
    )


def compute_camera_ray(u, v, fx, fy, cx, cy, qvec, tvec):
    xc = (float(u) - float(cx)) / float(fx)
    yc = (float(v) - float(cy)) / float(fy)
    zc = 1.0

    d_cam = np.array([xc, yc, zc], dtype=np.float64)
    norm = np.linalg.norm(d_cam)
    if norm > 0:
        d_cam /= norm

    R = qvec_to_rotmat(qvec)
    t = np.array(tvec, dtype=np.float64)

    C_world = -np.dot(R.T, t)
    d_world = np.dot(R.T, d_cam)
    d_world /= np.linalg.norm(d_world)

    return C_world, d_world


def intersect_rays_least_squares(centers, directions):
    centers = [np.array(c, dtype=np.float64) for c in centers]
    directions = [
        np.array(d, dtype=np.float64) / np.linalg.norm(d) for d in directions
    ]
    n = len(centers)

    if n == 0:
        return None, 0.0, 0.0

    if n == 1:
        return centers[0] + 5.0 * directions[0], 0.0, 0.0

    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    I = np.eye(3, dtype=np.float64)

    max_angle_deg = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cos_val = np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cos_val)))
            if angle > max_angle_deg:
                max_angle_deg = angle

    for C, d in zip(centers, directions):
        proj = I - np.outer(d, d)
        A += proj
        b += np.dot(proj, C)

    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        point = np.dot(np.linalg.pinv(A), b)

    residuals = []
    for C, d in zip(centers, directions):
        proj = I - np.outer(d, d)
        residuals.append(np.linalg.norm(np.dot(proj, point - C)))

    mean_res = float(np.mean(residuals)) if residuals else 0.0
    return point, mean_res, max_angle_deg


def intersect_ray_with_plane(C, d, plane_normal, plane_point):
    C = np.array(C, dtype=np.float64)
    d = np.array(d, dtype=np.float64) / np.linalg.norm(d)
    n = np.array(plane_normal, dtype=np.float64) / np.linalg.norm(plane_normal)
    p0 = np.array(plane_point, dtype=np.float64)

    denom = np.dot(n, d)
    if abs(denom) < 1e-6:
        return None

    t = np.dot(n, p0 - C) / denom
    if t <= 0.0:
        return None

    return C + t * d
