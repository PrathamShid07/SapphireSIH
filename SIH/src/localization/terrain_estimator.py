import numpy as np


def fit_plane_ransac(points, dist_thresh=0.15, max_iters=200):
    pts = np.array(points, dtype=np.float64)
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0, pts

    best_inliers = []
    best_plane = None

    n_pts = len(pts)
    for _ in range(max_iters):
        idx = np.random.choice(n_pts, 3, replace=False)
        p1, p2, p3 = pts[idx]

        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm

        d = -np.dot(n, p1)
        distances = np.abs(np.dot(pts, n) + d)
        inliers = np.where(distances < dist_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (n, d)

    if len(best_inliers) >= 3:
        inlier_pts = pts[best_inliers]
        centroid = np.mean(inlier_pts, axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - centroid)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        if normal[2] < 0:
            normal = -normal
        d = -np.dot(normal, centroid)
        return normal, d, inlier_pts

    normal = np.array([0.0, 0.0, 1.0])
    d = -float(np.median(pts[:, 2]))
    return normal, d, pts


class TerrainEstimator:
    def __init__(self, search_radius=10.0, dist_thresh=0.15):
        self.search_radius = search_radius
        self.dist_thresh = dist_thresh

    def estimate_ground_contact(self, position_xy, point_cloud_xyz):
        pts = np.array(point_cloud_xyz, dtype=np.float64)
        if len(pts) == 0:
            return np.array([position_xy[0], position_xy[1], 0.0]), np.array(
                [0.0, 0.0, 1.0]
            )

        dx = pts[:, 0] - position_xy[0]
        dy = pts[:, 1] - position_xy[1]
        dists_sq = dx**2 + dy**2
        nearby_mask = dists_sq < (self.search_radius**2)
        local_pts = pts[nearby_mask]

        if len(local_pts) < 5:
            sorted_indices = np.argsort(dists_sq)[: min(50, len(pts))]
            local_pts = pts[sorted_indices]

        normal, d, inliers = fit_plane_ransac(
            local_pts, dist_thresh=self.dist_thresh
        )

        px, py = float(position_xy[0]), float(position_xy[1])
        if abs(normal[2]) > 1e-4:
            pz = -(normal[0] * px + normal[1] * py + d) / normal[2]
        else:
            pz = float(np.median(local_pts[:, 2]))

        return np.array([px, py, pz], dtype=np.float64), normal
