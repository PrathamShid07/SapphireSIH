import numpy as np

CLASS_DIMENSION_PRIORS = {
    "car": {"length": 4.4, "width": 1.8, "height": 1.5},
    "van": {"length": 5.2, "width": 2.0, "height": 2.2},
    "truck": {"length": 8.0, "width": 2.5, "height": 3.2},
    "bus": {"length": 10.5, "width": 2.6, "height": 3.2},
}


def yaw_to_rotmat_and_quat(yaw_rad):
    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))

    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    half_yaw = yaw_rad / 2.0
    qw = float(np.cos(half_yaw))
    qx = 0.0
    qy = 0.0
    qz = float(np.sin(half_yaw))

    return R, [qw, qx, qy, qz]


class Box3DEstimator:
    def __init__(self, dimension_priors=None):
        self.priors = dimension_priors or CLASS_DIMENSION_PRIORS

    def estimate_3d_box(
        self,
        class_name,
        center_3d,
        ground_contact_3d,
        observations,
        triangulation_residual=0.0,
        baseline_angle_deg=0.0,
        is_dynamic=False,
    ):
        cname = class_name.lower()
        prior = self.priors.get(cname, self.priors["car"])

        p_len = prior["length"]
        p_wid = prior["width"]
        p_hgt = prior["height"]

        if center_3d is not None and ground_contact_3d is not None:
            est_h = 2.0 * abs(center_3d[2] - ground_contact_3d[2])
            if 0.5 * p_hgt <= est_h <= 2.0 * p_hgt:
                box_height = float(0.6 * p_hgt + 0.4 * est_h)
            else:
                box_height = float(p_hgt)
        else:
            box_height = float(p_hgt)

        box_length = float(p_len)
        box_width = float(p_wid)

        yaw_deg = 0.0
        yaw_rad = 0.0

        if is_dynamic and len(observations) >= 2:
            vels = [
                obs.get("velocity", [0, 0])
                for obs in observations
                if "velocity" in obs
            ]
            vx = float(np.mean([v[0] for v in vels])) if vels else 0.0
            vy = float(np.mean([v[1] for v in vels])) if vels else 0.0
            if abs(vx) > 0.1 or abs(vy) > 0.1:
                yaw_rad = float(np.arctan2(vy, vx))
                yaw_deg = float(np.degrees(yaw_rad))

        R_mat, quat = yaw_to_rotmat_and_quat(yaw_rad)

        det_scores = [
            obs.get("confidence", 0.5)
            for obs in observations
            if "confidence" in obs
        ]
        mean_score = float(np.mean(det_scores)) if det_scores else 0.5
        obs_factor = min(1.0, len(observations) / 4.0)
        ang_factor = min(1.0, max(0.2, baseline_angle_deg / 15.0))
        res_factor = float(np.exp(-triangulation_residual / 1.0))

        confidence = float(
            np.clip(
                0.4 * mean_score
                + 0.25 * obs_factor
                + 0.2 * ang_factor
                + 0.15 * res_factor,
                0.1,
                0.98,
            )
        )

        return {
            "dimensions": {
                "length": round(box_length, 3),
                "width": round(box_width, 3),
                "height": round(box_height, 3),
            },
            "orientation": {
                "yaw_deg": round(yaw_deg, 2),
                "rotation_matrix": [[round(x, 6) for x in row] for row in R_mat],
                "quaternion": [round(x, 6) for x in quat],
            },
            "confidence": round(confidence, 4),
        }
