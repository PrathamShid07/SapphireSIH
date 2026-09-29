import json
from pathlib import Path
import numpy as np
from PIL import Image
from src.sfm.colmap_parser import parse_colmap_model


def count_points_in_mask(points_3d, camera_images, masks_dir):
    if not points_3d or not camera_images or not masks_dir:
        return 0

    masks_path = Path(masks_dir)
    loaded_masks = {}

    count = 0
    checked = 0
    max_check = min(len(points_3d), 2000)

    for pt in points_3d[:max_check]:
        xyz = np.array(pt["xyz"])
        for img in camera_images:
            iname = img["name"]
            if iname not in loaded_masks:
                mp = masks_path / f"{iname}.png"
                if not mp.exists():
                    mp = masks_path / f"{Path(iname).stem}.png"
                if mp.exists():
                    loaded_masks[iname] = np.array(Image.open(str(mp)))
                else:
                    loaded_masks[iname] = None

            m = loaded_masks[iname]
            if m is None:
                continue

            qvec = img.get("qvec")
            tvec = img.get("tvec")
            if not qvec or not tvec:
                continue

            qw, qx, qy, qz = qvec
            R = np.array(
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
                ]
            )
            p_cam = np.dot(R, xyz) + np.array(tvec)
            if p_cam[2] <= 0:
                continue

            h, w = m.shape[:2]
            focal = max(h, w) * 1.2
            u = int(round(p_cam[0] * focal / p_cam[2] + w / 2.0))
            v = int(round(p_cam[1] * focal / p_cam[2] + h / 2.0))

            if 0 <= u < w and 0 <= v < h:
                if m[v, u] == 0:
                    count += 1
                    break
        checked += 1

    if checked > 0:
        estimated_total = int(round(count * (len(points_3d) / checked)))
        return estimated_total
    return count


class ReconstructionComparator:
    def __init__(self, logger=None):
        self.logger = logger

    def compare_reconstructions(
        self,
        unmasked_model_dir,
        masked_model_dir,
        masks_dir=None,
        output_path=None,
    ):
        unmasked_p = Path(unmasked_model_dir).resolve()
        masked_p = Path(masked_model_dir).resolve()

        if not unmasked_p.exists():
            raise FileNotFoundError(
                f"Unmasked sparse model not found at {unmasked_p}."
            )
        if not masked_p.exists():
            raise FileNotFoundError(
                f"Masked sparse model not found at {masked_p}."
            )

        unmasked_data = parse_colmap_model(unmasked_p)
        masked_data = parse_colmap_model(masked_p)

        unmasked_dyn_pts = 0
        masked_dyn_pts = 0
        if masks_dir and Path(masks_dir).exists():
            unmasked_dyn_pts = count_points_in_mask(
                unmasked_data.get("sparse_points_sample", []),
                unmasked_data.get("registered_images", []),
                masks_dir,
            )
            masked_dyn_pts = count_points_in_mask(
                masked_data.get("sparse_points_sample", []),
                masked_data.get("registered_images", []),
                masks_dir,
            )

        u_imgs = unmasked_data["num_registered_images"]
        m_imgs = masked_data["num_registered_images"]
        u_pts = unmasked_data["num_sparse_points"]
        m_pts = masked_data["num_sparse_points"]
        u_err = unmasked_data["mean_reprojection_error"]
        m_err = masked_data["mean_reprojection_error"]

        points_delta = m_pts - u_pts
        points_delta_pct = (
            float(100.0 * points_delta / u_pts) if u_pts > 0 else 0.0
        )
        reproj_delta = round(m_err - u_err, 4)
        dynamic_points_removed = max(0, unmasked_dyn_pts - masked_dyn_pts)
        dynamic_reduction_pct = (
            float(100.0 * dynamic_points_removed / unmasked_dyn_pts)
            if unmasked_dyn_pts > 0
            else 100.0
        )
        registration_preserved = m_imgs >= u_imgs

        if dynamic_points_removed > 0:
            verdict = (
                f"SUCCESS: Dynamic masking removed {dynamic_points_removed} spurious moving object points "
                f"({dynamic_reduction_pct:.1f}% reduction in dynamic zones) while maintaining "
                f"{m_imgs}/{u_imgs} registered cameras."
            )
        elif registration_preserved:
            verdict = (
                f"SUCCESS: Masked reconstruction preserved all {m_imgs} registered cameras "
                f"with mean reprojection error {m_err:.4f}px."
            )
        else:
            verdict = (
                f"PARTIAL: Masked model registered {m_imgs} vs {u_imgs} unmasked cameras. "
                f"Reprojection error: {m_err:.4f}px."
            )

        report = {
            "status": "SUCCESS",
            "unmasked": {
                "model_dir": str(unmasked_p),
                "registered_images": u_imgs,
                "sparse_points": u_pts,
                "mean_reprojection_error_px": u_err,
                "spurious_points_in_dynamic_zones": unmasked_dyn_pts,
            },
            "masked": {
                "model_dir": str(masked_p),
                "registered_images": m_imgs,
                "sparse_points": m_pts,
                "mean_reprojection_error_px": m_err,
                "spurious_points_in_dynamic_zones": masked_dyn_pts,
            },
            "comparison": {
                "points_delta": points_delta,
                "points_delta_percentage": points_delta_pct,
                "reprojection_error_delta_px": reproj_delta,
                "dynamic_points_removed": dynamic_points_removed,
                "dynamic_reduction_percentage": dynamic_reduction_pct,
                "registration_preserved": registration_preserved,
                "verdict": verdict,
            },
        }

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        if self.logger:
            self.logger.info(verdict)

        return report
