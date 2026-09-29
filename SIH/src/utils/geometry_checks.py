import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np

def compute_pca_point_cloud(
    points_xyz: np.ndarray,
    warn_threshold: float = 500.0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    if isinstance(points_xyz, tuple):
        raise TypeError("points_xyz must be an (N, 3) array, not a tuple; unpack read_ply_points() first")
    if points_xyz is None or len(points_xyz) < 4:
        return {
            "eigenvalue_ratio": 1.0,
            "eigenvalues": [0.0, 0.0, 0.0],
            "variance_explained": [0.3333, 0.3333, 0.3333],
            "dominant_axis": [1.0, 0.0, 0.0],
            "flattest_axis": [0.0, 0.0, 1.0],
            "is_flattened": False,
            "warn_threshold": float(warn_threshold),
            "num_points": len(points_xyz) if points_xyz is not None else 0,
            "check_performed": False,
            "status": "insufficient_data",
        }

    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3), got {pts.shape}")

    centroid = np.mean(pts, axis=0)
    centered = pts - centroid

    cov = np.cov(centered, rowvar=False)

    evals, evecs = np.linalg.eigh(cov)

    idx = np.argsort(evals)[::-1]
    evals = np.maximum(evals[idx], 1e-12)
    evecs = evecs[:, idx]

    total_var = np.sum(evals)
    variance_explained = (evals / total_var).tolist() if total_var > 0 else [0.3333, 0.3333, 0.3333]

    ratio = float(evals[0] / evals[2])
    dominant_axis = evecs[:, 0].tolist()
    flattest_axis = evecs[:, 2].tolist()

    is_flattened = bool(ratio > warn_threshold)

    if is_flattened and logger:
        logger.warning(
            f"Geometry check WARNING: Point cloud is severely flattened! "
            f"Largest/smallest eigenvalue ratio is {ratio:.1f}:1 (threshold: {warn_threshold:.1f}:1). "
            f"Dominant axis: [{dominant_axis[0]:.3f}, {dominant_axis[1]:.3f}, {dominant_axis[2]:.3f}], "
            f"Flattest axis: [{flattest_axis[0]:.3f}, {flattest_axis[1]:.3f}, {flattest_axis[2]:.3f}]."
        )

    return {
        "eigenvalue_ratio": float(round(ratio, 2)),
        "eigenvalues": [float(round(e, 6)) for e in evals],
        "variance_explained": [float(round(v, 4)) for v in variance_explained],
        "dominant_axis": [float(round(x, 4)) for x in dominant_axis],
        "flattest_axis": [float(round(x, 4)) for x in flattest_axis],
        "is_flattened": is_flattened,
        "warn_threshold": float(warn_threshold),
        "num_points": int(len(pts)),
        "check_performed": True,
        "status": "fail" if is_flattened else "pass",
    }

def check_ply_geometry(
    ply_path: Union[str, Path],
    warn_threshold: float = 500.0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    ppath = Path(ply_path).resolve()
    if not ppath.exists():
        if logger:
            logger.warning(f"PLY file not found for geometry check: {ppath}")
        result = compute_pca_point_cloud(np.empty((0, 3)), warn_threshold=warn_threshold, logger=logger)
        result["status"] = "error"
        result["check_status"] = "FILE_NOT_FOUND"
        return result

    try:
        from src.dense.point_cloud_cleaner import read_ply_points
        pts, _, _ = read_ply_points(ppath)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to read PLY file {ppath} for geometry check: {e}")
        result = compute_pca_point_cloud(np.empty((0, 3)), warn_threshold=warn_threshold, logger=logger)
        result["status"] = "error"
        result["check_status"] = "READ_FAILED"
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result = compute_pca_point_cloud(pts, warn_threshold=warn_threshold, logger=logger)
    result["check_status"] = "OK" if result["check_performed"] else "INSUFFICIENT_POINTS"
    return result
