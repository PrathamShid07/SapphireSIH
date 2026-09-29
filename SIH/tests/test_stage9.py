import json
from pathlib import Path
import numpy as np
import pytest

from src.localization.ray_triangulator import (
    compute_camera_ray,
    intersect_rays_least_squares,
    intersect_ray_with_plane,
    qvec_to_rotmat,
)
from src.localization.terrain_estimator import fit_plane_ransac, TerrainEstimator
from src.localization.box3d_estimator import (
    Box3DEstimator,
    yaw_to_rotmat_and_quat,
)
from src.localization.pointcloud_visualizer import (
    get_box_corners_3d,
    generate_wireframe_points,
    export_localized_scene_ply,
    generate_bev_visualization,
)
from src.localization.stage9_pipeline import run_stage9


def test_compute_camera_ray():
    qvec = [1.0, 0.0, 0.0, 0.0]
    tvec = [0.0, 0.0, -10.0]
    fx = fy = 1000.0
    cx = cy = 500.0
    u, v = 500.0, 500.0

    C, d = compute_camera_ray(u, v, fx, fy, cx, cy, qvec, tvec)
    assert np.allclose(C, [0.0, 0.0, 10.0])
    assert np.allclose(d, [0.0, 0.0, 1.0])
    assert abs(np.linalg.norm(d) - 1.0) < 1e-6


def test_intersect_rays_least_squares_exact():
    target_P = np.array([5.0, 12.0, 3.0], dtype=np.float64)

    centers = [
        np.array([0.0, 0.0, 10.0]),
        np.array([10.0, 0.0, 12.0]),
        np.array([5.0, 20.0, 15.0]),
    ]
    directions = [(target_P - C) / np.linalg.norm(target_P - C) for C in centers]

    recovered, residual, max_angle = intersect_rays_least_squares(
        centers, directions
    )
    assert np.allclose(recovered, target_P, atol=1e-4)
    assert residual < 1e-4
    assert max_angle > 10.0


def test_intersect_ray_with_plane():
    C = [0.0, 0.0, 10.0]
    d = [0.0, 0.0, -1.0]
    normal = [0.0, 0.0, 1.0]
    point = [0.0, 0.0, 2.0]

    hit = intersect_ray_with_plane(C, d, normal, point)
    assert hit is not None
    assert np.allclose(hit, [0.0, 0.0, 2.0])


def test_fit_plane_ransac():
    np.random.seed(42)
    n_pts = 50
    x = np.random.uniform(-10, 10, n_pts)
    y = np.random.uniform(-10, 10, n_pts)
    z = np.full(n_pts, 3.5) + np.random.normal(0, 0.01, n_pts)
    pts = np.vstack([x, y, z]).T

    normal, d, inliers = fit_plane_ransac(pts, dist_thresh=0.1)
    assert abs(abs(normal[2]) - 1.0) < 0.05
    assert len(inliers) >= 40


def test_terrain_estimator():
    pts = np.random.uniform(-5, 5, (100, 3))
    pts[:, 2] = 1.25

    estimator = TerrainEstimator(search_radius=10.0, dist_thresh=0.1)
    ground_pt, normal = estimator.estimate_ground_contact([0.0, 0.0], pts)
    assert abs(ground_pt[2] - 1.25) < 0.1
    assert abs(abs(normal[2]) - 1.0) < 0.1


def test_box3d_estimator():
    estimator = Box3DEstimator()
    obs = [{"confidence": 0.9, "velocity": [2.0, 1.0]}]
    res = estimator.estimate_3d_box(
        class_name="car",
        center_3d=[5.0, 5.0, 2.0],
        ground_contact_3d=[5.0, 5.0, 1.0],
        observations=obs,
        is_dynamic=True,
    )

    dims = res["dimensions"]
    assert dims["length"] > 3.0
    assert dims["width"] > 1.0
    assert dims["height"] > 1.0

    quat = res["orientation"]["quaternion"]
    assert abs(np.linalg.norm(quat) - 1.0) < 1e-4
    assert 0.0 < res["confidence"] <= 1.0


def test_get_box_corners_3d():
    center = [0.0, 0.0, 0.0]
    dims = {"length": 4.0, "width": 2.0, "height": 1.0}
    R = np.eye(3)
    corners = get_box_corners_3d(center, dims, R)
    assert corners.shape == (8, 3)
    assert np.allclose(corners[0], [2.0, 1.0, 0.5])


def test_visualizer_exports(tmp_path):
    out_ply = tmp_path / "scene.ply"
    out_png = tmp_path / "bev.png"

    objs = [
        {
            "track_id": 1,
            "class_name": "car",
            "position_3d": [5.0, 5.0, 1.0],
            "dimensions": {"length": 4.0, "width": 2.0, "height": 1.5},
            "orientation": {
                "yaw_deg": 30.0,
                "rotation_matrix": np.eye(3).tolist(),
            },
            "motion": {"is_dynamic": False},
        }
    ]

    export_localized_scene_ply(
        base_ply_path=tmp_path / "nonexistent.ply",
        objects_3d=objs,
        output_ply_path=out_ply,
    )
    assert out_ply.exists()

    pts = np.random.uniform(-10, 10, (50, 3))
    generate_bev_visualization(pts, objs, out_png)
    assert out_png.exists()


def test_stage9_pipeline_smoke(tmp_path):
    vdir = tmp_path / "test_video"
    sparse_dir = vdir / "sfm_masked" / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    tracking_dir = vdir / "tracking"
    tracking_dir.mkdir(parents=True)

    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("1 PINHOLE 1000 1000 1000 1000 500 500\n")

    with open(sparse_dir / "images.txt", "w") as f:
        f.write(
            "1 1.0 0.0 0.0 0.0 0.0 0.0 -10.0 1 frame1.jpg\n\n"
            "2 1.0 0.0 0.0 0.0 2.0 0.0 -10.0 1 frame2.jpg\n\n"
        )

    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("1 0.0 0.0 0.0 255 255 255 0.5 2 1 0 2 0\n")

    meta = [
        {
            "frame_index": 0,
            "image_name": "frame1.jpg",
            "objects": [
                {
                    "track_id": 1,
                    "bbox": [480, 480, 520, 520],
                    "confidence": 0.9,
                    "is_moving": False,
                }
            ],
        },
        {
            "frame_index": 1,
            "image_name": "frame2.jpg",
            "objects": [
                {
                    "track_id": 1,
                    "bbox": [480, 480, 520, 520],
                    "confidence": 0.9,
                    "is_moving": False,
                }
            ],
        },
    ]
    summary = {
        "1": {
            "track_id": 1,
            "class_name": "car",
            "is_dynamic": False,
            "avg_speed": 0.0,
        }
    }

    with open(tracking_dir / "tracking_metadata.json", "w") as f:
        json.dump(meta, f)
    with open(tracking_dir / "tracks_summary.json", "w") as f:
        json.dump(summary, f)

    out_loc = vdir / "localization"
    report = run_stage9(output_dir=out_loc)

    assert report["status"] == "SUCCESS"
    assert report["total_localized_objects"] == 1
    assert (out_loc / "objects_3d.json").exists()
    assert (out_loc / "stage9_report.json").exists()
    assert (out_loc / "localized_scene.ply").exists()
    assert (out_loc / "bev_visualization.png").exists()

    with open(out_loc / "objects_3d.json", "r") as f:
        objs = json.load(f)
        assert objs["total_objects"] == 1
        assert "ground_contact_3d" in objs["objects"][0]
        assert "dimensions" in objs["objects"][0]
        assert "orientation" in objs["objects"][0]
