import json
from pathlib import Path
import numpy as np
from plyfile import PlyData

from src.config.config_manager import ConfigManager
from src.utils.logger import setup_logger
from src.sfm.colmap_parser import parse_colmap_model
from .ray_triangulator import (
    compute_camera_ray,
    intersect_rays_least_squares,
    intersect_ray_with_plane,
)
from .terrain_estimator import TerrainEstimator
from .box3d_estimator import Box3DEstimator
from .pointcloud_visualizer import (
    export_localized_scene_ply,
    generate_bev_visualization,
)


def load_point_cloud_xyz(ply_path):
    p = Path(ply_path)
    if not p.exists():
        return np.zeros((0, 3), dtype=np.float64)
    try:
        ply = PlyData.read(str(p))
        v = ply["vertex"]
        return np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    except Exception:
        return np.zeros((0, 3), dtype=np.float64)


def run_stage9(video_path=None, output_dir=None, config_manager=None):
    if config_manager is None:
        config_manager = ConfigManager()

    if output_dir is None:
        base_out = config_manager.get_path("paths.outputs_dir")
        if video_path is not None:
            vstem = Path(video_path).stem
            output_dir = base_out / vstem / "localization"
        else:
            output_dir = base_out / "stage9_localization"
    else:
        output_dir = Path(output_dir).resolve()
        if output_dir.name != "localization":
            output_dir = output_dir / "localization"

    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = config_manager.get_path("paths.logs_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("stage9_localization", log_file=log_dir / "stage9.log")

    logger.info("Starting Stage 9 - 3D Object Localization")

    objects_3d_path = output_dir / "objects_3d.json"
    stage9_report_path = output_dir / "stage9_report.json"

    resume = config_manager.get("execution.resume", True)
    if resume and objects_3d_path.exists() and stage9_report_path.exists():
        logger.info(f"Existing Stage 9 report found at {stage9_report_path}. Resuming.")
        with open(stage9_report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    base_vdir = output_dir.parent
    sparse_model_dir = base_vdir / "sfm_masked" / "sparse" / "0"
    if not sparse_model_dir.exists():
        sparse_model_dir = base_vdir / "sfm" / "sparse" / "0"
    if not sparse_model_dir.exists():
        raise FileNotFoundError(
            f"Sparse reconstruction model not found under {base_vdir}. Run Stage 4 or Stage 8 first."
        )

    colmap_data = parse_colmap_model(sparse_model_dir)
    if not colmap_data["cameras"] or not colmap_data["registered_images"]:
        raise ValueError(
            "COLMAP model does not contain registered cameras or images."
        )

    cam_info = colmap_data["cameras"][0]
    fx, fy, cx, cy = cam_info["params"][:4]
    cam_images_map = {
        img["name"]: img for img in colmap_data["registered_images"]
    }

    ply_path = base_vdir / "sfm_masked" / "sparse_point_cloud.ply"
    if not ply_path.exists():
        ply_path = base_vdir / "sfm" / "sparse_point_cloud.ply"
    if not ply_path.exists():
        ply_path = base_vdir / "dense" / "fused.ply"

    logger.info(f"Loading point cloud from: {ply_path}")
    pt_cloud_xyz = load_point_cloud_xyz(ply_path)
    logger.info(f"Loaded {len(pt_cloud_xyz)} 3D points for terrain estimation.")

    tracking_dir = base_vdir / "tracking"
    meta_path = tracking_dir / "tracking_metadata.json"
    summary_path = tracking_dir / "tracks_summary.json"
    if not meta_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Tracking metadata not found at {tracking_dir}. Run Stage 7 first."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        tracking_metadata = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        tracks_summary = json.load(f)

    track_obs = {}
    for frame_entry in tracking_metadata:
        iname = frame_entry["image_name"]
        fidx = frame_entry["frame_index"]
        if iname not in cam_images_map:
            continue
        c_pose = cam_images_map[iname]

        for obj in frame_entry.get("objects", []):
            tid = obj["track_id"]
            if tid not in track_obs:
                track_obs[tid] = []
            obs_item = dict(obj)
            obs_item["image_name"] = iname
            obs_item["frame_index"] = fidx
            obs_item["qvec"] = c_pose["qvec"]
            obs_item["tvec"] = c_pose["tvec"]
            obs_item["camera_position"] = c_pose["position_xyz"]
            track_obs[tid].append(obs_item)

    terrain_estimator = TerrainEstimator(
        search_radius=config_manager.get("localization.ground_search_radius", 10.0),
        dist_thresh=config_manager.get(
            "localization.ground_ransac_distance_thresh", 0.15
        ),
    )
    box_estimator = Box3DEstimator()

    localized_objects = []
    logger.info(f"Localizing {len(tracks_summary)} tracks into 3D...")

    for tid_str, t_info in tracks_summary.items():
        try:
            tid = int(tid_str)
        except ValueError:
            tid = tid_str

        observations = track_obs.get(tid, [])
        if not observations:
            continue

        cname = t_info.get("class_name", "car")
        is_dynamic = t_info.get("is_dynamic", False)

        centers = []
        directions = []
        for obs in observations:
            box = obs["bbox"]
            u_c = (box[0] + box[2]) / 2.0
            v_c = (box[1] + box[3]) / 2.0

            C_world, d_world = compute_camera_ray(
                u=u_c,
                v=v_c,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                qvec=obs["qvec"],
                tvec=obs["tvec"],
            )
            centers.append(C_world)
            directions.append(d_world)

        p_3d, res_err, base_angle = intersect_rays_least_squares(
            centers, directions
        )

        if p_3d is None or base_angle < 1.0:
            if len(pt_cloud_xyz) > 0:
                ground_z = float(np.median(pt_cloud_xyz[:, 2]))
            else:
                ground_z = 0.0
            p_3d = intersect_ray_with_plane(
                centers[0], directions[0], [0.0, 0.0, 1.0], [0.0, 0.0, ground_z]
            )
            if p_3d is None:
                p_3d = centers[0] + 5.0 * directions[0]

        ground_pt, ground_normal = terrain_estimator.estimate_ground_contact(
            position_xy=p_3d[:2], point_cloud_xyz=pt_cloud_xyz
        )

        box_params = box_estimator.estimate_3d_box(
            class_name=cname,
            center_3d=p_3d,
            ground_contact_3d=ground_pt,
            observations=observations,
            triangulation_residual=res_err,
            baseline_angle_deg=base_angle,
            is_dynamic=is_dynamic,
        )

        vel_3d = [0.0, 0.0, 0.0]
        speed_mps = 0.0
        if is_dynamic:
            speed_2d = t_info.get("avg_speed", 0.0)
            yaw_rad = np.radians(box_params["orientation"]["yaw_deg"])
            vel_3d = [
                round(float(speed_2d * np.cos(yaw_rad)), 3),
                round(float(speed_2d * np.sin(yaw_rad)), 3),
                0.0,
            ]
            speed_mps = round(float(speed_2d), 3)

        obj_entry = {
            "track_id": tid,
            "class_name": cname,
            "confidence": box_params["confidence"],
            "position_3d": [round(float(x), 4) for x in p_3d],
            "ground_contact_3d": [round(float(x), 4) for x in ground_pt],
            "dimensions": box_params["dimensions"],
            "orientation": box_params["orientation"],
            "motion": {
                "is_dynamic": is_dynamic,
                "velocity_3d": vel_3d,
                "speed_mps": speed_mps,
            },
            "num_observations": len(observations),
            "observed_frames": [obs["frame_index"] for obs in observations],
        }
        localized_objects.append(obj_entry)

    objects_data = {
        "total_objects": len(localized_objects),
        "objects": localized_objects,
    }

    with open(objects_3d_path, "w", encoding="utf-8") as f:
        json.dump(objects_data, f, indent=2)

    logger.info(f"Saved {len(localized_objects)} localized 3D objects to: {objects_3d_path}")

    ply_out = output_dir / "localized_scene.ply"
    bev_out = output_dir / "bev_visualization.png"

    logger.info("Generating 3D scene wireframe PLY...")
    export_localized_scene_ply(
        base_ply_path=ply_path,
        objects_3d=localized_objects,
        output_ply_path=ply_out,
        logger=logger,
    )

    logger.info("Generating Bird's Eye View (BEV) visualization...")
    generate_bev_visualization(
        point_cloud_xyz=pt_cloud_xyz,
        objects_3d=localized_objects,
        output_png_path=bev_out,
    )

    dynamic_count = sum(
        1 for o in localized_objects if o["motion"]["is_dynamic"]
    )
    static_count = len(localized_objects) - dynamic_count

    class_distribution = {}
    for o in localized_objects:
        c = o["class_name"]
        class_distribution[c] = class_distribution.get(c, 0) + 1

    report = {
        "status": "SUCCESS",
        "total_localized_objects": len(localized_objects),
        "dynamic_objects_count": dynamic_count,
        "static_objects_count": static_count,
        "objects_by_class": class_distribution,
        "average_confidence": round(
            float(np.mean([o["confidence"] for o in localized_objects])), 4
        )
        if localized_objects
        else 0.0,
        "output_paths": {
            "localization_dir": str(output_dir),
            "objects_3d_json": str(objects_3d_path),
            "localized_scene_ply": str(ply_out),
            "bev_visualization_png": str(bev_out),
            "stage9_report_json": str(stage9_report_path),
        },
    }

    with open(stage9_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Stage 9 complete. Objects localized: {len(localized_objects)} ({dynamic_count} dynamic, {static_count} static).")
    logger.info(f"Report written to: {stage9_report_path}")

    return report
