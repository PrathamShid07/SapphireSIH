import json
from pathlib import Path
from PIL import Image
from src.config.config_manager import ConfigManager
from src.utils.logger import setup_logger
from src.utils.geometry_checks import check_ply_geometry
from .colmap_wrapper import ColmapWrapper
from .colmap_parser import parse_colmap_model
from src.utils.reconstruction_gate import resolve_reconstruction_inputs, load_reconstruction_manifest, join_poses_with_manifest

def run_stage4(video_path, output_dir=None, config_manager=None):
    if config_manager is None:
        config_manager = ConfigManager()

    vpath = Path(video_path).resolve()
    
    if output_dir is None:
        base_out = config_manager.get_path("paths.outputs_dir")
        output_dir = base_out / vpath.stem
    else:
        output_dir = Path(output_dir).resolve()

    log_dir = config_manager.get_path("paths.logs_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("stage4_sfm", log_file=log_dir / "stage4.log")

    logger.info(f"Starting Stage 4 COLMAP Structure-from-Motion for: {vpath}")

    recon_inputs = resolve_reconstruction_inputs(output_dir, config_manager)
    selected_dir = recon_inputs["image_dir"]
    input_images = [selected_dir / n for n in recon_inputs["image_names"]]
    total_input_images = len(input_images)
    image_names_filter = recon_inputs["image_names"] if recon_inputs["restrict_to_names"] else None
    logger.info(f"Reconstruction strategy: {recon_inputs['strategy']}")
    logger.info(f"Input image count: {total_input_images} from {selected_dir}")
    if recon_inputs["ineligible"]:
        logger.warning(f"{len(recon_inputs['ineligible'])} decoded frames excluded from reconstruction (see reconstruction manifest).")

    single_camera = config_manager.get("sfm.single_camera", True)
    custom_calib = config_manager.get("camera.custom_calibration")
    sensor_width_mm = config_manager.get("camera.sensor_width_mm")
    focal_length_mm = config_manager.get("camera.focal_length_mm")
    camera_model = config_manager.get("sfm.camera_model", "OPENCV")
    pca_warn_threshold = config_manager.get("sfm.pca_aspect_ratio_warn_threshold", 500.0)

    if single_camera and not custom_calib and (sensor_width_mm is None or focal_length_mm is None):
        raise ValueError(
            "Camera intrinsics prior missing: 'camera.sensor_width_mm' and 'camera.focal_length_mm' are null in config. "
            "When 'sfm.single_camera: true', providing accurate drone sensor dimensions and lens focal length "
            "(e.g., sensor_width_mm: 13.2 and focal_length_mm: 8.8 for a 1-inch sensor drone) is required to prevent "
            "scale drift and point cloud flattening. Please configure these fields in config/default_config.yaml."
        )

    camera_params_str = None
    if custom_calib:
        fx = custom_calib.get("fx", 1000.0)
        fy = custom_calib.get("fy", fx)
        cx = custom_calib.get("cx", 960.0)
        cy = custom_calib.get("cy", 540.0)
        k1 = custom_calib.get("k1", 0.0)
        k2 = custom_calib.get("k2", 0.0)
        p1 = custom_calib.get("p1", 0.0)
        p2 = custom_calib.get("p2", 0.0)
        if camera_model.upper() == "OPENCV":
            camera_params_str = f"{fx:.4f},{fy:.4f},{cx:.4f},{cy:.4f},{k1:.6f},{k2:.6f},{p1:.6f},{p2:.6f}"
        else:
            camera_params_str = f"{fx:.4f},{fy:.4f},{cx:.4f},{cy:.4f}"
    elif sensor_width_mm and focal_length_mm and total_input_images > 0:
        try:
            with Image.open(input_images[0]) as im:
                img_w, img_h = im.size
        except Exception:
            img_w, img_h = 1920, 1080
        fx = (float(focal_length_mm) / float(sensor_width_mm)) * img_w
        fy = fx
        cx = img_w / 2.0
        cy = img_h / 2.0
        if camera_model.upper() == "OPENCV":
            camera_params_str = f"{fx:.4f},{fy:.4f},{cx:.4f},{cy:.4f},0.0,0.0,0.0,0.0"
        else:
            camera_params_str = f"{fx:.4f},{fy:.4f},{cx:.4f},{cy:.4f}"
        logger.info(f"Initialized camera intrinsics prior: {camera_model} fx={fx:.2f}, fy={fy:.2f}, cx={cx:.1f}, cy={cy:.1f}")

    sfm_dir = output_dir / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)

    db_path = sfm_dir / "database.db"
    sparse_out_dir = sfm_dir / "sparse"
    ply_path = sfm_dir / "sparse_point_cloud.ply"
    camera_poses_path = sfm_dir / "camera_poses.json"
    sfm_report_path = sfm_dir / "sfm_report.json"

    colmap = ColmapWrapper(config_manager=config_manager)

    resume = config_manager.get("execution.resume", True)
    model_sub_dir = sparse_out_dir / "0"
    
    if not (resume and sfm_report_path.exists() and ply_path.exists()):
        logger.info("Running feature extraction...")
        colmap.extract_features(
            image_dir=selected_dir, db_path=db_path, camera_params=camera_params_str, image_names=image_names_filter
        )

        logger.info("Running feature matching...")
        colmap.match_features(db_path=db_path)

        logger.info("Running sparse mapping...")
        colmap.map_sparse(image_dir=selected_dir, db_path=db_path, sparse_dir=sparse_out_dir)

    failure_reasons = []
    if not model_sub_dir.exists() or not (model_sub_dir / "images.bin").exists() and not (model_sub_dir / "images.txt").exists():
        if (sparse_out_dir / "images.bin").exists() or (sparse_out_dir / "images.txt").exists():
            model_sub_dir = sparse_out_dir
        else:
            failure_reasons.append("COLMAP mapper did not produce any sparse 3D model (low overlap or feature match failure).")

    parsed_model = {}
    if not failure_reasons and model_sub_dir.exists():
        try:
            colmap.export_ply(input_model_dir=model_sub_dir, output_ply_path=ply_path)
        except Exception as e:
            logger.warning(f"PLY export warning: {e}")
            
        parsed_model = parse_colmap_model(model_sub_dir)

    num_registered = parsed_model.get("num_registered_images", 0)
    num_cameras = parsed_model.get("num_cameras", 0)
    num_points = parsed_model.get("num_sparse_points", 0)
    reg_percent = round((num_registered / total_input_images) * 100, 2) if total_input_images > 0 else 0.0

    if num_registered == 0:
        status = "FAILED"
        if not failure_reasons:
            failure_reasons.append("Zero images registered in sparse reconstruction.")
    elif num_registered < total_input_images:
        status = "PARTIAL"
        failure_reasons.append(f"{total_input_images - num_registered} keyframes failed to register due to insufficient overlap or baseline.")
    else:
        status = "SUCCESS"

    camera_poses = parsed_model.get("registered_images", [])
    with open(camera_poses_path, "w", encoding="utf-8") as f:
        json.dump(camera_poses, f, indent=2)

    poses_telemetry_path = None
    if recon_inputs["manifest_path"] is not None:
        manifest, _ = load_reconstruction_manifest(output_dir)
        poses_telemetry_path = sfm_dir / "registered_frames_telemetry.json"
        with open(poses_telemetry_path, "w", encoding="utf-8") as f:
            json.dump(join_poses_with_manifest(camera_poses, manifest), f, indent=2)

    logger.info("Performing geometry sanity check (PCA eigenvalue ratio)...")
    geom_sanity = check_ply_geometry(ply_path, warn_threshold=pca_warn_threshold, logger=logger)
    logger.info(f"Sparse PCA Eigenvalue Ratio: {geom_sanity['eigenvalue_ratio']}:1 (Flattened: {geom_sanity['is_flattened']})")

    report_data = {
        "status": status,
        "sfm_engine": "COLMAP",
        "camera_model": camera_model,
        "camera_params_prior": camera_params_str,
        "reconstruction_strategy": recon_inputs["strategy"],
        "reconstruction_manifest": str(recon_inputs["manifest_path"]) if recon_inputs["manifest_path"] else None,
        "mapper_init_min_tri_angle": colmap.init_min_tri_angle,
        "mapper_init_min_num_inliers": colmap.init_min_num_inliers,
        "masks_used": False,
        "total_input_images": total_input_images,
        "registered_image_count": num_registered,
        "registration_percentage": reg_percent,
        "num_cameras": num_cameras,
        "num_sparse_points": num_points,
        "mean_reprojection_error": parsed_model.get("mean_reprojection_error", 0.0),
        "geometry_sanity": geom_sanity,
        "failure_reasons": failure_reasons,
        "paths": {
            "database": str(db_path),
            "sparse_model_dir": str(model_sub_dir) if model_sub_dir.exists() else None,
            "sparse_point_cloud_ply": str(ply_path) if ply_path.exists() else None,
            "camera_poses_json": str(camera_poses_path),
            "registered_frames_telemetry_json": str(poses_telemetry_path) if poses_telemetry_path else None,
            "sfm_report_json": str(sfm_report_path)
        }
    }

    with open(sfm_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"Stage 4 COLMAP SfM complete. Status: {status}")
    logger.info(f"Registered images: {num_registered}/{total_input_images} ({reg_percent}%)")
    logger.info(f"Sparse 3D points: {num_points}")
    logger.info(f"Report written to: {sfm_report_path}")

    return report_data
