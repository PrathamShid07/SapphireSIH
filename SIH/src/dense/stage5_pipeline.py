import json
import shutil
from pathlib import Path
from src.config.config_manager import ConfigManager
from src.utils.logger import setup_logger
from src.utils.geometry_checks import check_ply_geometry
from src.utils.reconstruction_gate import (
    resolve_reconstruction_inputs,
    assert_sparse_model_matches_inputs,
    assert_registration_sufficient,
)
from src.sfm.colmap_parser import parse_colmap_model
from .mvs_runner import MVSRunner
from .point_cloud_cleaner import clean_point_cloud
from .mesh_processor import process_mesh_and_export, triangulate_point_cloud

def run_stage5(video_path, output_dir=None, config_manager=None):
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
    logger = setup_logger("stage5_dense", log_file=log_dir / "stage5.log")

    logger.info(f"Starting Stage 5 Dense Reconstruction for: {vpath}")

    recon_inputs = resolve_reconstruction_inputs(output_dir, config_manager)
    selected_dir = recon_inputs["image_dir"]
    sfm_sparse_dir = output_dir / "sfm" / "sparse" / "0"
    if not sfm_sparse_dir.exists():
        if (output_dir / "sfm" / "sparse").exists():
            sfm_sparse_dir = output_dir / "sfm" / "sparse"
        else:
            raise FileNotFoundError(
                f"Stage 4 sparse model not found at {sfm_sparse_dir}. Run Stage 4 first."
            )

    sfm_report_path = output_dir / "sfm" / "sfm_report.json"
    if sfm_report_path.exists():
        with open(sfm_report_path, "r", encoding="utf-8") as f:
            assert_registration_sufficient(json.load(f), config_manager.get("dense.min_registration_fraction"))
    parsed_sparse = parse_colmap_model(sfm_sparse_dir)
    assert_sparse_model_matches_inputs(
        [img["name"] for img in parsed_sparse.get("registered_images", [])], recon_inputs
    )

    dense_dir = output_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)

    fused_ply = dense_dir / "fused.ply"
    clean_fused_ply = dense_dir / "dense_point_cloud_cleaned.ply"
    raw_mesh_ply = dense_dir / "dense_mesh_raw.ply"
    report_path = dense_dir / "dense_reconstruction_report.json"

    undistort_marker = dense_dir / ".undistort_done"
    patch_match_marker = dense_dir / ".patch_match_done"
    fusion_marker = dense_dir / ".fusion_done"

    runner = MVSRunner(config_manager=config_manager)
    resume = config_manager.get("execution.resume", True)
    pca_warn_threshold = config_manager.get("dense.pca_aspect_ratio_warn_threshold", 500.0)

    runner.require_cuda()

    # Step 1: Image Undistortion
    if resume and undistort_marker.exists() and (dense_dir / "images").exists():
        logger.info("Skipping image_undistorter (cached outputs found)...")
    else:
        logger.info("Running COLMAP image_undistorter...")
        runner.run_image_undistorter(
            image_dir=selected_dir,
            input_model_dir=sfm_sparse_dir,
            dense_dir=dense_dir
        )
        undistort_marker.touch()

    # Step 2: PatchMatch Stereo (Depth Maps)
    depth_maps_dir = dense_dir / "stereo" / "depth_maps"
    if resume and patch_match_marker.exists() and depth_maps_dir.exists() and any(depth_maps_dir.iterdir()):
        logger.info("Skipping patch_match_stereo (cached depth maps found)...")
    else:
        logger.info("Running COLMAP patch_match_stereo (MVS depth maps)...")
        runner.run_patch_match_stereo(dense_dir=dense_dir)
        patch_match_marker.touch()

    # Step 3: Stereo Fusion (Dense Point Cloud)
    if resume and fusion_marker.exists() and fused_ply.exists():
        logger.info("Skipping stereo_fusion (cached fused.ply found)...")
    else:
        logger.info("Running COLMAP stereo_fusion (dense point cloud -> fused.ply)...")
        runner.run_stereo_fusion(dense_dir=dense_dir)
        fusion_marker.touch()

    # Step 4: Outlier Removal / Cleaning
    logger.info("Cleaning dense point cloud (statistical outlier removal)...")
    std_ratio = config_manager.get("dense.clean_outliers_std_ratio", 2.0)
    nb_neighbors = config_manager.get("dense.clean_outliers_nb_neighbors", 20)
    raw_cnt, clean_cnt, removed_cnt = clean_point_cloud(
        input_ply_path=fused_ply,
        output_ply_path=clean_fused_ply,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    # Step 5: Surface Meshing
    logger.info("Generating dense mesh...")
    meshed = False
    mesher_used = None

    native_meshers = {
        "poisson": lambda: runner.run_poisson_mesher(
            input_ply_path=clean_fused_ply, output_ply_path=raw_mesh_ply
        ),
        "delaunay": lambda: runner.run_delaunay_mesher(
            dense_dir=dense_dir, output_ply_path=raw_mesh_ply
        ),
    }
    # dense.mesher_type picks which COLMAP mesher to try first. Poisson does
    # implicit surface fitting from oriented normals and is far less sensitive
    # to residual point-cloud noise than delaunay_mesher's direct
    # triangulation, so it is the configured default; the other native mesher
    # is tried next, and the 2D XY fallback is the last resort.
    configured_type = getattr(runner, "mesher_type", "poisson")
    ordered_names = sorted(native_meshers.keys(), key=lambda name: name != configured_type)

    for name in ordered_names:
        try:
            logger.info(f"Attempting COLMAP {name}_mesher...")
            native_meshers[name]()
            if raw_mesh_ply.exists():
                meshed = True
                mesher_used = f"colmap_{name}"
                break
        except Exception as e:
            logger.warning(f"COLMAP {name}_mesher failed ({e}).")

    if not meshed:
        mesh_fallback_max_edge_length = config_manager.get("dense.mesh_fallback_max_edge_length", 5.0)
        mesh_fallback_max_slope_ratio = config_manager.get("dense.mesh_fallback_max_slope_ratio", 8.0)
        logger.warning(
            "Both COLMAP native meshers failed or are unavailable. Falling back to "
            "2D XY-projected Delaunay triangulation of the cleaned point cloud "
            "(max_edge_length=%s, max_slope_ratio=%s). This fallback cannot represent "
            "overhangs/vertical structures and is more sensitive to residual outliers - "
            "treat the resulting mesh as provisional.",
            mesh_fallback_max_edge_length, mesh_fallback_max_slope_ratio,
        )
        triangulate_point_cloud(
            input_ply_path=clean_fused_ply,
            output_mesh_path=raw_mesh_ply,
            max_edge_length=mesh_fallback_max_edge_length,
            max_slope_ratio=mesh_fallback_max_slope_ratio,
        )
        mesher_used = "xy_projected_fallback"

    logger.info("Cleaning mesh and exporting PLY, OBJ, GLB formats...")
    mesh_stats = process_mesh_and_export(
        input_mesh_path=raw_mesh_ply,
        output_dir=dense_dir,
        export_basename="dense_mesh"
    )

    ply_out = dense_dir / "dense_mesh.ply"
    obj_out = dense_dir / "dense_mesh.obj"
    glb_out = dense_dir / "dense_mesh.glb"

    logger.info("Performing dense point cloud geometry sanity check (PCA eigenvalue ratio)...")
    target_ply = clean_fused_ply if clean_fused_ply.exists() else fused_ply
    geom_sanity = check_ply_geometry(target_ply, warn_threshold=pca_warn_threshold, logger=logger)
    logger.info(f"Dense PCA Eigenvalue Ratio: {geom_sanity['eigenvalue_ratio']}:1 (Flattened: {geom_sanity['is_flattened']})")

    report_data = {
        "status": "SUCCESS" if raw_mesh_ply.exists() else "FAILED",
        "reconstruction_strategy": recon_inputs["strategy"],
        "reconstruction_manifest": str(recon_inputs["manifest_path"]) if recon_inputs["manifest_path"] else None,
        "registered_images_used": parsed_sparse.get("num_registered_images", 0),
        "patch_match_gpu_index": runner.patch_match_gpu_index,
        "mesher_used": mesher_used,
        "dense_point_count_raw": raw_cnt,
        "dense_point_count_cleaned": clean_cnt,
        "outliers_removed": removed_cnt,
        "mesh_vertex_count": mesh_stats.get("vertex_count", 0),
        "mesh_face_count": mesh_stats.get("face_count", 0),
        "geometry_sanity": geom_sanity,
        "output_paths": {
            "dense_dir": str(dense_dir),
            "dense_point_cloud_fused_ply": str(fused_ply) if fused_ply.exists() else None,
            "dense_point_cloud_cleaned_ply": str(clean_fused_ply) if clean_fused_ply.exists() else None,
            "dense_mesh_raw_ply": str(raw_mesh_ply) if raw_mesh_ply.exists() else None,
            "mesh_ply": str(ply_out) if ply_out.exists() else None,
            "mesh_obj": str(obj_out) if obj_out.exists() else None,
            "mesh_glb": str(glb_out) if glb_out.exists() else None,
            "dense_report_json": str(report_path),
        }
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    logger.info("Stage 5 Dense Reconstruction complete.")
    logger.info(f"Raw dense points: {raw_cnt}")
    logger.info(f"Cleaned dense points: {clean_cnt}")
    logger.info(f"Mesh vertices: {mesh_stats.get('vertex_count', 0)}, faces: {mesh_stats.get('face_count', 0)}")
    logger.info(f"Report written to: {report_path}")

    return report_data
