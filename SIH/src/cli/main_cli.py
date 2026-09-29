import argparse
import sys
from pathlib import Path
from src.config.config_manager import ConfigManager
from src.utils.logger import setup_logger
from src.utils.system_info import get_system_info, print_system_report
from src.preprocessing.stage2_pipeline import run_stage2
from src.metadata.stage3_pipeline import run_stage3
from src.sfm.stage4_pipeline import run_stage4
from src.dense.stage5_pipeline import run_stage5
from src.segmentation.stage6_pipeline import run_stage6
from src.detection.stage7_pipeline import run_stage7
from src.masking.stage8_pipeline import run_stage8
from src.localization.stage9_pipeline import run_stage9
from src.assets.stage10_pipeline import run_stage10
from src.assets.library_manager import LibraryManager
from src.fusion.stage11_pipeline import run_stage11
from src.texturing.stage12_pipeline import run_stage12
from src.config.config_manager import ConfigManager

def run_cli(args=None):
    parser = argparse.ArgumentParser(
        description="Drone Video to Hybrid 3D Digital Twin - Pipeline CLI (Stage 12)"
    )
    
    parser.add_argument(
        "--config", type=str, default=None, help="Path to custom configuration YAML file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("inspect-env", help="Inspect Python environment, hardware, GPU, COLMAP, FFmpeg")
    subparsers.add_parser("init-workspace", help="Initialize input/output directory structure")
    subparsers.add_parser("check-config", help="Validate configuration settings")
    subparsers.add_parser("version", help="Print version information")
    
    preprocess_parser = subparsers.add_parser("preprocess-video", help="Extract keyframes and evaluate quality")
    preprocess_parser.add_argument("--video", type=str, default=None, help="Path to input video file (defaults to input.video in the config)")
    preprocess_parser.add_argument("--output", type=str, default=None, help="Output directory")
    preprocess_parser.add_argument("--target-fps", type=float, default=None, help="Target frame sampling rate")

    meta_parser = subparsers.add_parser("extract-metadata", help="Extract camera intrinsics and flight telemetry metadata")
    meta_parser.add_argument("--video", type=str, default=None, help="Path to input video file (defaults to input.video in the config)")
    meta_parser.add_argument("--output", type=str, default=None, help="Output directory")

    sfm_parser = subparsers.add_parser("reconstruct-sparse", help="Perform COLMAP Structure-from-Motion sparse reconstruction")
    sfm_parser.add_argument("--video", type=str, default=None, help="Path to input video file (defaults to input.video in the config)")
    sfm_parser.add_argument("--output", type=str, default=None, help="Output directory")

    dense_parser = subparsers.add_parser("reconstruct-dense", help="Perform COLMAP MVS dense depth, point cloud fusion, mesh reconstruction")
    dense_parser.add_argument("--video", type=str, default=None, help="Path to input video file (defaults to input.video in the config)")
    dense_parser.add_argument("--output", type=str, default=None, help="Output directory")

    seg_parser = subparsers.add_parser("segment", help="Train and run UAVid semantic segmentation (Stage 6)")
    seg_parser.add_argument("--train-root", type=str, required=True, help="Root directory of UAVid training split")
    seg_parser.add_argument("--val-root", type=str, required=True, help="Root directory of UAVid validation split")
    seg_parser.add_argument("--output", type=str, default=None, help="Output directory")
    seg_parser.add_argument("--inference-only", action="store_true", help="Skip training, run inference + evaluation only")

    track_parser = subparsers.add_parser("track-vehicles", help="Detect and track vehicles across video or image sequence (Stage 7)")
    track_parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    track_parser.add_argument("--frames-dir", type=str, default=None, help="Path to directory containing input frames")
    track_parser.add_argument("--output", type=str, default=None, help="Output directory")
    track_parser.add_argument("--conf-threshold", type=float, default=None, help="Detection confidence threshold")

    mask_parser = subparsers.add_parser("mask-reconstruct", help="Perform dynamic object masking and masked SfM reconstruction (Stage 8)")
    mask_parser.add_argument("--video", type=str, default=None, help="Path to input video file (defaults to input.video in the config)")
    mask_parser.add_argument("--output", type=str, default=None, help="Output directory")

    loc_parser = subparsers.add_parser("localize-3d", help="Triangulate and localize tracked vehicles into 3D bounding boxes (Stage 9)")
    loc_parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    loc_parser.add_argument("--output", type=str, default=None, help="Output directory")

    assets_parser = subparsers.add_parser("manage-assets", help="Initialize and verify the 3D asset library (Stage 10)")
    assets_parser.add_argument("--force", action="store_true", help="Regenerate all curated assets even if they exist")

    match_parser = subparsers.add_parser("match-assets", help="Match 3D localized objects to asset library candidates (Stage 10)")
    match_parser.add_argument("--output", type=str, default=None, help="Output directory for matched_assets.json")

    fuse_parser = subparsers.add_parser("fuse-scene", help="Fuse reconstructed geometry with placed 3D assets (Stage 11)")
    fuse_parser.add_argument("--objects", type=str, default="outputs/sample_drone/localization/objects_3d.json", help="Path to objects_3d.json")
    fuse_parser.add_argument("--matched-assets", type=str, default="outputs/sample_drone/assets/matched_assets.json", help="Path to matched_assets.json")
    fuse_parser.add_argument("--dense-mesh", type=str, default="outputs/sample_drone/dense/dense_mesh.obj", help="Path to reconstructed mesh")
    fuse_parser.add_argument("--output", type=str, default="outputs/sample_drone/fusion", help="Output directory")

    tex_parser = subparsers.add_parser("texture-clean-mesh", help="Clean mesh, repair topology, correct normals, generate UVs/textures, and export LOD tiers (Stage 12)")
    tex_parser.add_argument("--input", type=str, default="outputs/sample_drone/fusion/fused_scene.obj", help="Path to input 3D mesh")
    tex_parser.add_argument("--quality", type=str, choices=["all", "high", "medium", "low"], default="all", help="Quality LOD tier to generate")
    tex_parser.add_argument("--output", type=str, default="outputs/sample_drone/textured", help="Output directory")

    parsed = parser.parse_args(args)
    
    cfg_manager = ConfigManager(config_path=parsed.config)
    log_dir = cfg_manager.get_path("paths.logs_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / cfg_manager.get("logging.filename", "pipeline.log")
    logger = setup_logger("pipeline_cli", log_file=log_file)
    
    if getattr(parsed, "video", None) is None and parsed.command in (
        "preprocess-video", "extract-metadata", "reconstruct-sparse", "reconstruct-dense", "mask-reconstruct"
    ):
        configured_video = cfg_manager.get_path("input.video")
        if configured_video is None:
            parser.error(f"{parsed.command} requires --video or input.video in the configuration")
        parsed.video = str(configured_video)

    if parsed.command == "version":
        version = cfg_manager.get("project.version", "0.1.0")
        print(f"Drone-to-Hybrid-3D-Digital-Twin version {version} (Stage 12)")
        return 0
        
    if parsed.command == "inspect-env":
        logger.info("Executing environment inspection...")
        info = get_system_info()
        report = print_system_report(info)
        print(report)
        logger.info("Environment inspection complete.")
        return 0
        
    if parsed.command == "init-workspace":
        logger.info("Initializing workspace directory structure...")
        created = cfg_manager.setup_directories()
        for p in created:
            logger.info(f"Directory verified/created: {p}")
        logger.info("Workspace initialization complete.")
        return 0

    if parsed.command == "check-config":
        logger.info("Validating configuration...")
        project_name = cfg_manager.get("project.name")
        stage = cfg_manager.get("project.stage")
        profile = cfg_manager.get("execution.profile")
        logger.info(f"Project Name: {project_name}")
        logger.info(f"Current Stage: {stage}")
        logger.info(f"Execution Profile: {profile}")
        logger.info("Configuration loaded successfully.")
        return 0

    if parsed.command == "preprocess-video":
        if parsed.target_fps is not None:
            cfg_manager.config["preprocessing"]["target_fps"] = parsed.target_fps
        manifest_data = run_stage2(parsed.video, output_dir=parsed.output, config_manager=cfg_manager)
        print(f"Preprocessing completed. Selected {manifest_data['sampling']['selected_frames_count']} keyframes.")
        return 0

    if parsed.command == "extract-metadata":
        res = run_stage3(parsed.video, output_dir=parsed.output, config_manager=cfg_manager)
        print(f"Metadata extraction completed. Calibration: {res['camera_metadata']['calibration_type']}, GPS: {res['telemetry_status']['gps_available']}")
        return 0

    if parsed.command == "reconstruct-sparse":
        res = run_stage4(parsed.video, output_dir=parsed.output, config_manager=cfg_manager)
        print(f"Sparse reconstruction status: {res['status']}. Registered: {res['registered_image_count']}/{res['total_input_images']} images ({res['registration_percentage']}%), Sparse Points: {res['num_sparse_points']}")
        return 0

    if parsed.command == "reconstruct-dense":
        res = run_stage5(parsed.video, output_dir=parsed.output, config_manager=cfg_manager)
        print(f"Dense reconstruction status: {res['status']}. Cleaned Points: {res['dense_point_count_cleaned']}, Mesh Vertices: {res['mesh_vertex_count']}, Faces: {res['mesh_face_count']}")
        return 0

    if parsed.command == "segment":
        if hasattr(parsed, "inference_only") and parsed.inference_only:
            cfg_manager.config.setdefault("segmentation", {})["inference_only"] = True
        res = run_stage6(
            train_root=parsed.train_root,
            val_root=parsed.val_root,
            output_dir=parsed.output,
            config_manager=cfg_manager,
        )
        eval_r = res.get("evaluation", {})
        miou = eval_r.get("mIoU", "N/A")
        pixel_acc = eval_r.get("pixel_accuracy", "N/A")
        print(f"Segmentation complete. mIoU={miou}, pixel_accuracy={pixel_acc}")
        return 0

    if parsed.command == "track-vehicles":
        if parsed.conf_threshold is not None:
            cfg_manager.config.setdefault("detection", {})["confidence_threshold"] = parsed.conf_threshold
        res = run_stage7(
            video_path=parsed.video,
            frames_dir=parsed.frames_dir,
            output_dir=parsed.output,
            config_manager=cfg_manager,
        )
        print(
            f"Tracking complete. Processed {res['total_frames_processed']} frames, "
            f"Identified {res['unique_tracks_count']} unique tracks "
            f"({res['dynamic_objects_count']} dynamic, {res['static_objects_count']} static)."
        )
        return 0

    if parsed.command == "mask-reconstruct":
        res = run_stage8(
            video_path=parsed.video,
            output_dir=parsed.output,
            config_manager=cfg_manager,
        )
        comp = res.get("comparison_summary", {})
        print(
            f"Dynamic Masking & SfM complete. "
            f"Masked 3D Points: {res['masked_sfm']['sparse_points']}, "
            f"Reproj Error: {res['masked_sfm']['mean_reprojection_error_px']}px. "
            f"Verdict: {comp.get('verdict', 'SUCCESS')}"
        )
        return 0

    if parsed.command == "localize-3d":
        res = run_stage9(
            video_path=parsed.video,
            output_dir=parsed.output,
            config_manager=cfg_manager,
        )
        print(
            f"3D Localization complete. Localized {res['total_localized_objects']} objects "
            f"({res['dynamic_objects_count']} dynamic, {res['static_objects_count']} static). "
            f"Avg Confidence: {res['average_confidence']}"
        )
        return 0

    if parsed.command == "manage-assets":
        from pathlib import Path
        assets_dir = Path(cfg_manager.get("assets.catalog_path", "data/assets/assets_catalog.json")).parent
        normalized_dir = Path(cfg_manager.get("assets.normalized_dir", "data/assets/normalized"))
        force = getattr(parsed, "force", False)
        mgr = LibraryManager(assets_dir=assets_dir, normalized_dir=normalized_dir)
        added = mgr.initialize_curated_library(force=force)
        summary = mgr.get_catalog_summary()
        verify = mgr.verify_library()
        print(
            f"Asset library ready. Total assets: {summary['total_assets']}, "
            f"New: {added}, By class: {summary['by_class']}, "
            f"Verification: {verify['status']}"
        )
        return 0

    if parsed.command == "match-assets":
        output_dir_arg = getattr(parsed, "output", None)
        res = run_stage10(output_dir=output_dir_arg, config_manager=cfg_manager)
        print(
            f"Asset matching complete. "
            f"Matched {res.get('matching', {}).get('total_matched', 0)}/"
            f"{res.get('matching', {}).get('total_input_objects', 0)} objects. "
            f"Library assets: {res.get('library', {}).get('total_assets', 0)}. "
            f"Status: {res['status']}"
        )
        return 0

    if parsed.command == "fuse-scene":
        res = run_stage11(
            objects_path=parsed.objects,
            matched_assets_path=parsed.matched_assets,
            dense_mesh_path=parsed.dense_mesh,
            output_dir=parsed.output,
            config=cfg_manager,
        )
        print(
            f"Scene fusion complete. Placed {res['total_placed_assets']} assets "
            f"({res['static_assets']} static, {res['dynamic_assets']} dynamic). "
            f"Floating corrected: {res['floating_corrected']}, "
            f"Penetrations corrected: {res['penetrations_corrected']}, "
            f"Collisions resolved: {res['collisions_resolved']}. "
            f"Exports: {list(res['exported_files'].keys())}"
        )
        return 0

    if parsed.command == "texture-clean-mesh":
        res = run_stage12(
            input_mesh_path=parsed.input,
            output_dir=parsed.output,
            quality_tier=parsed.quality,
            config=cfg_manager,
        )
        print(
            f"Texturing and mesh cleanup complete. Generated tiers: {list(res['tiers'].keys())}. "
            f"Cleaned faces: {res['clean_stats']['cleaned_faces']}, "
            f"Cleaned vertices: {res['clean_stats']['cleaned_vertices']}. "
            f"Master GLB: {res.get('master_exports', {}).get('glb', 'N/A')}"
        )
        return 0

    if parsed.command is None:
        parser.print_help()
        return 0

    logger.error(f"Unknown command: {parsed.command}")
    return 1
