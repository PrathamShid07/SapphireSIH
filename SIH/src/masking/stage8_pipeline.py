import json
import time
from pathlib import Path
from src.config.config_manager import ConfigManager
from src.utils.logger import setup_logger
from src.sfm.colmap_wrapper import ColmapWrapper
from src.utils.reconstruction_gate import assert_stage8_allowed, assert_masks_approved
from src.sfm.colmap_parser import parse_colmap_model
from .dynamic_mask_generator import DynamicMaskGenerator
from .mask_propagator import MaskPropagator
from .reconstruction_comparator import ReconstructionComparator
from . import colmap_db_pruner
from . import pose_comparator
from . import refinement_validator


def run_stage8(video_path, output_dir=None, config_manager=None):
    """
    Dispatches to the LEGACY or OPTIMIZED Stage 8 implementation based on
    `masking.reuse_stage4_poses` (default false). See
    docs/STAGE8_OPTIMIZATION.md for the full architecture, validation results,
    and rollback procedure.
    """
    if config_manager is None:
        config_manager = ConfigManager()

    log_dir = config_manager.get_path("paths.logs_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("stage8_masking", log_file=log_dir / "stage8.log")

    assert_stage8_allowed(config_manager)

    logger.info(f"Starting Stage 8 - Dynamic Object Masking for: {video_path}")

    reuse_stage4_poses = config_manager.get("masking.reuse_stage4_poses", False)
    if reuse_stage4_poses:
        logger.info(
            "masking.reuse_stage4_poses=true -> OPTIMIZED path "
            "(point_triangulator reuse of Stage 4 database/poses, EXPERIMENTAL until validated)."
        )
        return _run_stage8_optimized(video_path, output_dir, config_manager, logger)

    logger.info(
        "masking.reuse_stage4_poses=false (default) -> LEGACY path "
        "(full feature_extractor + matcher + mapper re-run)."
    )
    return _run_stage8_legacy(video_path, output_dir, config_manager, logger)


def _run_stage8_legacy(video_path, output_dir, config_manager, logger):
    """
    Original Stage 8 implementation, preserved unchanged in behavior. This is
    the default and the rollback target for the optimized path: if the
    optimized path's validation ever regresses, set
    `masking.reuse_stage4_poses: false` (the default) to fall back to this
    function with zero code changes required.

    The only addition relative to the original implementation is per-phase
    timing instrumentation (feature extraction / matching / mapper / total),
    required to produce a fair runtime comparison against the optimized path
    (see docs/STAGE8_OPTIMIZATION.md). Timing does not change any COLMAP
    invocation, argument, or output.
    """
    vpath = Path(video_path).resolve()
    vstem = vpath.stem

    if output_dir is None:
        base_out = config_manager.get_path("paths.outputs_dir")
        output_dir = base_out / vstem
    else:
        output_dir = Path(output_dir).resolve()

    selected_dir = output_dir / "processed" / "selected_frames"
    if not selected_dir.exists():
        raise FileNotFoundError(
            f"Selected frames not found at {selected_dir}. Run Stage 2 first."
        )

    tracking_dir = output_dir / "tracking"
    meta_path = tracking_dir / "tracking_metadata.json"
    summary_path = tracking_dir / "tracks_summary.json"
    if not meta_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Stage 7 tracking outputs not found at {tracking_dir}. Run Stage 7 first."
        )

    unmasked_sparse_dir = output_dir / "sfm" / "sparse" / "0"
    if not unmasked_sparse_dir.exists():
        if (output_dir / "sfm" / "sparse").exists():
            unmasked_sparse_dir = output_dir / "sfm" / "sparse"
        else:
            raise FileNotFoundError(
                f"Stage 4 unmasked SfM model not found at {unmasked_sparse_dir}. Run Stage 4 first."
            )

    masking_dir = output_dir / "masking"
    masking_dir.mkdir(parents=True, exist_ok=True)
    stage8_report_path = masking_dir / "stage8_report.json"
    comp_report_path = masking_dir / "masking_comparison_report.json"

    resume = config_manager.get("execution.resume", True)
    if resume and stage8_report_path.exists() and comp_report_path.exists():
        logger.info(f"Existing Stage 8 report found at {stage8_report_path}. Resuming.")
        with open(stage8_report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        tracking_meta = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        tracks_summary = json.load(f)

    logger.info("Propagating dynamic object tracks across temporal gaps...")
    propagator = MaskPropagator(
        max_gap=config_manager.get("masking.max_interpolation_gap", 3),
        motion_expansion=True,
        logger=logger,
    )
    propagated_meta = propagator.propagate_tracks(tracking_meta, tracks_summary)

    logger.info("Generating dynamic object masks...")
    dilation_px = config_manager.get("masking.dilation_px", 15)
    mask_gen = DynamicMaskGenerator(dilation_px=dilation_px, logger=logger)
    mask_result = mask_gen.generate_masks_from_tracking(
        tracking_metadata=propagated_meta,
        tracks_summary=tracks_summary,
        images_dir=selected_dir,
        output_dir=masking_dir,
    )

    masks_dir = Path(mask_result["masks_dir"])

    sfm_masked_dir = output_dir / "sfm_masked"
    sfm_masked_dir.mkdir(parents=True, exist_ok=True)
    masked_db = sfm_masked_dir / "database.db"
    masked_sparse = sfm_masked_dir / "sparse"

    wrapper = ColmapWrapper(config_manager=config_manager)

    timings = {}

    if not (resume and (masked_sparse / "0").exists()):
        if masked_db.exists():
            masked_db.unlink()

        logger.info(f"Extracting features with dynamic masks from {masks_dir}...")
        t0 = time.time()
        assert_masks_approved(config_manager, masks_dir)
        wrapper.extract_features(
            image_dir=selected_dir, db_path=masked_db, mask_path=masks_dir
        )
        timings["feature_extraction_seconds"] = round(time.time() - t0, 3)

        logger.info("Matching features for masked model...")
        t0 = time.time()
        wrapper.match_features(db_path=masked_db)
        timings["matching_seconds"] = round(time.time() - t0, 3)

        logger.info("Running COLMAP mapper for masked model...")
        t0 = time.time()
        wrapper.map_sparse(
            image_dir=selected_dir, db_path=masked_db, sparse_dir=masked_sparse
        )
        timings["mapper_seconds"] = round(time.time() - t0, 3)
        timings["total_seconds"] = round(
            timings["feature_extraction_seconds"]
            + timings["matching_seconds"]
            + timings["mapper_seconds"],
            3,
        )

        model_0 = masked_sparse / "0"
        if model_0.exists():
            ply_path = sfm_masked_dir / "sparse_point_cloud.ply"
            logger.info(f"Exporting masked sparse point cloud to {ply_path}...")
            wrapper.export_ply(input_model_dir=model_0, output_ply_path=ply_path)

    masked_model_dir = (
        masked_sparse / "0" if (masked_sparse / "0").exists() else masked_sparse
    )

    logger.info("Comparing masked reconstruction against unmasked baseline...")
    comparator = ReconstructionComparator(logger=logger)
    comparison = comparator.compare_reconstructions(
        unmasked_model_dir=unmasked_sparse_dir,
        masked_model_dir=masked_model_dir,
        masks_dir=masks_dir,
        output_path=comp_report_path,
    )

    report = {
        "status": "SUCCESS",
        "method": "legacy_full_colmap_rerun",
        "timings_seconds": timings,
        "mask_generation": {
            "total_masks": mask_result["total_masks"],
            "total_dynamic_objects_masked": mask_result["total_dynamic_objects"],
            "average_masked_pixel_percentage": mask_result[
                "average_masked_pixel_percentage"
            ],
            "masks_dir": str(masks_dir),
            "overlays_dir": mask_result["overlays_dir"],
        },
        "masked_sfm": {
            "sfm_masked_dir": str(sfm_masked_dir),
            "registered_images": comparison["masked"]["registered_images"],
            "sparse_points": comparison["masked"]["sparse_points"],
            "mean_reprojection_error_px": comparison["masked"][
                "mean_reprojection_error_px"
            ],
        },
        "comparison_summary": comparison["comparison"],
        "output_paths": {
            "masking_dir": str(masking_dir),
            "masks_dir": str(masks_dir),
            "overlays_dir": mask_result["overlays_dir"],
            "sfm_masked_dir": str(sfm_masked_dir),
            "masked_point_cloud_ply": str(
                sfm_masked_dir / "sparse_point_cloud.ply"
            )
            if (sfm_masked_dir / "sparse_point_cloud.ply").exists()
            else None,
            "masking_comparison_report_json": str(comp_report_path),
            "stage8_report_json": str(stage8_report_path),
        },
    }

    with open(stage8_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Stage 8 (legacy) Dynamic Object Masking complete.")
    logger.info(f"Verdict: {comparison['comparison']['verdict']}")
    logger.info(f"Report written to: {stage8_report_path}")

    return report


def _run_stage8_optimized(video_path, output_dir, config_manager, logger):
    """
    Optimized Stage 8: reuses Stage 4's COLMAP database and camera poses
    instead of re-running feature_extractor + matcher + mapper.

    Flow:
      1. Locate (never modify) Stage 4's database.db and sparse model.
      2. Generate dynamic-object masks exactly as the legacy path does.
      3. Copy Stage 4's database.db into an ISOLATED directory
         (`sfm_masked_optimized/`, distinct from both `sfm/` and
         `sfm_masked/`).
      4. Prune matches/two_view_geometries rows whose keypoint falls inside a
         dynamic-object mask, on the COPY only.
      5. Run COLMAP's point_triangulator against Stage 4's existing poses
         (fixed via --Mapper.fix_existing_frames) + the pruned database copy.
         feature_extractor, any matcher, and the incremental mapper are never
         invoked in this path.
      6. Compare against the Stage 4 baseline using the same
         ReconstructionComparator the legacy path uses.

    See docs/STAGE8_OPTIMIZATION.md for the full design rationale, the
    empirical COLMAP validation this is based on, and current benchmark
    results.
    """
    vpath = Path(video_path).resolve()
    vstem = vpath.stem

    if output_dir is None:
        base_out = config_manager.get_path("paths.outputs_dir")
        output_dir = base_out / vstem
    else:
        output_dir = Path(output_dir).resolve()

    selected_dir = output_dir / "processed" / "selected_frames"
    if not selected_dir.exists():
        raise FileNotFoundError(
            f"Selected frames not found at {selected_dir}. Run Stage 2 first."
        )

    tracking_dir = output_dir / "tracking"
    meta_path = tracking_dir / "tracking_metadata.json"
    summary_path = tracking_dir / "tracks_summary.json"
    if not meta_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Stage 7 tracking outputs not found at {tracking_dir}. Run Stage 7 first."
        )

    stage4_dir = output_dir / "sfm"
    stage4_db = stage4_dir / "database.db"
    stage4_model_dir = stage4_dir / "sparse" / "0"
    if not stage4_model_dir.exists():
        if (stage4_dir / "sparse").exists():
            stage4_model_dir = stage4_dir / "sparse"
        else:
            raise FileNotFoundError(
                f"Stage 4 unmasked SfM model not found at {stage4_model_dir}. Run Stage 4 first."
            )
    if not stage4_db.exists():
        raise FileNotFoundError(
            f"STOP: Stage 4 COLMAP database not found at {stage4_db}. The optimized Stage 8 "
            f"path (masking.reuse_stage4_poses: true) requires Stage 4's original database.db "
            f"to reuse its extracted features and matches - it cannot be reconstructed from the "
            f"sparse model alone. Run Stage 4 first, or set masking.reuse_stage4_poses: false "
            f"to use the legacy path."
        )

    stage4_model_check = parse_colmap_model(stage4_model_dir)
    if stage4_model_check.get("num_registered_images", 0) == 0:
        raise RuntimeError(
            f"STOP: Stage 4 model at {stage4_model_dir} has zero registered images. The "
            f"optimized Stage 8 path requires a valid, already-registered Stage 4 "
            f"reconstruction to reuse as fixed pose geometry. Fix Stage 4 first, or set "
            f"masking.reuse_stage4_poses: false to use the legacy path."
        )

    refine_reused_poses = config_manager.get("masking.refine_reused_poses", False)
    report_suffix = "optimized_refined" if refine_reused_poses else "optimized"

    masking_dir = output_dir / "masking"
    masking_dir.mkdir(parents=True, exist_ok=True)
    stage8_report_path = masking_dir / f"stage8_report_{report_suffix}.json"
    comp_report_path = masking_dir / f"masking_comparison_report_{report_suffix}.json"

    resume = config_manager.get("execution.resume", True)
    if resume and stage8_report_path.exists() and comp_report_path.exists():
        logger.info(f"Existing optimized Stage 8 report found at {stage8_report_path}. Resuming.")
        with open(stage8_report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        tracking_meta = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        tracks_summary = json.load(f)

    logger.info("[optimized] Propagating dynamic object tracks across temporal gaps...")
    propagator = MaskPropagator(
        max_gap=config_manager.get("masking.max_interpolation_gap", 3),
        motion_expansion=True,
        logger=logger,
    )
    propagated_meta = propagator.propagate_tracks(tracking_meta, tracks_summary)

    logger.info("[optimized] Generating dynamic object masks...")
    dilation_px = config_manager.get("masking.dilation_px", 15)
    mask_gen = DynamicMaskGenerator(dilation_px=dilation_px, logger=logger)
    mask_result = mask_gen.generate_masks_from_tracking(
        tracking_metadata=propagated_meta,
        tracks_summary=tracks_summary,
        images_dir=selected_dir,
        output_dir=masking_dir,
    )
    masks_dir = Path(mask_result["masks_dir"])

    # Isolated output directory: distinct from Stage 4's sfm/ (never touched),
    # the legacy path's sfm_masked/ (also never touched), and - when
    # refinement is enabled - distinct from the fixed-pose optimized output
    # too, so all variants can coexist for comparison (see
    # docs/STAGE8_OPTIMIZATION.md "Limited Pose Refinement" experiment).
    dir_name = "sfm_masked_optimized_refined" if refine_reused_poses else "sfm_masked_optimized"
    sfm_masked_opt_dir = output_dir / dir_name
    sfm_masked_opt_dir.mkdir(parents=True, exist_ok=True)
    pruned_db_path = sfm_masked_opt_dir / "database.db"
    triangulated_dir = sfm_masked_opt_dir / "sparse"
    refined_dir = sfm_masked_opt_dir / "sparse_refined"

    wrapper = ColmapWrapper(config_manager=config_manager)

    timings = {}
    prune_stats = {}
    missing_masks = []
    ba_diagnostics = None
    pose_delta_report = None
    validation_result = None

    ba_diagnostics_path = sfm_masked_opt_dir / "bundle_adjustment_diagnostics.json"

    already_done = (triangulated_dir / "points3D.bin").exists() or (
        triangulated_dir / "points3D.txt"
    ).exists()
    if refine_reused_poses:
        already_done = already_done and (
            (refined_dir / "points3D.bin").exists()
            or (refined_dir / "points3D.txt").exists()
        )

    if not (resume and already_done):
        t0 = time.time()
        logger.info(
            f"[optimized] Copying Stage 4 database ({stage4_db}) to isolated "
            f"working copy ({pruned_db_path})..."
        )
        colmap_db_pruner.copy_database(stage4_db, pruned_db_path)
        timings["database_copy_seconds"] = round(time.time() - t0, 3)

        t0 = time.time()
        image_name_to_id = colmap_db_pruner.load_image_name_to_id(pruned_db_path)
        image_masks, missing_masks = colmap_db_pruner.build_image_masks(
            masks_dir, image_name_to_id, logger=logger
        )
        prune_stats = colmap_db_pruner.prune_masked_correspondences(
            pruned_db_path, image_masks, logger=logger
        )
        timings["mask_pruning_seconds"] = round(time.time() - t0, 3)

        t0 = time.time()
        logger.info(
            "[optimized] Running point_triangulator against Stage 4's existing "
            "(fixed) poses + pruned database..."
        )
        try:
            wrapper.triangulate_points(
                image_dir=selected_dir,
                db_path=pruned_db_path,
                input_model_dir=stage4_model_dir,
                output_dir=triangulated_dir,
                fix_existing_frames=True,
                refine_intrinsics=False,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"STOP: point_triangulator could not consume the Stage 4 model/database "
                f"combination. This likely means either the pruned database's remaining "
                f"matches are insufficient for triangulation, or the Stage 4 model and "
                f"database are inconsistent (e.g. mismatched image filenames). Not falling "
                f"back to the legacy mapper automatically - investigate before retrying, or "
                f"set masking.reuse_stage4_poses: false. Original COLMAP error:\n{e}"
            ) from e
        timings["point_triangulation_seconds"] = round(time.time() - t0, 3)

        bundle_adjust_error = None
        if refine_reused_poses:
            t0 = time.time()
            logger.info(
                "[optimized][refine] Running one bundle_adjuster pass to refine camera "
                "poses (never intrinsics) against the pruned/triangulated observations..."
            )
            try:
                _, _, ba_diagnostics = wrapper.bundle_adjust(
                    input_model_dir=triangulated_dir,
                    output_dir=refined_dir,
                    refine_poses=True,
                    refine_points=True,
                    refine_focal_length=False,
                    refine_principal_point=False,
                    refine_extra_params=False,
                )
                with open(ba_diagnostics_path, "w", encoding="utf-8") as f:
                    json.dump(ba_diagnostics, f, indent=2)
            except RuntimeError as e:
                # bundle_adjuster failing is NOT treated as a fatal Stage 8
                # failure: the validation gate below will record this as
                # REFINEMENT_UNAVAILABLE and Stage 8 falls back to the
                # fixed-pose (point_triangulator-only) result, which remains
                # fully valid on its own.
                logger.warning(
                    f"[optimized][refine] bundle_adjuster command failed - refinement will "
                    f"be marked UNAVAILABLE and Stage 8 will fall back to the fixed-pose "
                    f"result. Error: {e}"
                )
                bundle_adjust_error = str(e)
            timings["pose_refinement_seconds"] = round(time.time() - t0, 3)
            if ba_diagnostics is not None:
                logger.info(f"[optimized][refine] bundle_adjuster diagnostics: {ba_diagnostics}")

            if bundle_adjust_error is None:
                stage4_poses = parse_colmap_model(stage4_model_dir).get("registered_images", [])
                refined_poses = parse_colmap_model(refined_dir).get("registered_images", [])
                pose_delta_report = pose_comparator.compare_poses(stage4_poses, refined_poses)
                logger.info(
                    f"[optimized][refine] Pose change vs Stage 4 (fixed-pose baseline would be "
                    f"exactly 0): max_translation={pose_delta_report['summary']['max_translation_delta']}, "
                    f"max_rotation_deg={pose_delta_report['summary']['max_rotation_delta_degrees']}"
                )

            logger.info("[optimized][refine] Running refinement validation/acceptance gate...")
            validation_result = refinement_validator.validate_refinement(
                fixed_model_dir=triangulated_dir,
                refined_model_dir=refined_dir,
                stage4_model_dir=stage4_model_dir,
                ba_diagnostics=ba_diagnostics,
                pose_delta_report=pose_delta_report,
                bundle_adjust_error=bundle_adjust_error,
                logger=logger,
            )

        final_model_dir = (
            refined_dir
            if refine_reused_poses and validation_result["decision"] == refinement_validator.REFINEMENT_ACCEPTED
            else triangulated_dir
        )
        if refine_reused_poses and validation_result["decision"] != refinement_validator.REFINEMENT_ACCEPTED:
            logger.warning(
                f"[optimized][refine] Refinement {validation_result['decision']} - "
                f"falling back to the fixed-pose (point_triangulator) result for the "
                f"Stage 8 output. Reason: {validation_result['reason']}"
            )

        timings["total_seconds"] = round(sum(timings.values()), 3)

        if (final_model_dir / "points3D.bin").exists() or (
            final_model_dir / "points3D.txt"
        ).exists():
            ply_path = sfm_masked_opt_dir / "sparse_point_cloud.ply"
            logger.info(f"[optimized] Exporting masked sparse point cloud to {ply_path}...")
            wrapper.export_ply(input_model_dir=final_model_dir, output_ply_path=ply_path)
    else:
        logger.info("[optimized] Resuming: pruned/triangulated model already exists.")
        if refine_reused_poses:
            if ba_diagnostics_path.exists():
                with open(ba_diagnostics_path, "r", encoding="utf-8") as f:
                    ba_diagnostics = json.load(f)
            stage4_poses = parse_colmap_model(stage4_model_dir).get("registered_images", [])
            refined_poses = parse_colmap_model(refined_dir).get("registered_images", [])
            pose_delta_report = pose_comparator.compare_poses(stage4_poses, refined_poses)
            validation_result = refinement_validator.validate_refinement(
                fixed_model_dir=triangulated_dir,
                refined_model_dir=refined_dir,
                stage4_model_dir=stage4_model_dir,
                ba_diagnostics=ba_diagnostics,
                pose_delta_report=pose_delta_report,
                bundle_adjust_error=None,
                logger=logger,
            )
        final_model_dir = (
            refined_dir
            if refine_reused_poses and validation_result["decision"] == refinement_validator.REFINEMENT_ACCEPTED
            else triangulated_dir
        )

    logger.info(
        "[optimized] Comparing optimized masked reconstruction against unmasked Stage 4 baseline..."
    )
    comparator = ReconstructionComparator(logger=logger)
    comparison = comparator.compare_reconstructions(
        unmasked_model_dir=stage4_model_dir,
        masked_model_dir=final_model_dir,
        masks_dir=masks_dir,
        output_path=comp_report_path,
    )

    refinement_accepted = (
        refine_reused_poses
        and validation_result is not None
        and validation_result["decision"] == refinement_validator.REFINEMENT_ACCEPTED
    )

    report = {
        "status": "SUCCESS",
        "method": (
            "optimized_point_triangulator_reuse_refined"
            if refinement_accepted
            else "optimized_point_triangulator_reuse"
        ),
        "reused_from_stage4": {
            "database": str(stage4_db),
            "sparse_model": str(stage4_model_dir),
            "note": (
                "Camera poses reused from Stage 4 as the INITIAL geometry for "
                "point_triangulator (--Mapper.fix_existing_frames 1, "
                "--refine_intrinsics 0). "
                + (
                    "One additional bundle_adjuster pass then refined poses and "
                    "points (never intrinsics) against the pruned observations, and "
                    "the validation gate ACCEPTED the result. "
                    if refinement_accepted
                    else (
                        "A bundle_adjuster refinement pass was attempted but the "
                        "validation gate did NOT accept it, so the fixed-pose "
                        "(point_triangulator-only) result was used instead. "
                        if refine_reused_poses
                        else "Poses were then used as-is (no further refinement was requested). "
                    )
                )
                + "feature_extractor, any matcher, and the incremental mapper "
                "were NOT re-run at any point."
            ),
        },
        "recomputed": {
            "matches_pruning": prune_stats,
            "images_without_mask": missing_masks,
            "point_triangulation": True,
            "pose_refinement_requested": refine_reused_poses,
            "pose_refinement_accepted": refinement_accepted,
        },
        "pose_refinement_diagnostics": ba_diagnostics,
        "pose_change_vs_stage4": pose_delta_report,
        "refinement_validation": validation_result,
        "timings_seconds": timings,
        "mask_generation": {
            "total_masks": mask_result["total_masks"],
            "total_dynamic_objects_masked": mask_result["total_dynamic_objects"],
            "average_masked_pixel_percentage": mask_result[
                "average_masked_pixel_percentage"
            ],
            "masks_dir": str(masks_dir),
            "overlays_dir": mask_result["overlays_dir"],
        },
        "masked_sfm": {
            "sfm_masked_dir": str(sfm_masked_opt_dir),
            "registered_images": comparison["masked"]["registered_images"],
            "sparse_points": comparison["masked"]["sparse_points"],
            "mean_reprojection_error_px": comparison["masked"][
                "mean_reprojection_error_px"
            ],
        },
        "comparison_summary": comparison["comparison"],
        "output_paths": {
            "masking_dir": str(masking_dir),
            "masks_dir": str(masks_dir),
            "overlays_dir": mask_result["overlays_dir"],
            "sfm_masked_optimized_dir": str(sfm_masked_opt_dir),
            "fixed_pose_model_dir": str(triangulated_dir) if triangulated_dir.exists() else None,
            "refined_model_dir": (
                str(refined_dir) if refine_reused_poses and refined_dir.exists() else None
            ),
            "final_model_dir_used": str(final_model_dir),
            "masked_point_cloud_ply": str(sfm_masked_opt_dir / "sparse_point_cloud.ply")
            if (sfm_masked_opt_dir / "sparse_point_cloud.ply").exists()
            else None,
            "masking_comparison_report_json": str(comp_report_path),
            "stage8_report_json": str(stage8_report_path),
        },
    }

    with open(stage8_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("[optimized] Stage 8 (optimized) Dynamic Object Masking complete.")
    logger.info(f"Verdict: {comparison['comparison']['verdict']}")
    logger.info(f"Report written to: {stage8_report_path}")

    return report
