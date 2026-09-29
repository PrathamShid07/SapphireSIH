import json
from pathlib import Path
from src.config.config_manager import ConfigManager
from src.utils.logger import setup_logger
from .visdrone_classes import VEHICLE_CLASS_NAMES
from .detector_trainer import VehicleDetector
from .sort_tracker import SortTracker
from .frame_processor import FrameProcessor


def run_stage7(video_path=None, frames_dir=None, output_dir=None, config_manager=None):
    if config_manager is None:
        config_manager = ConfigManager()

    if output_dir is None:
        base_out = config_manager.get_path("paths.outputs_dir")
        if video_path is not None:
            run_root = base_out / Path(video_path).stem
        elif frames_dir is not None:
            run_root = base_out / Path(frames_dir).parent.name
        else:
            run_root = base_out / "stage7_tracking"
        output_dir = run_root / "tracking" if run_root.name != "stage7_tracking" else run_root
    else:
        output_dir = Path(output_dir).resolve()
        if output_dir.name != "tracking":
            output_dir = output_dir / "tracking"
        run_root = output_dir.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = config_manager.get_path("paths.logs_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("stage7_tracking", log_file=log_dir / "stage7.log")

    logger.info("Starting Stage 7 - Detection, Instance Segmentation and Tracking")

    report_path = output_dir / "stage7_report.json"
    meta_path = output_dir / "tracking_metadata.json"
    summary_path = output_dir / "tracks_summary.json"

    resume = config_manager.get("execution.resume", True)
    if resume and report_path.exists() and meta_path.exists():
        logger.info(f"Existing Stage 7 report found at {report_path}. Resuming and skipping.")
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    image_paths = []
    if frames_dir is not None:
        fdir = Path(frames_dir)
        valid_exts = {".jpg", ".jpeg", ".png"}
        image_paths = sorted([f for f in fdir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    elif video_path is not None:
        stage2_selected = run_root / "processed" / "selected_frames"
        if stage2_selected.exists():
            valid_exts = {".jpg", ".jpeg", ".png"}
            image_paths = sorted([f for f in stage2_selected.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
        else:
            raise FileNotFoundError(
                f"Selected frames from Stage 2 not found at {stage2_selected}. Run Stage 2 first."
            )
    else:
        raise ValueError("Either video_path or frames_dir must be provided for Stage 7 tracking.")

    if not image_paths:
        raise ValueError("No frames found to track.")

    logger.info(f"Found {len(image_paths)} frames for tracking.")

    model_name = config_manager.get("detection.model", "fasterrcnn_mobilenet_v3_large_fpn")
    conf_threshold = config_manager.get("detection.confidence_threshold", 0.5)
    target_classes = config_manager.get("detection.target_classes", list(VEHICLE_CLASS_NAMES))

    max_age = config_manager.get("tracking.max_age", 5)
    min_hits = config_manager.get("tracking.min_hits", 2)
    iou_threshold = config_manager.get("tracking.iou_threshold", 0.3)

    logger.info(f"Initializing detector: {model_name}")
    detector = VehicleDetector(model_name=model_name)

    logger.info(f"Initializing SORT tracker (max_age={max_age}, min_hits={min_hits}, iou_thresh={iou_threshold})")
    tracker = SortTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    processor = FrameProcessor(
        detector=detector,
        tracker=tracker,
        conf_threshold=conf_threshold,
        target_classes=target_classes,
    )

    logger.info("Processing frame sequence...")
    proc_result = processor.process_frame_sequence(
        image_paths=image_paths,
        output_dir=output_dir,
        save_visualizations=True,
    )

    tracks_summary = proc_result["tracks_summary"]
    dynamic_count = sum(1 for t in tracks_summary.values() if t.get("is_dynamic", False))
    static_count = len(tracks_summary) - dynamic_count

    class_counts = {}
    for t in tracks_summary.values():
        c = t.get("class_name", "unknown")
        class_counts[c] = class_counts.get(c, 0) + 1

    report = {
        "status": "SUCCESS",
        "total_frames_processed": len(image_paths),
        "unique_tracks_count": len(tracks_summary),
        "dynamic_objects_count": dynamic_count,
        "static_objects_count": static_count,
        "vehicles_by_class": class_counts,
        "output_paths": {
            "tracking_dir": str(output_dir),
            "tracking_metadata_json": str(meta_path),
            "tracks_summary_json": str(summary_path),
            "visualizations_dir": str(output_dir / "visualizations"),
            "stage7_report_json": str(report_path),
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Stage 7 complete. Found {len(tracks_summary)} unique tracks ({dynamic_count} dynamic, {static_count} static).")
    logger.info(f"Report written to: {report_path}")

    return report
