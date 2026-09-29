import csv
import json
from pathlib import Path
from src.config.config_manager import ConfigManager
from src.utils.logger import setup_logger
from .camera_model import CameraIntrinsics
from .telemetry_parser import TelemetryParser, validate_telemetry_records
from .frame_synchronizer import FrameSynchronizer

def write_reconstruction_manifest_csv(path, rows):
    if not rows:
        return
    fields = [k for k in rows[0].keys() if k != "telemetry_invalid_reasons"] + ["telemetry_invalid_reasons"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = dict(r)
            row["telemetry_invalid_reasons"] = ";".join(r["telemetry_invalid_reasons"])
            writer.writerow(row)

def run_stage3(video_path, output_dir=None, config_manager=None):
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
    logger = setup_logger("stage3_metadata", log_file=log_dir / "stage3.log")

    logger.info(f"Starting Stage 3 Metadata Extraction for: {vpath}")

    stage2_manifest_path = output_dir / "manifest.json"
    if not stage2_manifest_path.exists():
        raise FileNotFoundError(
            f"Stage 2 manifest.json not found at {stage2_manifest_path}. Run Stage 2 preprocessing first."
        )

    with open(stage2_manifest_path, "r", encoding="utf-8") as f:
        stage2_manifest = json.load(f)

    video_meta = stage2_manifest.get("video_metadata", {})
    width = video_meta.get("width", 1920)
    height = video_meta.get("height", 1080)

    custom_calib = config_manager.get("camera.custom_calibration")
    sensor_width_mm = config_manager.get("camera.sensor_width_mm")
    focal_length_mm = config_manager.get("camera.focal_length_mm")

    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        custom_calib=custom_calib,
        sensor_width_mm=sensor_width_mm,
        focal_length_mm=focal_length_mm,
    )

    telem_parser = TelemetryParser(
        video_path=vpath,
        input_dir=vpath.parent,
        telemetry_path=config_manager.get_path("input.telemetry_csv"),
    )
    telemetry_data = telem_parser.extract()

    synchronizer = FrameSynchronizer(
        camera_intrinsics=intrinsics,
        max_time_diff_s=config_manager.get("telemetry.max_time_diff_s"),
        frame_index_tolerance=config_manager.get("telemetry.frame_index_tolerance", 1),
        require_valid_for_reconstruction=bool(config_manager.get("telemetry.require_valid_for_reconstruction", False)),
    )
    synced_frames, flight_summary = synchronizer.synchronize(stage2_manifest, telemetry_data)

    telemetry_validation = validate_telemetry_records(telemetry_data.get("records", []))
    reconstruction_manifest = None
    if stage2_manifest.get("reconstruction_sampling", {}).get("strategy") == "decoded_stride":
        reconstruction_manifest = synchronizer.build_reconstruction_manifest(stage2_manifest, telemetry_data)

    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    video_meta_path = meta_dir / "video_metadata.json"
    camera_meta_path = meta_dir / "camera_metadata.json"
    frame_meta_path = meta_dir / "frame_metadata.json"
    flight_meta_path = meta_dir / "flight_metadata.json"
    telemetry_status_path = meta_dir / "telemetry_status.json"

    with open(video_meta_path, "w", encoding="utf-8") as f:
        json.dump(video_meta, f, indent=2)

    with open(camera_meta_path, "w", encoding="utf-8") as f:
        json.dump(intrinsics.to_dict(), f, indent=2)

    with open(frame_meta_path, "w", encoding="utf-8") as f:
        json.dump(synced_frames, f, indent=2)

    with open(flight_meta_path, "w", encoding="utf-8") as f:
        json.dump(flight_summary, f, indent=2)

    reconstruction_manifest_path = None
    if reconstruction_manifest is not None:
        reconstruction_manifest_path = meta_dir / "reconstruction_manifest.json"
        with open(reconstruction_manifest_path, "w", encoding="utf-8") as f:
            json.dump(reconstruction_manifest, f, indent=2)
        write_reconstruction_manifest_csv(meta_dir / "reconstruction_manifest.csv", reconstruction_manifest["frames"])
        ms = reconstruction_manifest["summary"]
        logger.info(
            f"Reconstruction manifest: {ms['eligible_frame_count']}/{ms['reconstruction_frame_count']} frames eligible, "
            f"{ms['telemetry_rows_unassociated_count']} telemetry rows without a decoded frame, "
            f"max time diff {ms['max_observed_time_diff_s']}s"
        )

    telemetry_status = {
        "telemetry_detected": telemetry_data.get("telemetry_detected", False),
        "gps_available": telemetry_data.get("gps_available", False),
        "imu_available": telemetry_data.get("imu_available", False),
        "sources_checked": telemetry_data.get("sources_checked", []),
        "source_file": telemetry_data.get("source_file"),
        "reason": telemetry_data.get("reason"),
        "parse_error": telemetry_data.get("parse_error"),
        "validation": telemetry_validation,
        "trajectory_visualization": "GENERATED" if telemetry_data.get("gps_available") else "DEFERRED_TO_STAGE4_COLMAP",
    }

    with open(telemetry_status_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_status, f, indent=2)

    logger.info("Stage 3 Metadata extraction complete.")
    logger.info(f"Calibration Type: {intrinsics.calibration_type}")
    logger.info(f"GPS Available: {telemetry_data.get('gps_available', False)}")
    logger.info(f"IMU Available: {telemetry_data.get('imu_available', False)}")
    logger.info(f"Metadata written to: {meta_dir}")

    return {
        "status": "COMPLETED",
        "video_metadata": video_meta,
        "camera_metadata": intrinsics.to_dict(),
        "flight_summary": flight_summary,
        "telemetry_status": telemetry_status,
        "output_paths": {
            "metadata_dir": str(meta_dir),
            "video_metadata_path": str(video_meta_path),
            "camera_metadata_path": str(camera_meta_path),
            "frame_metadata_path": str(frame_meta_path),
            "flight_metadata_path": str(flight_meta_path),
            "telemetry_status_path": str(telemetry_status_path),
            "reconstruction_manifest_path": str(reconstruction_manifest_path) if reconstruction_manifest_path else None,
        }
    }
