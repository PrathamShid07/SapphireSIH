import json
from pathlib import Path
import pytest
from src.config.config_manager import ConfigManager
from src.metadata.camera_model import CameraIntrinsics
from src.metadata.telemetry_parser import TelemetryParser
from src.metadata.coordinate_system import CoordinateSystem
from src.metadata.frame_synchronizer import FrameSynchronizer
from src.metadata.stage3_pipeline import run_stage3
from src.preprocessing.stage2_pipeline import run_stage2
from tests.test_stage2 import create_synthetic_video

def test_camera_intrinsics_default():
    cam = CameraIntrinsics(width=1920, height=1080)
    d = cam.to_dict()
    assert d["width"] == 1920
    assert d["height"] == 1080
    assert d["fx"] == 2304.0
    assert d["fy"] == 2304.0
    assert d["cx"] == 960.0
    assert d["cy"] == 540.0
    assert d["calibration_type"] == "ESTIMATED_DEFAULT"

def test_camera_intrinsics_custom():
    custom = {
        "fx": 1500.0, "fy": 1500.0, "cx": 960.0, "cy": 540.0,
        "k1": -0.1, "k2": 0.02, "p1": 0.0, "p2": 0.0,
        "calibration_type": "CALIBRATED", "model": "OPENCV"
    }
    cam = CameraIntrinsics(width=1920, height=1080, custom_calib=custom)
    d = cam.to_dict()
    assert d["fx"] == 1500.0
    assert d["calibration_type"] == "CALIBRATED"
    assert d["distortion"]["k1"] == -0.1

def test_telemetry_parser_no_data(tmp_path):
    parser = TelemetryParser(input_dir=tmp_path)
    res = parser.extract()
    assert res["telemetry_detected"] is False
    assert res["gps_available"] is False
    assert res["imu_available"] is False
    assert res["source_file"] is None

def test_telemetry_parser_csv(tmp_path):
    csv_file = tmp_path / "gps.csv"
    csv_file.write_text(
        "timestamp,latitude,longitude,altitude,yaw,pitch,roll\n"
        "0.0,12.9716,77.5946,920.5,45.0,0.0,0.0\n"
        "1.0,12.9717,77.5947,921.0,46.0,1.0,0.0\n",
        encoding="utf-8"
    )
    parser = TelemetryParser(input_dir=tmp_path)
    res = parser.extract()
    assert res["telemetry_detected"] is True
    assert res["gps_available"] is True
    assert len(res["records"]) == 2

def test_coordinate_system():
    cs_ungeo = CoordinateSystem()
    assert cs_ungeo.georeferenced is False
    x, y, z = cs_ungeo.latlon_to_local_enu(12.9716, 77.5946, 920.0)
    assert x is None
    
    cs_geo = CoordinateSystem(ref_lat=12.9716, ref_lon=77.5946, ref_alt=920.0)
    assert cs_geo.georeferenced is True
    e, n, u = cs_geo.latlon_to_local_enu(12.9716, 77.5946, 925.0)
    assert e == 0.0
    assert n == 0.0
    assert u == 5.0

def test_stage3_pipeline(tmp_path):
    video_file = tmp_path / "drone.mp4"
    create_synthetic_video(video_file, num_frames=20, fps=10.0)
    
    out_dir = tmp_path / "stage3_out"
    cfg = ConfigManager(root_dir=tmp_path)
    
    run_stage2(video_file, output_dir=out_dir, config_manager=cfg)
    res = run_stage3(video_file, output_dir=out_dir, config_manager=cfg)
    
    assert res["status"] == "COMPLETED"
    meta_dir = Path(res["output_paths"]["metadata_dir"])
    assert (meta_dir / "video_metadata.json").exists()
    assert (meta_dir / "camera_metadata.json").exists()
    assert (meta_dir / "frame_metadata.json").exists()
    assert (meta_dir / "flight_metadata.json").exists()
    assert (meta_dir / "telemetry_status.json").exists()

    with open(meta_dir / "telemetry_status.json", "r", encoding="utf-8") as f:
        ts = json.load(f)
    assert ts["gps_available"] is False
    assert ts["trajectory_visualization"] == "DEFERRED_TO_STAGE4_COLMAP"
