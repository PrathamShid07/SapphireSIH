import struct
import json
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
from src.sfm.colmap_wrapper import ColmapWrapper
from src.sfm.colmap_parser import parse_colmap_model, read_cameras_binary, read_images_binary, read_points3D_binary
from src.utils.system_info import check_colmap_installation
from src.utils.geometry_checks import compute_pca_point_cloud, check_ply_geometry
from src.config.config_manager import ConfigManager

def test_colmap_wrapper_init(tmp_path):
    col_info = check_colmap_installation()
    if col_info["available"]:
        wrapper = ColmapWrapper()
        assert wrapper.colmap_exe is not None
        assert Path(wrapper.colmap_exe).exists()
    else:
        dummy = tmp_path / "colmap"
        dummy.touch()
        wrapper = ColmapWrapper(colmap_path=dummy)
        assert wrapper.colmap_exe == str(dummy)

def test_colmap_parser_text_model(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    
    cameras_txt = model_dir / "cameras.txt"
    cameras_txt.write_text("1 OPENCV 1920 1080 2304 2304 960 540 0.0 0.0 0.0 0.0\n", encoding="utf-8")
    
    images_txt = model_dir / "images.txt"
    images_txt.write_text("1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 image_001.jpg\n10 20 1\n", encoding="utf-8")
    
    points_txt = model_dir / "points3D.txt"
    points_txt.write_text("1 1.0 2.0 3.0 255 0 0 0.5 1 0 0\n", encoding="utf-8")
    
    parsed = parse_colmap_model(model_dir)
    assert parsed["num_cameras"] == 1
    assert parsed["num_registered_images"] == 1
    assert parsed["num_sparse_points"] == 1
    assert parsed["cameras"][0]["width"] == 1920
    assert parsed["registered_images"][0]["name"] == "image_001.jpg"

def test_geometry_checks_pca():
    # 1. 3D isotropic cloud (not flattened)
    np.random.seed(42)
    pts_3d = np.random.normal(0, 1, (500, 3))
    res_3d = compute_pca_point_cloud(pts_3d, warn_threshold=500.0)
    assert res_3d["is_flattened"] is False
    assert res_3d["eigenvalue_ratio"] < 10.0

    # 2. Severe flat 2D plane with tiny z noise (flattened ratio > 500)
    x = np.random.uniform(-100, 100, 1000)
    y = np.random.uniform(-100, 100, 1000)
    z = np.random.normal(0, 0.001, 1000)
    flat_pts = np.column_stack([x, y, z])
    res_flat = compute_pca_point_cloud(flat_pts, warn_threshold=500.0)
    assert res_flat["is_flattened"] is True
    assert res_flat["eigenvalue_ratio"] > 500.0
    assert len(res_flat["dominant_axis"]) == 3
    assert len(res_flat["flattest_axis"]) == 3

def test_camera_prior_validation_missing(tmp_path):
    from src.sfm.stage4_pipeline import run_stage4
    cfg = ConfigManager()
    cfg.config["camera"]["sensor_width_mm"] = None
    cfg.config["camera"]["focal_length_mm"] = None
    cfg.config["camera"]["custom_calibration"] = None
    cfg.config["sfm"]["single_camera"] = True

    frames_dir = tmp_path / "processed" / "selected_frames"
    frames_dir.mkdir(parents=True)
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(frames_dir / "frame_0001.jpg")

    with pytest.raises(ValueError, match="Camera intrinsics prior missing"):
        run_stage4("dummy.mp4", output_dir=tmp_path, config_manager=cfg)
