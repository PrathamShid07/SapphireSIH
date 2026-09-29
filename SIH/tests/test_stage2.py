import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from src.config.config_manager import ConfigManager
from src.preprocessing.video_metadata import extract_video_metadata
from src.preprocessing.quality_analyzer import QualityAnalyzer
from src.preprocessing.frame_extractor import VideoPreprocessor
from src.preprocessing.stage2_pipeline import run_stage2

def create_synthetic_video(file_path, num_frames=30, fps=10.0, width=320, height=240):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    
    for i in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if i == 5:
            img = cv2.GaussianBlur(img, (21, 21), 0)
        elif i == 10:
            img.fill(10)
        elif i == 15:
            img.fill(245)
        else:
            cv2.rectangle(img, (30 + i * 2, 30), (150 + i * 2, 150), (0, 255, i * 8 % 255), -1)
            cv2.putText(img, f"Frame {i}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        out.write(img)
        
    out.release()

def test_video_metadata_extraction(tmp_path):
    video_file = tmp_path / "test_meta.mp4"
    create_synthetic_video(video_file, num_frames=20, fps=10.0)
    
    meta = extract_video_metadata(video_file)
    assert meta["file_name"] == "test_meta.mp4"
    assert meta["total_frames"] == 20
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["fps"] == 10.0
    assert meta["duration_sec"] == 2.0

def test_quality_analyzer():
    analyzer = QualityAnalyzer({
        "min_blur_score": 50.0,
        "min_brightness": 20.0,
        "max_brightness": 230.0,
        "min_contrast": 15.0,
        "max_similarity": 0.95
    })
    
    good_img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(good_img, (50, 50), (200, 200), (255, 255, 255), -1)
    
    is_sel, reason, scores, gray = analyzer.analyze(good_img)
    assert is_sel is True
    assert reason is None
    assert scores["blur_score"] > 50.0
    
    dark_img = np.zeros((240, 320, 3), dtype=np.uint8)
    dark_img.fill(5)
    is_sel_dark, reason_dark, _, _ = analyzer.analyze(dark_img)
    assert is_sel_dark is False
    assert reason_dark in ["UNDEREXPOSED", "BLURRY", "LOW_CONTRAST"]

def test_video_preprocessor_pipeline(tmp_path):
    video_file = tmp_path / "sample.mp4"
    create_synthetic_video(video_file, num_frames=30, fps=10.0)
    
    out_dir = tmp_path / "output_run"
    cfg = ConfigManager(root_dir=tmp_path)
    
    manifest_data = run_stage2(video_file, output_dir=out_dir, config_manager=cfg)
    
    assert manifest_data["status"] == "COMPLETED"
    assert manifest_data["sampling"]["total_video_frames"] == 30
    assert manifest_data["sampling"]["extracted_frames_count"] > 0
    assert manifest_data["sampling"]["selected_frames_count"] > 0
    
    manifest_path = Path(manifest_data["output_paths"]["manifest_path"])
    assert manifest_path.exists()
    
    selected_dir = Path(manifest_data["output_paths"]["selected_dir"])
    assert selected_dir.exists()
    
    contact_sheet = Path(manifest_data["output_paths"]["contact_sheet_path"])
    assert contact_sheet.exists()

def test_resume_safety(tmp_path):
    video_file = tmp_path / "resume_test.mp4"
    create_synthetic_video(video_file, num_frames=15, fps=5.0)
    
    out_dir = tmp_path / "output_resume"
    cfg = ConfigManager(root_dir=tmp_path)
    
    m1 = run_stage2(video_file, output_dir=out_dir, config_manager=cfg)
    m2 = run_stage2(video_file, output_dir=out_dir, config_manager=cfg)
    
    assert m1["sampling"]["selected_frames_count"] == m2["sampling"]["selected_frames_count"]
