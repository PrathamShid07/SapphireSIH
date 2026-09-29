import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import torch

from src.detection.visdrone_classes import (
    VISDRONE_CLASSES,
    VEHICLE_CLASS_NAMES,
    COCO_TO_VISDRONE,
    get_track_color,
)
from src.detection.sort_tracker import (
    KalmanBoxTracker,
    SortTracker,
    compute_iou,
    bbox_to_z,
    x_to_bbox,
)
from src.detection.detector_trainer import (
    VisDroneDetectionDataset,
    VehicleDetector,
    evaluate_detector,
)
from src.detection.frame_processor import (
    FrameProcessor,
    draw_tracking_visualization,
)
from src.detection.stage7_pipeline import run_stage7


def test_compute_iou():
    bb1 = [10.0, 10.0, 50.0, 50.0]
    bb2 = [10.0, 10.0, 50.0, 50.0]
    assert abs(compute_iou(bb1, bb2) - 1.0) < 1e-5

    bb3 = [60.0, 60.0, 100.0, 100.0]
    assert compute_iou(bb1, bb3) == 0.0

    bb4 = [30.0, 10.0, 70.0, 50.0]
    iou = compute_iou(bb1, bb4)
    assert 0.0 < iou < 1.0


def test_bbox_z_roundtrip():
    bbox = np.array([20.0, 30.0, 80.0, 110.0])
    z = bbox_to_z(bbox)
    x = np.zeros((7, 1))
    x[:4] = z
    recovered = x_to_bbox(x)
    assert np.allclose(bbox, recovered, atol=1e-4)


def test_kalman_box_tracker_state_and_prediction():
    bbox = [100.0, 100.0, 160.0, 180.0]
    tracker = KalmanBoxTracker(bbox, score=0.95, class_id=4, class_name="car")

    assert tracker.hits == 0
    assert tracker.time_since_update == 0

    pred_box = tracker.predict()
    assert len(pred_box) == 4
    assert tracker.time_since_update == 1

    tracker.update([102.0, 103.0, 162.0, 183.0], score=0.96)
    assert tracker.hits == 1
    assert tracker.time_since_update == 0

    vx, vy = tracker.get_velocity()
    assert isinstance(vx, float)
    assert isinstance(vy, float)
    assert len(tracker.history) >= 2


def test_sort_tracker_id_persistence():
    tracker = SortTracker(max_age=3, min_hits=1, iou_threshold=0.2)

    track_ids = []
    for i in range(5):
        box = [100.0 + i * 5.0, 100.0, 150.0 + i * 5.0, 150.0]
        dets = [{"bbox": box, "score": 0.9, "class_id": 4, "class_name": "car"}]
        active = tracker.update(dets)
        assert len(active) == 1
        track_ids.append(active[0]["track_id"])

    assert len(set(track_ids)) == 1


def test_sort_tracker_age_out():
    tracker = SortTracker(max_age=2, min_hits=1, iou_threshold=0.2)
    dets = [{"bbox": [50.0, 50.0, 100.0, 100.0], "score": 0.9, "class_name": "car"}]
    active = tracker.update(dets)
    assert len(active) == 1

    tracker.update([])
    tracker.update([])
    active_after = tracker.update([])
    assert len(active_after) == 0
    assert len(tracker.trackers) == 0


def test_visdrone_classes_and_palette():
    assert "car" in VEHICLE_CLASS_NAMES
    assert "truck" in VEHICLE_CLASS_NAMES
    assert "bus" in VEHICLE_CLASS_NAMES
    assert "van" in VEHICLE_CLASS_NAMES

    assert 3 in COCO_TO_VISDRONE
    assert COCO_TO_VISDRONE[3] == (4, "car")

    for tid in range(10):
        color = get_track_color(tid)
        assert len(color) == 3
        for c in color:
            assert 0 <= c <= 255


def test_visdrone_dataset_parsing(tmp_path):
    img_dir = tmp_path / "images"
    anno_dir = tmp_path / "annotations"
    img_dir.mkdir()
    anno_dir.mkdir()

    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img.save(str(img_dir / "00001.jpg"))

    anno_content = "50,60,100,80,1,4,0,0\n200,220,120,90,1,6,0,0\n10,10,30,40,1,1,0,0\n"
    with open(anno_dir / "00001.txt", "w", encoding="utf-8") as f:
        f.write(anno_content)

    dataset = VisDroneDetectionDataset(
        images_dir=img_dir,
        annotations_dir=anno_dir,
        target_classes=["car", "truck"],
    )
    assert len(dataset) == 1
    tensor, target, p = dataset[0]
    assert tensor.shape == (3, 480, 640)
    assert len(target["boxes"]) == 2
    assert int(target["labels"][0]) == 4
    assert int(target["labels"][1]) == 6


def test_vehicle_detector_predict_format():
    detector = VehicleDetector(model_name="fasterrcnn_mobilenet_v3_large_fpn")
    synthetic_img = np.zeros((300, 400, 3), dtype=np.uint8)
    preds = detector.predict([synthetic_img], conf_threshold=0.1)

    assert isinstance(preds, list)
    assert len(preds) == 1
    for d in preds[0]:
        assert "bbox" in d
        assert len(d["bbox"]) == 4
        assert "score" in d
        assert "class_name" in d
        assert "class_id" in d


class MockDetector:
    def __init__(self):
        self.call_count = 0

    def predict(self, images, conf_threshold=0.5, target_classes=None):
        self.call_count += 1
        x1 = 50.0 + self.call_count * 10.0
        y1 = 50.0
        x2 = 120.0 + self.call_count * 10.0
        y2 = 100.0
        dets = [
            {
                "bbox": [x1, y1, x2, y2],
                "score": 0.92,
                "class_id": 4,
                "class_name": "car",
            }
        ]
        return [dets]


def test_frame_processor_tracking_and_vis(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    out_dir = tmp_path / "track_out"

    img_paths = []
    for i in range(4):
        img_p = frames_dir / f"frame_{i:03d}.jpg"
        Image.new("RGB", (320, 240), color=(50, 50, 50)).save(str(img_p))
        img_paths.append(img_p)

    mock_det = MockDetector()
    tracker = SortTracker(max_age=3, min_hits=1, iou_threshold=0.2)
    processor = FrameProcessor(detector=mock_det, tracker=tracker)

    res = processor.process_frame_sequence(img_paths, out_dir, save_visualizations=True)

    assert Path(res["tracking_metadata_path"]).exists()
    assert Path(res["tracks_summary_path"]).exists()
    assert res["total_frames_processed"] == 4
    assert res["unique_tracks_count"] == 1

    vis_dir = out_dir / "visualizations"
    assert vis_dir.exists()
    assert len(list(vis_dir.glob("*.jpg"))) == 4

    with open(res["tracking_metadata_path"], "r", encoding="utf-8") as f:
        meta = json.load(f)
        assert len(meta) == 4
        assert meta[0]["active_objects_count"] == 1
        assert "velocity" in meta[0]["objects"][0]


def test_stage7_pipeline_smoke(tmp_path):
    frames_dir = tmp_path / "input_frames"
    frames_dir.mkdir()
    out_dir = tmp_path / "stage7_out"

    for i in range(3):
        Image.new("RGB", (320, 240), color=(80, 80, 80)).save(str(frames_dir / f"f_{i:02d}.jpg"))

    report = run_stage7(frames_dir=frames_dir, output_dir=out_dir)

    assert report["status"] == "SUCCESS"
    assert report["total_frames_processed"] == 3
    assert Path(report["output_paths"]["stage7_report_json"]).exists()
    assert Path(report["output_paths"]["tracking_metadata_json"]).exists()
    assert Path(report["output_paths"]["tracks_summary_json"]).exists()
