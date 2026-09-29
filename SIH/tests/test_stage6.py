import json
import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image

from src.segmentation.uavid_classes import (
    RGB_TO_CLASS,
    CLASS_TO_RGB,
    NUM_CLASSES,
    CLASSES,
    rgb_label_to_class_index,
    class_index_to_rgb,
)
from src.segmentation.dataset import UAVidDataset
from src.segmentation.model import SegmentationModel
from src.segmentation.evaluator import evaluate_segmentation


TRAIN_ROOT = Path("data/raw/uavid_train")
VAL_ROOT = Path("data/raw/uavid_val")


def test_rgb_to_class_mapping():
    assert len(RGB_TO_CLASS) == NUM_CLASSES
    for rgb, idx in RGB_TO_CLASS.items():
        assert 0 <= idx < NUM_CLASSES
        assert CLASS_TO_RGB[idx] == rgb


def test_rgb_label_to_class_index_all_colors():
    h, w = 8, 8
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    colors = list(RGB_TO_CLASS.keys())
    for i, (r, g, b) in enumerate(colors):
        rgb[i, 0] = [r, g, b]

    result = rgb_label_to_class_index(rgb)
    assert result.shape == (h, w)
    for i, color in enumerate(colors):
        expected = RGB_TO_CLASS[color]
        assert result[i, 0] == expected, f"Color {color} should map to class {expected}, got {result[i, 0]}"


def test_class_index_to_rgb_roundtrip():
    class_map = np.arange(NUM_CLASSES, dtype=np.uint8).reshape(1, NUM_CLASSES)
    rgb_out = class_index_to_rgb(class_map)
    assert rgb_out.shape == (1, NUM_CLASSES, 3)
    for c in range(NUM_CLASSES):
        expected = CLASS_TO_RGB[c]
        got = tuple(rgb_out[0, c])
        assert got == expected, f"class {c} roundtrip failed: expected {expected}, got {got}"


def test_uavid_dataset_loads():
    if not TRAIN_ROOT.exists():
        pytest.skip("UAVid training data not found")
    ds = UAVidDataset(split_root=TRAIN_ROOT, input_size=(256, 256), augment=False, max_samples=2)
    assert len(ds) == 2
    img, lbl, meta = ds[0]
    assert img.shape == (3, 256, 256)
    assert lbl.shape == (256, 256)
    assert lbl.dtype == torch.int64
    assert int(lbl.min()) >= 0
    assert int(lbl.max()) < NUM_CLASSES
    assert "image_path" in meta


def test_model_forward():
    model = SegmentationModel(model_name="deeplabv3_mobilenet_v3_large", num_classes=NUM_CLASSES)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert "out" in out
    assert out["out"].shape == (1, NUM_CLASSES, 256, 256)


def test_model_predict():
    model = SegmentationModel(model_name="deeplabv3_mobilenet_v3_large", num_classes=NUM_CLASSES)
    x = torch.randn(1, 3, 256, 256)
    pred = model.predict(x)
    assert pred.shape == (1, 256, 256)
    assert int(pred.min()) >= 0
    assert int(pred.max()) < NUM_CLASSES


def test_model_save_load(tmp_path):
    model = SegmentationModel()
    ckpt_path = tmp_path / "test_ckpt.pth"
    model.save(ckpt_path)
    assert ckpt_path.exists()
    model2 = SegmentationModel()
    model2.load(ckpt_path)


def test_evaluator_metrics(tmp_path):
    mask_dir = tmp_path / "masks"
    gt_dir = tmp_path / "gt_root" / "seq1" / "Labels"
    mask_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    h, w = 64, 64
    gt = np.zeros((h, w, 3), dtype=np.uint8)
    r, g, b = list(RGB_TO_CLASS.keys())[0]
    gt[:, :] = [r, g, b]

    pred = gt.copy()

    Image.fromarray(gt).save(str(gt_dir / "000000.png"))
    Image.fromarray(pred).save(str(mask_dir / "seq1_000000_mask.png"))

    manifest = [{
        "image_path": str(tmp_path / "gt_root" / "seq1" / "Images" / "000000.png"),
        "mask_path": str(mask_dir / "seq1_000000_mask.png"),
        "overlay_path": str(mask_dir / "seq1_000000_overlay.png"),
    }]

    result = evaluate_segmentation(
        manifest=manifest,
        gt_root=str(tmp_path / "gt_root"),
        output_dir=tmp_path / "eval_out",
    )
    assert result["status"] == "SUCCESS"
    assert abs(result["mIoU"] - 1.0) < 1e-6, f"Expected mIoU=1.0 for perfect pred, got {result['mIoU']}"
    assert abs(result["pixel_accuracy"] - 1.0) < 1e-6


def test_inference_writes_masks(tmp_path):
    if not VAL_ROOT.exists():
        pytest.skip("UAVid val data not found")
    from src.segmentation.inference import run_inference

    ds = UAVidDataset(split_root=VAL_ROOT, input_size=(256, 256), augment=False, max_samples=2)
    model = SegmentationModel()
    manifest = run_inference(model=model, dataset=ds, output_dir=tmp_path / "inf_out", batch_size=1)

    assert len(manifest) == 2
    for entry in manifest:
        assert Path(entry["mask_path"]).exists()
        assert Path(entry["overlay_path"]).exists()
        mask = np.array(Image.open(entry["mask_path"]))
        assert mask.shape[2] == 3
