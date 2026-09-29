import json
import warnings
from pathlib import Path
import numpy as np
from PIL import Image
from .uavid_classes import NUM_CLASSES, CLASSES, rgb_label_to_class_index


def evaluate_segmentation(manifest, gt_root, output_dir, logger=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intersection = np.zeros(NUM_CLASSES, dtype=np.int64)
    union = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_correct = 0
    total_pixels = 0
    evaluated = 0
    skipped = 0

    for entry in manifest:
        img_path = Path(entry["image_path"])
        mask_path = Path(entry["mask_path"])

        seq_name = img_path.parent.parent.name
        frame_name = img_path.name
        gt_path = Path(gt_root) / seq_name / "Labels" / frame_name

        if not gt_path.exists():
            warnings.warn(f"GT label not found, skipping: {gt_path}")
            skipped += 1
            continue

        if not mask_path.exists():
            warnings.warn(f"Predicted mask not found, skipping: {mask_path}")
            skipped += 1
            continue

        gt_rgb = np.array(Image.open(gt_path).convert("RGB"))
        pred_rgb = np.array(Image.open(mask_path).convert("RGB"))

        if gt_rgb.shape != pred_rgb.shape:
            gt_pil = Image.fromarray(gt_rgb).resize(
                (pred_rgb.shape[1], pred_rgb.shape[0]), Image.NEAREST
            )
            gt_rgb = np.array(gt_pil)

        gt = rgb_label_to_class_index(gt_rgb)
        pred = rgb_label_to_class_index(pred_rgb)

        total_correct += int((pred == gt).sum())
        total_pixels += int(gt.size)

        for c in range(NUM_CLASSES):
            gt_c = gt == c
            pred_c = pred == c
            intersection[c] += int((gt_c & pred_c).sum())
            union[c] += int((gt_c | pred_c).sum())

        evaluated += 1

    if evaluated == 0:
        result = {"status": "NO_PREDICTIONS_EVALUATED", "evaluated": 0, "skipped": skipped}
        _write(output_dir, result)
        return result

    iou_per_class = {}
    for c in range(NUM_CLASSES):
        if union[c] == 0:
            iou_per_class[CLASSES[c]] = None
        else:
            iou_per_class[CLASSES[c]] = float(intersection[c]) / float(union[c])

    valid_ious = [v for v in iou_per_class.values() if v is not None]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    pixel_accuracy = float(total_correct) / float(total_pixels) if total_pixels > 0 else 0.0

    result = {
        "status": "SUCCESS",
        "evaluated_frames": evaluated,
        "skipped_frames": skipped,
        "mIoU": miou,
        "pixel_accuracy": pixel_accuracy,
        "per_class_IoU": iou_per_class,
    }

    if logger:
        logger.info(f"Evaluation: mIoU={miou:.4f}, pixel_acc={pixel_accuracy:.4f}, frames={evaluated}")

    _write(output_dir, result)
    return result


def _write(output_dir, result):
    eval_path = output_dir / "segmentation_evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
