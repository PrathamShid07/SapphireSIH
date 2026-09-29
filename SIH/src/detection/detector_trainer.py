import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .visdrone_classes import (
    VISDRONE_CLASSES,
    COCO_TO_VISDRONE,
    VEHICLE_CLASS_NAMES,
)


class VisDroneDetectionDataset(Dataset):
    def __init__(self, images_dir, annotations_dir=None, target_classes=None, max_samples=None):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir) if annotations_dir else None
        self.target_classes = set(target_classes) if target_classes else None
        self.samples = []

        valid_exts = {".jpg", ".jpeg", ".png"}
        image_files = sorted(
            [f for f in self.images_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        )

        for img_path in image_files:
            anno_path = None
            if self.annotations_dir:
                candidate = self.annotations_dir / f"{img_path.stem}.txt"
                if candidate.exists():
                    anno_path = candidate
            self.samples.append((img_path, anno_path))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def _parse_annotation(self, anno_path, width, height):
        boxes = []
        labels = []
        if not anno_path or not anno_path.exists():
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue

                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_w = float(parts[2])
                bbox_h = float(parts[3])
                category = int(parts[5])

                if category == 0:
                    continue

                class_name = VISDRONE_CLASSES.get(category, "unknown")
                if self.target_classes and class_name not in self.target_classes:
                    continue

                x1 = max(0.0, bbox_left)
                y1 = max(0.0, bbox_top)
                x2 = min(float(width), bbox_left + bbox_w)
                y2 = min(float(height), bbox_top + bbox_h)

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(category)

        if not boxes:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        img_path, anno_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        img_arr = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1))

        boxes, labels = self._parse_annotation(anno_path, w, h)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([h, w]),
        }

        return img_tensor, target, str(img_path)


class VehicleDetector:
    def __init__(self, model_name="fasterrcnn_mobilenet_v3_large_fpn", num_classes=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

        if num_classes is not None:
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, images, conf_threshold=0.5, target_classes=None):
        self.model.eval()
        target_names = set(target_classes) if target_classes else VEHICLE_CLASS_NAMES

        tensors = []
        sizes = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_img = Image.open(str(img)).convert("RGB")
            elif isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] == 3:
                    pil_img = Image.fromarray(img)
                else:
                    pil_img = Image.fromarray(img).convert("RGB")
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            elif isinstance(img, torch.Tensor):
                tensors.append(img.to(self.device))
                sizes.append((img.shape[1], img.shape[2]))
                continue
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            arr = np.array(pil_img, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)).to(self.device))
            sizes.append(pil_img.size)

        batch_detections = []
        with torch.no_grad():
            outputs = self.model(tensors)

        for out in outputs:
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()

            dets = []
            for box, score, label in zip(boxes, scores, labels):
                if score < conf_threshold:
                    continue

                coco_id = int(label)
                if coco_id in COCO_TO_VISDRONE:
                    vd_id, vd_name = COCO_TO_VISDRONE[coco_id]
                else:
                    vd_id = coco_id
                    vd_name = VISDRONE_CLASSES.get(coco_id, f"class_{coco_id}")

                if target_names and vd_name not in target_names:
                    continue

                dets.append(
                    {
                        "bbox": [float(b) for b in box],
                        "score": float(score),
                        "class_id": vd_id,
                        "class_name": vd_name,
                    }
                )
            batch_detections.append(dets)

        return batch_detections

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(p))

    def load(self, path):
        state = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(state)
        return self


def collate_fn(batch):
    return tuple(zip(*batch))


def train_detector(
    detector,
    dataset,
    output_dir,
    epochs=5,
    batch_size=2,
    learning_rate=0.0001,
    resume=True,
    logger=None,
):
    out_p = Path(output_dir)
    ckpt_dir = out_p / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = ckpt_dir / "model_detector_final.pth"

    if resume and final_ckpt.exists():
        if logger:
            logger.info(f"Existing checkpoint found at {final_ckpt}. Skipping training.")
        detector.load(final_ckpt)
        return {"skipped": True, "checkpoint": str(final_ckpt)}

    detector.model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(
        [p for p in detector.model.parameters() if p.requires_grad], lr=learning_rate
    )

    history = []
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for images, targets, _ in loader:
            images = [img.to(detector.device) for img in images]
            targets = [{k: v.to(detector.device) for k, v in t.items()} for t in targets]

            loss_dict = detector.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        history.append({"epoch": epoch, "loss": avg_loss})
        if logger:
            logger.info(f"Epoch {epoch}/{epochs} - Detector Loss: {avg_loss:.4f}")

    detector.save(final_ckpt)
    return {
        "skipped": False,
        "checkpoint": str(final_ckpt),
        "epochs": epochs,
        "history": history,
    }


def evaluate_detector(detector, dataset, iou_threshold=0.5, conf_threshold=0.5, logger=None):
    detector.model.eval()

    total_gt = 0
    total_det = 0
    true_positives = 0

    for idx in range(len(dataset)):
        img_tensor, target, img_path = dataset[idx]
        gt_boxes = target["boxes"].numpy()

        dets = detector.predict([img_tensor], conf_threshold=conf_threshold)[0]

        total_gt += len(gt_boxes)
        total_det += len(dets)

        if len(gt_boxes) == 0 or len(dets) == 0:
            continue

        pred_boxes = np.array([d["bbox"] for d in dets])
        matched_gt = set()

        for pb in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gb in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                xx1 = max(pb[0], gb[0])
                yy1 = max(pb[1], gb[1])
                xx2 = min(pb[2], gb[2])
                yy2 = min(pb[3], gb[3])
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                area_p = (pb[2] - pb[0]) * (pb[3] - pb[1])
                area_g = (gb[2] - gb[0]) * (gb[3] - gb[1])
                union = area_p + area_g - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                true_positives += 1
                matched_gt.add(best_gt_idx)

    precision = float(true_positives / total_det) if total_det > 0 else 0.0
    recall = float(true_positives / total_gt) if total_gt > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    eval_result = {
        "status": "SUCCESS",
        "evaluated_samples": len(dataset),
        "total_gt_boxes": total_gt,
        "total_det_boxes": total_det,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou_threshold": iou_threshold,
    }

    if logger:
        logger.info(
            f"Detector Evaluation: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

    return eval_result
