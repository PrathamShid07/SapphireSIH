import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def generate_frame_mask(
    image_shape,
    dynamic_bboxes,
    dilation_px=15,
    semantic_mask=None,
    dynamic_semantic_classes=None,
):
    h, w = image_shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)

    for box in dynamic_bboxes:
        x1, y1, x2, y2 = box
        bx1 = max(0, int(round(x1 - dilation_px)))
        by1 = max(0, int(round(y1 - dilation_px)))
        bx2 = min(w, int(round(x2 + dilation_px)))
        by2 = min(h, int(round(y2 + dilation_px)))

        if bx2 > bx1 and by2 > by1:
            mask[by1:by2, bx1:bx2] = 0

    if semantic_mask is not None and dynamic_semantic_classes:
        if semantic_mask.shape[:2] != (h, w):
            sem_pil = Image.fromarray(semantic_mask).resize((w, h), Image.NEAREST)
            semantic_mask = np.array(sem_pil)

        dyn_mask = np.isin(semantic_mask, list(dynamic_semantic_classes))
        if dilation_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
            )
            dyn_mask = cv2.dilate(dyn_mask.astype(np.uint8), kernel).astype(bool)

        mask[dyn_mask] = 0

    return mask


def create_mask_overlay(image_np, mask):
    h, w = image_np.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = image_np.copy()
    masked_pixels = mask == 0

    red_tint = np.zeros_like(image_np)
    red_tint[:, :] = [255, 0, 0]

    overlay[masked_pixels] = (
        0.5 * image_np[masked_pixels] + 0.5 * red_tint[masked_pixels]
    ).astype(np.uint8)

    return overlay


class DynamicMaskGenerator:
    def __init__(self, dilation_px=15, logger=None):
        self.dilation_px = dilation_px
        self.logger = logger

    def generate_masks_from_tracking(
        self,
        tracking_metadata,
        tracks_summary,
        images_dir,
        output_dir,
        semantic_masks_dir=None,
        dynamic_semantic_classes=None,
    ):
        out_dir = Path(output_dir)
        masks_dir = out_dir / "masks"
        overlays_dir = out_dir / "overlays"
        masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        img_dir = Path(images_dir)
        dynamic_track_ids = set()
        for tid_str, t_info in tracks_summary.items():
            if t_info.get("is_dynamic", False):
                try:
                    dynamic_track_ids.add(int(tid_str))
                except ValueError:
                    dynamic_track_ids.add(tid_str)

        manifest = []
        total_dynamic_objects = 0
        total_masked_pixels = 0
        total_image_pixels = 0

        for frame_data in tracking_metadata:
            img_name = frame_data["image_name"]
            img_path = img_dir / img_name
            if not img_path.exists():
                img_path = Path(frame_data.get("image_path", img_path))

            if not img_path.exists():
                continue

            pil_img = Image.open(str(img_path)).convert("RGB")
            img_np = np.array(pil_img)
            h, w = img_np.shape[:2]

            dyn_boxes = []
            for obj in frame_data.get("objects", []):
                tid = obj.get("track_id")
                is_moving = obj.get("is_moving", False)
                if is_moving or (tid in dynamic_track_ids):
                    dyn_boxes.append(obj["bbox"])
                    total_dynamic_objects += 1

            sem_mask = None
            if semantic_masks_dir:
                sem_p = Path(semantic_masks_dir) / f"{Path(img_name).stem}_mask.png"
                if sem_p.exists():
                    sem_mask = np.array(Image.open(str(sem_p)))

            mask = generate_frame_mask(
                image_shape=(h, w),
                dynamic_bboxes=dyn_boxes,
                dilation_px=self.dilation_px,
                semantic_mask=sem_mask,
                dynamic_semantic_classes=dynamic_semantic_classes,
            )

            mask_img = Image.fromarray(mask)
            mask_fname_std = f"{img_name}.png"
            mask_path_std = masks_dir / mask_fname_std
            mask_img.save(str(mask_path_std))

            mask_fname_colmap = f"{img_name}.png"
            if not img_name.endswith(".png"):
                mask_path_extra = masks_dir / f"{Path(img_name).stem}.png"
                mask_img.save(str(mask_path_extra))

            overlay_bgr = create_mask_overlay(
                cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), mask
            )
            overlay_path = overlays_dir / f"{Path(img_name).stem}_overlay.jpg"
            cv2.imwrite(str(overlay_path), overlay_bgr)

            masked_cnt = int(np.sum(mask == 0))
            frame_pixels = h * w
            total_masked_pixels += masked_cnt
            total_image_pixels += frame_pixels

            manifest.append(
                {
                    "image_name": img_name,
                    "mask_path": str(mask_path_std),
                    "overlay_path": str(overlay_path),
                    "dynamic_objects_count": len(dyn_boxes),
                    "masked_pixel_percentage": float(
                        100.0 * masked_cnt / frame_pixels
                    )
                    if frame_pixels > 0
                    else 0.0,
                }
            )

        manifest_path = out_dir / "mask_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        avg_masked_pct = (
            float(100.0 * total_masked_pixels / total_image_pixels)
            if total_image_pixels > 0
            else 0.0
        )

        if self.logger:
            self.logger.info(
                f"Generated {len(manifest)} masks. Dynamic objects masked: {total_dynamic_objects}. "
                f"Average masked area: {avg_masked_pct:.2f}%"
            )

        return {
            "masks_dir": str(masks_dir),
            "overlays_dir": str(overlays_dir),
            "manifest_path": str(manifest_path),
            "total_masks": len(manifest),
            "total_dynamic_objects": total_dynamic_objects,
            "average_masked_pixel_percentage": avg_masked_pct,
            "manifest": manifest,
        }
