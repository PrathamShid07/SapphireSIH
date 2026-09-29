import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from .uavid_classes import class_index_to_rgb


def run_inference(model, dataset, output_dir, batch_size=1, logger=None):
    output_dir = Path(output_dir)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    manifest = []

    sample_idx = 0
    with torch.no_grad():
        for imgs, labels, metas in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out["out"].argmax(dim=1).cpu().numpy()

            for i in range(len(preds)):
                pred = preds[i]
                img_path = Path(metas["image_path"][i])
                stem = img_path.stem
                seq_name = img_path.parent.parent.name

                colored_mask = class_index_to_rgb(pred)
                mask_img = Image.fromarray(colored_mask.astype(np.uint8))

                orig_img = Image.open(img_path).convert("RGB")
                orig_resized = orig_img.resize(
                    (colored_mask.shape[1], colored_mask.shape[0]), Image.BILINEAR
                )
                orig_arr = np.array(orig_resized, dtype=np.float32)
                overlay_arr = (0.6 * orig_arr + 0.4 * colored_mask.astype(np.float32)).clip(0, 255).astype(np.uint8)
                overlay_img = Image.fromarray(overlay_arr)

                mask_fname = f"{seq_name}_{stem}_mask.png"
                overlay_fname = f"{seq_name}_{stem}_overlay.png"
                mask_img.save(str(mask_dir / mask_fname))
                overlay_img.save(str(overlay_dir / overlay_fname))

                manifest.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_dir / mask_fname),
                    "overlay_path": str(overlay_dir / overlay_fname),
                })
                sample_idx += 1

    manifest_path = output_dir / "inference_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if logger:
        logger.info(f"Inference complete: {sample_idx} frames processed. Manifest: {manifest_path}")

    return manifest
