import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .uavid_classes import rgb_label_to_class_index


class UAVidDataset(Dataset):
    def __init__(self, split_root, input_size=(512, 512), augment=False, max_samples=None):
        self.input_size = input_size
        self.augment = augment
        self.samples = []

        split_root = Path(split_root)
        for seq_dir in sorted(split_root.glob("seq*")):
            img_dir = seq_dir / "Images"
            lbl_dir = seq_dir / "Labels"
            if not img_dir.exists() or not lbl_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*.png")):
                lbl_path = lbl_dir / img_path.name
                if lbl_path.exists():
                    self.samples.append((img_path, lbl_path))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        label_rgb = np.array(Image.open(lbl_path).convert("RGB"))

        image = image.resize(self.input_size, Image.BILINEAR)
        label_rgb_resized = np.array(
            Image.fromarray(label_rgb).resize(self.input_size, Image.NEAREST)
        )

        if self.augment and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label_rgb_resized = label_rgb_resized[:, ::-1, :].copy()

        img_arr = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_arr = (img_arr - mean) / std

        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1))
        label_tensor = torch.from_numpy(rgb_label_to_class_index(label_rgb_resized)).long()

        return img_tensor, label_tensor, {"image_path": str(img_path), "label_path": str(lbl_path)}
