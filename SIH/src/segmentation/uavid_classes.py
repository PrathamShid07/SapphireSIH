import numpy as np

CLASSES = [
    "Background clutter",
    "Building",
    "Road",
    "Static car",
    "Tree",
    "Low vegetation",
    "Human",
    "Moving car",
]

NUM_CLASSES = len(CLASSES)

RGB_TO_CLASS = {
    (0, 0, 0): 0,
    (128, 0, 0): 1,
    (128, 64, 128): 2,
    (192, 0, 192): 3,
    (0, 128, 0): 4,
    (128, 128, 0): 5,
    (64, 64, 0): 6,
    (64, 0, 128): 7,
}

CLASS_TO_RGB = {v: k for k, v in RGB_TO_CLASS.items()}

PALETTE = np.array([CLASS_TO_RGB[i] for i in range(NUM_CLASSES)], dtype=np.uint8)

_rgb_array = np.zeros((256, 256, 256), dtype=np.uint8)
for (r, g, b), idx in RGB_TO_CLASS.items():
    _rgb_array[r, g, b] = idx


def rgb_label_to_class_index(rgb_array: np.ndarray) -> np.ndarray:
    r = rgb_array[:, :, 0].astype(np.int32)
    g = rgb_array[:, :, 1].astype(np.int32)
    b = rgb_array[:, :, 2].astype(np.int32)
    return _rgb_array[r, g, b]


def class_index_to_rgb(class_map: np.ndarray) -> np.ndarray:
    return PALETTE[class_map.clip(0, NUM_CLASSES - 1)]
