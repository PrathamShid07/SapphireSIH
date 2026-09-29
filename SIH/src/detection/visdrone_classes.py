import colorsys

VISDRONE_CLASSES = {
    0: "ignored regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

VISDRONE_CLASS_TO_ID = {v: k for k, v in VISDRONE_CLASSES.items()}

VEHICLE_CLASS_IDS = {4, 5, 6, 9}
VEHICLE_CLASS_NAMES = {"car", "van", "truck", "bus"}

COCO_TO_VISDRONE = {
    1: (1, "pedestrian"),
    2: (3, "bicycle"),
    3: (4, "car"),
    4: (10, "motor"),
    6: (9, "bus"),
    8: (6, "truck"),
}

CLASS_COLORS = {
    "car": (0, 255, 0),
    "van": (255, 165, 0),
    "truck": (255, 0, 0),
    "bus": (0, 165, 255),
    "pedestrian": (255, 255, 0),
    "people": (255, 255, 0),
    "bicycle": (0, 255, 255),
    "motor": (255, 0, 255),
    "tricycle": (128, 0, 255),
    "awning-tricycle": (255, 128, 0),
    "ignored regions": (128, 128, 128),
    "others": (200, 200, 200),
}


def get_track_color(track_id: int):
    golden_ratio = 0.618033988749895
    hue = (track_id * golden_ratio) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)
