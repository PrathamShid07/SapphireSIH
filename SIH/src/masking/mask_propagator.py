import copy
import numpy as np


def interpolate_box(box_start, box_end, alpha):
    return [
        float((1.0 - alpha) * box_start[0] + alpha * box_end[0]),
        float((1.0 - alpha) * box_start[1] + alpha * box_end[1]),
        float((1.0 - alpha) * box_start[2] + alpha * box_end[2]),
        float((1.0 - alpha) * box_start[3] + alpha * box_end[3]),
    ]


class MaskPropagator:
    def __init__(self, max_gap=3, motion_expansion=True, logger=None):
        self.max_gap = max_gap
        self.motion_expansion = motion_expansion
        self.logger = logger

    def propagate_tracks(self, tracking_metadata, tracks_summary=None):
        updated_meta = copy.deepcopy(tracking_metadata)
        track_appearances = {}

        for f_idx, frame in enumerate(updated_meta):
            for obj in frame.get("objects", []):
                tid = obj["track_id"]
                if tid not in track_appearances:
                    track_appearances[tid] = []
                track_appearances[tid].append((f_idx, obj))

        interpolated_count = 0
        for tid, appearances in track_appearances.items():
            appearances.sort(key=lambda x: x[0])
            for i in range(len(appearances) - 1):
                f1, obj1 = appearances[i]
                f2, obj2 = appearances[i + 1]
                gap = f2 - f1 - 1
                if 0 < gap <= self.max_gap:
                    for f in range(f1 + 1, f2):
                        alpha = float(f - f1) / float(f2 - f1)
                        interp_box = interpolate_box(
                            obj1["bbox"], obj2["bbox"], alpha
                        )
                        interp_vx = (1.0 - alpha) * obj1.get("velocity", [0, 0])[
                            0
                        ] + alpha * obj2.get("velocity", [0, 0])[0]
                        interp_vy = (1.0 - alpha) * obj1.get("velocity", [0, 0])[
                            1
                        ] + alpha * obj2.get("velocity", [0, 0])[1]

                        interp_obj = {
                            "track_id": tid,
                            "class_name": obj1.get("class_name", "car"),
                            "confidence": float(
                                (1.0 - alpha) * obj1.get("confidence", 0.5)
                                + alpha * obj2.get("confidence", 0.5)
                            ),
                            "bbox": interp_box,
                            "velocity": [float(interp_vx), float(interp_vy)],
                            "speed": float(np.sqrt(interp_vx**2 + interp_vy**2)),
                            "is_moving": obj1.get("is_moving", False)
                            or obj2.get("is_moving", False),
                            "interpolated": True,
                        }
                        updated_meta[f]["objects"].append(interp_obj)
                        updated_meta[f]["active_objects_count"] = len(
                            updated_meta[f]["objects"]
                        )
                        interpolated_count += 1

        if self.motion_expansion:
            for frame in updated_meta:
                for obj in frame.get("objects", []):
                    vx, vy = obj.get("velocity", [0.0, 0.0])
                    if abs(vx) > 0.5 or abs(vy) > 0.5:
                        b = obj["bbox"]
                        obj["bbox"] = [
                            min(b[0], b[0] + vx),
                            min(b[1], b[1] + vy),
                            max(b[2], b[2] + vx),
                            max(b[3], b[3] + vy),
                        ]

        if self.logger:
            self.logger.info(
                f"Mask propagation complete. Interpolated {interpolated_count} missing object instances across gaps."
            )

        return updated_meta
