import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .sort_tracker import SortTracker
from .visdrone_classes import get_track_color, CLASS_COLORS


def draw_tracking_visualization(image_np, active_tracks):
    vis_img = image_np.copy()

    for trk in active_tracks:
        tid = trk["track_id"]
        box = trk["bbox"]
        cname = trk.get("class_name", "vehicle")
        score = trk.get("score", 1.0)
        history = trk.get("history", [])

        color = get_track_color(tid)

        x1, y1, x2, y2 = [int(round(b)) for b in box]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        label = f"ID #{tid} {cname} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, scale, thickness)

        bg_y1 = max(0, y1 - label_h - baseline - 4)
        bg_y2 = y1
        cv2.rectangle(vis_img, (x1, bg_y1), (x1 + label_w + 4, bg_y2), color, -1)
        cv2.putText(
            vis_img,
            label,
            (x1 + 2, bg_y2 - baseline - 2),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

        if len(history) > 1:
            pts = [np.array(pt, dtype=np.int32) for pt in history]
            for p_idx in range(1, len(pts)):
                cv2.line(vis_img, tuple(pts[p_idx - 1]), tuple(pts[p_idx]), color, 2)

    return vis_img


class FrameProcessor:
    def __init__(
        self,
        detector,
        tracker=None,
        conf_threshold=0.5,
        target_classes=None,
        motion_threshold=2.0,
    ):
        self.detector = detector
        self.tracker = tracker if tracker is not None else SortTracker()
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes
        self.motion_threshold = motion_threshold

    def process_frame_sequence(self, image_paths, output_dir, save_visualizations=True):
        out_dir = Path(output_dir)
        vis_dir = out_dir / "visualizations"
        if save_visualizations:
            vis_dir.mkdir(parents=True, exist_ok=True)

        frame_metadata = []
        tracks_lifecycle = {}

        for frame_idx, img_path in enumerate(image_paths):
            p = Path(img_path)
            pil_img = Image.open(str(p)).convert("RGB")
            img_np = np.array(pil_img)
            h, w = img_np.shape[:2]

            dets = self.detector.predict(
                [pil_img],
                conf_threshold=self.conf_threshold,
                target_classes=self.target_classes,
            )[0]

            active_tracks = self.tracker.update(dets)

            frame_objs = []
            for trk in active_tracks:
                tid = trk["track_id"]
                cname = trk.get("class_name", "vehicle")
                score = trk.get("score", 1.0)
                box = trk["bbox"]
                vx, vy = trk.get("velocity", [0.0, 0.0])
                speed = float(np.sqrt(vx**2 + vy**2))
                is_moving = speed >= self.motion_threshold

                frame_objs.append(
                    {
                        "track_id": tid,
                        "class_name": cname,
                        "confidence": score,
                        "bbox": box,
                        "velocity": [vx, vy],
                        "speed": speed,
                        "is_moving": is_moving,
                    }
                )

                if tid not in tracks_lifecycle:
                    tracks_lifecycle[tid] = {
                        "track_id": tid,
                        "class_name": cname,
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "total_hits": 1,
                        "confidences": [score],
                        "trajectory": [trk["bbox"]],
                        "speeds": [speed],
                    }
                else:
                    t_info = tracks_lifecycle[tid]
                    t_info["last_frame"] = frame_idx
                    t_info["total_hits"] += 1
                    t_info["confidences"].append(score)
                    t_info["trajectory"].append(trk["bbox"])
                    t_info["speeds"].append(speed)

            frame_entry = {
                "frame_index": frame_idx,
                "image_name": p.name,
                "image_path": str(p),
                "image_size": [w, h],
                "active_objects_count": len(frame_objs),
                "objects": frame_objs,
            }

            if save_visualizations:
                vis_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                annotated_bgr = draw_tracking_visualization(vis_bgr, active_tracks)
                vis_path = vis_dir / f"track_{p.stem}.jpg"
                cv2.imwrite(str(vis_path), annotated_bgr)
                frame_entry["visualization_path"] = str(vis_path)

            frame_metadata.append(frame_entry)

        tracks_summary = {}
        for tid, t_info in tracks_lifecycle.items():
            speeds = t_info["speeds"]
            max_speed = float(np.max(speeds)) if speeds else 0.0
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            is_dynamic = max_speed >= self.motion_threshold

            traj = t_info["trajectory"]
            if len(traj) >= 2:
                start_c = (
                    (traj[0][0] + traj[0][2]) / 2.0,
                    (traj[0][1] + traj[0][3]) / 2.0,
                )
                end_c = (
                    (traj[-1][0] + traj[-1][2]) / 2.0,
                    (traj[-1][1] + traj[-1][3]) / 2.0,
                )
                total_displacement = float(
                    np.sqrt((end_c[0] - start_c[0]) ** 2 + (end_c[1] - start_c[1]) ** 2)
                )
            else:
                total_displacement = 0.0

            tracks_summary[str(tid)] = {
                "track_id": tid,
                "class_name": t_info["class_name"],
                "first_frame": t_info["first_frame"],
                "last_frame": t_info["last_frame"],
                "total_detections": t_info["total_hits"],
                "avg_confidence": float(np.mean(t_info["confidences"])),
                "avg_speed": avg_speed,
                "max_speed": max_speed,
                "total_displacement": total_displacement,
                "is_dynamic": is_dynamic or (total_displacement > 10.0),
            }

        meta_path = out_dir / "tracking_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(frame_metadata, f, indent=2)

        summary_path = out_dir / "tracks_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(tracks_summary, f, indent=2)

        return {
            "tracking_metadata_path": str(meta_path),
            "tracks_summary_path": str(summary_path),
            "total_frames_processed": len(image_paths),
            "unique_tracks_count": len(tracks_summary),
            "frame_metadata": frame_metadata,
            "tracks_summary": tracks_summary,
        }
