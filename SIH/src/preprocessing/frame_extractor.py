import json
from pathlib import Path
import cv2
import numpy as np
from .video_metadata import extract_video_metadata
from .quality_analyzer import QualityAnalyzer

class VideoPreprocessor:
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        if config_manager:
            self.target_fps = config_manager.get("preprocessing.target_fps", 2.0)
            self.thresholds = config_manager.get("preprocessing.quality_thresholds")
            self.contact_cfg = config_manager.get("preprocessing.contact_sheet", {})
            self.resume = config_manager.get("execution.resume", True)
            recon_cfg = config_manager.get("preprocessing.reconstruction", {}) or {}
        else:
            self.target_fps = 2.0
            self.thresholds = None
            self.contact_cfg = {}
            self.resume = True
            recon_cfg = {}

        self.recon_strategy = recon_cfg.get("strategy", "selected_keyframes")
        self.recon_stride = int(recon_cfg.get("stride", 1))
        if self.recon_strategy not in ("selected_keyframes", "decoded_stride"):
            raise ValueError(
                f"Unknown preprocessing.reconstruction.strategy '{self.recon_strategy}'. "
                "Expected 'selected_keyframes' or 'decoded_stride'."
            )
        if self.recon_stride < 1:
            raise ValueError("preprocessing.reconstruction.stride must be >= 1")

        self.analyzer = QualityAnalyzer(self.thresholds)

    def process_video(self, video_path, output_dir=None):
        vpath = Path(video_path).resolve()
        if output_dir is None:
            if self.config_manager:
                base_out = self.config_manager.get_path("paths.outputs_dir")
            else:
                base_out = Path("outputs").resolve()
            output_dir = base_out / vpath.stem
        else:
            output_dir = Path(output_dir).resolve()

        extracted_dir = output_dir / "interim" / "extracted_frames"
        selected_dir = output_dir / "processed" / "selected_frames"
        manifest_path = output_dir / "manifest.json"
        recon_dir = output_dir / "processed" / "reconstruction_frames"
        contact_sheet_path = output_dir / "contact_sheet.jpg"

        extracted_dir.mkdir(parents=True, exist_ok=True)
        selected_dir.mkdir(parents=True, exist_ok=True)
        use_decoded_recon = self.recon_strategy == "decoded_stride"
        if use_decoded_recon:
            recon_dir.mkdir(parents=True, exist_ok=True)
        recon_sampling = {"strategy": self.recon_strategy, "stride": self.recon_stride if use_decoded_recon else None}

        if self.resume and manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                if manifest_data.get("status") == "COMPLETED" and manifest_data.get("reconstruction_sampling") == recon_sampling:
                    return manifest_data
            except Exception:
                pass

        metadata = extract_video_metadata(vpath)
        container_fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        decoded_count = metadata["decoded_frame_count"]
        decoded_ts = metadata["decoded_timestamps_sec"]
        timestamps_valid = metadata["timestamps_valid"] and decoded_count > 0
        video_fps = metadata["decoded_fps"] if metadata["decoded_fps"] > 0 else container_fps
        timestamp_source = "decoder_pos_msec" if timestamps_valid else "container_fps_fallback"

        if self.target_fps and self.target_fps > 0 and video_fps > 0:
            step = max(1, int(round(video_fps / self.target_fps)))
        else:
            step = 1

        cap = cv2.VideoCapture(str(vpath))
        
        frames_manifest = []
        selected_count = 0
        rejected_count = 0
        prev_selected_gray = None
        selected_frame_records = []
        reconstruction_frames = []

        frame_idx = 0
        extracted_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if timestamps_valid and frame_idx < decoded_count:
                frame_ts = decoded_ts[frame_idx]
            else:
                frame_ts = frame_idx / container_fps if container_fps > 0 else 0.0

            if use_decoded_recon and frame_idx % self.recon_stride == 0:
                recon_name = f"recon_{frame_idx:05d}.jpg"
                recon_path = recon_dir / recon_name
                cv2.imwrite(str(recon_path), frame)
                reconstruction_frames.append({
                    "frame_id": f"recon_{frame_idx:05d}",
                    "decoded_frame_index": frame_idx,
                    "video_time_s": round(frame_ts, 6),
                    "timestamp_source": timestamp_source,
                    "image_name": recon_name,
                    "image_path": str(recon_path),
                })

            if frame_idx % step == 0:
                extracted_idx += 1
                frame_id = f"frame_{extracted_idx:05d}"
                timestamp_sec = round(frame_ts, 6)
                
                raw_frame_name = f"{frame_id}.jpg"
                raw_frame_path = extracted_dir / raw_frame_name
                cv2.imwrite(str(raw_frame_path), frame)

                is_selected, rejection_reason, scores, gray = self.analyzer.analyze(frame, prev_selected_gray)

                selected_frame_path = None
                if is_selected:
                    selected_count += 1
                    prev_selected_gray = gray
                    sel_frame_name = f"selected_{extracted_idx:05d}.jpg"
                    sel_path = selected_dir / sel_frame_name
                    cv2.imwrite(str(sel_path), frame)
                    selected_frame_path = str(sel_path)
                    selected_frame_records.append((sel_path, frame_id, timestamp_sec, scores))
                else:
                    rejected_count += 1

                record = {
                    "frame_id": frame_id,
                    "original_frame_number": frame_idx,
                    "decoded_frame_index": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "timestamp_source": timestamp_source,
                    "quality_scores": scores,
                    "selected": is_selected,
                    "rejection_reason": rejection_reason,
                    "raw_frame_path": str(raw_frame_path),
                    "selected_frame_path": selected_frame_path,
                }
                frames_manifest.append(record)

            frame_idx += 1

        cap.release()

        if frame_idx != decoded_count:
            raise RuntimeError(
                f"Decoded {frame_idx} frames while extracting but the decoder probe counted "
                f"{decoded_count}; refusing to associate frames with timestamps/telemetry."
            )

        contact_sheet_created = self.generate_contact_sheet(
            selected_frame_records, contact_sheet_path
        )

        manifest_data = {
            "status": "COMPLETED",
            "video_metadata": metadata,
            "reconstruction_sampling": recon_sampling,
            "reconstruction_frames": reconstruction_frames,
            "sampling": {
                "target_fps": self.target_fps,
                "step": step,
                "total_video_frames": decoded_count,
                "container_frame_count": total_frames,
                "decoded_frame_count": decoded_count,
                "timestamp_source": timestamp_source,
                "reconstruction_frames_count": len(reconstruction_frames) if use_decoded_recon else None,
                "extracted_frames_count": len(frames_manifest),
                "selected_frames_count": selected_count,
                "rejected_frames_count": rejected_count,
                "selection_rate": round(selected_count / len(frames_manifest), 4) if frames_manifest else 0.0,
            },
            "output_paths": {
                "output_dir": str(output_dir),
                "extracted_dir": str(extracted_dir),
                "selected_dir": str(selected_dir),
                "reconstruction_dir": str(recon_dir) if use_decoded_recon else None,
                "manifest_path": str(manifest_path),
                "contact_sheet_path": str(contact_sheet_path) if contact_sheet_created else None,
            },
            "frames": frames_manifest,
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return manifest_data

    def generate_contact_sheet(self, selected_records, output_path):
        if not selected_records:
            return False

        cols = self.contact_cfg.get("cols", 5)
        tw = self.contact_cfg.get("thumb_width", 320)
        th = self.contact_cfg.get("thumb_height", 240)

        num_items = len(selected_records)
        rows = int(np.ceil(num_items / cols))

        canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)

        for i, (path, fid, ts, scores) in enumerate(selected_records):
            img = cv2.imread(str(path))
            if img is None:
                continue
            
            thumb = cv2.resize(img, (tw, th))
            
            label = f"{fid} ({ts}s)"
            cv2.putText(thumb, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            r = i // cols
            c = i % cols
            canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = thumb

        cv2.imwrite(str(output_path), canvas)
        return True
