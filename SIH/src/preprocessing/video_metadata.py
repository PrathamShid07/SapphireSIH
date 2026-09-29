from pathlib import Path
import cv2
import numpy as np

def probe_decoded_frames(video_path):
    vpath = Path(video_path).resolve()
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {vpath}")

    timestamps = []
    while True:
        ok = cap.grab()
        if not ok:
            break
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    cap.release()

    count = len(timestamps)
    ts = np.asarray(timestamps, dtype=np.float64)
    diffs = np.diff(ts) if count > 1 else np.asarray([])
    monotonic = bool(np.all(diffs > 0)) if diffs.size else True
    timestamps_valid = monotonic and (count < 2 or float(ts[-1]) > 0.0)

    if timestamps_valid and diffs.size:
        median_dt = float(np.median(diffs))
        decoded_fps = 1.0 / median_dt if median_dt > 0 else 0.0
    else:
        decoded_fps = 0.0

    return {
        "decoded_frame_count": count,
        "decoded_timestamps_sec": [round(float(t), 6) for t in ts],
        "decoded_fps": round(decoded_fps, 4),
        "decoded_first_timestamp_sec": round(float(ts[0]), 6) if count else None,
        "decoded_last_timestamp_sec": round(float(ts[-1]), 6) if count else None,
        "decoded_duration_sec": round(float(ts[-1] - ts[0]) + (1.0 / decoded_fps if decoded_fps > 0 else 0.0), 6) if count else 0.0,
        "timestamps_monotonic": monotonic,
        "timestamps_valid": bool(timestamps_valid),
    }

def extract_video_metadata(video_path):
    vpath = Path(video_path).resolve()
    if not vpath.exists():
        raise FileNotFoundError(f"Video file does not exist: {vpath}")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {vpath}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    cap.release()

    duration = total_frames / fps if fps > 0 else 0.0
    aspect_ratio = round(width / height, 4) if height > 0 else 0.0

    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    decoded = probe_decoded_frames(vpath)
    decoded_count = decoded["decoded_frame_count"]

    return {
        "file_name": vpath.name,
        "file_path": str(vpath),
        "file_size_bytes": vpath.stat().st_size,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_sec": round(duration, 3),
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "fourcc": fourcc_str,
        "container_fps": round(fps, 3),
        "container_frame_count": total_frames,
        "container_duration_sec": round(duration, 3),
        "decoded_frame_count": decoded_count,
        "decoded_fps": decoded["decoded_fps"],
        "decoded_first_timestamp_sec": decoded["decoded_first_timestamp_sec"],
        "decoded_last_timestamp_sec": decoded["decoded_last_timestamp_sec"],
        "decoded_duration_sec": decoded["decoded_duration_sec"],
        "decoded_timestamps_sec": decoded["decoded_timestamps_sec"],
        "timestamps_monotonic": decoded["timestamps_monotonic"],
        "timestamps_valid": decoded["timestamps_valid"],
        "container_frame_count_mismatch": total_frames != decoded_count,
        "container_vs_decoded_frame_delta": total_frames - decoded_count,
    }
