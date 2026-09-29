import csv
import json
import math
from datetime import datetime
from pathlib import Path

FIELD_ALIASES = {
    "frame_index": ("frame_index", "frame", "frame_number"),
    "video_time_s": ("video_time_s", "video_time_sec", "time_sec", "video_time"),
    "latitude": ("latitude", "drone_lat", "lat"),
    "longitude": ("longitude", "drone_lon", "lon", "lng"),
    "altitude": ("altitude", "drone_altitude_m", "altitude_m", "alt"),
    "yaw": ("yaw",),
    "pitch": ("pitch",),
    "roll": ("roll",),
}


def _lookup(row, field):
    for name in FIELD_ALIASES[field]:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _to_float(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_int(value):
    f = _to_float(value)
    if f is None or f != int(f):
        return None
    return int(f)


def _parse_wall_clock(value):
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value).strip()).timestamp()
    except ValueError:
        return None


def gps_validity(lat, lon, alt):
    reasons = []
    if lat is None or not (-90.0 <= lat <= 90.0):
        reasons.append("invalid_latitude")
    if lon is None or not (-180.0 <= lon <= 180.0):
        reasons.append("invalid_longitude")
    if alt is None:
        reasons.append("invalid_altitude")
    return (not reasons), reasons


class TelemetryParser:
    def __init__(self, video_path=None, input_dir=None, telemetry_path=None):
        self.telemetry_path = Path(telemetry_path).resolve() if telemetry_path else None
        self.video_path = Path(video_path).resolve() if video_path else None
        if input_dir:
            self.input_dir = Path(input_dir).resolve()
        elif self.video_path:
            self.input_dir = self.video_path.parent
        else:
            self.input_dir = None

    def find_sidecar_telemetry(self):
        if not self.input_dir or not self.input_dir.exists():
            return None
        
        candidates = [
            "gps.csv", "telemetry.csv", "flight_log.csv", "metadata.csv",
            "gps.json", "telemetry.json", "flight_log.json", "metadata.json",
            f"{self.video_path.stem}.csv" if self.video_path else None,
            f"{self.video_path.stem}.json" if self.video_path else None,
            f"{self.video_path.stem}.srt" if self.video_path else None,
        ]
        
        if self.video_path:
            candidates = candidates[-3:] + candidates[:-3]

        for cand in candidates:
            if cand:
                p = self.input_dir / cand
                if p.exists():
                    return p
        return None

    def parse_csv_telemetry(self, csv_path):
        records = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_number, row in enumerate(reader):
                video_time = _to_float(_lookup(row, "video_time_s"))
                raw_ts = row.get("timestamp")
                ts_float = _to_float(raw_ts)
                wall_clock = None if ts_float is not None else _parse_wall_clock(raw_ts)
                lat = _to_float(_lookup(row, "latitude"))
                lon = _to_float(_lookup(row, "longitude"))
                alt = _to_float(_lookup(row, "altitude"))
                gps_ok, gps_reasons = gps_validity(lat, lon, alt)
                rec = {
                    "row_number": row_number,
                    "frame_index": _to_int(_lookup(row, "frame_index")),
                    "video_time_s": video_time,
                    "timestamp": video_time if video_time is not None else (ts_float if ts_float is not None else 0.0),
                    "timestamp_raw": raw_ts,
                    "wall_clock_epoch_s": wall_clock,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    "gps_valid": gps_ok,
                    "gps_invalid_reasons": gps_reasons,
                    "yaw": _to_float(_lookup(row, "yaw")),
                    "pitch": _to_float(_lookup(row, "pitch")),
                    "roll": _to_float(_lookup(row, "roll")),
                }
                records.append(rec)
        return records

    def parse_json_telemetry(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "telemetry" in data:
            return data["telemetry"]
        return []

    def extract(self):
        sources_checked = ["video_container_telemetry", "sidecar_metadata_files"]
        if self.telemetry_path is not None and not self.telemetry_path.exists():
            return {
                "telemetry_detected": False,
                "gps_available": False,
                "imu_available": False,
                "sources_checked": [str(self.telemetry_path)],
                "source_file": None,
                "reason": f"Configured telemetry file does not exist: {self.telemetry_path}",
                "records": [],
            }
        sidecar = self.telemetry_path or self.find_sidecar_telemetry()
        parse_error = None
        
        if sidecar:
            sources_checked.append(str(sidecar))
            ext = sidecar.suffix.lower()
            try:
                if ext == ".csv":
                    records = self.parse_csv_telemetry(sidecar)
                elif ext == ".json":
                    records = self.parse_json_telemetry(sidecar)
                else:
                    records = []
                    
                if records:
                    return {
                        "telemetry_detected": True,
                        "gps_available": any(r.get("latitude") is not None for r in records),
                        "imu_available": any(r.get("yaw") is not None for r in records),
                        "sources_checked": sources_checked,
                        "source_file": str(sidecar),
                        "records": records,
                    }
            except Exception as e:
                parse_error = f"{type(e).__name__}: {e}"

        result = {
            "telemetry_detected": False,
            "gps_available": False,
            "imu_available": False,
            "sources_checked": sources_checked,
            "source_file": None,
            "reason": "No valid GPS/IMU telemetry stream or sidecar file detected",
            "records": [],
        }
        if parse_error:
            result["reason"] = f"Telemetry sidecar {sidecar} could not be parsed: {parse_error}"
            result["parse_error"] = parse_error
        return result


def validate_telemetry_records(records):
    n = len(records)
    frame_indices = [r.get("frame_index") for r in records if r.get("frame_index") is not None]
    times = [r.get("video_time_s") for r in records if r.get("video_time_s") is not None]
    wall = [r.get("wall_clock_epoch_s") for r in records if r.get("wall_clock_epoch_s") is not None]

    seen = set()
    duplicates = []
    for i in frame_indices:
        if i in seen and i not in duplicates:
            duplicates.append(i)
        seen.add(i)
    missing = []
    if frame_indices:
        missing = sorted(set(range(min(frame_indices), max(frame_indices) + 1)) - seen)

    def strictly_increasing(vals):
        return all(b > a for a, b in zip(vals, vals[1:]))

    invalid_gps_rows = [r["row_number"] for r in records if r.get("gps_valid") is False]

    return {
        "row_count": n,
        "frame_index_min": min(frame_indices) if frame_indices else None,
        "frame_index_max": max(frame_indices) if frame_indices else None,
        "duplicate_frame_indices": duplicates,
        "missing_frame_indices": missing,
        "video_time_s_min": min(times) if times else None,
        "video_time_s_max": max(times) if times else None,
        "video_time_s_present_for_all_rows": len(times) == n,
        "video_time_s_strictly_increasing": strictly_increasing(times) if times else None,
        "wall_clock_strictly_increasing": strictly_increasing(wall) if wall else None,
        "gps_valid_row_count": n - len(invalid_gps_rows),
        "gps_invalid_row_numbers": invalid_gps_rows,
    }
