import bisect


def _record_time(rec):
    t = rec.get("video_time_s")
    if t is None:
        t = rec.get("timestamp")
    return t


def associate_frames_with_telemetry(frames, records, max_time_diff_s=None, frame_index_tolerance=1):
    timed = [(i, _record_time(r)) for i, r in enumerate(records) if _record_time(r) is not None]
    timed.sort(key=lambda item: item[1])
    times = [t for _, t in timed]

    associations = []
    used_counts = {}

    for frame in frames:
        ft = frame["video_time_s"]
        fidx = frame.get("decoded_frame_index")
        result = {
            "frame_id": frame.get("frame_id"),
            "decoded_frame_index": fidx,
            "video_time_s": ft,
            "telemetry_record_index": None,
            "telemetry_row_number": None,
            "telemetry_frame_index": None,
            "telemetry_video_time_s": None,
            "telemetry_match_time_diff_s": None,
            "frame_index_delta": None,
            "frame_index_consistent": None,
            "latitude": None,
            "longitude": None,
            "altitude_m": None,
            "telemetry_valid": False,
            "telemetry_invalid_reasons": [],
            "record": None,
        }

        if not times:
            result["telemetry_invalid_reasons"].append("no_telemetry_records")
            associations.append(result)
            continue

        pos = bisect.bisect_left(times, ft)
        candidates = [p for p in (pos - 1, pos) if 0 <= p < len(times)]
        best = min(candidates, key=lambda p: abs(times[p] - ft))
        rec_idx = timed[best][0]
        rec = records[rec_idx]
        diff = abs(times[best] - ft)
        result["telemetry_match_time_diff_s"] = round(diff, 6)

        if max_time_diff_s is not None and diff > max_time_diff_s:
            result["telemetry_invalid_reasons"].append("time_diff_exceeds_tolerance")
            associations.append(result)
            continue

        used_counts[rec_idx] = used_counts.get(rec_idx, 0) + 1
        result["telemetry_record_index"] = rec_idx
        result["telemetry_row_number"] = rec.get("row_number", rec_idx)
        result["telemetry_frame_index"] = rec.get("frame_index")
        result["telemetry_video_time_s"] = times[best]
        result["record"] = rec
        result["latitude"] = rec.get("latitude")
        result["longitude"] = rec.get("longitude")
        result["altitude_m"] = rec.get("altitude")

        if rec.get("frame_index") is not None and fidx is not None:
            delta = fidx - rec["frame_index"]
            result["frame_index_delta"] = delta
            result["frame_index_consistent"] = abs(delta) <= frame_index_tolerance

        gps_ok = rec.get("gps_valid")
        if gps_ok is None:
            gps_ok = rec.get("latitude") is not None and rec.get("longitude") is not None
            reasons = [] if gps_ok else ["missing_gps"]
        else:
            reasons = list(rec.get("gps_invalid_reasons", []))
        result["telemetry_valid"] = bool(gps_ok)
        result["telemetry_invalid_reasons"].extend(reasons)
        associations.append(result)

    for a in associations:
        idx = a["telemetry_record_index"]
        a["telemetry_row_shared_with_other_frame"] = bool(idx is not None and used_counts.get(idx, 0) > 1)

    return associations, used_counts


def classify_unmatched_telemetry(records, used_counts, decoded_range, eps=1e-6):
    first, last = decoded_range if decoded_range else (None, None)
    unmatched = []
    for i, r in enumerate(records):
        if i in used_counts:
            continue
        t = _record_time(r)
        if t is None:
            reason = "no_video_time"
        elif last is not None and t > last + eps:
            reason = "beyond_decoded_video_range"
        elif first is not None and t < first - eps:
            reason = "before_decoded_video_range"
        else:
            reason = "not_nearest_to_any_associated_frame"
        unmatched.append({
            "row_number": r.get("row_number", i),
            "frame_index": r.get("frame_index"),
            "video_time_s": t,
            "reason": reason,
        })
    return unmatched
