from pathlib import Path
from .coordinate_system import CoordinateSystem
from .telemetry_association import associate_frames_with_telemetry, classify_unmatched_telemetry


class FrameSynchronizer:
    def __init__(self, camera_intrinsics, max_time_diff_s=None, frame_index_tolerance=1, require_valid_for_reconstruction=False):
        self.camera_intrinsics = camera_intrinsics
        self.max_time_diff_s = max_time_diff_s
        self.frame_index_tolerance = frame_index_tolerance
        self.require_valid_for_reconstruction = require_valid_for_reconstruction

    def _build_coord_sys(self, telemetry_data):
        telemetry_records = telemetry_data.get("records", [])
        if telemetry_data.get("gps_available", False) and telemetry_records:
            ref_rec = next(
                (r for r in telemetry_records if r.get("latitude") is not None and r.get("longitude") is not None and r.get("gps_valid", True)),
                None,
            )
            if ref_rec:
                return CoordinateSystem(
                    ref_lat=ref_rec["latitude"],
                    ref_lon=ref_rec["longitude"],
                    ref_alt=ref_rec.get("altitude", 0.0)
                )
        return CoordinateSystem()

    def _decoded_range(self, stage2_manifest):
        vm = stage2_manifest.get("video_metadata", {})
        first = vm.get("decoded_first_timestamp_sec")
        last = vm.get("decoded_last_timestamp_sec")
        if first is None or last is None:
            return None
        return first, last

    def synchronize(self, stage2_manifest, telemetry_data):
        frames = stage2_manifest.get("frames", [])
        telemetry_records = telemetry_data.get("records", [])
        has_gps = telemetry_data.get("gps_available", False)
        coord_sys = self._build_coord_sys(telemetry_data)

        assoc_input = [
            {
                "frame_id": f["frame_id"],
                "decoded_frame_index": f.get("decoded_frame_index", f.get("original_frame_number")),
                "video_time_s": f["timestamp_sec"],
            }
            for f in frames
        ]
        associations, used_counts = associate_frames_with_telemetry(
            assoc_input, telemetry_records, self.max_time_diff_s, self.frame_index_tolerance
        )

        synced_frames = []
        gps_count = 0
        orientation_count = 0

        for f_entry, assoc in zip(frames, associations):
            ts = f_entry["timestamp_sec"]
            fid = f_entry["frame_id"]
            matched_telem = assoc["record"]

            lat = assoc["latitude"]
            lon = assoc["longitude"]
            alt = assoc["altitude_m"]
            yaw = matched_telem.get("yaw") if matched_telem else None
            pitch = matched_telem.get("pitch") if matched_telem else None
            roll = matched_telem.get("roll") if matched_telem else None

            usable = assoc["telemetry_valid"] and lat is not None and lon is not None
            if usable:
                gps_count += 1
                x, y, z = coord_sys.latlon_to_local_enu(lat, lon, alt if alt is not None else 0.0)
            else:
                lat = lon = alt = None
                x, y, z = None, None, None

            if usable and yaw is not None:
                orientation_count += 1

            synced_record = {
                "frame_id": fid,
                "original_frame_number": f_entry["original_frame_number"],
                "decoded_frame_index": assoc["decoded_frame_index"],
                "timestamp_sec": ts,
                "selected": f_entry["selected"],
                "rejection_reason": f_entry.get("rejection_reason"),
                "quality_scores": f_entry.get("quality_scores"),
                "camera_intrinsics": self.camera_intrinsics.to_dict(),
                "telemetry": {
                    "available": usable,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    "yaw": yaw if usable else None,
                    "pitch": pitch if usable else None,
                    "roll": roll if usable else None,
                    "local_cartesian_xyz": {"x": x, "y": y, "z": z} if x is not None else None,
                    "match": {
                        "telemetry_row_number": assoc["telemetry_row_number"],
                        "telemetry_frame_index": assoc["telemetry_frame_index"],
                        "telemetry_video_time_s": assoc["telemetry_video_time_s"],
                        "time_diff_s": assoc["telemetry_match_time_diff_s"],
                        "frame_index_delta": assoc["frame_index_delta"],
                        "frame_index_consistent": assoc["frame_index_consistent"],
                        "invalid_reasons": assoc["telemetry_invalid_reasons"],
                    },
                },
                "paths": {
                    "raw_frame_path": f_entry.get("raw_frame_path"),
                    "selected_frame_path": f_entry.get("selected_frame_path"),
                }
            }
            synced_frames.append(synced_record)

        decoded_range = self._decoded_range(stage2_manifest)
        unmatched_rows = classify_unmatched_telemetry(
            telemetry_records, used_counts, decoded_range
        )
        beyond = [u for u in unmatched_rows if u["reason"] == "beyond_decoded_video_range"]
        video_meta = stage2_manifest.get("video_metadata", {})

        flight_summary = {
            "total_frames_evaluated": len(frames),
            "total_selected_frames": stage2_manifest.get("sampling", {}).get("selected_frames_count", 0),
            "gps_available": has_gps,
            "imu_available": telemetry_data.get("imu_available", False),
            "frames_with_gps": gps_count,
            "frames_with_orientation": orientation_count,
            "timestamp_coverage": {
                "start_time_sec": frames[0]["timestamp_sec"] if frames else 0.0,
                "end_time_sec": frames[-1]["timestamp_sec"] if frames else 0.0,
                "duration_sec": round(frames[-1]["timestamp_sec"] - frames[0]["timestamp_sec"], 3) if len(frames) > 1 else 0.0,
            },
            "coordinate_system": coord_sys.to_dict(),
            "synchronization_status": "SYNCHRONIZED_WITHOUT_GPS" if not has_gps else "SYNCHRONIZED_WITH_GPS",
            "telemetry_association": {
                "method": "nearest_video_time_s",
                "max_time_diff_s": self.max_time_diff_s,
                "frame_index_tolerance": self.frame_index_tolerance,
                "telemetry_row_count": len(telemetry_records),
                "decoded_frame_count": video_meta.get("decoded_frame_count"),
                "container_frame_count": video_meta.get("container_frame_count"),
                "decoded_time_range_s": list(decoded_range) if decoded_range else None,
                "telemetry_rows_beyond_decoded_video": len(beyond),
                "telemetry_rows_beyond_decoded_video_row_numbers": [u["row_number"] for u in beyond],
                "telemetry_rows_unassociated_total_for_evaluated_frames": len(unmatched_rows),
            },
        }

        return synced_frames, flight_summary

    def build_reconstruction_manifest(self, stage2_manifest, telemetry_data):
        recon_frames = stage2_manifest.get("reconstruction_frames", [])
        telemetry_records = telemetry_data.get("records", [])
        coord_sys = self._build_coord_sys(telemetry_data)
        associations, used_counts = associate_frames_with_telemetry(
            recon_frames, telemetry_records, self.max_time_diff_s, self.frame_index_tolerance
        )

        rows = []
        for frame, assoc in zip(recon_frames, associations):
            image_exists = Path(frame["image_path"]).exists()
            usable = assoc["telemetry_valid"]
            if usable:
                e, n, u = coord_sys.latlon_to_local_enu(assoc["latitude"], assoc["longitude"], assoc["altitude_m"])
            else:
                e = n = u = None
            eligible = image_exists and (usable or not self.require_valid_for_reconstruction)
            rows.append({
                "frame_index": assoc["decoded_frame_index"],
                "video_time_s": assoc["video_time_s"],
                "image_name": frame["image_name"],
                "image_path": frame["image_path"],
                "latitude": assoc["latitude"] if usable else None,
                "longitude": assoc["longitude"] if usable else None,
                "altitude_m": assoc["altitude_m"] if usable else None,
                "local_enu_east_m": e,
                "local_enu_north_m": n,
                "local_enu_up_m": u,
                "telemetry_row_number": assoc["telemetry_row_number"],
                "telemetry_frame_index": assoc["telemetry_frame_index"],
                "telemetry_video_time_s": assoc["telemetry_video_time_s"],
                "telemetry_match_time_diff_s": assoc["telemetry_match_time_diff_s"],
                "frame_index_delta": assoc["frame_index_delta"],
                "frame_index_consistent": assoc["frame_index_consistent"],
                "telemetry_row_shared_with_other_frame": assoc["telemetry_row_shared_with_other_frame"],
                "telemetry_valid": usable,
                "telemetry_invalid_reasons": assoc["telemetry_invalid_reasons"],
                "timestamp_source": frame.get("timestamp_source"),
                "image_exists": image_exists,
                "reconstruction_eligible": bool(eligible),
            })

        decoded_range = self._decoded_range(stage2_manifest)
        unmatched_rows = classify_unmatched_telemetry(
            telemetry_records, used_counts, decoded_range
        )
        video_meta = stage2_manifest.get("video_metadata", {})
        inconsistent = [r for r in rows if r["frame_index_consistent"] is False]
        deltas = [abs(r["frame_index_delta"]) for r in rows if r["frame_index_delta"] is not None]
        diffs = [r["telemetry_match_time_diff_s"] for r in rows if r["telemetry_match_time_diff_s"] is not None]

        summary = {
            "container_frame_count": video_meta.get("container_frame_count"),
            "decoded_frame_count": video_meta.get("decoded_frame_count"),
            "container_frame_count_mismatch": video_meta.get("container_frame_count_mismatch"),
            "telemetry_row_count": len(telemetry_records),
            "reconstruction_frame_count": len(rows),
            "eligible_frame_count": sum(1 for r in rows if r["reconstruction_eligible"]),
            "ineligible_frame_count": sum(1 for r in rows if not r["reconstruction_eligible"]),
            "frames_with_valid_telemetry": sum(1 for r in rows if r["telemetry_valid"]),
            "require_valid_telemetry_for_reconstruction": self.require_valid_for_reconstruction,
            "association_method": "nearest_video_time_s",
            "max_time_diff_s": self.max_time_diff_s,
            "max_observed_time_diff_s": max(diffs) if diffs else None,
            "frame_index_tolerance": self.frame_index_tolerance,
            "frames_with_frame_index_disagreement": len(inconsistent),
            "max_abs_frame_index_delta": max(deltas) if deltas else None,
            "telemetry_rows_used": len(used_counts),
            "telemetry_rows_shared_by_multiple_frames": sum(1 for c in used_counts.values() if c > 1),
            "telemetry_rows_unassociated": unmatched_rows,
            "telemetry_rows_unassociated_count": len(unmatched_rows),
            "coordinate_system": coord_sys.to_dict(),
        }
        return {"summary": summary, "frames": rows}
