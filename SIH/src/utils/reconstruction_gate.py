import json
from pathlib import Path

class ReconstructionGateError(RuntimeError):
    pass

def load_reconstruction_manifest(output_dir):
    manifest_path = Path(output_dir) / "metadata" / "reconstruction_manifest.json"
    if not manifest_path.exists():
        raise ReconstructionGateError(
            f"Reconstruction manifest not found at {manifest_path}. "
            "preprocessing.reconstruction.strategy='decoded_stride' requires Stage 2 and Stage 3 to have run "
            "so every reconstruction image has an explicit telemetry association."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f), manifest_path

def resolve_reconstruction_inputs(output_dir, config_manager):
    output_dir = Path(output_dir)
    strategy = config_manager.get("preprocessing.reconstruction.strategy", "selected_keyframes")

    if strategy == "selected_keyframes":
        image_dir = output_dir / "processed" / "selected_frames"
        if not image_dir.exists() or not list(image_dir.glob("*.jpg")):
            raise FileNotFoundError(f"Selected keyframe images not found in {image_dir}. Run Stage 2 first.")
        return {
            "strategy": strategy,
            "image_dir": image_dir,
            "image_names": sorted(p.name for p in image_dir.glob("*.jpg")),
            "manifest_path": None,
            "restrict_to_names": False,
            "ineligible": [],
        }

    if strategy != "decoded_stride":
        raise ReconstructionGateError(f"Unknown preprocessing.reconstruction.strategy '{strategy}'.")

    manifest, manifest_path = load_reconstruction_manifest(output_dir)
    image_dir = output_dir / "processed" / "reconstruction_frames"
    rows = manifest.get("frames", [])
    eligible = [r for r in rows if r.get("reconstruction_eligible")]
    ineligible = [
        {"frame_index": r.get("frame_index"), "reasons": r.get("telemetry_invalid_reasons") or ["image_missing" if not r.get("image_exists") else "not_eligible"]}
        for r in rows if not r.get("reconstruction_eligible")
    ]
    if not eligible:
        raise ReconstructionGateError(
            f"No reconstruction-eligible frames in {manifest_path} ({len(rows)} frames listed, "
            f"{len(ineligible)} ineligible). Refusing to run reconstruction."
        )
    missing = [r["image_name"] for r in eligible if not (image_dir / r["image_name"]).exists()]
    if missing:
        raise ReconstructionGateError(
            f"{len(missing)} manifest-eligible images are missing from {image_dir} (first: {missing[0]}). "
            "Re-run Stage 2 and Stage 3."
        )
    return {
        "strategy": strategy,
        "image_dir": image_dir,
        "image_names": sorted(r["image_name"] for r in eligible),
        "manifest_path": manifest_path,
        "restrict_to_names": True,
        "ineligible": ineligible,
    }

def assert_sparse_model_matches_inputs(registered_names, recon_inputs):
    allowed = set(recon_inputs["image_names"])
    foreign = sorted(set(registered_names) - allowed)
    if foreign:
        raise ReconstructionGateError(
            f"The sparse model registers {len(foreign)} images that are not in the reconstruction image set "
            f"(first: {foreign[0]}). Stage 4 and Stage 5 must consume the same reconstruction manifest; re-run Stage 4."
        )

def assert_registration_sufficient(sfm_report, min_fraction):
    if min_fraction is None:
        return
    total = sfm_report.get("total_input_images") or 0
    registered = sfm_report.get("registered_image_count") or 0
    fraction = registered / total if total else 0.0
    if fraction < float(min_fraction):
        raise ReconstructionGateError(
            f"Stage 4 registered {registered}/{total} images ({fraction:.2%}), below dense.min_registration_fraction="
            f"{min_fraction}. Refusing to spend a GPU run on an inadequate sparse model."
        )

def masks_approved(config_manager):
    return bool(config_manager.get("masking.approved_for_reconstruction", True))

def assert_masks_approved(config_manager, mask_path=None):
    if mask_path is not None and not masks_approved(config_manager):
        raise ReconstructionGateError(
            "Dynamic masks are not approved for reconstruction on this dataset "
            "(masking.approved_for_reconstruction is false). Refusing to pass masks to COLMAP."
        )

def assert_stage8_allowed(config_manager):
    if not masks_approved(config_manager):
        raise ReconstructionGateError(
            "Stage 8 dynamic masking is not approved for this dataset "
            "(masking.approved_for_reconstruction is false). Reconstruction must use unmasked imagery."
        )
    strategy = config_manager.get("preprocessing.reconstruction.strategy", "selected_keyframes")
    if strategy != "selected_keyframes":
        raise ReconstructionGateError(
            f"Stage 8 tracks and masks the selected keyframes, but reconstruction strategy is '{strategy}'. "
            "Its masked SfM would not be comparable with the Stage 4 model."
        )

def join_poses_with_manifest(registered_images, manifest):
    rows_by_name = {r["image_name"]: r for r in manifest.get("frames", [])}
    registered_names = set()
    joined = []
    for img in registered_images:
        name = img.get("name")
        registered_names.add(name)
        row = rows_by_name.get(name)
        joined.append({
            "image_name": name,
            "colmap_image_id": img.get("id"),
            "in_manifest": row is not None,
            "frame_index": row.get("frame_index") if row else None,
            "video_time_s": row.get("video_time_s") if row else None,
            "latitude": row.get("latitude") if row else None,
            "longitude": row.get("longitude") if row else None,
            "altitude_m": row.get("altitude_m") if row else None,
            "position_xyz_colmap": img.get("position_xyz"),
            "local_enu_east_m": row.get("local_enu_east_m") if row else None,
            "local_enu_north_m": row.get("local_enu_north_m") if row else None,
            "local_enu_up_m": row.get("local_enu_up_m") if row else None,
            "telemetry_match_time_diff_s": row.get("telemetry_match_time_diff_s") if row else None,
        })
    unregistered = [
        r["image_name"] for r in manifest.get("frames", [])
        if r.get("reconstruction_eligible") and r["image_name"] not in registered_names
    ]
    return {"registered": joined, "eligible_but_unregistered": unregistered}
