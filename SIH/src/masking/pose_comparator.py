"""
Pure comparison utility for measuring how much camera poses moved between two
COLMAP sparse models. Used to quantify what a "limited pose refinement" step
(src/masking/stage8_pipeline.py's optional post-triangulation bundle_adjuster
pass) actually does to Stage 4's original poses - see
docs/STAGE8_OPTIMIZATION.md for the full experiment writeup.

Does not run COLMAP or touch any file; operates on the dicts returned by
src.sfm.colmap_parser.parse_colmap_model().
"""

import math


def _quat_angle_diff_degrees(qvec1, qvec2):
    """Angle (degrees) between two rotations expressed as COLMAP (qw,qx,qy,qz) quaternions."""
    dot = sum(a * b for a, b in zip(qvec1, qvec2))
    dot = max(-1.0, min(1.0, abs(dot)))  # quaternion sign ambiguity: q and -q are the same rotation
    return 2.0 * math.degrees(math.acos(dot))


def compare_poses(reference_images, comparison_images):
    """
    reference_images / comparison_images: lists of the "registered_images"
    dicts produced by src.sfm.colmap_parser.parse_colmap_model() (each having
    "name", "qvec", "tvec").

    Returns per-image translation distance and rotation angle (degrees)
    between the two models, plus max/mean summary stats, for every image
    present in both. Images present in only one model are reported separately
    rather than silently ignored.
    """
    ref_by_name = {img["name"]: img for img in reference_images}
    cmp_by_name = {img["name"]: img for img in comparison_images}

    common_names = sorted(set(ref_by_name) & set(cmp_by_name))
    only_in_reference = sorted(set(ref_by_name) - set(cmp_by_name))
    only_in_comparison = sorted(set(cmp_by_name) - set(ref_by_name))

    per_image = []
    for name in common_names:
        ref = ref_by_name[name]
        cmp_ = cmp_by_name[name]
        translation_delta = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(ref["tvec"], cmp_["tvec"]))
        )
        rotation_delta_deg = _quat_angle_diff_degrees(ref["qvec"], cmp_["qvec"])
        per_image.append({
            "name": name,
            "translation_delta": round(translation_delta, 8),
            "rotation_delta_degrees": round(rotation_delta_deg, 8),
        })

    translations = [p["translation_delta"] for p in per_image]
    rotations = [p["rotation_delta_degrees"] for p in per_image]

    summary = {
        "num_compared": len(per_image),
        "only_in_reference": only_in_reference,
        "only_in_comparison": only_in_comparison,
        "max_translation_delta": round(max(translations), 8) if translations else 0.0,
        "mean_translation_delta": round(sum(translations) / len(translations), 8) if translations else 0.0,
        "max_rotation_delta_degrees": round(max(rotations), 8) if rotations else 0.0,
        "mean_rotation_delta_degrees": round(sum(rotations) / len(rotations), 8) if rotations else 0.0,
    }

    return {"per_image": per_image, "summary": summary}
