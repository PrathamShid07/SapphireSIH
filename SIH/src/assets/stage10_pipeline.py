import json
import logging
import time
from pathlib import Path
from typing import Optional

from .library_manager import LibraryManager
from .asset_matcher import AssetMatcher

logger = logging.getLogger(__name__)


def run_stage10(
    output_dir: Optional[Path] = None,
    config_manager=None,
) -> dict:
    start_time = time.time()

    if config_manager is not None:
        assets_dir = Path(config_manager.get("assets.catalog_path", "data/assets/assets_catalog.json")).parent
        normalized_dir = Path(config_manager.get("assets.normalized_dir", "data/assets/normalized"))
        top_k = int(config_manager.get("assets.top_k_candidates", 3))
        weights = config_manager.get("assets.matching_weights", {})
        class_w = float(weights.get("class_weight", 0.5))
        dim_w = float(weights.get("dimension_weight", 0.3))
        shape_w = float(weights.get("shape_weight", 0.15))
        ctx_w = float(weights.get("context_weight", 0.05))
    else:
        assets_dir = Path("data/assets")
        normalized_dir = assets_dir / "normalized"
        top_k = 3
        class_w, dim_w, shape_w, ctx_w = 0.5, 0.3, 0.15, 0.05

    if output_dir is None:
        output_dir = Path("outputs/sample_drone/assets")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    library = LibraryManager(assets_dir=assets_dir, normalized_dir=normalized_dir)

    logger.info("Initializing curated 3D asset library...")
    added = library.initialize_curated_library()
    catalog_summary = library.get_catalog_summary()
    logger.info(f"Library ready: {catalog_summary}")

    verify_result = library.verify_library()
    if verify_result["status"] != "OK":
        logger.warning(f"Library has {len(verify_result['issues'])} issues: {verify_result['issues']}")

    localization_dir = output_dir.parent / "localization"
    objects_3d_path = localization_dir / "objects_3d.json"

    if not objects_3d_path.exists():
        msg = f"Stage 9 output not found: {objects_3d_path}"
        logger.error(msg)
        return {
            "status": "FAILED",
            "reason": msg,
            "elapsed_seconds": round(time.time() - start_time, 2),
        }

    logger.info(f"Loading Stage 9 localized objects: {objects_3d_path}")
    with open(objects_3d_path, "r") as f:
        localized = json.load(f)

    objects = localized.get("objects", [])
    logger.info(f"Matching {len(objects)} 3D objects against asset library...")

    matcher = AssetMatcher(
        library=library,
        class_weight=class_w,
        dimension_weight=dim_w,
        shape_weight=shape_w,
        context_weight=ctx_w,
        top_k=top_k,
    )

    matched_objects = []
    total_with_match = 0
    for obj in objects:
        result = matcher.match_object(obj)
        matched_objects.append(result)
        if result["best_match"] is not None:
            total_with_match += 1

    matched_assets_path = output_dir / "matched_assets.json"
    with open(matched_assets_path, "w") as f:
        json.dump(
            {
                "total_input_objects": len(objects),
                "total_matched": total_with_match,
                "top_k": top_k,
                "matches": matched_objects,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved matched assets: {matched_assets_path}")

    elapsed = round(time.time() - start_time, 2)
    report = {
        "status": "SUCCESS",
        "elapsed_seconds": elapsed,
        "library": {
            "total_assets": catalog_summary["total_assets"],
            "by_class": catalog_summary["by_class"],
            "newly_generated": added,
            "verification": verify_result["status"],
        },
        "matching": {
            "total_input_objects": len(objects),
            "total_matched": total_with_match,
            "match_rate": round(total_with_match / max(len(objects), 1), 4),
        },
    }

    report_path = output_dir / "stage10_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Stage 10 report saved: {report_path}")

    return report
