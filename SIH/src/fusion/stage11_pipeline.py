from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import trimesh

from src.assets.library_manager import LibraryManager
from src.assets.asset_matcher import AssetMatcher
from .placement_record import (
    FusedSceneManifest,
    PlacedAsset,
    PlacementTransform,
    PlacementValidation,
)
from .terrain_aligner import TerrainAligner, TerrainElevationModel
from .collision_checker import CollisionChecker
from .scene_composer import SceneComposer


def run_stage11(
    objects_path: Union[str, Path] = "outputs/sample_drone/localization/objects_3d.json",
    matched_assets_path: Optional[Union[str, Path]] = "outputs/sample_drone/assets/matched_assets.json",
    dense_mesh_path: Optional[Union[str, Path]] = "outputs/sample_drone/dense/dense_mesh.obj",
    output_dir: Union[str, Path] = "outputs/sample_drone/fusion",
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = {}
    if config is not None:
        if hasattr(config, "get"):
            cfg = config.get("fusion", {}) or {}
        elif isinstance(config, dict):
            cfg = config.get("fusion", {}) or {}

    max_float_tol = float(cfg.get("max_float_tolerance_m", 0.25))
    max_pen_tol = float(cfg.get("max_penetration_tolerance_m", 0.15))
    coll_iou_thresh = float(cfg.get("collision_iou_threshold", 0.1))
    snap_ground = bool(cfg.get("snap_to_ground", True))
    align_normal = bool(cfg.get("align_to_terrain_normal", True))
    max_tilt = float(cfg.get("max_terrain_tilt_deg", 25.0))
    export_fmts = list(cfg.get("export_formats", ["glb", "obj", "ply"]))

    obj_p = Path(objects_path)
    if not obj_p.exists():
        raise FileNotFoundError(f"Objects 3D file not found: {obj_p}")

    with open(obj_p, "r", encoding="utf-8") as f:
        obj_data = json.load(f)
    raw_objects = obj_data.get("objects", [])

    matches_by_track: Dict[int, Dict[str, Any]] = {}
    if matched_assets_path is not None and Path(matched_assets_path).exists():
        with open(matched_assets_path, "r", encoding="utf-8") as f:
            match_data = json.load(f)
        for m in match_data.get("matches", []):
            matches_by_track[int(m["track_id"])] = m
    else:
        lib = LibraryManager()
        matcher = AssetMatcher(lib)
        match_res = matcher.match_all(raw_objects)
        for m in match_res.get("matches", []):
            matches_by_track[int(m["track_id"])] = m

    loaded_mesh: Optional[trimesh.Trimesh] = None
    if dense_mesh_path is not None:
        m_path = Path(dense_mesh_path)
        if m_path.exists():
            try:
                loaded = trimesh.load(str(m_path), force="mesh")
                if isinstance(loaded, trimesh.Trimesh):
                    loaded_mesh = loaded
            except Exception:
                loaded_mesh = None

    terrain_model = TerrainElevationModel(mesh=loaded_mesh)
    terrain_aligner = TerrainAligner(terrain_model=terrain_model, max_tilt_deg=max_tilt)
    collision_checker = CollisionChecker(
        float_tolerance_m=max_float_tol,
        penetration_tolerance_m=max_pen_tol,
        collision_iou_threshold=coll_iou_thresh,
        snap_to_ground=snap_ground,
    )

    placed_assets: List[PlacedAsset] = []
    floating_count = 0
    penetration_count = 0

    for obj in raw_objects:
        track_id = int(obj.get("track_id", 0))
        class_name = str(obj.get("class_name", "car"))
        pos_3d = list(obj.get("position_3d", [0.0, 0.0, 0.0]))
        ground_3d = list(obj.get("ground_contact_3d", pos_3d))
        dims = dict(obj.get("dimensions", {"length": 4.4, "width": 1.8, "height": 1.4}))
        conf = float(obj.get("confidence", 1.0))
        is_dynamic = bool(obj.get("motion", {}).get("is_dynamic", False))
        yaw_deg = float(obj.get("orientation", {}).get("yaw_deg", 0.0))

        match_info = matches_by_track.get(track_id)
        asset_id = "curated_sedan"
        subcategory = "sedan"
        mesh_file = "data/assets/normalized/curated_sedan.glb"
        asset_dims = dims

        if match_info and match_info.get("top_candidates"):
            top = match_info["top_candidates"][0]
            asset_id = str(top.get("asset_id", asset_id))
            subcategory = str(top.get("subcategory", subcategory))
            mesh_file = str(top.get("mesh_file", mesh_file))
            if "dimensions" in top:
                asset_dims = top["dimensions"]

        z_elev, normal = terrain_model.query_elevation_and_normal(ground_3d[0], ground_3d[1])
        if loaded_mesh is None:
            z_elev = ground_3d[2]

        is_tree = (class_name.lower() == "tree")
        if align_normal:
            rot_mat = terrain_aligner.compute_alignment_matrix(yaw_deg, normal, is_tree=is_tree)
        else:
            rot_mat = terrain_aligner.compute_alignment_matrix(yaw_deg, np.array([0, 0, 1]), is_tree=is_tree)

        scale_vec = terrain_aligner.compute_asset_scale(dims, asset_dims, class_name=class_name)

        base_pos = np.array([ground_3d[0], ground_3d[1], ground_3d[2]], dtype=np.float64)
        transform = PlacementTransform.from_components(base_pos, rot_mat, scale_vec)

        placed_asset = PlacedAsset(
            track_id=track_id,
            asset_id=asset_id,
            class_name=class_name,
            subcategory=subcategory,
            is_dynamic=is_dynamic,
            confidence=conf,
            mesh_file=mesh_file,
            transform=transform,
            ground_contact_3d=[float(v) for v in base_pos],
            bounding_box_3d={"dimensions": dims},
            validation=PlacementValidation(),
        )

        collision_checker.check_ground_contact(placed_asset, z_elev)

        if placed_asset.validation.is_floating:
            floating_count += 1
        if placed_asset.validation.terrain_penetration:
            penetration_count += 1

        placed_assets.append(placed_asset)

    resolved_assets, resolved_collisions = collision_checker.check_and_resolve_collisions(placed_assets)

    composer = SceneComposer()
    scene, transformed_meshes = composer.compose_scene(
        placed_assets=resolved_assets,
        base_mesh=loaded_mesh,
        base_mesh_path=dense_mesh_path,
    )

    exported_files = composer.export_scene(
        scene=scene,
        transformed_meshes=transformed_meshes,
        output_dir=out_dir,
        formats=export_fmts,
    )

    static_count = sum(1 for a in resolved_assets if not a.is_dynamic)
    dynamic_count = sum(1 for a in resolved_assets if a.is_dynamic)

    manifest = FusedSceneManifest(
        timestamp=datetime.now().isoformat(),
        base_mesh_file=str(dense_mesh_path) if dense_mesh_path else None,
        total_placed_assets=len(resolved_assets),
        static_assets=static_count,
        dynamic_assets=dynamic_count,
        floating_detected=floating_count,
        penetrations_detected=penetration_count,
        collisions_resolved=resolved_collisions,
        exported_files=exported_files,
        placed_assets=[a.to_dict() for a in resolved_assets],
    )

    manifest_path = out_dir / "fused_scene_manifest.json"
    manifest.save(manifest_path)

    elapsed = time.time() - t0
    report = {
        "status": "COMPLETED",
        "elapsed_seconds": round(elapsed, 4),
        "total_placed_assets": len(resolved_assets),
        "static_assets": static_count,
        "dynamic_assets": dynamic_count,
        "floating_corrected": floating_count,
        "penetrations_corrected": penetration_count,
        "collisions_resolved": resolved_collisions,
        "base_mesh_included": loaded_mesh is not None,
        "exported_files": exported_files,
        "manifest_path": str(manifest_path),
    }

    report_path = out_dir / "stage11_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
