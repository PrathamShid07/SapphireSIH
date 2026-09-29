import json
from pathlib import Path
import numpy as np
import pytest
import trimesh

from src.fusion.placement_record import (
    FusedSceneManifest,
    PlacedAsset,
    PlacementTransform,
    PlacementValidation,
)
from src.fusion.terrain_aligner import TerrainAligner, TerrainElevationModel
from src.fusion.collision_checker import (
    CollisionChecker,
    compute_3d_aabb_iou,
    compute_aabb,
    compute_oriented_box_corners,
)
from src.fusion.scene_composer import SceneComposer
from src.fusion.stage11_pipeline import run_stage11


def test_placement_transform_matrix():
    pos = np.array([10.0, 20.0, 5.0])
    rot = np.eye(3)
    scale = np.array([1.5, 1.2, 1.0])
    transform = PlacementTransform.from_components(pos, rot, scale)

    mat = transform.to_numpy()
    assert mat.shape == (4, 4)
    assert np.allclose(mat[:3, 3], pos)
    assert np.allclose(mat[3, :], [0, 0, 0, 1])
    assert transform.yaw_deg == 0.0
    assert transform.pitch_deg == 0.0
    assert transform.roll_deg == 0.0


def test_placement_record_roundtrip():
    val = PlacementValidation(
        is_floating=True,
        float_distance_m=0.35,
        was_adjusted=True,
        adjustment_details="Snapped down.",
    )
    t = PlacementTransform.from_components(
        np.array([1.0, 2.0, 3.0]),
        np.eye(3),
        np.array([1.0, 1.0, 1.0]),
    )
    asset = PlacedAsset(
        track_id=42,
        asset_id="curated_sedan",
        class_name="car",
        subcategory="sedan",
        is_dynamic=False,
        confidence=0.88,
        mesh_file="data/assets/normalized/curated_sedan.glb",
        transform=t,
        ground_contact_3d=[1.0, 2.0, 3.0],
        bounding_box_3d={"dimensions": {"length": 4.4, "width": 1.8, "height": 1.4}},
        validation=val,
    )
    data = asset.to_dict()
    reloaded = PlacedAsset.from_dict(data)

    assert reloaded.track_id == 42
    assert reloaded.asset_id == "curated_sedan"
    assert reloaded.class_name == "car"
    assert reloaded.validation.is_floating is True
    assert reloaded.validation.float_distance_m == 0.35
    assert reloaded.validation.was_adjusted is True


def test_terrain_elevation_flat():
    box = trimesh.creation.box(extents=[20.0, 20.0, 0.1])
    box.apply_translation([0.0, 0.0, 5.0])
    model = TerrainElevationModel(mesh=box)

    elev, normal = model.query_elevation_and_normal(0.0, 0.0)
    assert abs(elev - 5.0) < 0.2
    assert normal[2] > 0.9


def test_terrain_elevation_sloped():
    xs = np.linspace(-10, 10, 20)
    ys = np.linspace(-10, 10, 20)
    xx, yy = np.meshgrid(xs, ys)
    zz = 0.2 * xx + 3.0
    pts = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
    model = TerrainElevationModel(points=pts)

    elev, normal = model.query_elevation_and_normal(0.0, 0.0)
    assert abs(elev - 3.0) < 0.5
    assert abs(normal[0]) > 0.05
    assert normal[2] > 0.5


def test_terrain_aligner_car_vs_tree():
    aligner = TerrainAligner(max_tilt_deg=25.0)
    sloped_normal = np.array([-0.2, 0.0, 0.98])
    sloped_normal /= np.linalg.norm(sloped_normal)

    r_car = aligner.compute_alignment_matrix(yaw_deg=0.0, terrain_normal=sloped_normal, is_tree=False)
    up_car = np.dot(r_car, [0, 0, 1])
    assert abs(up_car[0]) > 0.05

    r_tree = aligner.compute_alignment_matrix(yaw_deg=0.0, terrain_normal=sloped_normal, is_tree=True)
    up_tree = np.dot(r_tree, [0, 0, 1])
    assert np.allclose(up_tree, [0, 0, 1], atol=1e-5)


def test_terrain_aligner_scale():
    aligner = TerrainAligner()
    scale_car = aligner.compute_asset_scale(
        {"length": 4.8, "width": 2.0, "height": 1.6},
        {"length": 4.4, "width": 1.8, "height": 1.4},
        class_name="car",
    )
    assert scale_car[0] > 1.0
    assert scale_car[1] > 1.0
    assert scale_car[2] > 1.0

    scale_tree = aligner.compute_asset_scale(
        {"length": 8.0, "width": 8.0, "height": 10.0},
        {"length": 6.0, "width": 6.0, "height": 7.0},
        class_name="tree",
    )
    assert scale_tree[2] == pytest.approx(10.0 / 7.0, rel=1e-3)


def test_collision_checker_floating_and_penetration():
    checker = CollisionChecker(float_tolerance_m=0.25, penetration_tolerance_m=0.15, snap_to_ground=True)

    t1 = PlacementTransform.from_components(np.array([0, 0, 2.0]), np.eye(3), np.array([1, 1, 1]))
    floating_asset = PlacedAsset(
        track_id=1,
        asset_id="sedan",
        class_name="car",
        subcategory="sedan",
        is_dynamic=False,
        confidence=0.9,
        mesh_file="test.glb",
        transform=t1,
        ground_contact_3d=[0, 0, 2.0],
        bounding_box_3d={"dimensions": {"length": 4.4, "width": 1.8, "height": 1.4}},
    )

    checker.check_ground_contact(floating_asset, terrain_z=0.5)
    assert floating_asset.validation.is_floating is True
    assert floating_asset.validation.float_distance_m == pytest.approx(1.5, abs=1e-3)
    assert floating_asset.validation.was_adjusted is True
    assert floating_asset.transform.position[2] == pytest.approx(0.5, abs=1e-3)

    t2 = PlacementTransform.from_components(np.array([0, 0, -1.0]), np.eye(3), np.array([1, 1, 1]))
    sunken_asset = PlacedAsset(
        track_id=2,
        asset_id="sedan",
        class_name="car",
        subcategory="sedan",
        is_dynamic=False,
        confidence=0.9,
        mesh_file="test.glb",
        transform=t2,
        ground_contact_3d=[0, 0, -1.0],
        bounding_box_3d={"dimensions": {"length": 4.4, "width": 1.8, "height": 1.4}},
    )

    checker.check_ground_contact(sunken_asset, terrain_z=0.0)
    assert sunken_asset.validation.terrain_penetration is True
    assert sunken_asset.validation.penetration_depth_m == pytest.approx(1.0, abs=1e-3)
    assert sunken_asset.validation.was_adjusted is True
    assert sunken_asset.transform.position[2] == pytest.approx(0.0, abs=1e-3)


def test_collision_checker_inter_asset():
    checker = CollisionChecker(collision_iou_threshold=0.1)

    t1 = PlacementTransform.from_components(np.array([0, 0, 0]), np.eye(3), np.array([1, 1, 1]))
    t2 = PlacementTransform.from_components(np.array([0.5, 0.2, 0]), np.eye(3), np.array([1, 1, 1]))
    t3 = PlacementTransform.from_components(np.array([50.0, 50.0, 0]), np.eye(3), np.array([1, 1, 1]))

    a1 = PlacedAsset(1, "sedan", "car", "sedan", False, 0.95, "m1.glb", t1, [0, 0, 0], {"dimensions": {"length": 4.4, "width": 1.8, "height": 1.4}})
    a2 = PlacedAsset(2, "sedan", "car", "sedan", False, 0.60, "m2.glb", t2, [0.5, 0.2, 0], {"dimensions": {"length": 4.4, "width": 1.8, "height": 1.4}})
    a3 = PlacedAsset(3, "sedan", "car", "sedan", False, 0.85, "m3.glb", t3, [50, 50, 0], {"dimensions": {"length": 4.4, "width": 1.8, "height": 1.4}})

    resolved, count = checker.check_and_resolve_collisions([a1, a2, a3])
    assert count == 1
    assert len(resolved) == 2
    track_ids = [a.track_id for a in resolved]
    assert 1 in track_ids
    assert 3 in track_ids
    assert 2 not in track_ids


def test_scene_composer_and_export(tmp_path):
    box_mesh = trimesh.creation.box(extents=[10, 10, 1])
    box_mesh_path = tmp_path / "base.obj"
    box_mesh.export(str(box_mesh_path))

    asset_mesh = trimesh.creation.box(extents=[2, 1, 1])
    asset_path = tmp_path / "asset.glb"
    with open(asset_path, "wb") as f:
        f.write(asset_mesh.export(file_type="glb"))

    t = PlacementTransform.from_components(np.array([2.0, 2.0, 1.0]), np.eye(3), np.array([1, 1, 1]))
    placed = PlacedAsset(
        track_id=10,
        asset_id="test_box",
        class_name="car",
        subcategory="box",
        is_dynamic=False,
        confidence=0.9,
        mesh_file=str(asset_path),
        transform=t,
        ground_contact_3d=[2.0, 2.0, 1.0],
        bounding_box_3d={"dimensions": {"length": 2, "width": 1, "height": 1}},
    )

    composer = SceneComposer()
    scene, transformed = composer.compose_scene([placed], base_mesh_path=box_mesh_path)
    assert len(scene.geometry) >= 2

    exported = composer.export_scene(scene, transformed, tmp_path / "exports", formats=["glb", "obj", "ply"])
    assert "glb" in exported
    assert "obj" in exported
    assert "ply" in exported

    assert Path(exported["glb"]).exists() and Path(exported["glb"]).stat().st_size > 0
    assert Path(exported["obj"]).exists() and Path(exported["obj"]).stat().st_size > 0
    assert Path(exported["ply"]).exists() and Path(exported["ply"]).stat().st_size > 0

    reloaded_glb = trimesh.load(exported["glb"])
    assert reloaded_glb is not None


def test_stage11_full_pipeline(tmp_path):
    base_mesh = trimesh.creation.box(extents=[30, 30, 0.5])
    base_mesh_path = tmp_path / "dense_mesh.obj"
    base_mesh.export(str(base_mesh_path))

    asset_mesh = trimesh.creation.box(extents=[4, 2, 1.5])
    asset_path = tmp_path / "sedan.glb"
    with open(asset_path, "wb") as f:
        f.write(asset_mesh.export(file_type="glb"))

    objects_data = {
        "total_objects": 2,
        "objects": [
            {
                "track_id": 1,
                "class_name": "car",
                "confidence": 0.85,
                "position_3d": [0.0, 0.0, 0.75],
                "ground_contact_3d": [0.0, 0.0, 0.0],
                "dimensions": {"length": 4.4, "width": 1.8, "height": 1.4},
                "orientation": {"yaw_deg": 15.0},
                "motion": {"is_dynamic": False, "speed_mps": 0.0},
            },
            {
                "track_id": 2,
                "class_name": "car",
                "confidence": 0.90,
                "position_3d": [5.0, 5.0, 2.0],
                "ground_contact_3d": [5.0, 5.0, 2.0],
                "dimensions": {"length": 4.4, "width": 1.8, "height": 1.4},
                "orientation": {"yaw_deg": 0.0},
                "motion": {"is_dynamic": False, "speed_mps": 0.0},
            },
        ],
    }
    objects_path = tmp_path / "objects_3d.json"
    with open(objects_path, "w", encoding="utf-8") as f:
        json.dump(objects_data, f)

    matched_data = {
        "total_matched": 2,
        "matches": [
            {
                "track_id": 1,
                "top_candidates": [
                    {
                        "asset_id": "curated_sedan",
                        "subcategory": "sedan",
                        "mesh_file": str(asset_path),
                        "suggested_scale": 1.0,
                    }
                ],
            },
            {
                "track_id": 2,
                "top_candidates": [
                    {
                        "asset_id": "curated_sedan",
                        "subcategory": "sedan",
                        "mesh_file": str(asset_path),
                        "suggested_scale": 1.0,
                    }
                ],
            },
        ],
    }
    matched_path = tmp_path / "matched_assets.json"
    with open(matched_path, "w", encoding="utf-8") as f:
        json.dump(matched_data, f)

    output_dir = tmp_path / "output_fusion"
    report = run_stage11(
        objects_path=objects_path,
        matched_assets_path=matched_path,
        dense_mesh_path=base_mesh_path,
        output_dir=output_dir,
    )

    assert report["status"] == "COMPLETED"
    assert report["total_placed_assets"] == 2
    assert report["base_mesh_included"] is True
    assert (output_dir / "fused_scene.glb").exists()
    assert (output_dir / "fused_scene.obj").exists()
    assert (output_dir / "fused_scene.ply").exists()
    assert (output_dir / "fused_scene_manifest.json").exists()
    assert (output_dir / "stage11_report.json").exists()
