import json
import math
from pathlib import Path
import numpy as np
import pytest
import trimesh

from src.assets.asset_record import AssetDimensions, AssetRecord, MatchingScore
from src.assets.asset_normalizer import normalize_mesh, normalize_and_export
from src.assets.curated_models import (
    make_sedan, make_suv, make_hatchback, make_city_bus,
    make_box_truck, make_cargo_van, make_oak_tree, make_pine_tree,
    CURATED_GENERATORS, SUBCATEGORY_TO_CLASS,
)
from src.assets.library_manager import LibraryManager
from src.assets.asset_matcher import AssetMatcher, _class_score, _dimension_score, _shape_score
from src.assets.stage10_pipeline import run_stage10


def _make_simple_box_mesh(l, w, h):
    return trimesh.creation.box(extents=[l, w, h])


def test_asset_record_validate():
    rec = AssetRecord(
        asset_id="test_car",
        source="test",
        license="CC0",
        class_name="car",
        subcategory="sedan",
        dimensions=AssetDimensions(length=4.4, width=1.8, height=1.5),
        coordinate_system="right-handed",
        up_axis="+Z",
        forward_axis="+X",
        origin_description="ground center",
        texture_available=False,
        mesh_file="test.glb",
    )
    errors = rec.validate()
    assert errors == []


def test_asset_record_validate_invalid():
    rec = AssetRecord(
        asset_id="",
        source="test",
        license="CC0",
        class_name="car",
        subcategory="sedan",
        dimensions=AssetDimensions(length=-1.0, width=1.8, height=1.5),
        coordinate_system="right-handed",
        up_axis="invalid_axis",
        forward_axis="+X",
        origin_description="ground center",
        texture_available=False,
        mesh_file="test.glb",
    )
    errors = rec.validate()
    assert any("asset_id" in e for e in errors)
    assert any("length" in e for e in errors)
    assert any("up_axis" in e for e in errors)


def test_asset_record_roundtrip():
    rec = AssetRecord(
        asset_id="car_001",
        source="curated",
        license="CC0",
        class_name="car",
        subcategory="sedan",
        dimensions=AssetDimensions(4.4, 1.8, 1.5),
        coordinate_system="right-handed",
        up_axis="+Z",
        forward_axis="+X",
        origin_description="ground center",
        texture_available=False,
        mesh_file="car_001.glb",
        tags=["car", "sedan"],
    )
    d = rec.to_dict()
    rec2 = AssetRecord.from_dict(d)
    assert rec2.asset_id == rec.asset_id
    assert rec2.dimensions.length == rec.dimensions.length
    assert rec2.tags == rec.tags


def test_normalize_mesh_ground_at_zero():
    box = _make_simple_box_mesh(3.0, 2.0, 1.5)
    box.apply_translation([5.0, -3.0, 7.0])
    normed, diag = normalize_mesh(box, target_length=4.0, target_width=2.0, target_height=1.0)
    assert abs(normed.bounds[0][2]) < 1e-5, f"Ground Z should be 0, got {normed.bounds[0][2]}"
    assert normed.bounds[0][2] >= -1e-5


def test_normalize_mesh_dimensions():
    box = _make_simple_box_mesh(3.0, 1.5, 2.0)
    normed, diag = normalize_mesh(box, target_length=5.0, target_width=2.5, target_height=1.8)
    nl = normed.bounds[1][0] - normed.bounds[0][0]
    nw = normed.bounds[1][1] - normed.bounds[0][1]
    nh = normed.bounds[1][2] - normed.bounds[0][2]
    assert abs(nl - 5.0) < 0.05, f"Expected length ~5.0, got {nl}"
    assert abs(nw - 2.5) < 0.05, f"Expected width ~2.5, got {nw}"
    assert abs(nh - 1.8) < 0.05, f"Expected height ~1.8, got {nh}"


def test_normalize_mesh_center_xy():
    box = _make_simple_box_mesh(4.0, 2.0, 1.5)
    normed, _ = normalize_mesh(box, target_length=4.0, target_width=2.0, target_height=1.5)
    cx = (normed.bounds[0][0] + normed.bounds[1][0]) / 2.0
    cy = (normed.bounds[0][1] + normed.bounds[1][1]) / 2.0
    assert abs(cx) < 1e-4, f"Center X should be 0, got {cx}"
    assert abs(cy) < 1e-4, f"Center Y should be 0, got {cy}"


def test_normalize_and_export(tmp_path):
    box = _make_simple_box_mesh(3.0, 1.5, 1.0)
    out_glb = tmp_path / "test.glb"
    out_obj = tmp_path / "test.obj"
    diag = normalize_and_export(
        input_path=None,
        output_glb=out_glb,
        output_obj=out_obj,
        target_length=4.4,
        target_width=1.8,
        target_height=1.5,
        mesh=box,
    )
    assert out_glb.exists()
    assert out_obj.exists()
    assert abs(diag["ground_z_min"]) < 1e-5
    assert abs(diag["normalized_length"] - 4.4) < 0.05


def test_curated_models_basic_geometry():
    for fn in [make_sedan, make_suv, make_hatchback, make_city_bus, make_box_truck, make_cargo_van]:
        mesh = fn()
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert mesh.bounds[0][2] >= -1e-4, f"{fn.__name__}: ground not at Z=0, min Z={mesh.bounds[0][2]}"


def test_curated_trees():
    for fn in [make_oak_tree, make_pine_tree]:
        mesh = fn()
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        h = mesh.bounds[1][2] - mesh.bounds[0][2]
        assert h > 3.0, f"{fn.__name__}: height too small: {h}"


def test_curated_generators_completeness():
    for subcategory in CURATED_GENERATORS:
        assert subcategory in SUBCATEGORY_TO_CLASS, f"{subcategory} missing from SUBCATEGORY_TO_CLASS"
    cars = [k for k, v in SUBCATEGORY_TO_CLASS.items() if v == "car"]
    trees = [k for k, v in SUBCATEGORY_TO_CLASS.items() if v == "tree"]
    assert len(cars) >= 5
    assert len(trees) >= 5


def test_library_manager_initialize(tmp_path):
    mgr = LibraryManager(
        assets_dir=tmp_path / "assets",
        normalized_dir=tmp_path / "assets" / "normalized",
    )
    added = mgr.initialize_curated_library()
    assert added > 0
    summary = mgr.get_catalog_summary()
    assert summary["total_assets"] >= 10
    assert "car" in summary["by_class"]
    assert "tree" in summary["by_class"]
    assert summary["by_class"]["car"] >= 5


def test_library_manager_resume_safe(tmp_path):
    mgr = LibraryManager(
        assets_dir=tmp_path / "assets",
        normalized_dir=tmp_path / "assets" / "normalized",
    )
    added_first = mgr.initialize_curated_library()
    added_second = mgr.initialize_curated_library()
    assert added_second == 0, "Second call should add 0 since assets exist"


def test_library_manager_list_and_query(tmp_path):
    mgr = LibraryManager(tmp_path / "assets", tmp_path / "assets" / "normalized")
    mgr.initialize_curated_library()
    cars = mgr.list_assets(class_name="car")
    assert len(cars) >= 5
    buses = mgr.list_assets(class_name="bus")
    assert len(buses) >= 2
    trees = mgr.list_assets(class_name="tree")
    assert len(trees) >= 5


def test_library_manager_verify(tmp_path):
    mgr = LibraryManager(tmp_path / "assets", tmp_path / "assets" / "normalized")
    mgr.initialize_curated_library()
    result = mgr.verify_library()
    assert result["total_assets"] > 0
    assert result["missing_files"] == 0
    assert result["status"] == "OK"


def test_class_score():
    assert _class_score("car", "car", "sedan") == 1.0
    assert _class_score("truck", "truck", "box_truck") == 1.0
    assert _class_score("car", "tree", "oak_tree") == 0.0
    assert _class_score("car", "car", "suv") == 1.0


def test_dimension_score_perfect():
    q = AssetDimensions(4.4, 1.8, 1.5)
    c = AssetDimensions(4.4, 1.8, 1.5)
    score, err = _dimension_score(q, c)
    assert abs(score - 1.0) < 1e-4


def test_dimension_score_degradation():
    q = AssetDimensions(4.4, 1.8, 1.5)
    c = AssetDimensions(10.5, 2.6, 3.2)
    score, _ = _dimension_score(q, c)
    assert score < 0.4


def test_shape_score_matching_subcategory():
    q = AssetDimensions(4.4, 1.8, 1.5)
    s = _shape_score(q, "sedan")
    assert s > 0.6


def test_asset_matcher_basic(tmp_path):
    mgr = LibraryManager(tmp_path / "assets", tmp_path / "assets" / "normalized")
    mgr.initialize_curated_library()
    matcher = AssetMatcher(library=mgr, top_k=3)
    results = matcher.match(
        class_name="car",
        dimensions=AssetDimensions(4.4, 1.8, 1.5),
        is_dynamic=False,
    )
    assert len(results) >= 1
    assert results[0].total_score > 0.5
    for r in results:
        assert r.class_name == "car"


def test_asset_matcher_no_cross_class(tmp_path):
    mgr = LibraryManager(tmp_path / "assets", tmp_path / "assets" / "normalized")
    mgr.initialize_curated_library()
    matcher = AssetMatcher(library=mgr, top_k=5)
    results = matcher.match(
        class_name="bus",
        dimensions=AssetDimensions(10.5, 2.6, 3.2),
    )
    assert all(r.class_name == "bus" for r in results)


def test_asset_matcher_match_object(tmp_path):
    mgr = LibraryManager(tmp_path / "assets", tmp_path / "assets" / "normalized")
    mgr.initialize_curated_library()
    matcher = AssetMatcher(library=mgr, top_k=3)
    obj = {
        "track_id": 42,
        "class_name": "car",
        "dimensions": {"length": 4.4, "width": 1.8, "height": 1.5},
        "motion": {"is_dynamic": True, "speed_mps": 8.0},
    }
    result = matcher.match_object(obj)
    assert result["track_id"] == 42
    assert result["best_match"] is not None
    assert result["best_match"]["class_name"] == "car"
    assert len(result["top_candidates"]) >= 1


def test_stage10_pipeline_smoke(tmp_path):
    localization_dir = tmp_path / "localization"
    localization_dir.mkdir()
    objects_3d = {
        "total_objects": 2,
        "objects": [
            {
                "track_id": 1,
                "class_name": "car",
                "dimensions": {"length": 4.4, "width": 1.8, "height": 1.5},
                "motion": {"is_dynamic": False, "speed_mps": 0.0},
            },
            {
                "track_id": 2,
                "class_name": "bus",
                "dimensions": {"length": 10.5, "width": 2.6, "height": 3.2},
                "motion": {"is_dynamic": True, "speed_mps": 5.0},
            },
        ],
    }
    with open(localization_dir / "objects_3d.json", "w") as f:
        json.dump(objects_3d, f)

    output_dir = tmp_path / "assets"

    class MockConfig:
        def get(self, key, default=None):
            cfg = {
                "assets.catalog_path": str(output_dir / "assets_catalog.json"),
                "assets.normalized_dir": str(output_dir / "normalized"),
                "assets.top_k_candidates": 3,
                "assets.matching_weights": {
                    "class_weight": 0.5,
                    "dimension_weight": 0.3,
                    "shape_weight": 0.15,
                    "context_weight": 0.05,
                },
            }
            return cfg.get(key, default)

    report = run_stage10(output_dir=output_dir, config_manager=MockConfig())
    assert report["status"] == "SUCCESS"
    assert report["matching"]["total_input_objects"] == 2
    assert report["matching"]["total_matched"] == 2
    assert (output_dir / "matched_assets.json").exists()
    assert (output_dir / "stage10_report.json").exists()

    with open(output_dir / "matched_assets.json") as f:
        ma = json.load(f)
    assert ma["total_input_objects"] == 2
    assert ma["total_matched"] == 2
    assert ma["matches"][0]["best_match"] is not None
    assert ma["matches"][1]["best_match"] is not None
