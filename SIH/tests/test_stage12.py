from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import trimesh

from src.texturing.mesh_cleaner import MeshCleaner
from src.texturing.uv_processor import UVProcessor
from src.texturing.texture_optimizer import TextureOptimizer
from src.texturing.lod_generator import LODGenerator
from src.texturing.stage12_pipeline import run_stage12


def test_mesh_cleaner_degenerate_and_duplicates(tmp_path):
    v = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [10, 10, 10],
    ], dtype=np.float64)
    f = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 0, 1],
        [0, 1, 2],
    ])
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    cleaner = MeshCleaner(hole_max_size=0, min_component_faces=0)
    cleaned, stats = cleaner.clean(mesh, temp_dir=tmp_path)

    assert stats["cleaned_faces"] == 2
    assert stats["cleaned_vertices"] == 4


def test_mesh_cleaner_hole_filling(tmp_path):
    box = trimesh.creation.box(extents=[2, 2, 2])
    faces_with_hole = box.faces[:-1]
    open_box = trimesh.Trimesh(vertices=box.vertices, faces=faces_with_hole, process=False)
    assert not open_box.is_watertight

    cleaner = MeshCleaner(hole_max_size=30)
    closed_mesh, stats = cleaner.clean(open_box, temp_dir=tmp_path)

    assert len(closed_mesh.faces) >= len(faces_with_hole)


def test_uv_processor_box():
    mesh = trimesh.creation.box()
    proc = UVProcessor(method="box")
    mesh_uv, stats = proc.compute_uv(mesh)

    assert hasattr(mesh_uv.visual, "uv")
    assert mesh_uv.visual.uv is not None
    assert len(mesh_uv.visual.uv) == len(mesh_uv.vertices)
    assert stats["uv_bounds"][0] >= 0.0
    assert stats["uv_bounds"][1] <= 1.0


def test_uv_processor_planar():
    mesh = trimesh.creation.box()
    proc = UVProcessor(method="planar")
    mesh_uv, stats = proc.compute_uv(mesh)

    assert stats["method"] == "planar"
    assert len(mesh_uv.visual.uv) == len(mesh_uv.vertices)
    assert np.all(mesh_uv.visual.uv >= 0.0)
    assert np.all(mesh_uv.visual.uv <= 1.0)


def test_texture_optimizer_bake_and_resize():
    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = np.random.randint(50, 250, (len(mesh.vertices), 4))
    proc = UVProcessor(method="box")
    mesh, _ = proc.compute_uv(mesh)

    opt = TextureOptimizer()
    tex = opt.bake_vertex_colors(mesh, texture_size=256)

    assert isinstance(tex, Image.Image)
    assert tex.size == (256, 256)

    arr = np.asarray(tex)
    assert not np.all(arr == opt.default_color), (
        "baked texture is a flat fallback fill; original vertex colors were lost "
        "(UVProcessor.compute_uv replaces mesh.visual and must stash colors for "
        "TextureOptimizer to recover)"
    )

    resized = opt.resize_texture(tex, target_size=128)
    assert resized.size == (128, 128)


def test_texture_optimizer_blend():
    img1 = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img2 = Image.new("RGB", (64, 64), color=(0, 255, 0))

    opt = TextureOptimizer()
    blended = opt.blend_multi_textures([img1, img2], weights=[0.5, 0.5])
    arr = np.asarray(blended)

    assert arr.shape == (64, 64, 3)
    assert abs(int(arr[0, 0, 0]) - 127) <= 2
    assert abs(int(arr[0, 0, 1]) - 127) <= 2


def test_lod_generator_decimation(tmp_path):
    ico = trimesh.creation.icosphere(subdivisions=3)
    orig_faces = len(ico.faces)
    assert orig_faces > 500

    lod = LODGenerator(high_faces=1000, medium_faces=300, low_faces=100)
    dec_mesh, stats = lod.generate_tier(ico, "medium", temp_dir=tmp_path)

    assert len(dec_mesh.faces) <= 350
    assert len(dec_mesh.faces) > 50
    assert stats["tier"] == "medium"


def test_stage12_pipeline_e2e(tmp_path):
    box = trimesh.creation.box(extents=[10, 10, 2])
    box.visual.vertex_colors = np.random.randint(60, 240, (len(box.vertices), 4))
    input_obj = tmp_path / "test_input.obj"
    box.export(str(input_obj))

    output_dir = tmp_path / "textured_out"
    report = run_stage12(
        input_mesh_path=input_obj,
        output_dir=output_dir,
        quality_tier="all",
    )

    assert report["status"] == "COMPLETED"
    assert "high" in report["tiers"]
    assert "medium" in report["tiers"]
    assert "low" in report["tiers"]

    master_glb = output_dir / "master" / "high_scene.glb"
    assert master_glb.exists() and master_glb.stat().st_size > 0

    med_obj = output_dir / "medium" / "medium_scene.obj"
    assert med_obj.exists() and med_obj.stat().st_size > 0

    low_ply = output_dir / "low" / "low_scene.ply"
    assert low_ply.exists() and low_ply.stat().st_size > 0

    assert (output_dir / "stage12_report.json").exists()
