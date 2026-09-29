import numpy as np
from pathlib import Path
import pytest
import trimesh
from src.dense.mvs_runner import MVSRunner
from src.dense.point_cloud_cleaner import clean_point_cloud, write_ply_points, read_ply_points
from src.dense.mesh_processor import process_mesh_and_export
from src.utils.system_info import check_colmap_installation
from src.config.config_manager import ConfigManager

def test_mvs_runner_init(tmp_path):
    cfg = ConfigManager()
    col_info = check_colmap_installation()
    if col_info["available"]:
        runner = MVSRunner(config_manager=cfg)
        assert runner.wrapper.colmap_exe is not None
        assert runner.stereo_fusion_min_num_pixels == 5
        assert runner.patch_match_geom_consistency is True
        assert runner.stereo_fusion_max_reproj_error == 1.5
    else:
        dummy = tmp_path / "colmap"
        dummy.touch()
        from src.sfm.colmap_wrapper import ColmapWrapper
        runner = MVSRunner.__new__(MVSRunner)
        runner.config_manager = cfg
        runner.wrapper = ColmapWrapper(colmap_path=dummy)
        runner.max_image_size = 2000
        runner.patch_match_gpu = True
        runner.patch_match_geom_consistency = True
        runner.patch_match_min_triangulation_angle = 4.0
        runner.patch_match_filter_min_ncc = 0.15
        runner.patch_match_filter_geom_consistency_max_cost = 1.0
        runner.stereo_fusion_min_num_pixels = 5
        runner.stereo_fusion_max_num_pixels = 10000
        runner.stereo_fusion_max_reproj_error = 1.5
        runner.stereo_fusion_max_depth_error = 0.005
        runner.stereo_fusion_max_normal_error = 10.0
        assert runner.stereo_fusion_min_num_pixels == 5

def test_point_cloud_cleaner(tmp_path):
    in_ply = tmp_path / "raw.ply"
    out_ply = tmp_path / "clean.ply"
    
    np.random.seed(42)
    pts = np.random.normal(0, 1, (200, 3))
    outliers = np.array([[50, 50, 50], [-50, -50, -50], [100, 100, 100]])
    all_pts = np.vstack([pts, outliers])
    
    write_ply_points(in_ply, all_pts)
    
    raw_cnt, clean_cnt, removed_cnt = clean_point_cloud(
        input_ply_path=in_ply,
        output_ply_path=out_ply,
        nb_neighbors=10,
        std_ratio=1.5
    )
    
    assert raw_cnt == 203
    assert clean_cnt < raw_cnt
    assert removed_cnt > 0
    assert out_ply.exists()

def test_mesh_processor(tmp_path):
    mesh_in = tmp_path / "input_mesh.ply"
    out_dir = tmp_path / "mesh_out"
    
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh.export(str(mesh_in))
    
    stats = process_mesh_and_export(input_mesh_path=mesh_in, output_dir=out_dir)
    
    assert stats["vertex_count"] > 0
    assert stats["face_count"] > 0
    assert Path(stats["exported_files"]["ply"]).exists()
    assert Path(stats["exported_files"]["obj"]).exists()
    assert Path(stats["exported_files"]["glb"]).exists()
