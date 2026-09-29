import re
from pathlib import Path
from src.sfm.colmap_wrapper import ColmapWrapper
from src.utils.system_info import check_colmap_installation

class MVSRunner:
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.wrapper = ColmapWrapper(config_manager=config_manager)
        if config_manager:
            self.max_image_size = config_manager.get("dense.max_image_size", 2000)
            self.patch_match_gpu = config_manager.get("dense.patch_match_gpu", True)
            self.patch_match_gpu_index = str(config_manager.get("dense.patch_match_gpu_index", "0"))
            self.patch_match_geom_consistency = config_manager.get("dense.patch_match_geom_consistency", True)
            self.patch_match_min_triangulation_angle = config_manager.get("dense.patch_match_min_triangulation_angle", 4.0)
            self.patch_match_filter_min_ncc = config_manager.get("dense.patch_match_filter_min_ncc", 0.15)
            self.patch_match_filter_geom_consistency_max_cost = config_manager.get("dense.patch_match_filter_geom_consistency_max_cost", 1.0)

            self.stereo_fusion_min_num_pixels = config_manager.get("dense.stereo_fusion_min_num_pixels", 5)
            self.stereo_fusion_max_num_pixels = config_manager.get("dense.stereo_fusion_max_num_pixels", 10000)
            self.stereo_fusion_max_reproj_error = config_manager.get("dense.stereo_fusion_max_reproj_error", 1.5)
            self.stereo_fusion_max_depth_error = config_manager.get("dense.stereo_fusion_max_depth_error", 0.005)
            self.stereo_fusion_max_normal_error = config_manager.get("dense.stereo_fusion_max_normal_error", 10.0)
            self.mesher_type = config_manager.get("dense.mesher_type", "poisson")
        else:
            self.max_image_size = 2000
            self.patch_match_gpu = True
            self.patch_match_gpu_index = "0"
            self.patch_match_geom_consistency = True
            self.patch_match_min_triangulation_angle = 4.0
            self.patch_match_filter_min_ncc = 0.15
            self.patch_match_filter_geom_consistency_max_cost = 1.0

            self.stereo_fusion_min_num_pixels = 5
            self.stereo_fusion_max_num_pixels = 10000
            self.stereo_fusion_max_reproj_error = 1.5
            self.stereo_fusion_max_depth_error = 0.005
            self.stereo_fusion_max_normal_error = 10.0
            self.mesher_type = "poisson"

    def run_image_undistorter(self, image_dir, input_model_dir, dense_dir):
        dense_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "image_undistorter",
            "--image_path", str(image_dir),
            "--input_path", str(input_model_dir),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
        ]
        if self.max_image_size and self.max_image_size > 0:
            args.extend(["--max_image_size", str(self.max_image_size)])
        return self.wrapper.run_cmd(args)

    def require_cuda(self):
        if not self.patch_match_gpu:
            raise RuntimeError(
                "dense.patch_match_gpu is false, but COLMAP PatchMatch Stereo has no CPU implementation. "
                "Dense reconstruction requires a CUDA GPU."
            )
        if not re.fullmatch(r"\d+(,\d+)*", self.patch_match_gpu_index):
            raise RuntimeError(
                f"dense.patch_match_gpu_index must be explicit CUDA device indices such as '0' or '0,1', "
                f"got '{self.patch_match_gpu_index}' (-1 would select all GPUs implicitly)."
            )
        info = check_colmap_installation()
        if not info.get("cuda_support"):
            raise RuntimeError(
                f"COLMAP at {info.get('path')} ({info.get('version')}) was built without CUDA; "
                "patch_match_stereo cannot run. Run Stage 5 on a CUDA-enabled COLMAP build (e.g. the SPIT RTX A6000 node). "
                "No depth maps or dense point cloud have been produced."
            )

    def run_patch_match_stereo(self, dense_dir):
        args = [
            "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.gpu_index", self.patch_match_gpu_index,
            "--PatchMatchStereo.geom_consistency", "true" if self.patch_match_geom_consistency else "false",
            "--PatchMatchStereo.min_triangulation_angle", str(self.patch_match_min_triangulation_angle),
            "--PatchMatchStereo.filter_min_ncc", str(self.patch_match_filter_min_ncc),
            "--PatchMatchStereo.filter_geom_consistency_max_cost", str(self.patch_match_filter_geom_consistency_max_cost),
        ]
        return self.wrapper.run_cmd(args)

    def run_stereo_fusion(self, dense_dir, output_ply_path=None):
        if output_ply_path is None:
            output_ply_path = dense_dir / "fused.ply"
        normal_err_str = str(int(self.stereo_fusion_max_normal_error)) if self.stereo_fusion_max_normal_error == int(self.stereo_fusion_max_normal_error) else str(self.stereo_fusion_max_normal_error)
        args = [
            "stereo_fusion",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(output_ply_path),
            "--StereoFusion.min_num_pixels", str(self.stereo_fusion_min_num_pixels),
            "--StereoFusion.max_num_pixels", str(self.stereo_fusion_max_num_pixels),
            "--StereoFusion.max_reproj_error", str(self.stereo_fusion_max_reproj_error),
            "--StereoFusion.max_depth_error", str(self.stereo_fusion_max_depth_error),
            "--StereoFusion.max_normal_error", normal_err_str,
        ]
        return self.wrapper.run_cmd(args)

    def run_delaunay_mesher(self, dense_dir, output_ply_path):
        args = [
            "delaunay_mesher",
            "--input_path", str(dense_dir),
            "--output_path", str(output_ply_path),
        ]
        return self.wrapper.run_cmd(args)

    def run_poisson_mesher(self, input_ply_path, output_ply_path):
        args = [
            "poisson_mesher",
            "--input_path", str(input_ply_path),
            "--output_path", str(output_ply_path),
        ]
        return self.wrapper.run_cmd(args)
