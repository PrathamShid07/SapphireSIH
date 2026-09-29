import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from src.utils.system_info import check_colmap_installation

logger = logging.getLogger("colmap_wrapper")


def parse_bundle_adjustment_report(stderr_text):
    """
    Parse COLMAP's own "Bundle adjustment report" block out of bundle_adjuster's
    stderr, e.g.:

        Bundle adjustment report
            Residuals : 4398
           Parameters : 2168
           Iterations : 101
                 Time : 0.057025 [s]
         Initial cost : 0.254696 [px]
           Final cost : 0.242732 [px]
          Termination : NO_CONVERGENCE

    Returns a dict with the parsed fields, or None if no such block is found
    (e.g. bundle_adjuster failed before reaching that stage). This is a
    direct, literal parse of COLMAP's own diagnostic text - it does not
    compute, infer, or threshold anything itself.
    """
    if not stderr_text or "Bundle adjustment report" not in stderr_text:
        return None

    patterns = {
        "residuals": r"Residuals\s*:\s*(\d+)",
        "parameters": r"Parameters\s*:\s*(\d+)",
        "iterations": r"Iterations\s*:\s*(\d+)",
        "time_seconds": r"Time\s*:\s*([\d.]+)",
        "initial_cost_px": r"Initial cost\s*:\s*([\d.]+)",
        "final_cost_px": r"Final cost\s*:\s*([\d.]+)",
        "termination": r"Termination\s*:\s*(\S+)",
    }
    report = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stderr_text)
        if not match:
            continue
        value = match.group(1)
        if key == "termination":
            report[key] = value
        elif key in ("time_seconds", "initial_cost_px", "final_cost_px"):
            report[key] = float(value)
        else:
            report[key] = int(value)
    return report if report else None


class ColmapWrapper:
    def __init__(self, config_manager=None, colmap_path=None):
        self.config_manager = config_manager
        if colmap_path:
            self.colmap_exe = str(colmap_path)
        else:
            col_info = check_colmap_installation()
            if not col_info["available"]:
                raise RuntimeError("COLMAP executable not found in system or C:\\COLMAP")
            self.colmap_exe = col_info["path"]

        if config_manager:
            self.matcher_type = config_manager.get("sfm.matcher_type", "sequential")
            self.camera_model = config_manager.get("sfm.camera_model", "OPENCV")
            self.single_camera = config_manager.get("sfm.single_camera", True)
            self.use_gpu = config_manager.get("sfm.use_gpu", True)
            self.resume = config_manager.get("execution.resume", True)
            self.init_min_tri_angle = config_manager.get("sfm.mapper.init_min_tri_angle")
            self.init_min_num_inliers = config_manager.get("sfm.mapper.init_min_num_inliers")
            gpu_index = config_manager.get("sfm.gpu_index")
            self.gpu_index = None if gpu_index is None else str(gpu_index)
            self.sift_max_num_features = config_manager.get("sfm.features.max_num_features")
            self.sequential_overlap = config_manager.get("sfm.sequential.overlap")
        else:
            self.matcher_type = "sequential"
            self.camera_model = "OPENCV"
            self.single_camera = True
            self.use_gpu = True
            self.resume = True
            self.init_min_tri_angle = None
            self.init_min_num_inliers = None
            self.gpu_index = None
            self.sift_max_num_features = None
            self.sequential_overlap = None

        if self.gpu_index is not None and not re.fullmatch(r"-?\d+(,\d+)*", self.gpu_index):
            raise ValueError(f"sfm.gpu_index must be a CUDA device index or comma-separated list, got '{self.gpu_index}'")

        matcher_sub = "sequential_matcher" if self.matcher_type == "sequential" else "exhaustive_matcher"
        self.extractor_gpu_prefix = self._first_supported_prefix(
            "feature_extractor", ("--FeatureExtraction", "--SiftExtraction"), "use_gpu")
        self.matcher_gpu_prefix = self._first_supported_prefix(
            matcher_sub, ("--FeatureMatching", "--SiftMatching"), "use_gpu")
        self._has_sift_gpu_extractor_flag = self.extractor_gpu_prefix is not None
        self._has_sift_gpu_matcher_flag = self.matcher_gpu_prefix is not None
        self.gpu_flags_resolved = bool(self.extractor_gpu_prefix and self.matcher_gpu_prefix)
        if self.use_gpu and not self.gpu_flags_resolved:
            logger.warning(
                "sfm.use_gpu is true but this COLMAP build exposes no recognised GPU flag for "
                "feature extraction and/or matching; those steps will use COLMAP's own default device."
            )

    def _first_supported_prefix(self, subcommand, prefixes, option):
        for prefix in prefixes:
            if self._check_help(subcommand, f"{prefix}.{option}"):
                return prefix
        return None

    def gpu_args(self, prefix):
        if not self.use_gpu or prefix is None:
            return []
        args = [f"{prefix}.use_gpu", "1"]
        if self.gpu_index is not None:
            args.extend([f"{prefix}.gpu_index", self.gpu_index])
        return args

    def _check_help(self, subcommand, flag_name):
        try:
            res = subprocess.run([self.colmap_exe, subcommand, "-h"], capture_output=True, text=True, timeout=5)
            output = (res.stdout or "") + (res.stderr or "")
            return flag_name in output
        except Exception:
            return False

    def run_cmd(self, args):
        cmd = [self.colmap_exe] + [str(a) for a in args]
        logger.info(f"Executing COLMAP command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            err_msg = res.stderr or res.stdout
            raise RuntimeError(f"COLMAP command failed: {' '.join(cmd)}\nError: {err_msg}")
        return res.stdout

    def run_cmd_capture_stderr(self, args):
        """
        Like run_cmd, but also returns stderr. COLMAP's glog-based diagnostics
        (e.g. bundle_adjuster's convergence report: residuals, initial/final
        cost, termination status) are written to stderr even on success, which
        run_cmd() discards. Needed by bundle_adjust() to surface that report
        rather than silently losing it.
        """
        cmd = [self.colmap_exe] + [str(a) for a in args]
        logger.info(f"Executing COLMAP command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            err_msg = res.stderr or res.stdout
            raise RuntimeError(f"COLMAP command failed: {' '.join(cmd)}\nError: {err_msg}")
        return res.stdout, res.stderr

    def extract_features(self, image_dir, db_path, mask_path=None, camera_params=None, image_names=None):
        args = [
            "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(image_dir),
            "--ImageReader.camera_model", self.camera_model,
            "--ImageReader.single_camera", "1" if self.single_camera else "0",
        ]
        if camera_params:
            args.extend(["--ImageReader.camera_params", str(camera_params)])
        args.extend(self.gpu_args(self.extractor_gpu_prefix))
        if self.sift_max_num_features is not None:
            flag = "--SiftExtraction.max_num_features"
            if not self._check_help("feature_extractor", flag):
                raise RuntimeError(f"sfm.features.max_num_features is set but this COLMAP build does not expose {flag}.")
            args.extend([flag, str(int(self.sift_max_num_features))])
        if mask_path is not None:
            args.extend(["--ImageReader.mask_path", str(mask_path)])
        if image_names is not None:
            list_path = Path(db_path).parent / "image_list.txt"
            list_path.write_text("\n".join(image_names) + "\n", encoding="utf-8")
            if self._check_help("feature_extractor", "--ImageReader.image_list_path"):
                flag = "--ImageReader.image_list_path"
            elif self._check_help("feature_extractor", "--image_list_path"):
                flag = "--image_list_path"
            else:
                raise RuntimeError(
                    "This COLMAP build's feature_extractor has no image list option; cannot restrict "
                    "feature extraction to the manifest-eligible reconstruction frames."
                )
            args.extend([flag, str(list_path)])
        return self.run_cmd(args)

    def match_features(self, db_path):
        matcher_cmd = "sequential_matcher" if self.matcher_type == "sequential" else "exhaustive_matcher"
        args = [
            matcher_cmd,
            "--database_path", str(db_path),
        ]
        args.extend(self.gpu_args(self.matcher_gpu_prefix))
        if self.matcher_type == "sequential" and self.sequential_overlap is not None:
            flag = "--SequentialMatching.overlap"
            if not self._check_help("sequential_matcher", flag):
                raise RuntimeError(f"sfm.sequential.overlap is set but this COLMAP build does not expose {flag}.")
            args.extend([flag, str(int(self.sequential_overlap))])
        return self.run_cmd(args)

    def map_sparse(self, image_dir, db_path, sparse_dir):
        sparse_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "mapper",
            "--database_path", str(db_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
        ]
        args.extend(self.mapper_init_args())
        return self.run_cmd(args)

    def mapper_init_args(self):
        args = []
        for value, flag in (
            (self.init_min_tri_angle, "--Mapper.init_min_tri_angle"),
            (self.init_min_num_inliers, "--Mapper.init_min_num_inliers"),
        ):
            if value is None:
                continue
            if not self._check_help("mapper", flag):
                raise RuntimeError(
                    f"sfm.mapper is configured to set {flag}={value} but this COLMAP build's mapper does not expose it."
                )
            args.extend([flag, str(value)])
        return args

    def supports_point_triangulator_fixed_frames(self):
        """
        Whether this COLMAP build's `point_triangulator` exposes
        `--Mapper.fix_existing_frames`, which is required to safely reuse an
        existing model's camera poses as fixed geometry (see
        docs/STAGE8_OPTIMIZATION.md). Without this flag, point_triangulator's
        internal bundle-adjustment pass could silently re-optimize poses,
        invalidating the "reuse Stage 4 poses as-is" assumption the optimized
        Stage 8 path depends on.
        """
        return self._check_help("point_triangulator", "--Mapper.fix_existing_frames")

    def triangulate_points(
        self,
        image_dir,
        db_path,
        input_model_dir,
        output_dir,
        fix_existing_frames=True,
        refine_intrinsics=False,
    ):
        """
        Run COLMAP's point_triangulator: re-triangulate 3D points from the
        matches in db_path using the camera poses/intrinsics already present
        in input_model_dir, WITHOUT running feature_extractor, a matcher, or
        the incremental `mapper`. Images are matched to the existing model by
        filename (COLMAP's own `--clear_points` default behavior).

        Raises RuntimeError up front (refuses to silently proceed) if
        fix_existing_frames=True is requested but this COLMAP build doesn't
        support the required flag.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if fix_existing_frames and not self.supports_point_triangulator_fixed_frames():
            raise RuntimeError(
                "STOP: this COLMAP build's `point_triangulator` does not expose "
                "--Mapper.fix_existing_frames. Reusing Stage 4's camera poses as fixed "
                "geometry cannot be guaranteed safe without this flag - point_triangulator's "
                "internal bundle adjustment could silently move poses on older COLMAP builds. "
                "Refusing to proceed with an unverified workaround. Upgrade COLMAP, or set "
                "masking.reuse_stage4_poses: false to use the legacy Stage 8 path."
            )

        args = [
            "point_triangulator",
            "--database_path", str(db_path),
            "--image_path", str(image_dir),
            "--input_path", str(input_model_dir),
            "--output_path", str(output_dir),
            "--refine_intrinsics", "1" if refine_intrinsics else "0",
        ]
        if fix_existing_frames:
            args.extend(["--Mapper.fix_existing_frames", "1"])
        return self.run_cmd(args)

    def bundle_adjust(
        self,
        input_model_dir,
        output_dir,
        refine_poses=True,
        refine_points=True,
        refine_focal_length=False,
        refine_principal_point=False,
        refine_extra_params=False,
    ):
        """
        Run COLMAP's bundle_adjuster on an EXISTING model (poses + points),
        refining it in place (into output_dir). Does not touch a database and
        does not triangulate new points from raw matches - it only refines
        what is already in input_model_dir.

        Camera intrinsics (focal length, principal point, distortion/"extra
        params") default to NOT being refined - they are only touched if the
        caller explicitly passes refine_focal_length/refine_principal_point/
        refine_extra_params=True. This mirrors COLMAP's own
        `--BundleAdjustment.refine_*` flags directly; nothing is inferred or
        defaulted to "on" behind the caller's back.

        Returns (stdout, stderr, diagnostics) where diagnostics is the parsed
        "Bundle adjustment report" block (residuals, parameters, iterations,
        initial/final cost, termination status) if COLMAP printed one, else
        None. Termination status is COLMAP's own convergence verdict - this
        method does not invent or apply any additional accept/reject
        threshold; it only surfaces what COLMAP itself reported.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "bundle_adjuster",
            "--input_path", str(input_model_dir),
            "--output_path", str(output_dir),
            "--BundleAdjustment.refine_rig_from_world", "1" if refine_poses else "0",
            "--BundleAdjustment.refine_points3D", "1" if refine_points else "0",
            "--BundleAdjustment.refine_focal_length", "1" if refine_focal_length else "0",
            "--BundleAdjustment.refine_principal_point", "1" if refine_principal_point else "0",
            "--BundleAdjustment.refine_extra_params", "1" if refine_extra_params else "0",
        ]
        stdout, stderr = self.run_cmd_capture_stderr(args)
        diagnostics = parse_bundle_adjustment_report(stderr)
        return stdout, stderr, diagnostics

    def export_ply(self, input_model_dir, output_ply_path):
        args = [
            "model_converter",
            "--input_path", str(input_model_dir),
            "--output_path", str(output_ply_path),
            "--output_type", "PLY",
        ]
        return self.run_cmd(args)

    def export_txt(self, input_model_dir, output_txt_dir):
        output_txt_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "model_converter",
            "--input_path", str(input_model_dir),
            "--output_path", str(output_txt_dir),
            "--output_type", "TXT",
        ]
        return self.run_cmd(args)
