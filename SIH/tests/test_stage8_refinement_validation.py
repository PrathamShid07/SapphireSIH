"""
Tests for the Stage 8 refinement validation/acceptance gate:
src/masking/refinement_validator.py, and its integration into
src/masking/stage8_pipeline.py's optimized-refined path.

See docs/STAGE8_OPTIMIZATION.md section 9 for the empirical evidence this
gate's design is based on (a converged, healthy case vs. a NO_CONVERGENCE,
~51-degree-divergent case).
"""

import json
from pathlib import Path

import cv2
import pytest

from src.config.config_manager import ConfigManager
from src.masking import refinement_validator, stage8_pipeline
from src.sfm.colmap_wrapper import ColmapWrapper
from src.utils.system_info import check_colmap_installation


IDENTITY_QVEC = (1.0, 0.0, 0.0, 0.0)
DEFAULT_CAMERA_PARAMS = (1500.0, 1500.0, 960.0, 540.0, 0.0, 0.0, 0.0, 0.0)


def _write_text_model(model_dir, camera_params=DEFAULT_CAMERA_PARAMS, images=None, points=None):
    """
    Writes a minimal COLMAP text-format model, matching the exact format
    src.sfm.colmap_parser.parse_colmap_model expects (same convention already
    used by tests/test_stage8.py and tests/test_stage4.py).

    images: list of dicts {id, qvec(4), tvec(3), name}
    points: list of tuples (x, y, z, error)
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    cam_line = f"1 OPENCV 1920 1080 {' '.join(str(p) for p in camera_params)}\n"
    (model_dir / "cameras.txt").write_text("# cameras\n" + cam_line, encoding="utf-8")

    images = images or []
    img_lines = []
    for img in images:
        q = img["qvec"]
        t = img["tvec"]
        img_lines.append(
            f"{img['id']} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {img['name']}\n\n"
        )
    (model_dir / "images.txt").write_text("# images\n" + "".join(img_lines), encoding="utf-8")

    points = points or []
    pts_lines = []
    for i, p in enumerate(points, start=1):
        pts_lines.append(f"{i} {p[0]} {p[1]} {p[2]} 255 0 0 {p[3]}\n")
    (model_dir / "points3D.txt").write_text("# points\n" + "".join(pts_lines), encoding="utf-8")

    return model_dir


def _make_images(n, tvec_offset=(0.0, 0.0, 0.0), qvec=IDENTITY_QVEC):
    return [
        {
            "id": i + 1,
            "qvec": qvec,
            "tvec": (float(i) + tvec_offset[0], tvec_offset[1], tvec_offset[2]),
            "name": f"frame_{i:04d}.jpg",
        }
        for i in range(n)
    ]


def _make_points(n, error=0.3):
    return [(float(i), float(i), 5.0, error) for i in range(n)]


def _healthy_ba_diagnostics(termination="CONVERGENCE", initial=0.30, final=0.28):
    return {
        "residuals": 1000, "parameters": 500, "iterations": 10,
        "time_seconds": 0.02, "initial_cost_px": initial, "final_cost_px": final,
        "termination": termination,
    }


def _pose_delta(max_t, max_r):
    return {
        "per_image": [],
        "summary": {
            "num_compared": 3, "only_in_reference": [], "only_in_comparison": [],
            "max_translation_delta": max_t, "mean_translation_delta": max_t / 2,
            "max_rotation_delta_degrees": max_r, "mean_rotation_delta_degrees": max_r / 2,
        },
    }


class TestSuccessfulConvergence:
    def test_healthy_refinement_is_accepted(self, tmp_path):
        stage4_dir = _write_text_model(tmp_path / "stage4", images=_make_images(3), points=_make_points(10))
        fixed_dir = _write_text_model(tmp_path / "fixed", images=_make_images(3), points=_make_points(10, error=0.30))
        refined_dir = _write_text_model(
            tmp_path / "refined",
            images=_make_images(3, tvec_offset=(0.01, 0.0, 0.0)),
            points=_make_points(10, error=0.28),
        )
        result = refinement_validator.validate_refinement(
            fixed_model_dir=fixed_dir,
            refined_model_dir=refined_dir,
            stage4_model_dir=stage4_dir,
            ba_diagnostics=_healthy_ba_diagnostics(),
            pose_delta_report=_pose_delta(max_t=0.02, max_r=0.3),
        )
        assert result["decision"] == refinement_validator.REFINEMENT_ACCEPTED
        assert result["checks"]["bundle_adjuster_converged"]["passed"] is True
        assert result["checks"]["intrinsics_unchanged"]["passed"] is True
        assert result["checks"]["registered_image_count_consistent"]["passed"] is True
        assert result["checks"]["sparse_point_count_consistent"]["passed"] is True
        assert result["fixed_path_metrics"]["registered_images"] == 3
        assert result["refined_path_metrics"]["registered_images"] == 3

    def test_diagnostic_only_checks_never_gate_a_healthy_result(self, tmp_path):
        # Even though pose deltas and reprojection delta are recorded, they
        # must not carry a "passed" verdict that could fail an otherwise-clean result.
        stage4_dir = _write_text_model(tmp_path / "stage4", images=_make_images(2), points=_make_points(5))
        fixed_dir = _write_text_model(tmp_path / "fixed", images=_make_images(2), points=_make_points(5))
        refined_dir = _write_text_model(tmp_path / "refined", images=_make_images(2), points=_make_points(5))
        result = refinement_validator.validate_refinement(
            fixed_model_dir=fixed_dir, refined_model_dir=refined_dir, stage4_model_dir=stage4_dir,
            ba_diagnostics=_healthy_ba_diagnostics(),
            pose_delta_report=_pose_delta(max_t=0.0, max_r=0.0),
        )
        assert result["checks"]["max_translation_change"]["evidence_based"] is False
        assert result["checks"]["max_translation_change"]["passed"] is None
        assert result["checks"]["max_rotation_change_degrees"]["passed"] is None
        assert result["checks"]["reprojection_error_before_after"]["passed"] is None


class TestNoConvergence:
    def test_no_convergence_is_rejected(self, tmp_path):
        stage4_dir = _write_text_model(tmp_path / "stage4", images=_make_images(3), points=_make_points(10))
        fixed_dir = _write_text_model(tmp_path / "fixed", images=_make_images(3), points=_make_points(10))
        refined_dir = _write_text_model(tmp_path / "refined", images=_make_images(3), points=_make_points(10))
        result = refinement_validator.validate_refinement(
            fixed_model_dir=fixed_dir, refined_model_dir=refined_dir, stage4_model_dir=stage4_dir,
            ba_diagnostics=_healthy_ba_diagnostics(termination="NO_CONVERGENCE"),
            pose_delta_report=_pose_delta(max_t=0.02, max_r=0.3),
        )
        assert result["decision"] == refinement_validator.REFINEMENT_REJECTED
        assert result["checks"]["bundle_adjuster_converged"]["passed"] is False
        assert "NO_CONVERGENCE" in result["reason"]


class TestExtremePoseDivergence:
    def test_extreme_divergence_is_rejected_via_termination_not_magnitude(self, tmp_path):
        """
        Reproduces the empirically observed failure mode from
        docs/STAGE8_OPTIMIZATION.md section 9.2: extreme pose divergence
        (~51 degrees) co-occurred with NO_CONVERGENCE. The gate rejects this
        via the termination check (evidence-based), NOT by applying an
        invented numeric threshold to the pose-delta magnitude itself - the
        pose-delta checks remain diagnostic-only (passed=None) even here.
        """
        stage4_dir = _write_text_model(tmp_path / "stage4", images=_make_images(3), points=_make_points(10))
        fixed_dir = _write_text_model(tmp_path / "fixed", images=_make_images(3), points=_make_points(10))
        refined_dir = _write_text_model(
            tmp_path / "refined",
            images=_make_images(3, tvec_offset=(4.0, 0.0, 0.0)),
            points=_make_points(10),
        )
        result = refinement_validator.validate_refinement(
            fixed_model_dir=fixed_dir, refined_model_dir=refined_dir, stage4_model_dir=stage4_dir,
            ba_diagnostics=_healthy_ba_diagnostics(termination="NO_CONVERGENCE"),
            pose_delta_report=_pose_delta(max_t=3.99, max_r=51.2),
        )
        assert result["decision"] == refinement_validator.REFINEMENT_REJECTED
        # The large numbers are recorded...
        assert result["checks"]["max_translation_change"]["value"] == 3.99
        assert result["checks"]["max_rotation_change_degrees"]["value"] == 51.2
        # ...but did NOT themselves gate the decision - termination did.
        assert result["checks"]["max_translation_change"]["passed"] is None
        assert result["checks"]["max_rotation_change_degrees"]["passed"] is None
        assert result["checks"]["bundle_adjuster_converged"]["passed"] is False

    def test_non_finite_pose_delta_is_rejected_as_invalid(self, tmp_path):
        stage4_dir = _write_text_model(tmp_path / "stage4", images=_make_images(2), points=_make_points(5))
        fixed_dir = _write_text_model(tmp_path / "fixed", images=_make_images(2), points=_make_points(5))
        refined_dir = _write_text_model(tmp_path / "refined", images=_make_images(2), points=_make_points(5))
        result = refinement_validator.validate_refinement(
            fixed_model_dir=fixed_dir, refined_model_dir=refined_dir, stage4_model_dir=stage4_dir,
            ba_diagnostics=_healthy_ba_diagnostics(),
            pose_delta_report=_pose_delta(max_t=float("nan"), max_r=0.1),
        )
        assert result["decision"] == refinement_validator.REFINEMENT_REJECTED
        assert result["checks"]["pose_deltas_finite"]["passed"] is False


class TestMissingModelOutput:
    def test_missing_output_files_is_unavailable(self, tmp_path):
        stage4_dir = _write_text_model(tmp_path / "stage4", images=_make_images(2), points=_make_points(5))
        fixed_dir = _write_text_model(tmp_path / "fixed", images=_make_images(2), points=_make_points(5))
        empty_refined_dir = tmp_path / "refined_empty"
        empty_refined_dir.mkdir()  # no cameras/images/points3D files at all

        result = refinement_validator.validate_refinement(
            fixed_model_dir=fixed_dir, refined_model_dir=empty_refined_dir, stage4_model_dir=stage4_dir,
            ba_diagnostics=_healthy_ba_diagnostics(),
            pose_delta_report=_pose_delta(max_t=0.01, max_r=0.1),
        )
        assert result["decision"] == refinement_validator.REFINEMENT_UNAVAILABLE
        assert result["checks"]["output_files_present"]["passed"] is False
        assert result["refined_path_metrics"] is None

    def test_bundle_adjust_command_failure_is_unavailable(self, tmp_path):
        stage4_dir = _write_text_model(tmp_path / "stage4", images=_make_images(2), points=_make_points(5))
        fixed_dir = _write_text_model(tmp_path / "fixed", images=_make_images(2), points=_make_points(5))
        result = refinement_validator.validate_refinement(
            fixed_model_dir=fixed_dir, refined_model_dir=tmp_path / "never_created",
            stage4_model_dir=stage4_dir,
            ba_diagnostics=None, pose_delta_report=None,
            bundle_adjust_error="COLMAP command failed: bundle_adjuster exit code 1",
        )
        assert result["decision"] == refinement_validator.REFINEMENT_UNAVAILABLE
        assert "bundle_adjuster command failed" in result["reason"]
        # Fixed-path metrics must still be reported even when refinement is unavailable.
        assert result["fixed_path_metrics"]["registered_images"] == 2

    def test_unparseable_diagnostics_is_unavailable(self, tmp_path):
        stage4_dir = _write_text_model(tmp_path / "stage4", images=_make_images(2), points=_make_points(5))
        fixed_dir = _write_text_model(tmp_path / "fixed", images=_make_images(2), points=_make_points(5))
        refined_dir = _write_text_model(tmp_path / "refined", images=_make_images(2), points=_make_points(5))
        result = refinement_validator.validate_refinement(
            fixed_model_dir=fixed_dir, refined_model_dir=refined_dir, stage4_model_dir=stage4_dir,
            ba_diagnostics=None,  # e.g. "Bundle adjustment report" not found in COLMAP output
            pose_delta_report=_pose_delta(max_t=0.01, max_r=0.1),
        )
        assert result["decision"] == refinement_validator.REFINEMENT_UNAVAILABLE


class TestIntrinsicsChanged:
    def test_changed_intrinsics_is_rejected(self, tmp_path):
        stage4_dir = _write_text_model(tmp_path / "stage4", images=_make_images(2), points=_make_points(5))
        fixed_dir = _write_text_model(tmp_path / "fixed", images=_make_images(2), points=_make_points(5))
        changed_params = (1520.0, 1520.0, 960.0, 540.0, 0.0, 0.0, 0.0, 0.0)  # focal length drifted
        refined_dir = _write_text_model(
            tmp_path / "refined", camera_params=changed_params, images=_make_images(2), points=_make_points(5)
        )
        result = refinement_validator.validate_refinement(
            fixed_model_dir=fixed_dir, refined_model_dir=refined_dir, stage4_model_dir=stage4_dir,
            ba_diagnostics=_healthy_ba_diagnostics(),
            pose_delta_report=_pose_delta(max_t=0.01, max_r=0.1),
        )
        assert result["decision"] == refinement_validator.REFINEMENT_REJECTED
        assert result["checks"]["intrinsics_unchanged"]["passed"] is False
        assert "intrinsics changed" in result["reason"]


class TestInvalidEmptyReconstruction:
    def test_empty_refined_reconstruction_is_rejected(self, tmp_path):
        stage4_dir = _write_text_model(tmp_path / "stage4", images=_make_images(3), points=_make_points(10))
        fixed_dir = _write_text_model(tmp_path / "fixed", images=_make_images(3), points=_make_points(10))
        # Files exist (passes criterion 9) but contain zero images/points.
        refined_dir = _write_text_model(tmp_path / "refined", images=[], points=[])
        result = refinement_validator.validate_refinement(
            fixed_model_dir=fixed_dir, refined_model_dir=refined_dir, stage4_model_dir=stage4_dir,
            ba_diagnostics=_healthy_ba_diagnostics(),
            pose_delta_report=_pose_delta(max_t=0.0, max_r=0.0),
        )
        assert result["decision"] == refinement_validator.REFINEMENT_REJECTED
        assert result["checks"]["model_valid"]["passed"] is False
        assert result["checks"]["registered_image_count_consistent"]["passed"] is False

    def test_cost_increase_is_rejected(self, tmp_path):
        # A converged BA that somehow reports a higher final cost than initial
        # cost is internally inconsistent - reject even if termination says CONVERGENCE.
        stage4_dir = _write_text_model(tmp_path / "stage4", images=_make_images(2), points=_make_points(5))
        fixed_dir = _write_text_model(tmp_path / "fixed", images=_make_images(2), points=_make_points(5))
        refined_dir = _write_text_model(tmp_path / "refined", images=_make_images(2), points=_make_points(5))
        result = refinement_validator.validate_refinement(
            fixed_model_dir=fixed_dir, refined_model_dir=refined_dir, stage4_model_dir=stage4_dir,
            ba_diagnostics=_healthy_ba_diagnostics(termination="CONVERGENCE", initial=0.2, final=0.5),
            pose_delta_report=_pose_delta(max_t=0.01, max_r=0.1),
        )
        assert result["decision"] == refinement_validator.REFINEMENT_REJECTED
        assert result["checks"]["cost_decreased_or_equal"]["passed"] is False


# ---------------------------------------------------------------------------
# Integration: successful fallback to the fixed-pose result when
# bundle_adjuster fails, using real COLMAP for everything except the
# (monkeypatched) bundle_adjuster failure itself.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_VIDEO = REPO_ROOT / "data" / "input" / "sample_drone.mp4"

_colmap_info = check_colmap_installation()
_skip_reason = None
if not _colmap_info["available"]:
    _skip_reason = "COLMAP is not installed on this machine"
elif not SAMPLE_VIDEO.exists():
    _skip_reason = f"sample video not found at {SAMPLE_VIDEO}"


def _extract_spread_frames(video_path, frames_dir, n=12):
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(total * i / n) for i in range(n)]
    count = 0
    for target in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(frames_dir / f"frame_{count:04d}.jpg"), frame)
            count += 1
    cap.release()
    return count


def _make_config(tmp_path, overrides=None):
    cfg = ConfigManager()
    cfg.root_dir = tmp_path
    cfg.config["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    cfg.config["paths"]["logs_dir"] = str(tmp_path / "logs")
    cfg.config["execution"]["resume"] = False
    if overrides:
        for dotted, value in overrides.items():
            keys = dotted.split(".")
            d = cfg.config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
    return cfg


@pytest.mark.skipif(_skip_reason is not None, reason=str(_skip_reason))
class TestGracefulFallbackRealColmapIntegration:
    def test_bundle_adjuster_failure_falls_back_to_fixed_pose_result(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "run"
        frames_dir = output_dir / "processed" / "selected_frames"
        n_frames = _extract_spread_frames(SAMPLE_VIDEO, frames_dir, n=12)
        assert n_frames >= 10

        cfg = _make_config(tmp_path, {
            "masking.reuse_stage4_poses": True,
            "masking.refine_reused_poses": True,
            "sfm.matcher_type": "exhaustive",
            "camera.custom_calibration": {
                "fx": 1500.0, "fy": 1500.0, "cx": 960.0, "cy": 540.0,
                "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
            },
        })

        from src.sfm.stage4_pipeline import run_stage4
        stage4_report = run_stage4(str(SAMPLE_VIDEO), output_dir=output_dir, config_manager=cfg)
        assert stage4_report["registered_image_count"] > 0

        tracking_dir = output_dir / "tracking"
        tracking_dir.mkdir(parents=True)
        meta = [
            {
                "frame_index": i,
                "image_name": f"frame_{i:04d}.jpg",
                "objects": [{"track_id": 1, "bbox": [0.0, 0.0, 640.0, 540.0], "is_moving": True}],
            }
            for i in range(n_frames)
        ]
        with open(tracking_dir / "tracking_metadata.json", "w") as f:
            json.dump(meta, f)
        with open(tracking_dir / "tracks_summary.json", "w") as f:
            json.dump({"1": {"is_dynamic": True}}, f)

        # Force bundle_adjuster to fail, simulating a COLMAP-level failure,
        # while feature reuse/pruning/triangulation all run for real.
        def failing_bundle_adjust(self, *a, **k):
            raise RuntimeError("COLMAP command failed: simulated bundle_adjuster crash")

        monkeypatch.setattr(ColmapWrapper, "bundle_adjust", failing_bundle_adjust)

        report = stage8_pipeline.run_stage8(str(SAMPLE_VIDEO), output_dir=output_dir, config_manager=cfg)

        assert report["status"] == "SUCCESS"
        assert report["method"] == "optimized_point_triangulator_reuse"  # fallback, not "_refined"
        assert report["refinement_validation"]["decision"] == refinement_validator.REFINEMENT_UNAVAILABLE
        assert "simulated bundle_adjuster crash" in report["refinement_validation"]["reason"]
        assert report["recomputed"]["pose_refinement_requested"] is True
        assert report["recomputed"]["pose_refinement_accepted"] is False

        # The fixed-pose model must still be present and used as the actual result.
        fixed_dir = Path(report["output_paths"]["fixed_pose_model_dir"])
        assert fixed_dir.exists()
        assert report["output_paths"]["final_model_dir_used"] == str(fixed_dir)
        assert report["masked_sfm"]["registered_images"] == report["refinement_validation"]["fixed_path_metrics"]["registered_images"]
        assert report["masked_sfm"]["registered_images"] == stage4_report["registered_image_count"]

    def test_real_colmap_refinement_gate_produces_consistent_outcome(self, tmp_path):
        """
        Companion positive-path test using the SAME realistic scenario as
        docs/STAGE8_OPTIMIZATION.md section 9.3's production benchmark
        (which observed CONVERGENCE). NOTE: real COLMAP's bundle_adjuster
        convergence has since been observed to be nondeterministic run-to-run
        even on this exact benchmark (see the sibling test in
        test_stage8_pose_refinement.py) - Ceres's BA solver has no fixed
        seed here, so a hard "must be ACCEPTED" assertion would be flaky.
        This test instead checks that whichever decision the gate reaches is
        handled consistently (i.e. the invariants of that decision hold),
        which is what actually matters for correctness. The purely
        deterministic ACCEPTED-path assertions live in
        TestSuccessfulConvergence above, using synthetic diagnostics.
        """
        output_dir = tmp_path / "run"
        frames_dir = output_dir / "processed" / "selected_frames"
        n_frames = _extract_spread_frames(SAMPLE_VIDEO, frames_dir, n=15)
        assert n_frames >= 10

        cfg = _make_config(tmp_path, {
            "masking.reuse_stage4_poses": True,
            "masking.refine_reused_poses": True,
            "sfm.matcher_type": "exhaustive",
            "camera.custom_calibration": {
                "fx": 1500.0, "fy": 1500.0, "cx": 960.0, "cy": 540.0,
                "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
            },
        })

        from src.sfm.stage4_pipeline import run_stage4
        stage4_report = run_stage4(str(SAMPLE_VIDEO), output_dir=output_dir, config_manager=cfg)
        assert stage4_report["registered_image_count"] > 0

        tracking_dir = output_dir / "tracking"
        tracking_dir.mkdir(parents=True)
        meta = [
            {
                "frame_index": i,
                "image_name": f"frame_{i:04d}.jpg",
                "objects": [{"track_id": 1, "bbox": [0.0, 0.0, 640.0, 540.0], "is_moving": True}],
            }
            for i in range(n_frames)
        ]
        with open(tracking_dir / "tracking_metadata.json", "w") as f:
            json.dump(meta, f)
        with open(tracking_dir / "tracks_summary.json", "w") as f:
            json.dump({"1": {"is_dynamic": True}}, f)

        report = stage8_pipeline.run_stage8(str(SAMPLE_VIDEO), output_dir=output_dir, config_manager=cfg)

        assert report["status"] == "SUCCESS"
        decision = report["refinement_validation"]["decision"]
        assert decision in {
            refinement_validator.REFINEMENT_ACCEPTED,
            refinement_validator.REFINEMENT_REJECTED,
            refinement_validator.REFINEMENT_UNAVAILABLE,
        }

        fixed_dir = Path(report["output_paths"]["fixed_pose_model_dir"])
        assert fixed_dir.exists()  # fixed-pose result always preserved regardless of decision

        if decision == refinement_validator.REFINEMENT_ACCEPTED:
            assert report["method"] == "optimized_point_triangulator_reuse_refined"
            assert report["recomputed"]["pose_refinement_accepted"] is True
            refined_dir = Path(report["output_paths"]["refined_model_dir"])
            assert report["output_paths"]["final_model_dir_used"] == str(refined_dir)
            assert fixed_dir != refined_dir
        else:
            assert report["method"] == "optimized_point_triangulator_reuse"
            assert report["recomputed"]["pose_refinement_accepted"] is False
            assert report["output_paths"]["final_model_dir_used"] == str(fixed_dir)
            # Attempted refinement's own diagnostics must still be recorded for review.
            assert report["pose_refinement_diagnostics"] is not None
