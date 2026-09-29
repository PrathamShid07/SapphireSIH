"""
Focused tests for the "limited pose refinement" follow-up experiment:
src/masking/pose_comparator.py, ColmapWrapper.bundle_adjust() /
parse_bundle_adjustment_report(), and the masking.refine_reused_poses config
gate in src/masking/stage8_pipeline.py.

See docs/STAGE8_OPTIMIZATION.md for the full experiment writeup and the
empirical COLMAP findings this is built on.
"""

import json
import stat
from pathlib import Path

import cv2
import pytest

from src.config.config_manager import ConfigManager
from src.masking import pose_comparator, stage8_pipeline
from src.sfm.colmap_wrapper import ColmapWrapper, parse_bundle_adjustment_report
from src.utils.system_info import check_colmap_installation


# ---------------------------------------------------------------------------
# parse_bundle_adjustment_report
# ---------------------------------------------------------------------------

class TestParseBundleAdjustmentReport:
    def test_parses_real_colmap_output(self):
        # Captured verbatim from a real COLMAP 4.2.0 bundle_adjuster run
        # (see docs/STAGE8_OPTIMIZATION.md).
        stderr_text = """
I20260928 11:09:20.220536 bundle_adjustment.cc:46] === Global bundle adjustment ===
I20260928 11:09:20.277358 bundle_adjustment_ceres.cc:1167] Bundle adjustment report
    Residuals : 4398
   Parameters : 2168
   Iterations : 101
         Time : 0.0555398 [s]
 Initial cost : 0.254696 [px]
   Final cost : 0.242732 [px]
  Termination : NO_CONVERGENCE
"""
        report = parse_bundle_adjustment_report(stderr_text)
        assert report == {
            "residuals": 4398,
            "parameters": 2168,
            "iterations": 101,
            "time_seconds": 0.0555398,
            "initial_cost_px": 0.254696,
            "final_cost_px": 0.242732,
            "termination": "NO_CONVERGENCE",
        }

    def test_parses_convergence_status(self):
        stderr_text = "Bundle adjustment report\n  Termination : CONVERGENCE\n"
        report = parse_bundle_adjustment_report(stderr_text)
        assert report["termination"] == "CONVERGENCE"

    def test_returns_none_when_no_report_present(self):
        assert parse_bundle_adjustment_report("some unrelated log output") is None
        assert parse_bundle_adjustment_report("") is None
        assert parse_bundle_adjustment_report(None) is None


# ---------------------------------------------------------------------------
# pose_comparator
# ---------------------------------------------------------------------------

class TestPoseComparator:
    def test_identical_poses_have_zero_delta(self):
        images = [
            {"name": "a.jpg", "qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [1.0, 2.0, 3.0]},
            {"name": "b.jpg", "qvec": [0.0, 1.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]},
        ]
        result = pose_comparator.compare_poses(images, images)
        assert result["summary"]["max_translation_delta"] == 0.0
        assert result["summary"]["max_rotation_delta_degrees"] == 0.0
        assert result["summary"]["num_compared"] == 2

    def test_translation_only_change(self):
        ref = [{"name": "a.jpg", "qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]}]
        moved = [{"name": "a.jpg", "qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [3.0, 4.0, 0.0]}]
        result = pose_comparator.compare_poses(ref, moved)
        assert result["per_image"][0]["translation_delta"] == pytest.approx(5.0)
        assert result["per_image"][0]["rotation_delta_degrees"] == pytest.approx(0.0)

    def test_rotation_only_change_90_degrees(self):
        import math
        # 90 degree rotation about Z: qw=cos(45deg), qz=sin(45deg)
        half = math.radians(45)
        ref = [{"name": "a.jpg", "qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]}]
        rotated = [{
            "name": "a.jpg",
            "qvec": [math.cos(half), 0.0, 0.0, math.sin(half)],
            "tvec": [0.0, 0.0, 0.0],
        }]
        result = pose_comparator.compare_poses(ref, rotated)
        assert result["per_image"][0]["rotation_delta_degrees"] == pytest.approx(90.0, abs=1e-6)

    def test_quaternion_sign_ambiguity_is_zero_delta(self):
        # q and -q represent the identical rotation in COLMAP's convention.
        ref = [{"name": "a.jpg", "qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]}]
        negated = [{"name": "a.jpg", "qvec": [-1.0, 0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]}]
        result = pose_comparator.compare_poses(ref, negated)
        assert result["per_image"][0]["rotation_delta_degrees"] == pytest.approx(0.0, abs=1e-6)

    def test_reports_images_present_in_only_one_model(self):
        ref = [
            {"name": "a.jpg", "qvec": [1, 0, 0, 0], "tvec": [0, 0, 0]},
            {"name": "b.jpg", "qvec": [1, 0, 0, 0], "tvec": [0, 0, 0]},
        ]
        cmp_ = [{"name": "a.jpg", "qvec": [1, 0, 0, 0], "tvec": [0, 0, 0]}]
        result = pose_comparator.compare_poses(ref, cmp_)
        assert result["summary"]["num_compared"] == 1
        assert result["summary"]["only_in_reference"] == ["b.jpg"]
        assert result["summary"]["only_in_comparison"] == []


# ---------------------------------------------------------------------------
# bundle_adjust capability wiring (fake colmap, no real COLMAP needed)
# ---------------------------------------------------------------------------

class _FakeColmapBA:
    """Fake colmap binary that only needs to answer bundle_adjuster calls
    for capability-check-style tests (not used for the real integration test)."""

    def __init__(self, tmp_path):
        self.path = tmp_path / "fake_colmap_ba.sh"
        script = """#!/bin/bash
if [[ "$2" == "-h" ]]; then
  echo "generic help output"
  exit 0
fi
if [[ "$1" == "bundle_adjuster" ]]; then
  echo "some info" 1>&2
  echo "Bundle adjustment report" 1>&2
  echo "    Residuals : 100" 1>&2
  echo "   Parameters : 50" 1>&2
  echo "   Iterations : 10" 1>&2
  echo "         Time : 0.01 [s]" 1>&2
  echo " Initial cost : 1.0 [px]" 1>&2
  echo "   Final cost : 0.5 [px]" 1>&2
  echo "  Termination : CONVERGENCE" 1>&2
  exit 0
fi
echo "fake colmap: unsupported invocation: $@" 1>&2
exit 1
"""
        self.path.write_text(script)
        self.path.chmod(self.path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


class TestBundleAdjustWrapper:
    def test_bundle_adjust_returns_parsed_diagnostics(self, tmp_path):
        fake = _FakeColmapBA(tmp_path)
        wrapper = ColmapWrapper(colmap_path=fake.path)
        stdout, stderr, diagnostics = wrapper.bundle_adjust(
            input_model_dir=tmp_path / "in", output_dir=tmp_path / "out"
        )
        assert diagnostics["termination"] == "CONVERGENCE"
        assert diagnostics["residuals"] == 100
        assert "Bundle adjustment report" in stderr

    def test_bundle_adjust_defaults_never_refine_intrinsics(self, tmp_path, monkeypatch):
        fake = _FakeColmapBA(tmp_path)
        wrapper = ColmapWrapper(colmap_path=fake.path)

        captured_args = {}
        original_run = wrapper.run_cmd_capture_stderr

        def spy(args):
            captured_args["args"] = args
            return original_run(args)

        monkeypatch.setattr(wrapper, "run_cmd_capture_stderr", spy)
        wrapper.bundle_adjust(input_model_dir=tmp_path / "in", output_dir=tmp_path / "out")

        args = captured_args["args"]
        assert "--BundleAdjustment.refine_focal_length" in args
        idx = args.index("--BundleAdjustment.refine_focal_length")
        assert args[idx + 1] == "0"
        idx = args.index("--BundleAdjustment.refine_principal_point")
        assert args[idx + 1] == "0"
        idx = args.index("--BundleAdjustment.refine_extra_params")
        assert args[idx + 1] == "0"
        # poses ARE refined by default
        idx = args.index("--BundleAdjustment.refine_rig_from_world")
        assert args[idx + 1] == "1"

    def test_bundle_adjust_can_explicitly_enable_intrinsics_refinement(self, tmp_path, monkeypatch):
        fake = _FakeColmapBA(tmp_path)
        wrapper = ColmapWrapper(colmap_path=fake.path)
        captured_args = {}
        original_run = wrapper.run_cmd_capture_stderr

        def spy(args):
            captured_args["args"] = args
            return original_run(args)

        monkeypatch.setattr(wrapper, "run_cmd_capture_stderr", spy)
        wrapper.bundle_adjust(
            input_model_dir=tmp_path / "in",
            output_dir=tmp_path / "out",
            refine_focal_length=True,
        )
        args = captured_args["args"]
        idx = args.index("--BundleAdjustment.refine_focal_length")
        assert args[idx + 1] == "1"


# ---------------------------------------------------------------------------
# Config gating: refine_reused_poses only takes effect when
# reuse_stage4_poses is also true, and produces isolated output.
# ---------------------------------------------------------------------------

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


class TestRefinementConfigGating:
    def test_refine_flag_ignored_when_reuse_stage4_poses_false(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path, {
            "masking.reuse_stage4_poses": False,
            "masking.refine_reused_poses": True,
        })
        called = {}
        monkeypatch.setattr(
            stage8_pipeline, "_run_stage8_legacy",
            lambda *a, **k: called.setdefault("legacy", True) or {"status": "SUCCESS"},
        )
        stage8_pipeline.run_stage8("dummy.mp4", output_dir=tmp_path, config_manager=cfg)
        # refine_reused_poses must never route to the optimized path on its own
        assert called.get("legacy") is True

    def test_default_refine_flag_is_false(self):
        cfg = ConfigManager()
        assert cfg.get("masking.refine_reused_poses", None) is False


# ---------------------------------------------------------------------------
# Real-COLMAP integration test for the refined variant. Reuses the same
# recipe validated in tests/test_stage8_optimized.py and
# docs/STAGE8_OPTIMIZATION.md.
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


@pytest.mark.skipif(_skip_reason is not None, reason=str(_skip_reason))
class TestRefinedPathRealColmapIntegration:
    def test_refined_path_never_touches_intrinsics_and_produces_isolated_output(self, tmp_path):
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
        stage4_intrinsics = stage4_report["camera_params_prior"]

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
        # NOTE: real COLMAP's bundle_adjuster convergence is empirically
        # nondeterministic run-to-run on this exact benchmark (see
        # docs/STAGE8_OPTIMIZATION.md section 9's follow-up validation-gate
        # experiment) - a bundle_adjuster call was made and diagnostics/pose
        # deltas were computed either way, but whether the validation gate
        # ACCEPTS or REJECTS/marks it UNAVAILABLE is not asserted here; that
        # decision logic itself is covered exhaustively in
        # tests/test_stage8_refinement_validation.py. This test only checks
        # invariants that must hold regardless of the gate's outcome.
        assert report["pose_refinement_diagnostics"] is not None
        assert "termination" in report["pose_refinement_diagnostics"]
        assert report["pose_change_vs_stage4"] is not None
        assert report["refinement_validation"] is not None
        assert report["refinement_validation"]["decision"] in {
            "REFINEMENT_ACCEPTED", "REFINEMENT_REJECTED", "REFINEMENT_UNAVAILABLE",
        }
        if report["refinement_validation"]["decision"] == "REFINEMENT_ACCEPTED":
            assert report["method"] == "optimized_point_triangulator_reuse_refined"
        else:
            assert report["method"] == "optimized_point_triangulator_reuse"

        # Output isolated from both the legacy and fixed-pose optimized paths.
        opt_dir = Path(report["output_paths"]["sfm_masked_optimized_dir"])
        assert opt_dir.name == "sfm_masked_optimized_refined"
        assert not (output_dir / "sfm_masked").exists()
        assert not (output_dir / "sfm_masked_optimized").exists()

        # Intrinsics must be byte-identical to Stage 4's - bundle_adjuster was
        # run with all three refine_* intrinsics flags off. This holds
        # regardless of convergence: intrinsics preservation is enforced by
        # the flags passed to bundle_adjuster, not by whether it converged.
        from src.sfm.colmap_parser import parse_colmap_model
        refined_model = parse_colmap_model(opt_dir / "sparse_refined")
        stage4_model = parse_colmap_model(output_dir / "sfm" / "sparse" / "0")
        assert refined_model["cameras"][0]["params"] == stage4_model["cameras"][0]["params"]

        assert "pose_refinement_seconds" in report["timings_seconds"]
        assert "feature_extraction_seconds" not in report["timings_seconds"]
        assert "matching_seconds" not in report["timings_seconds"]
