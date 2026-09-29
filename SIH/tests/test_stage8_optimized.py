"""
Focused tests for the optimized Stage 8 path (masking.reuse_stage4_poses)
added in src/masking/stage8_pipeline.py, src/masking/colmap_db_pruner.py and
src/sfm/colmap_wrapper.py's triangulate_points/supports_point_triangulator_fixed_frames.

See docs/STAGE8_OPTIMIZATION.md for the full design and validation writeup.
"""

import hashlib
import json
import os
import stat
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.config.config_manager import ConfigManager
from src.masking import stage8_pipeline
from src.sfm.colmap_wrapper import ColmapWrapper
from src.utils.system_info import check_colmap_installation


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash_dir(path):
    digest = hashlib.sha256()
    for f in sorted(Path(path).rglob("*")):
        if f.is_file():
            digest.update(f.name.encode())
            digest.update(f.read_bytes())
    return digest.hexdigest()


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


class TestDispatch:
    def test_default_config_uses_legacy_path(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        called = {}
        monkeypatch.setattr(
            stage8_pipeline, "_run_stage8_legacy",
            lambda *a, **k: called.setdefault("legacy", True) or {"status": "SUCCESS"},
        )
        monkeypatch.setattr(
            stage8_pipeline, "_run_stage8_optimized",
            lambda *a, **k: called.setdefault("optimized", True) or {"status": "SUCCESS"},
        )
        stage8_pipeline.run_stage8("dummy.mp4", output_dir=tmp_path, config_manager=cfg)
        assert called.get("legacy") is True
        assert "optimized" not in called

    def test_explicit_true_uses_optimized_path(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path, {"masking.reuse_stage4_poses": True})
        called = {}
        monkeypatch.setattr(
            stage8_pipeline, "_run_stage8_legacy",
            lambda *a, **k: called.setdefault("legacy", True) or {"status": "SUCCESS"},
        )
        monkeypatch.setattr(
            stage8_pipeline, "_run_stage8_optimized",
            lambda *a, **k: called.setdefault("optimized", True) or {"status": "SUCCESS"},
        )
        stage8_pipeline.run_stage8("dummy.mp4", output_dir=tmp_path, config_manager=cfg)
        assert called.get("optimized") is True
        assert "legacy" not in called

    def test_explicit_false_uses_legacy_path(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path, {"masking.reuse_stage4_poses": False})
        called = {}
        monkeypatch.setattr(
            stage8_pipeline, "_run_stage8_legacy",
            lambda *a, **k: called.setdefault("legacy", True) or {"status": "SUCCESS"},
        )
        stage8_pipeline.run_stage8("dummy.mp4", output_dir=tmp_path, config_manager=cfg)
        assert called.get("legacy") is True


def _make_selected_frames(output_dir, n=2):
    frames_dir = output_dir / "processed" / "selected_frames"
    frames_dir.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(
            frames_dir / f"frame_{i:04d}.jpg"
        )
    return frames_dir


def _make_tracking_outputs(output_dir, n=2):
    tracking_dir = output_dir / "tracking"
    tracking_dir.mkdir(parents=True)
    meta = [
        {"frame_index": i, "image_name": f"frame_{i:04d}.jpg", "objects": []}
        for i in range(n)
    ]
    with open(tracking_dir / "tracking_metadata.json", "w") as f:
        json.dump(meta, f)
    with open(tracking_dir / "tracks_summary.json", "w") as f:
        json.dump({}, f)


class TestOptimizedStopConditions:
    def test_missing_selected_frames_raises(self, tmp_path):
        cfg = _make_config(tmp_path, {"masking.reuse_stage4_poses": True})
        with pytest.raises(FileNotFoundError, match="Selected frames not found"):
            stage8_pipeline.run_stage8("dummy.mp4", output_dir=tmp_path, config_manager=cfg)

    def test_missing_tracking_outputs_raises(self, tmp_path):
        _make_selected_frames(tmp_path)
        cfg = _make_config(tmp_path, {"masking.reuse_stage4_poses": True})
        with pytest.raises(FileNotFoundError, match="Stage 7 tracking outputs not found"):
            stage8_pipeline.run_stage8("dummy.mp4", output_dir=tmp_path, config_manager=cfg)

    def test_missing_stage4_model_raises(self, tmp_path):
        _make_selected_frames(tmp_path)
        _make_tracking_outputs(tmp_path)
        cfg = _make_config(tmp_path, {"masking.reuse_stage4_poses": True})
        with pytest.raises(FileNotFoundError, match="Stage 4 unmasked SfM model not found"):
            stage8_pipeline.run_stage8("dummy.mp4", output_dir=tmp_path, config_manager=cfg)

    def test_missing_stage4_database_raises_explicit_stop(self, tmp_path):
        _make_selected_frames(tmp_path)
        _make_tracking_outputs(tmp_path)
        model_dir = tmp_path / "sfm" / "sparse" / "0"
        model_dir.mkdir(parents=True)
        (model_dir / "cameras.txt").write_text("# empty\n")
        (model_dir / "images.txt").write_text("# empty\n")
        (model_dir / "points3D.txt").write_text("# empty\n")
        # Deliberately do NOT create sfm/database.db

        cfg = _make_config(tmp_path, {"masking.reuse_stage4_poses": True})
        with pytest.raises(FileNotFoundError, match="STOP.*database.db"):
            stage8_pipeline.run_stage8("dummy.mp4", output_dir=tmp_path, config_manager=cfg)

    def test_zero_registered_images_raises_explicit_stop(self, tmp_path):
        _make_selected_frames(tmp_path)
        _make_tracking_outputs(tmp_path)
        model_dir = tmp_path / "sfm" / "sparse" / "0"
        model_dir.mkdir(parents=True)
        (model_dir / "cameras.txt").write_text("# empty\n")
        (model_dir / "images.txt").write_text("# empty\n")
        (model_dir / "points3D.txt").write_text("# empty\n")
        (tmp_path / "sfm" / "database.db").write_bytes(b"not a real db, just needs to exist")

        cfg = _make_config(tmp_path, {"masking.reuse_stage4_poses": True})
        with pytest.raises(RuntimeError, match="STOP.*zero registered images"):
            stage8_pipeline.run_stage8("dummy.mp4", output_dir=tmp_path, config_manager=cfg)


class _FakeColmap:
    """A fake `colmap` executable used to deterministically test the
    fix_existing_frames capability-check without needing a real COLMAP build."""

    def __init__(self, tmp_path, has_fix_existing_frames):
        self.path = tmp_path / "fake_colmap.sh"
        flag_line = "--Mapper.fix_existing_frames arg (=0)" if has_fix_existing_frames else "no such flag here"
        script = f"""#!/bin/bash
if [[ "$2" == "-h" ]]; then
  if [[ "$1" == "point_triangulator" ]]; then
    echo "{flag_line}"
  else
    echo "generic help output"
  fi
  exit 0
fi
echo "fake colmap: unsupported invocation for this test: $@" 1>&2
exit 1
"""
        self.path.write_text(script)
        self.path.chmod(self.path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


class TestPointTriangulatorCapabilityCheck:
    def test_supports_flag_true(self, tmp_path):
        fake = _FakeColmap(tmp_path, has_fix_existing_frames=True)
        wrapper = ColmapWrapper(colmap_path=fake.path)
        assert wrapper.supports_point_triangulator_fixed_frames() is True

    def test_supports_flag_false(self, tmp_path):
        fake = _FakeColmap(tmp_path, has_fix_existing_frames=False)
        wrapper = ColmapWrapper(colmap_path=fake.path)
        assert wrapper.supports_point_triangulator_fixed_frames() is False

    def test_triangulate_points_stops_explicitly_when_unsupported(self, tmp_path):
        fake = _FakeColmap(tmp_path, has_fix_existing_frames=False)
        wrapper = ColmapWrapper(colmap_path=fake.path)
        with pytest.raises(RuntimeError, match="STOP.*fix_existing_frames"):
            wrapper.triangulate_points(
                image_dir=tmp_path,
                db_path=tmp_path / "db.db",
                input_model_dir=tmp_path / "model",
                output_dir=tmp_path / "out",
                fix_existing_frames=True,
            )


# ---------------------------------------------------------------------------
# Real-COLMAP integration test. Uses the repo's own sample_drone.mp4 (already
# committed at data/input/sample_drone.mp4) sampled across its full duration,
# reproducing the exact recipe validated manually in docs/STAGE8_OPTIMIZATION.md.
# Skips cleanly if COLMAP isn't installed or the sample video is missing.
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
class TestOptimizedPathRealColmapIntegration:
    def test_optimized_path_preserves_stage4_and_produces_valid_model(self, tmp_path):
        output_dir = tmp_path / "run"
        frames_dir = _make_selected_frames(output_dir, n=0)  # create parent dirs only
        frames_dir.rmdir()
        n_frames = _extract_spread_frames(SAMPLE_VIDEO, frames_dir, n=12)
        assert n_frames >= 10, "expected the sample video to yield enough spread frames"

        cfg = _make_config(tmp_path, {
            "masking.reuse_stage4_poses": True,
            "sfm.matcher_type": "exhaustive",
            "camera.custom_calibration": {
                "fx": 1500.0, "fy": 1500.0, "cx": 960.0, "cy": 540.0,
                "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
            },
        })

        from src.sfm.stage4_pipeline import run_stage4
        stage4_report = run_stage4(str(SAMPLE_VIDEO), output_dir=output_dir, config_manager=cfg)
        assert stage4_report["registered_image_count"] > 0, (
            "Stage 4 baseline must actually register images for this test to be meaningful"
        )

        stage4_db = output_dir / "sfm" / "database.db"
        stage4_model_dir = output_dir / "sfm" / "sparse" / "0"
        db_hash_before = _hash_file(stage4_db)
        model_hash_before = _hash_dir(stage4_model_dir)

        # Synthetic Stage 7 tracking output: mark a fixed region of every frame
        # as a "dynamic object" (stand-in for a real detector/tracker, since
        # running the actual Stage 6/7 models requires downloading pretrained
        # weights that are not available in this offline test environment).
        _make_tracking_outputs(output_dir, n=n_frames)
        tracking_dir = output_dir / "tracking"
        meta = [
            {
                "frame_index": i,
                "image_name": f"frame_{i:04d}.jpg",
                "objects": [
                    {"track_id": 1, "bbox": [0.0, 0.0, 960.0, 540.0], "is_moving": True}
                ],
            }
            for i in range(n_frames)
        ]
        with open(tracking_dir / "tracking_metadata.json", "w") as f:
            json.dump(meta, f)
        with open(tracking_dir / "tracks_summary.json", "w") as f:
            json.dump({"1": {"is_dynamic": True}}, f)

        report = stage8_pipeline.run_stage8(str(SAMPLE_VIDEO), output_dir=output_dir, config_manager=cfg)

        assert report["status"] == "SUCCESS"
        assert report["method"] == "optimized_point_triangulator_reuse"

        # Stage 4 artifacts must be byte-for-byte unchanged.
        assert _hash_file(stage4_db) == db_hash_before, "Stage 4 database.db was modified!"
        assert _hash_dir(stage4_model_dir) == model_hash_before, "Stage 4 sparse model was modified!"

        # Registered image count preserved (poses reused, not re-solved).
        assert report["masked_sfm"]["registered_images"] == stage4_report["registered_image_count"]

        # Outputs are isolated from both Stage 4's sfm/ and the legacy sfm_masked/.
        opt_dir = Path(report["output_paths"]["sfm_masked_optimized_dir"])
        assert opt_dir.name == "sfm_masked_optimized"
        assert opt_dir != output_dir / "sfm"
        assert not (output_dir / "sfm_masked").exists()  # legacy path never ran

        # Some points should have been removed relative to the unmasked baseline,
        # since a real (large) region of every frame was marked dynamic.
        assert report["comparison_summary"]["points_delta"] <= 0

        # feature_extraction/matching times must be ABSENT (never run) - only
        # copy/prune/triangulate phases should be timed.
        assert "feature_extraction_seconds" not in report["timings_seconds"]
        assert "matching_seconds" not in report["timings_seconds"]
        assert "point_triangulation_seconds" in report["timings_seconds"]

    def test_legacy_path_still_works_end_to_end(self, tmp_path):
        output_dir = tmp_path / "run"
        frames_dir = _make_selected_frames(output_dir, n=0)
        frames_dir.rmdir()
        n_frames = _extract_spread_frames(SAMPLE_VIDEO, frames_dir, n=8)
        assert n_frames >= 6

        cfg = _make_config(tmp_path, {
            "masking.reuse_stage4_poses": False,
            "sfm.matcher_type": "exhaustive",
            "camera.custom_calibration": {
                "fx": 1500.0, "fy": 1500.0, "cx": 960.0, "cy": 540.0,
                "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
            },
        })

        from src.sfm.stage4_pipeline import run_stage4
        stage4_report = run_stage4(str(SAMPLE_VIDEO), output_dir=output_dir, config_manager=cfg)
        assert stage4_report["registered_image_count"] > 0

        _make_tracking_outputs(output_dir, n=n_frames)
        tracking_dir = output_dir / "tracking"
        meta = [
            {"frame_index": i, "image_name": f"frame_{i:04d}.jpg", "objects": []}
            for i in range(n_frames)
        ]
        with open(tracking_dir / "tracking_metadata.json", "w") as f:
            json.dump(meta, f)
        with open(tracking_dir / "tracks_summary.json", "w") as f:
            json.dump({}, f)

        report = stage8_pipeline.run_stage8(str(SAMPLE_VIDEO), output_dir=output_dir, config_manager=cfg)

        assert report["status"] == "SUCCESS"
        assert report["method"] == "legacy_full_colmap_rerun"
        assert "feature_extraction_seconds" in report["timings_seconds"]
        assert "matching_seconds" in report["timings_seconds"]
        assert "mapper_seconds" in report["timings_seconds"]
        assert (output_dir / "sfm_masked").exists()
        assert not (output_dir / "sfm_masked_optimized").exists()
