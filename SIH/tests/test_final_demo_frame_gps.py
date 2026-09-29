import json
import stat
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from src.config.config_manager import ConfigManager
from src.preprocessing.video_metadata import extract_video_metadata, probe_decoded_frames
from src.preprocessing.stage2_pipeline import run_stage2
from src.metadata.stage3_pipeline import run_stage3
from src.metadata.telemetry_parser import TelemetryParser, validate_telemetry_records
from src.metadata.telemetry_association import associate_frames_with_telemetry, classify_unmatched_telemetry
from src.metadata.camera_model import CameraIntrinsics
from src.metadata.frame_synchronizer import FrameSynchronizer
from src.sfm.colmap_wrapper import ColmapWrapper
from src.dense.mvs_runner import MVSRunner
from src.dense.point_cloud_cleaner import write_ply_points
from src.utils.geometry_checks import check_ply_geometry, compute_pca_point_cloud
from src.utils.reconstruction_gate import (
    ReconstructionGateError,
    resolve_reconstruction_inputs,
    assert_masks_approved,
    assert_stage8_allowed,
    join_poses_with_manifest,
)
from tests.test_stage2 import create_synthetic_video

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL_DEMO_VIDEO = REPO_ROOT / "data" / "input" / "final_demo.mp4"
FINAL_DEMO_CSV = REPO_ROOT / "data" / "input" / "final_demo.csv"

CSV_HEADER = "frame_index,video_time_s,timestamp,drone_lat,drone_lon,drone_altitude_m\n"


def write_final_demo_style_csv(path, rows=12, dt=0.0411, bad_gps_rows=()):
    lines = [CSV_HEADER]
    for i in range(rows):
        lat = "" if i in bad_gps_rows else f"{55.9431 + i * 1e-6:.6f}"
        lines.append(
            f"{i},{i * dt:.6f},2026-09-28 12:00:{i * dt:09.6f},{lat},-3.187300,{80.0 + i * 0.05:.6f}\n"
        )
    Path(path).write_text("".join(lines), encoding="utf-8")


def final_demo_config(tmp_path, **overrides):
    cfg = ConfigManager(config_path=REPO_ROOT / "config" / "final_demo.yaml", root_dir=tmp_path)
    cfg.config["input"]["telemetry_csv"] = None
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        node = cfg.config
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def test_probe_decoded_frames_synthetic(tmp_path):
    video = tmp_path / "v.mp4"
    create_synthetic_video(video, num_frames=20, fps=10.0)
    probe = probe_decoded_frames(video)
    assert probe["decoded_frame_count"] == 20
    assert probe["timestamps_valid"] is True
    assert probe["decoded_timestamps_sec"][1] == pytest.approx(0.1, abs=1e-3)
    assert probe["decoded_fps"] == pytest.approx(10.0, abs=0.1)
    meta = extract_video_metadata(video)
    assert meta["container_frame_count"] == meta["decoded_frame_count"] == 20
    assert meta["container_frame_count_mismatch"] is False


@pytest.mark.real_data
@pytest.mark.skipif(not FINAL_DEMO_VIDEO.exists(), reason="final_demo.mp4 not present")
def test_final_demo_container_metadata_is_stale():
    meta = extract_video_metadata(FINAL_DEMO_VIDEO)
    assert meta["container_frame_count"] == 97
    assert meta["decoded_frame_count"] == 75
    assert meta["container_frame_count_mismatch"] is True
    assert meta["container_vs_decoded_frame_delta"] == 22
    assert meta["decoded_fps"] == pytest.approx(25.0, abs=0.1)
    assert meta["decoded_last_timestamp_sec"] == pytest.approx(2.96, abs=1e-3)
    assert meta["timestamps_monotonic"] is True


def test_parser_reads_final_demo_schema(tmp_path):
    write_final_demo_style_csv(tmp_path / "clip.csv", rows=5, bad_gps_rows=(3,))
    res = TelemetryParser(video_path=tmp_path / "clip.mp4").extract()
    assert res["telemetry_detected"] is True
    assert res["gps_available"] is True
    recs = res["records"]
    assert len(recs) == 5
    assert recs[1]["frame_index"] == 1
    assert recs[1]["video_time_s"] == pytest.approx(0.0411)
    assert recs[1]["timestamp"] == pytest.approx(0.0411)
    assert recs[1]["latitude"] == pytest.approx(55.943101)
    assert recs[1]["altitude"] == pytest.approx(80.05)
    assert recs[1]["wall_clock_epoch_s"] is not None
    assert recs[1]["gps_valid"] is True
    assert recs[3]["gps_valid"] is False
    assert "invalid_latitude" in recs[3]["gps_invalid_reasons"]


def test_parser_legacy_schema_still_supported(tmp_path):
    (tmp_path / "gps.csv").write_text(
        "time_sec,latitude,longitude,altitude\n0.0,12.9,77.5,900\n1.0,12.91,77.51,901\n", encoding="utf-8"
    )
    res = TelemetryParser(input_dir=tmp_path).extract()
    assert [r["timestamp"] for r in res["records"]] == [0.0, 1.0]
    assert all(r["gps_valid"] for r in res["records"])


def test_parser_surfaces_parse_failure_instead_of_silently_dropping(tmp_path):
    (tmp_path / "telemetry.json").write_text("{not json", encoding="utf-8")
    res = TelemetryParser(input_dir=tmp_path).extract()
    assert res["telemetry_detected"] is False
    assert "parse_error" in res
    assert "could not be parsed" in res["reason"]


def test_parser_prefers_video_named_sidecar_over_generic_gps_csv(tmp_path):
    (tmp_path / "gps.csv").write_text("time_sec,latitude,longitude,altitude\n0.0,1.0,1.0,1.0\n", encoding="utf-8")
    write_final_demo_style_csv(tmp_path / "clip.csv", rows=3)
    res = TelemetryParser(video_path=tmp_path / "clip.mp4").extract()
    assert res["source_file"].endswith("clip.csv")


def test_validate_telemetry_records_reports_gaps_and_duplicates():
    recs = [
        {"row_number": 0, "frame_index": 0, "video_time_s": 0.0, "gps_valid": True},
        {"row_number": 1, "frame_index": 1, "video_time_s": 0.04, "gps_valid": True},
        {"row_number": 2, "frame_index": 1, "video_time_s": 0.03, "gps_valid": False},
        {"row_number": 3, "frame_index": 5, "video_time_s": 0.2, "gps_valid": True},
    ]
    v = validate_telemetry_records(recs)
    assert v["duplicate_frame_indices"] == [1]
    assert v["missing_frame_indices"] == [2, 3, 4]
    assert v["video_time_s_strictly_increasing"] is False
    assert v["gps_invalid_row_numbers"] == [2]
    assert v["gps_valid_row_count"] == 3


def make_records(n, dt=0.0411, t0=0.0):
    return [
        {
            "row_number": i,
            "frame_index": i,
            "video_time_s": t0 + i * dt,
            "timestamp": t0 + i * dt,
            "latitude": 55.0 + i * 1e-6,
            "longitude": -3.0,
            "altitude": 80.0 + i,
            "gps_valid": True,
            "gps_invalid_reasons": [],
        }
        for i in range(n)
    ]


def make_frames(n, dt=0.04):
    return [{"frame_id": f"f{i}", "decoded_frame_index": i, "video_time_s": i * dt} for i in range(n)]


def test_association_matches_by_time_not_row_number():
    records = make_records(97)
    frames = make_frames(75)
    assoc, used = associate_frames_with_telemetry(frames, records, max_time_diff_s=0.05, frame_index_tolerance=1)
    last = assoc[74]
    assert last["telemetry_row_number"] == 72
    assert last["telemetry_row_number"] != last["decoded_frame_index"]
    assert last["telemetry_match_time_diff_s"] < 0.005
    assert last["frame_index_delta"] == 2
    assert last["frame_index_consistent"] is False
    assert assoc[0]["telemetry_row_number"] == 0
    assert assoc[0]["frame_index_consistent"] is True
    assert all(a["telemetry_valid"] for a in assoc)
    assert max(a["telemetry_match_time_diff_s"] for a in assoc) <= 0.0206


def test_trailing_telemetry_rows_are_not_associated_with_nonexistent_frames():
    records = make_records(97)
    frames = make_frames(75)
    assoc, used = associate_frames_with_telemetry(frames, records, max_time_diff_s=0.05)
    assert max(used) == 72
    unmatched = classify_unmatched_telemetry(records, used, (0.0, 2.96))
    beyond = [u["row_number"] for u in unmatched if u["reason"] == "beyond_decoded_video_range"]
    assert beyond == list(range(73, 97))
    assert len(unmatched) == 24


def test_time_tolerance_rejects_association():
    records = make_records(5, t0=1.0)
    assoc, used = associate_frames_with_telemetry(make_frames(3), records, max_time_diff_s=0.05)
    assert all(a["telemetry_valid"] is False for a in assoc)
    assert all("time_diff_exceeds_tolerance" in a["telemetry_invalid_reasons"] for a in assoc)
    assert all(a["latitude"] is None for a in assoc)
    assert used == {}


def test_invalid_gps_row_marks_frame_invalid_and_hides_coordinates():
    records = make_records(6)
    records[2]["gps_valid"] = False
    records[2]["gps_invalid_reasons"] = ["invalid_latitude"]
    records[2]["latitude"] = None
    assoc, _ = associate_frames_with_telemetry(make_frames(6), records, max_time_diff_s=0.05)
    bad = [a for a in assoc if a["telemetry_row_number"] == 2]
    assert bad and all(a["telemetry_valid"] is False for a in bad)
    assert all("invalid_latitude" in a["telemetry_invalid_reasons"] for a in bad)


def test_missing_telemetry_yields_invalid_frames():
    assoc, used = associate_frames_with_telemetry(make_frames(3), [], max_time_diff_s=0.05)
    assert all(a["telemetry_valid"] is False for a in assoc)
    assert used == {}


def test_shared_telemetry_rows_are_flagged():
    records = make_records(97)
    assoc, used = associate_frames_with_telemetry(make_frames(75), records, max_time_diff_s=0.05)
    shared = [a for a in assoc if a["telemetry_row_shared_with_other_frame"]]
    assert len(shared) == sum(c for c in used.values() if c > 1)
    assert len(used) == 73


def run_stage2_and_3(tmp_path, cfg, frames=20, fps=10.0, telemetry_rows=30, dt=0.1, bad_gps_rows=()):
    video = tmp_path / "clip.mp4"
    create_synthetic_video(video, num_frames=frames, fps=fps)
    write_final_demo_style_csv(tmp_path / "clip.csv", rows=telemetry_rows, dt=dt, bad_gps_rows=bad_gps_rows)
    out_dir = tmp_path / "out"
    run_stage2(video, output_dir=out_dir, config_manager=cfg)
    res = run_stage3(video, output_dir=out_dir, config_manager=cfg)
    return out_dir, res


def test_stage2_reconstruction_frames_use_decoded_frames_and_timestamps(tmp_path):
    cfg = final_demo_config(tmp_path)
    cfg.config["preprocessing"]["target_fps"] = 2.0
    out_dir, _ = run_stage2_and_3(tmp_path, cfg)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    recon = manifest["reconstruction_frames"]
    assert len(recon) == manifest["video_metadata"]["decoded_frame_count"] == 20
    assert [r["decoded_frame_index"] for r in recon] == list(range(20))
    assert recon[7]["video_time_s"] == pytest.approx(0.7, abs=1e-3)
    assert recon[7]["timestamp_source"] == "decoder_pos_msec"
    assert all(Path(r["image_path"]).exists() for r in recon)
    assert manifest["sampling"]["total_video_frames"] == 20
    assert manifest["sampling"]["reconstruction_frames_count"] == 20
    assert len(manifest["frames"]) < len(recon)


def test_stage2_default_strategy_writes_no_reconstruction_frames(tmp_path):
    video = tmp_path / "clip.mp4"
    create_synthetic_video(video, num_frames=20, fps=10.0)
    cfg = ConfigManager(root_dir=tmp_path)
    manifest = run_stage2(video, output_dir=tmp_path / "out", config_manager=cfg)
    assert manifest["reconstruction_frames"] == []
    assert not (tmp_path / "out" / "processed" / "reconstruction_frames").exists()


def test_stage2_stride_subsamples_reconstruction_frames(tmp_path):
    video = tmp_path / "clip.mp4"
    create_synthetic_video(video, num_frames=20, fps=10.0)
    cfg = final_demo_config(tmp_path, **{"preprocessing.reconstruction.stride": 4})
    manifest = run_stage2(video, output_dir=tmp_path / "out", config_manager=cfg)
    assert [r["decoded_frame_index"] for r in manifest["reconstruction_frames"]] == [0, 4, 8, 12, 16]


def test_stage2_reruns_when_reconstruction_strategy_changes(tmp_path):
    video = tmp_path / "clip.mp4"
    create_synthetic_video(video, num_frames=20, fps=10.0)
    out = tmp_path / "out"
    first = run_stage2(video, output_dir=out, config_manager=ConfigManager(root_dir=tmp_path))
    assert first["reconstruction_frames"] == []
    second = run_stage2(video, output_dir=out, config_manager=final_demo_config(tmp_path))
    assert len(second["reconstruction_frames"]) == 20


def test_reconstruction_manifest_end_to_end_with_trailing_telemetry(tmp_path):
    cfg = final_demo_config(tmp_path)
    out_dir, res = run_stage2_and_3(tmp_path, cfg, frames=20, fps=10.0, telemetry_rows=30, dt=0.1)
    manifest_path = Path(res["output_paths"]["reconstruction_manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["frames"]
    summary = manifest["summary"]

    assert len(rows) == 20
    required = {
        "frame_index", "video_time_s", "image_path", "latitude", "longitude", "altitude_m",
        "telemetry_match_time_diff_s", "telemetry_valid", "reconstruction_eligible",
        "local_enu_east_m", "local_enu_north_m", "local_enu_up_m",
    }
    assert required <= set(rows[0].keys())
    assert all(r["reconstruction_eligible"] for r in rows)
    assert rows[5]["frame_index"] == 5
    assert rows[5]["telemetry_row_number"] == 5
    assert rows[5]["latitude"] == pytest.approx(55.943105)
    assert rows[5]["altitude_m"] == pytest.approx(80.25)
    assert rows[5]["telemetry_match_time_diff_s"] < 0.01
    assert summary["decoded_frame_count"] == 20
    assert summary["telemetry_row_count"] == 30
    assert summary["telemetry_rows_unassociated_count"] == 10
    assert {u["reason"] for u in summary["telemetry_rows_unassociated"]} == {"beyond_decoded_video_range"}
    assert not any(r["frame_index"] >= 20 for r in rows)
    assert (manifest_path.parent / "reconstruction_manifest.csv").exists()

    flight = json.loads((manifest_path.parent / "flight_metadata.json").read_text())
    assert flight["telemetry_association"]["telemetry_rows_beyond_decoded_video"] == 10
    telem_status = json.loads((manifest_path.parent / "telemetry_status.json").read_text())
    assert telem_status["validation"]["row_count"] == 30


def test_frames_without_valid_gps_are_not_reconstruction_eligible(tmp_path):
    cfg = final_demo_config(tmp_path)
    out_dir, res = run_stage2_and_3(tmp_path, cfg, telemetry_rows=30, dt=0.1, bad_gps_rows=(4, 5))
    rows = json.loads(Path(res["output_paths"]["reconstruction_manifest_path"]).read_text())["frames"]
    ineligible = [r["frame_index"] for r in rows if not r["reconstruction_eligible"]]
    assert ineligible == [4, 5]
    assert all(r["latitude"] is None for r in rows if r["frame_index"] in (4, 5))

    inputs = resolve_reconstruction_inputs(out_dir, cfg)
    assert "recon_00004.jpg" not in inputs["image_names"]
    assert len(inputs["image_names"]) == 18
    assert inputs["restrict_to_names"] is True
    assert [i["frame_index"] for i in inputs["ineligible"]] == [4, 5]


def test_gps_not_required_keeps_frames_eligible(tmp_path):
    cfg = final_demo_config(tmp_path, **{"telemetry.require_valid_for_reconstruction": False})
    out_dir, res = run_stage2_and_3(tmp_path, cfg, telemetry_rows=30, dt=0.1, bad_gps_rows=(4,))
    rows = json.loads(Path(res["output_paths"]["reconstruction_manifest_path"]).read_text())["frames"]
    assert all(r["reconstruction_eligible"] for r in rows)
    assert rows[4]["telemetry_valid"] is False


def test_gate_requires_manifest_for_decoded_stride(tmp_path):
    cfg = final_demo_config(tmp_path)
    (tmp_path / "processed" / "reconstruction_frames").mkdir(parents=True)
    with pytest.raises(ReconstructionGateError, match="Reconstruction manifest not found"):
        resolve_reconstruction_inputs(tmp_path, cfg)


def test_gate_rejects_missing_eligible_images_and_empty_manifest(tmp_path):
    cfg = final_demo_config(tmp_path)
    meta = tmp_path / "metadata"
    meta.mkdir()
    (tmp_path / "processed" / "reconstruction_frames").mkdir(parents=True)
    (meta / "reconstruction_manifest.json").write_text(json.dumps({"summary": {}, "frames": [
        {"image_name": "recon_00000.jpg", "reconstruction_eligible": True, "frame_index": 0}
    ]}))
    with pytest.raises(ReconstructionGateError, match="missing"):
        resolve_reconstruction_inputs(tmp_path, cfg)
    (meta / "reconstruction_manifest.json").write_text(json.dumps({"summary": {}, "frames": [
        {"image_name": "recon_00000.jpg", "reconstruction_eligible": False, "frame_index": 0, "telemetry_invalid_reasons": ["missing_gps"]}
    ]}))
    with pytest.raises(ReconstructionGateError, match="No reconstruction-eligible"):
        resolve_reconstruction_inputs(tmp_path, cfg)


def test_gate_selected_keyframes_strategy_is_legacy_behaviour(tmp_path):
    cfg = ConfigManager(root_dir=tmp_path)
    d = tmp_path / "processed" / "selected_frames"
    d.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(d / "selected_00001.jpg")
    inputs = resolve_reconstruction_inputs(tmp_path, cfg)
    assert inputs["image_dir"] == d
    assert inputs["restrict_to_names"] is False
    assert inputs["image_names"] == ["selected_00001.jpg"]


def test_mask_gate_blocks_unapproved_masks():
    cfg_final = ConfigManager(config_path=REPO_ROOT / "config" / "final_demo.yaml")
    with pytest.raises(ReconstructionGateError, match="not approved"):
        assert_masks_approved(cfg_final, Path("/some/masks"))
    assert_masks_approved(cfg_final, None)
    with pytest.raises(ReconstructionGateError, match="Stage 8"):
        assert_stage8_allowed(cfg_final)
    cfg_default = ConfigManager()
    assert_masks_approved(cfg_default, Path("/some/masks"))
    assert_stage8_allowed(cfg_default)


def test_stage8_refuses_to_run_for_final_demo(tmp_path):
    from src.masking.stage8_pipeline import run_stage8
    cfg = final_demo_config(tmp_path)
    with pytest.raises(ReconstructionGateError):
        run_stage8("final_demo.mp4", output_dir=tmp_path, config_manager=cfg)


def test_stage8_refuses_when_reconstruction_frames_differ_from_keyframes():
    cfg = ConfigManager()
    cfg.config["preprocessing"]["reconstruction"]["strategy"] = "decoded_stride"
    with pytest.raises(ReconstructionGateError, match="not be comparable"):
        assert_stage8_allowed(cfg)


def test_join_poses_with_manifest_reports_unregistered():
    manifest = {"frames": [
        {"image_name": "a.jpg", "frame_index": 0, "video_time_s": 0.0, "latitude": 1.0, "longitude": 2.0, "altitude_m": 3.0,
         "reconstruction_eligible": True},
        {"image_name": "b.jpg", "frame_index": 1, "video_time_s": 0.04, "latitude": 1.1, "longitude": 2.1, "altitude_m": 3.1,
         "reconstruction_eligible": True},
    ]}
    joined = join_poses_with_manifest([{"id": 7, "name": "a.jpg", "position_xyz": [0, 0, 0]}, {"id": 8, "name": "zzz.jpg"}], manifest)
    assert joined["registered"][0]["latitude"] == 1.0
    assert joined["registered"][0]["frame_index"] == 0
    assert joined["registered"][1]["in_manifest"] is False
    assert joined["eligible_but_unregistered"] == ["b.jpg"]


def make_fake_colmap(tmp_path, help_text):
    exe = tmp_path / "colmap"
    exe.write_text("#!/bin/sh\n" + "".join(f"echo '{line}'\n" for line in help_text.split("\n")))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_final_demo_config_values_are_explicit_and_dataset_specific():
    final = ConfigManager(config_path=REPO_ROOT / "config" / "final_demo.yaml")
    default = ConfigManager()
    assert final.get("sfm.mapper.init_min_tri_angle") == 5.0
    assert final.get("preprocessing.reconstruction.strategy") == "decoded_stride"
    assert final.get("preprocessing.reconstruction.stride") == 1
    assert final.get("telemetry.require_valid_for_reconstruction") is True
    assert final.get("masking.approved_for_reconstruction") is False
    assert final.get("paths.logs_dir") == "logs/final_demo"
    assert final.get("sfm.camera_model") == default.get("sfm.camera_model")
    assert default.get("sfm.mapper.init_min_tri_angle") is None
    assert default.get("preprocessing.reconstruction.strategy") == "selected_keyframes"
    assert default.get("masking.approved_for_reconstruction") is True


def test_mapper_init_args_only_emitted_when_configured(tmp_path):
    exe = make_fake_colmap(tmp_path, "--Mapper.init_min_tri_angle arg (=16)\n--Mapper.init_min_num_inliers arg (=100)")
    default_wrapper = ColmapWrapper(config_manager=ConfigManager(), colmap_path=exe)
    assert default_wrapper.mapper_init_args() == []
    final_wrapper = ColmapWrapper(config_manager=ConfigManager(config_path=REPO_ROOT / "config" / "final_demo.yaml"), colmap_path=exe)
    assert final_wrapper.mapper_init_args() == ["--Mapper.init_min_tri_angle", "5.0"]


def test_mapper_init_args_fail_loudly_when_flag_unsupported(tmp_path):
    exe = make_fake_colmap(tmp_path, "no relevant flags here")
    wrapper = ColmapWrapper(config_manager=ConfigManager(config_path=REPO_ROOT / "config" / "final_demo.yaml"), colmap_path=exe)
    with pytest.raises(RuntimeError, match="init_min_tri_angle"):
        wrapper.mapper_init_args()


def test_map_sparse_passes_initializer_threshold(tmp_path, monkeypatch):
    exe = make_fake_colmap(tmp_path, "--Mapper.init_min_tri_angle arg (=16)")
    wrapper = ColmapWrapper(config_manager=ConfigManager(config_path=REPO_ROOT / "config" / "final_demo.yaml"), colmap_path=exe)
    captured = {}
    monkeypatch.setattr(wrapper, "run_cmd", lambda args: captured.setdefault("args", list(args)))
    wrapper.map_sparse(tmp_path / "img", tmp_path / "db.db", tmp_path / "sparse")
    args = captured["args"]
    assert args[0] == "mapper"
    assert args[args.index("--Mapper.init_min_tri_angle") + 1] == "5.0"


def test_extract_features_restricts_to_manifest_names(tmp_path, monkeypatch):
    exe = make_fake_colmap(tmp_path, "--ImageReader.image_list_path arg")
    wrapper = ColmapWrapper(config_manager=ConfigManager(), colmap_path=exe)
    captured = {}
    monkeypatch.setattr(wrapper, "run_cmd", lambda args: captured.setdefault("args", list(args)))
    db = tmp_path / "sfm" / "database.db"
    db.parent.mkdir()
    wrapper.extract_features(tmp_path / "img", db, image_names=["recon_00000.jpg", "recon_00002.jpg"])
    flag_idx = captured["args"].index("--ImageReader.image_list_path")
    list_file = Path(captured["args"][flag_idx + 1])
    assert list_file.read_text().split() == ["recon_00000.jpg", "recon_00002.jpg"]


def test_extract_features_fails_when_image_list_flag_missing(tmp_path, monkeypatch):
    exe = make_fake_colmap(tmp_path, "nothing")
    wrapper = ColmapWrapper(config_manager=ConfigManager(), colmap_path=exe)
    monkeypatch.setattr(wrapper, "run_cmd", lambda args: None)
    with pytest.raises(RuntimeError, match="image list"):
        wrapper.extract_features(tmp_path, tmp_path / "db.db", image_names=["a.jpg"])


def test_colmap_cuda_detection_distinguishes_with_and_without(tmp_path, monkeypatch):
    from src.utils.system_info import check_colmap_installation
    d1 = tmp_path / "nocuda"
    d1.mkdir()
    make_fake_colmap(d1, "COLMAP 3.9 -- Structure-from-Motion and Multi-View Stereo (without CUDA)")
    monkeypatch.setenv("PATH", str(d1))
    assert check_colmap_installation()["cuda_support"] is False
    d2 = tmp_path / "cuda"
    d2.mkdir()
    make_fake_colmap(d2, "COLMAP 3.9 -- Structure-from-Motion and Multi-View Stereo (with CUDA)")
    monkeypatch.setenv("PATH", str(d2))
    assert check_colmap_installation()["cuda_support"] is True
    d3 = tmp_path / "gpu4"
    d3.mkdir()
    make_fake_colmap(d3, "COLMAP 4.2.0 (Commit Unknown on Unknown without GPU support)")
    monkeypatch.setenv("PATH", str(d3))
    assert check_colmap_installation()["cuda_support"] is False


def test_dense_stage_requires_cuda_explicitly(tmp_path, monkeypatch):
    exe = make_fake_colmap(tmp_path, "help")
    runner = MVSRunner.__new__(MVSRunner)
    runner.wrapper = ColmapWrapper(colmap_path=exe)
    runner.patch_match_gpu = True
    runner.patch_match_gpu_index = "0"
    monkeypatch.setattr("src.dense.mvs_runner.check_colmap_installation",
                        lambda: {"available": True, "path": "/x/colmap", "version": "v", "cuda_support": False})
    with pytest.raises(RuntimeError, match="without CUDA"):
        runner.require_cuda()
    monkeypatch.setattr("src.dense.mvs_runner.check_colmap_installation",
                        lambda: {"available": True, "path": "/x/colmap", "version": "v", "cuda_support": True})
    runner.require_cuda()
    runner.patch_match_gpu = False
    with pytest.raises(RuntimeError, match="no CPU implementation"):
        runner.require_cuda()


def test_patch_match_uses_configured_gpu_index_never_minus_one(tmp_path, monkeypatch):
    exe = make_fake_colmap(tmp_path, "help")
    runner = MVSRunner(config_manager=ConfigManager()) if False else MVSRunner.__new__(MVSRunner)
    runner.wrapper = ColmapWrapper(colmap_path=exe)
    cfg = ConfigManager()
    runner.patch_match_gpu = True
    runner.patch_match_gpu_index = str(cfg.get("dense.patch_match_gpu_index"))
    runner.patch_match_geom_consistency = True
    runner.patch_match_min_triangulation_angle = 4.0
    runner.patch_match_filter_min_ncc = 0.15
    runner.patch_match_filter_geom_consistency_max_cost = 1.0
    captured = {}
    monkeypatch.setattr(runner.wrapper, "run_cmd", lambda args: captured.setdefault("args", list(args)))
    runner.run_patch_match_stereo(tmp_path)
    a = captured["args"]
    assert a[a.index("--PatchMatchStereo.gpu_index") + 1] == "0"


def test_check_ply_geometry_actually_analyses_points(tmp_path):
    rng = np.random.default_rng(0)
    flat = np.column_stack([rng.uniform(-100, 100, 2000), rng.uniform(-100, 100, 2000), rng.normal(0, 0.001, 2000)])
    ply = tmp_path / "flat.ply"
    write_ply_points(ply, flat.astype(np.float32))
    res = check_ply_geometry(ply, warn_threshold=500.0)
    assert res["check_status"] == "OK"
    assert res["check_performed"] is True
    assert res["num_points"] == 2000
    assert res["is_flattened"] is True
    assert res["eigenvalue_ratio"] > 500.0

    blob = rng.normal(0, 1, (2000, 3)).astype(np.float32)
    ply2 = tmp_path / "blob.ply"
    write_ply_points(ply2, blob)
    res2 = check_ply_geometry(ply2, warn_threshold=500.0)
    assert res2["num_points"] == 2000
    assert res2["is_flattened"] is False
    assert res2["eigenvalue_ratio"] < 10.0


def test_check_ply_geometry_reports_missing_unreadable_and_tiny(tmp_path):
    missing = check_ply_geometry(tmp_path / "nope.ply")
    assert missing["check_status"] == "FILE_NOT_FOUND"
    assert missing["check_performed"] is False
    bad = tmp_path / "bad.ply"
    bad.write_text("not a ply")
    unreadable = check_ply_geometry(bad)
    assert unreadable["check_status"] == "READ_FAILED"
    assert "error" in unreadable
    tiny = tmp_path / "tiny.ply"
    write_ply_points(tiny, np.zeros((3, 3), dtype=np.float32))
    assert check_ply_geometry(tiny)["check_status"] == "INSUFFICIENT_POINTS"


def test_compute_pca_rejects_tuple_and_bad_shapes():
    with pytest.raises(TypeError):
        compute_pca_point_cloud((np.zeros((10, 3)), None, None))
    with pytest.raises(ValueError):
        compute_pca_point_cloud(np.zeros((10, 2)))


@pytest.mark.real_data
@pytest.mark.skipif(not (FINAL_DEMO_VIDEO.exists() and FINAL_DEMO_CSV.exists()), reason="final_demo inputs not present")
def test_final_demo_real_inputs_map_only_decoded_frames(tmp_path):
    meta = extract_video_metadata(FINAL_DEMO_VIDEO)
    parsed = TelemetryParser(video_path=FINAL_DEMO_VIDEO).extract()
    assert parsed["source_file"].endswith("final_demo.csv")
    assert len(parsed["records"]) == 97
    validation = validate_telemetry_records(parsed["records"])
    assert validation["duplicate_frame_indices"] == []
    assert validation["missing_frame_indices"] == []
    assert validation["video_time_s_strictly_increasing"] is True
    assert validation["gps_valid_row_count"] == 97

    frames = [
        {"frame_id": f"recon_{i:05d}", "decoded_frame_index": i, "video_time_s": t}
        for i, t in enumerate(meta["decoded_timestamps_sec"])
    ]
    assert len(frames) == 75
    assoc, used = associate_frames_with_telemetry(frames, parsed["records"], max_time_diff_s=0.05)
    assert all(a["telemetry_valid"] for a in assoc)
    assert max(used) <= 72
    unmatched = classify_unmatched_telemetry(
        parsed["records"], used, (meta["decoded_first_timestamp_sec"], meta["decoded_last_timestamp_sec"])
    )
    assert {u["reason"] for u in unmatched} == {"beyond_decoded_video_range"}
    assert len(unmatched) == 24


def test_geometry_status_values(tmp_path):
    rng = np.random.default_rng(1)
    flat = np.column_stack([rng.uniform(-100, 100, 500), rng.uniform(-100, 100, 500), rng.normal(0, 0.001, 500)])
    p1 = tmp_path / "flat.ply"
    write_ply_points(p1, flat.astype(np.float32))
    assert check_ply_geometry(p1)["status"] == "fail"
    p2 = tmp_path / "blob.ply"
    write_ply_points(p2, rng.normal(0, 1, (500, 3)).astype(np.float32))
    assert check_ply_geometry(p2)["status"] == "pass"
    p3 = tmp_path / "tiny.ply"
    write_ply_points(p3, np.zeros((2, 3), dtype=np.float32))
    assert check_ply_geometry(p3)["status"] == "insufficient_data"
    assert check_ply_geometry(tmp_path / "missing.ply")["status"] == "error"
    bad = tmp_path / "bad.ply"
    bad.write_text("nope")
    assert check_ply_geometry(bad)["status"] == "error"


def test_final_demo_effective_config_is_complete_and_repo_relative():
    cfg = ConfigManager(config_path=REPO_ROOT / "config" / "final_demo.yaml")
    assert cfg.get("input.video") == "data/input/final_demo.mp4"
    assert cfg.get("input.telemetry_csv") == "data/input/final_demo.csv"
    assert cfg.get("preprocessing.reconstruction.strategy") == "decoded_stride"
    assert cfg.get("preprocessing.reconstruction.stride") == 1
    assert cfg.get("telemetry.max_time_diff_s") == 0.05
    assert cfg.get("sfm.matcher_type") == "sequential"
    assert cfg.get("sfm.sequential.overlap") == 10
    assert cfg.get("sfm.camera_model") == "OPENCV"
    assert cfg.get("sfm.gpu_index") == "0"
    assert cfg.get("sfm.features.max_num_features") == 8192
    assert cfg.get("sfm.mapper.init_min_tri_angle") == 5.0
    assert cfg.get("dense.patch_match_gpu") is True
    assert cfg.get("dense.patch_match_gpu_index") == "0"
    assert cfg.get("masking.approved_for_reconstruction") is False
    assert cfg.get("dense.clean_outliers_std_ratio") == ConfigManager().get("dense.clean_outliers_std_ratio")
    for key in ("input.video", "input.telemetry_csv", "paths.outputs_dir", "paths.logs_dir"):
        assert not Path(cfg.get(key)).is_absolute()
    assert cfg.get_path("paths.logs_dir") == REPO_ROOT / "logs" / "final_demo"
    assert cfg.get_path("paths.outputs_dir") / FINAL_DEMO_VIDEO.stem == REPO_ROOT / "outputs" / "final_demo"


def test_default_config_leaves_dataset_specific_colmap_settings_unset():
    d = ConfigManager()
    assert d.get("sfm.gpu_index") is None
    assert d.get("sfm.features.max_num_features") is None
    assert d.get("sfm.sequential.overlap") is None
    assert d.get("sfm.mapper.init_min_tri_angle") is None
    assert d.get("dense.min_registration_fraction") is None
    assert d.get("input.video") is None


FULL_HELP = "\n".join([
    "--FeatureExtraction.use_gpu arg", "--FeatureExtraction.gpu_index arg",
    "--FeatureMatching.use_gpu arg", "--FeatureMatching.gpu_index arg",
    "--SiftExtraction.max_num_features arg", "--SequentialMatching.overlap arg",
    "--Mapper.init_min_tri_angle arg", "--ImageReader.image_list_path arg",
])


def test_final_demo_gpu_flags_are_explicit_for_extraction_and_matching(tmp_path, monkeypatch):
    exe = make_fake_colmap(tmp_path, FULL_HELP)
    wrapper = ColmapWrapper(config_manager=ConfigManager(config_path=REPO_ROOT / "config" / "final_demo.yaml"), colmap_path=exe)
    calls = []
    monkeypatch.setattr(wrapper, "run_cmd", lambda args: calls.append(list(args)))
    wrapper.extract_features(tmp_path, tmp_path / "db.db")
    wrapper.match_features(tmp_path / "db.db")
    extract, match = calls
    assert extract[extract.index("--FeatureExtraction.use_gpu") + 1] == "1"
    assert extract[extract.index("--FeatureExtraction.gpu_index") + 1] == "0"
    assert extract[extract.index("--SiftExtraction.max_num_features") + 1] == "8192"
    assert match[0] == "sequential_matcher"
    assert match[match.index("--FeatureMatching.gpu_index") + 1] == "0"
    assert match[match.index("--SequentialMatching.overlap") + 1] == "10"


def test_gpu_flags_use_legacy_sift_names_on_colmap_3x(tmp_path, monkeypatch):
    exe = make_fake_colmap(tmp_path, "--SiftExtraction.use_gpu arg\n--SiftExtraction.gpu_index arg\n--SiftMatching.use_gpu arg\n--SiftMatching.gpu_index arg")
    wrapper = ColmapWrapper(config_manager=ConfigManager(config_path=REPO_ROOT / "config" / "final_demo.yaml"), colmap_path=exe)
    assert wrapper.gpu_args(wrapper.extractor_gpu_prefix) == ["--SiftExtraction.use_gpu", "1", "--SiftExtraction.gpu_index", "0"]
    assert wrapper.gpu_args(wrapper.matcher_gpu_prefix) == ["--SiftMatching.use_gpu", "1", "--SiftMatching.gpu_index", "0"]


def test_default_config_passes_no_optional_colmap_flags(tmp_path, monkeypatch):
    exe = make_fake_colmap(tmp_path, FULL_HELP)
    cfg = ConfigManager()
    cfg.config["sfm"]["use_gpu"] = False
    wrapper = ColmapWrapper(config_manager=cfg, colmap_path=exe)
    calls = []
    monkeypatch.setattr(wrapper, "run_cmd", lambda args: calls.append(list(args)))
    wrapper.match_features(tmp_path / "db.db")
    assert calls[0] == ["sequential_matcher", "--database_path", str(tmp_path / "db.db")]


def test_unsupported_configured_feature_flags_fail_loudly(tmp_path, monkeypatch):
    exe = make_fake_colmap(tmp_path, "--FeatureExtraction.use_gpu arg")
    wrapper = ColmapWrapper(config_manager=ConfigManager(config_path=REPO_ROOT / "config" / "final_demo.yaml"), colmap_path=exe)
    monkeypatch.setattr(wrapper, "run_cmd", lambda args: None)
    with pytest.raises(RuntimeError, match="max_num_features"):
        wrapper.extract_features(tmp_path, tmp_path / "db.db")
    with pytest.raises(RuntimeError, match="overlap"):
        wrapper.match_features(tmp_path / "db.db")


def test_invalid_gpu_index_is_rejected(tmp_path):
    exe = make_fake_colmap(tmp_path, FULL_HELP)
    cfg = ConfigManager(config_path=REPO_ROOT / "config" / "final_demo.yaml")
    cfg.config["sfm"]["gpu_index"] = "all"
    with pytest.raises(ValueError, match="gpu_index"):
        ColmapWrapper(config_manager=cfg, colmap_path=exe)


def test_patch_match_gpu_index_minus_one_is_rejected(tmp_path, monkeypatch):
    exe = make_fake_colmap(tmp_path, "help")
    runner = MVSRunner.__new__(MVSRunner)
    runner.wrapper = ColmapWrapper(colmap_path=exe)
    runner.patch_match_gpu = True
    runner.patch_match_gpu_index = "-1"
    monkeypatch.setattr("src.dense.mvs_runner.check_colmap_installation",
                        lambda: {"available": True, "path": "/x", "version": "v", "cuda_support": True})
    with pytest.raises(RuntimeError, match="explicit CUDA device"):
        runner.require_cuda()
    runner.patch_match_gpu_index = "0,1"
    runner.require_cuda()


def test_sparse_model_must_match_reconstruction_image_set():
    from src.utils.reconstruction_gate import assert_sparse_model_matches_inputs, assert_registration_sufficient
    inputs = {"image_names": ["recon_00000.jpg", "recon_00001.jpg"]}
    assert_sparse_model_matches_inputs(["recon_00000.jpg"], inputs)
    with pytest.raises(ReconstructionGateError, match="not in the reconstruction image set"):
        assert_sparse_model_matches_inputs(["recon_00000.jpg", "selected_00001.jpg"], inputs)
    assert_registration_sufficient({"total_input_images": 75, "registered_image_count": 3}, None)
    assert_registration_sufficient({"total_input_images": 75, "registered_image_count": 75}, 0.9)
    with pytest.raises(ReconstructionGateError, match="3/75"):
        assert_registration_sufficient({"total_input_images": 75, "registered_image_count": 3}, 0.9)


def test_configured_telemetry_path_takes_precedence_and_missing_is_reported(tmp_path):
    write_final_demo_style_csv(tmp_path / "other.csv", rows=3)
    write_final_demo_style_csv(tmp_path / "clip.csv", rows=6)
    res = TelemetryParser(video_path=tmp_path / "clip.mp4", telemetry_path=tmp_path / "other.csv").extract()
    assert len(res["records"]) == 3
    missing = TelemetryParser(video_path=tmp_path / "clip.mp4", telemetry_path=tmp_path / "nope.csv").extract()
    assert missing["telemetry_detected"] is False
    assert "does not exist" in missing["reason"]


def test_original_csv_is_never_modified(tmp_path):
    csv_path = tmp_path / "clip.csv"
    write_final_demo_style_csv(csv_path, rows=10)
    before = csv_path.read_bytes()
    cfg = final_demo_config(tmp_path)
    video = tmp_path / "clip.mp4"
    create_synthetic_video(video, num_frames=8, fps=10.0)
    run_stage2(video, output_dir=tmp_path / "out", config_manager=cfg)
    run_stage3(video, output_dir=tmp_path / "out", config_manager=cfg)
    assert csv_path.read_bytes() == before
