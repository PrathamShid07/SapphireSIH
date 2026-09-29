import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.masking.dynamic_mask_generator import (
    generate_frame_mask,
    create_mask_overlay,
    DynamicMaskGenerator,
)
from src.masking.mask_propagator import MaskPropagator
from src.masking.reconstruction_comparator import (
    ReconstructionComparator,
    count_points_in_mask,
)
from src.sfm.colmap_wrapper import ColmapWrapper


def test_generate_frame_mask():
    shape = (200, 300, 3)
    dyn_boxes = [[50.0, 60.0, 100.0, 120.0]]
    mask_no_dil = generate_frame_mask(shape, dyn_boxes, dilation_px=0)

    assert mask_no_dil.shape == (200, 300)
    assert mask_no_dil[0, 0] == 255
    assert mask_no_dil[80, 80] == 0

    mask_dil = generate_frame_mask(shape, dyn_boxes, dilation_px=10)
    assert np.sum(mask_dil == 0) > np.sum(mask_no_dil == 0)
    assert mask_dil[55, 45] == 0


def test_generate_frame_mask_with_semantic():
    shape = (100, 100, 3)
    sem = np.zeros((100, 100), dtype=np.uint8)
    sem[20:40, 20:40] = 7

    mask = generate_frame_mask(
        shape,
        dynamic_bboxes=[],
        dilation_px=0,
        semantic_mask=sem,
        dynamic_semantic_classes={7},
    )
    assert mask[0, 0] == 255
    assert mask[30, 30] == 0


def test_create_mask_overlay():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.full((100, 100), 255, dtype=np.uint8)
    mask[20:40, 20:40] = 0

    overlay = create_mask_overlay(img, mask)
    assert overlay.shape == (100, 100, 3)
    assert overlay[0, 0, 0] == 128
    assert overlay[30, 30, 0] > overlay[30, 30, 1]


def test_mask_propagator_interpolation():
    meta = [
        {
            "frame_index": 0,
            "objects": [
                {"track_id": 1, "bbox": [10.0, 10.0, 30.0, 30.0], "is_moving": True}
            ],
        },
        {"frame_index": 1, "objects": []},
        {
            "frame_index": 2,
            "objects": [
                {"track_id": 1, "bbox": [50.0, 50.0, 70.0, 70.0], "is_moving": True}
            ],
        },
    ]

    propagator = MaskPropagator(max_gap=2, motion_expansion=False)
    updated = propagator.propagate_tracks(meta)

    assert len(updated[1]["objects"]) == 1
    interp_obj = updated[1]["objects"][0]
    assert interp_obj["track_id"] == 1
    assert interp_obj["interpolated"] is True
    assert np.allclose(interp_obj["bbox"], [30.0, 30.0, 50.0, 50.0])


def test_dynamic_mask_generator_from_tracking(tmp_path):
    img_dir = tmp_path / "images"
    out_dir = tmp_path / "mask_out"
    img_dir.mkdir()

    for i in range(2):
        Image.new("RGB", (320, 240), color=(100, 100, 100)).save(
            str(img_dir / f"frame_{i:02d}.jpg")
        )

    meta = [
        {
            "frame_index": 0,
            "image_name": "frame_00.jpg",
            "objects": [
                {
                    "track_id": 5,
                    "bbox": [50.0, 50.0, 100.0, 100.0],
                    "is_moving": True,
                }
            ],
        },
        {
            "frame_index": 1,
            "image_name": "frame_01.jpg",
            "objects": [
                {
                    "track_id": 5,
                    "bbox": [60.0, 50.0, 110.0, 100.0],
                    "is_moving": True,
                }
            ],
        },
    ]
    summary = {"5": {"is_dynamic": True}}

    gen = DynamicMaskGenerator(dilation_px=5)
    res = gen.generate_masks_from_tracking(meta, summary, img_dir, out_dir)

    assert res["total_masks"] == 2
    assert Path(res["manifest_path"]).exists()
    assert (out_dir / "masks" / "frame_00.jpg.png").exists()
    assert (out_dir / "overlays" / "frame_00_overlay.jpg").exists()


def test_reconstruction_comparator_mock(tmp_path):
    unmasked_dir = tmp_path / "unmasked"
    masked_dir = tmp_path / "masked"
    unmasked_dir.mkdir()
    masked_dir.mkdir()

    u_cam = "# Camera list with one line of data per camera:\n1 PINHOLE 100 100 100 100 50 50\n"
    u_img = "# Image list with two lines of data per image:\n1 1 0 0 0 0 0 0 1 frame1.jpg\n\n2 1 0 0 0 0 0 0 1 frame2.jpg\n\n"
    u_pts = "# 3D point list with one line of data per point:\n1 0 0 1 255 0 0 0.8 2 1 0 2 0\n2 0 0 2 0 255 0 0.6 2 1 1 2 1\n"

    m_cam = u_cam
    m_img = u_img
    m_pts = "# 3D point list with one line of data per point:\n1 0 0 1 255 0 0 0.5 2 1 0 2 0\n"

    for d, c, i, p in [
        (unmasked_dir, u_cam, u_img, u_pts),
        (masked_dir, m_cam, m_img, m_pts),
    ]:
        with open(d / "cameras.txt", "w") as f:
            f.write(c)
        with open(d / "images.txt", "w") as f:
            f.write(i)
        with open(d / "points3D.txt", "w") as f:
            f.write(p)

    comparator = ReconstructionComparator()
    report = comparator.compare_reconstructions(
        unmasked_model_dir=unmasked_dir,
        masked_model_dir=masked_dir,
        output_path=tmp_path / "comp.json",
    )

    assert report["status"] == "SUCCESS"
    assert report["unmasked"]["sparse_points"] == 2
    assert report["masked"]["sparse_points"] == 1
    assert report["comparison"]["points_delta"] == -1
    assert report["comparison"]["registration_preserved"] is True
    assert (tmp_path / "comp.json").exists()


def test_colmap_wrapper_mask_arg(tmp_path):
    from src.utils.system_info import check_colmap_installation
    col_info = check_colmap_installation()
    if col_info["available"]:
        wrapper = ColmapWrapper()
        assert wrapper.colmap_exe is not None
    else:
        dummy = tmp_path / "colmap"
        dummy.touch()
        wrapper = ColmapWrapper(colmap_path=dummy)
        assert wrapper.colmap_exe == str(dummy)
