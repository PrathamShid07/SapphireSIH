"""
Focused, fast unit tests for src/masking/colmap_db_pruner.py.

These build a tiny hand-crafted SQLite database matching COLMAP 4.2.0's real
schema (verified empirically against an actual COLMAP installation - see
docs/STAGE8_OPTIMIZATION.md) so the pruning logic can be tested deterministically
without needing a real COLMAP binary or a real reconstruction.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.masking import colmap_db_pruner as pruner


def _create_minimal_colmap_db(db_path, keypoints_by_image, matches_by_pair):
    """
    keypoints_by_image: {image_id: Nx2 array-like of (x, y)}
    matches_by_pair: {(image_id1, image_id2): Mx2 array-like of (feat_idx1, feat_idx2)}
                      (image_id1 must be < image_id2, matching COLMAP's convention)
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, "
        "cols INTEGER, data BLOB)"
    )
    cur.execute(
        "CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER, "
        "cols INTEGER, data BLOB)"
    )
    cur.execute(
        "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER, "
        "cols INTEGER, data BLOB)"
    )
    cur.execute(
        "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
    )

    for image_id, pts in keypoints_by_image.items():
        arr = np.asarray(pts, dtype=np.float32)
        cur.execute(
            "INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)",
            (image_id, f"frame_{image_id:04d}.jpg", 1),
        )
        cur.execute(
            "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (image_id, arr.shape[0], arr.shape[1], arr.tobytes()),
        )

    for (id1, id2), match_rows in matches_by_pair.items():
        assert id1 < id2, "COLMAP always stores pairs with image_id1 < image_id2"
        pair_id = id1 * pruner.MAX_IMAGE_ID + id2
        arr = np.asarray(match_rows, dtype=np.uint32)
        for table in ("matches", "two_view_geometries"):
            cur.execute(
                f"INSERT INTO {table} (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (pair_id, arr.shape[0], arr.shape[1], arr.tobytes()),
            )

    conn.commit()
    conn.close()


def _read_match_rows(db_path, table, pair_id):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(f"SELECT rows, cols, data FROM {table} WHERE pair_id = ?", (pair_id,))
    rows, cols, data = cur.fetchone()
    conn.close()
    if rows == 0:
        return np.zeros((0, cols), dtype=np.uint32)
    return np.frombuffer(data, dtype=np.uint32).reshape(rows, cols)


class TestPairIdEncoding:
    def test_pair_id_roundtrip(self):
        # Verified against a real COLMAP 4.2.0 database in docs/STAGE8_OPTIMIZATION.md
        for id1, id2 in [(1, 2), (1, 6), (3, 5), (4, 6), (5, 6)]:
            pair_id = id1 * pruner.MAX_IMAGE_ID + id2
            decoded1, decoded2 = pruner.pair_id_to_image_ids(pair_id)
            assert (decoded1, decoded2) == (id1, id2)

    def test_known_real_colmap_pair_ids(self):
        # Exact pair_id values captured from a real COLMAP 4.2.0 database.db
        # produced by feature_extractor + sequential_matcher on real drone footage.
        assert pruner.pair_id_to_image_ids(2147483649) == (1, 2)
        assert pruner.pair_id_to_image_ids(4294967297) == (2, 3)
        assert pruner.pair_id_to_image_ids(10737418241) == (5, 6)


class TestCopyDatabase:
    def test_copy_creates_independent_file(self, tmp_path):
        src = tmp_path / "src.db"
        _create_minimal_colmap_db(src, {1: [[0, 0]], 2: [[0, 0]]}, {})
        dst = tmp_path / "nested" / "dst.db"

        result = pruner.copy_database(src, dst)

        assert result == dst
        assert dst.exists()
        src_bytes = src.read_bytes()
        dst_bytes = dst.read_bytes()
        assert src_bytes == dst_bytes

        # Mutating the copy must never affect the source.
        conn = sqlite3.connect(str(dst))
        conn.execute("DELETE FROM keypoints")
        conn.commit()
        conn.close()
        conn2 = sqlite3.connect(str(src))
        remaining = conn2.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
        conn2.close()
        assert remaining == 2

    def test_copy_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pruner.copy_database(tmp_path / "does_not_exist.db", tmp_path / "dst.db")


class TestLoadHelpers:
    def test_load_image_name_to_id(self, tmp_path):
        db = tmp_path / "db.db"
        _create_minimal_colmap_db(db, {1: [[0, 0]], 2: [[0, 0]]}, {})
        mapping = pruner.load_image_name_to_id(db)
        assert mapping == {"frame_0001.jpg": 1, "frame_0002.jpg": 2}

    def test_load_keypoint_coordinates(self, tmp_path):
        db = tmp_path / "db.db"
        _create_minimal_colmap_db(db, {1: [[10.0, 20.0], [30.0, 40.0]]}, {})
        coords = pruner.load_keypoint_coordinates(db)
        assert coords[1].shape == (2, 2)
        assert np.allclose(coords[1], [[10.0, 20.0], [30.0, 40.0]])


class TestPruneMaskedCorrespondences:
    def test_removes_matches_touching_masked_keypoints(self, tmp_path):
        # Image 1 has 3 keypoints: two in the "static" left half, one in the
        # "dynamic" right half. Image 2 has 3 keypoints, all static.
        db = tmp_path / "db.db"
        kp1 = [[10.0, 10.0], [20.0, 20.0], [90.0, 10.0]]  # index 2 is at x=90 (masked)
        kp2 = [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]
        matches = [[0, 0], [1, 1], [2, 2]]  # row 2 references masked feature 2 of image 1
        _create_minimal_colmap_db(db, {1: kp1, 2: kp2}, {(1, 2): matches})

        mask = np.full((100, 100), 255, dtype=np.uint8)
        mask[:, 50:] = 0  # right half is "dynamic" (masked)

        stats = pruner.prune_masked_correspondences(db, {1: mask})

        pair_id = 1 * pruner.MAX_IMAGE_ID + 2
        remaining = _read_match_rows(db, "matches", pair_id)
        assert remaining.shape[0] == 2
        assert list(map(tuple, remaining)) == [(0, 0), (1, 1)]

        remaining_tvg = _read_match_rows(db, "two_view_geometries", pair_id)
        assert remaining_tvg.shape[0] == 2

        assert stats["matches"]["rows_before"] == 3
        assert stats["matches"]["rows_after"] == 2
        assert stats["matches"]["rows_removed"] == 1
        assert stats["matches"]["pairs_touched"] == 1

    def test_removes_when_either_side_is_masked(self, tmp_path):
        db = tmp_path / "db.db"
        kp1 = [[10.0, 10.0], [10.0, 10.0]]
        kp2 = [[90.0, 10.0], [10.0, 10.0]]  # index 0 of image 2 is masked
        matches = [[0, 0], [1, 1]]
        _create_minimal_colmap_db(db, {1: kp1, 2: kp2}, {(1, 2): matches})

        mask = np.full((100, 100), 255, dtype=np.uint8)
        mask[:, 50:] = 0

        pruner.prune_masked_correspondences(db, {2: mask})

        pair_id = 1 * pruner.MAX_IMAGE_ID + 2
        remaining = _read_match_rows(db, "matches", pair_id)
        # Row 0 touches image2's masked feature 0 -> removed. Row 1 is fully static -> kept.
        assert list(map(tuple, remaining)) == [(1, 1)]

    def test_no_mask_for_either_image_leaves_pair_untouched(self, tmp_path):
        db = tmp_path / "db.db"
        kp1 = [[10.0, 10.0]]
        kp2 = [[10.0, 10.0]]
        matches = [[0, 0]]
        _create_minimal_colmap_db(db, {1: kp1, 2: kp2}, {(1, 2): matches})

        stats = pruner.prune_masked_correspondences(db, {})  # no masks at all

        pair_id = 1 * pruner.MAX_IMAGE_ID + 2
        remaining = _read_match_rows(db, "matches", pair_id)
        assert remaining.shape[0] == 1
        assert stats["matches"]["rows_removed"] == 0
        assert stats["matches"]["pairs_touched"] == 0

    def test_all_correspondences_removed_when_fully_masked(self, tmp_path):
        db = tmp_path / "db.db"
        kp1 = [[90.0, 90.0], [95.0, 95.0]]
        kp2 = [[10.0, 10.0], [20.0, 20.0]]
        matches = [[0, 0], [1, 1]]
        _create_minimal_colmap_db(db, {1: kp1, 2: kp2}, {(1, 2): matches})

        mask = np.full((100, 100), 255, dtype=np.uint8)
        mask[:, 50:] = 0  # covers both of image 1's keypoints

        pruner.prune_masked_correspondences(db, {1: mask})

        pair_id = 1 * pruner.MAX_IMAGE_ID + 2
        remaining = _read_match_rows(db, "matches", pair_id)
        assert remaining.shape[0] == 0

    def test_keypoints_table_is_never_modified(self, tmp_path):
        db = tmp_path / "db.db"
        kp1 = [[90.0, 90.0], [10.0, 10.0]]
        kp2 = [[10.0, 10.0], [20.0, 20.0]]
        matches = [[0, 0], [1, 1]]
        _create_minimal_colmap_db(db, {1: kp1, 2: kp2}, {(1, 2): matches})

        mask = np.full((100, 100), 255, dtype=np.uint8)
        mask[:, 50:] = 0

        before = pruner.load_keypoint_coordinates(db)
        pruner.prune_masked_correspondences(db, {1: mask})
        after = pruner.load_keypoint_coordinates(db)

        # Pruning must only touch matches/two_view_geometries, never keypoints.
        assert np.array_equal(before[1], after[1])
        assert np.array_equal(before[2], after[2])
        assert len(after[1]) == 2  # nothing deleted, no reindexing needed


class TestBuildImageMasks:
    def test_matches_dynamic_mask_generator_naming_convention(self, tmp_path):
        from PIL import Image as PILImage

        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        mask_arr = np.full((10, 10), 255, dtype=np.uint8)
        # DynamicMaskGenerator writes "<image_name>.png", e.g. "frame_0001.jpg.png"
        PILImage.fromarray(mask_arr).save(masks_dir / "frame_0001.jpg.png")

        image_masks, missing = pruner.build_image_masks(
            masks_dir, {"frame_0001.jpg": 1, "frame_0002.jpg": 2}
        )

        assert 1 in image_masks
        assert 2 not in image_masks
        assert missing == ["frame_0002.jpg"]
