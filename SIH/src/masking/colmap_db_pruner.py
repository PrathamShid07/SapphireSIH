"""
Utilities for pruning dynamic-object correspondences out of a COPY of an
existing COLMAP database, so that Stage 8's optimized path can reuse Stage 4's
extracted features/matches/poses instead of re-running feature_extractor and
matcher from scratch.

Empirically verified against a real COLMAP 4.2.0 database (see
docs/STAGE8_OPTIMIZATION.md for the validation run) on this machine:

- ``matches`` and ``two_view_geometries`` tables are keyed by ``pair_id``,
  encoded as ``pair_id = image_id1 * 2147483647 + image_id2`` with
  ``image_id1 < image_id2`` (COLMAP always swaps the pair before encoding).
- Both tables store their correspondences as a ``(rows, cols)`` uint32 array
  in the ``data`` BLOB, where each row is ``[feature_idx_in_image1,
  feature_idx_in_image2]`` — 0-indexed row numbers into that image's
  ``keypoints`` array.
- ``keypoints.data`` is a ``(rows, cols)`` float32 array; columns 0 and 1 are
  always the pixel (x, y) location regardless of whether the remaining
  columns encode scale/orientation (cols=4) or a full 2x2 affine shape
  (cols=6).

This module never re-extracts features or re-matches images: it only removes
existing correspondence rows whose keypoint falls inside a dynamic-object
mask. It operates entirely on a database file the caller has already copied
in isolation — it never opens or writes to Stage 4's original database.
"""

import shutil
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

# COLMAP's pair_id encoding constant (verified empirically against COLMAP
# 4.2.0's own database.cc convention: pair_id = image_id1 * MAX_IMAGE_ID +
# image_id2, image_id1 < image_id2). This is stable across COLMAP versions
# that share this schema (see docs/STAGE8_OPTIMIZATION.md for how to re-verify
# it against a different COLMAP build before trusting it blindly).
MAX_IMAGE_ID = 2147483647


def pair_id_to_image_ids(pair_id):
    """Decode a COLMAP matches/two_view_geometries pair_id into (image_id1, image_id2)."""
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def copy_database(src_db_path, dst_db_path):
    """Create an isolated copy of a COLMAP database. Never modifies src_db_path."""
    src = Path(src_db_path)
    dst = Path(dst_db_path)
    if not src.exists():
        raise FileNotFoundError(
            f"Source COLMAP database not found at {src}. Run Stage 4 first."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copyfile(str(src), str(dst))
    return dst


def load_image_name_to_id(db_path):
    """Return {image_filename: database image_id} for every image in db_path."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT image_id, name FROM images")
        return {name: image_id for image_id, name in cur.fetchall()}
    finally:
        conn.close()


def load_keypoint_coordinates(db_path):
    """Return {image_id: Nx2 float32 array of (x, y) pixel coordinates}."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT image_id, rows, cols, data FROM keypoints")
        coords = {}
        for image_id, rows, cols, data in cur.fetchall():
            if rows == 0 or data is None:
                coords[image_id] = np.zeros((0, 2), dtype=np.float32)
                continue
            arr = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
            coords[image_id] = arr[:, :2].copy()
        return coords
    finally:
        conn.close()


def build_image_masks(masks_dir, image_name_to_id, logger=None):
    """
    Load per-image dynamic-object mask arrays keyed by COLMAP image_id.

    Follows the same file-naming convention DynamicMaskGenerator writes
    (``<image_filename>.png``, e.g. ``frame_0001.jpg.png``), matching the
    convention already relied on by reconstruction_comparator.count_points_in_mask.

    Images with no mask file are treated as fully static for this run (no
    pruning applied to their keypoints) and reported back in ``missing`` so
    the caller can log/report the gap rather than silently ignoring it.
    """
    masks_path = Path(masks_dir)
    image_masks = {}
    missing = []
    for name, image_id in image_name_to_id.items():
        candidates = [masks_path / f"{name}.png", masks_path / f"{Path(name).stem}.png"]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            missing.append(name)
            continue
        image_masks[image_id] = np.array(PILImage.open(str(found)))

    if missing and logger:
        preview = missing[:5]
        suffix = "..." if len(missing) > 5 else ""
        logger.warning(
            f"No dynamic-object mask found for {len(missing)}/{len(image_name_to_id)} "
            f"images; those images are treated as fully static (no pruning applied) "
            f"for this optimized Stage 8 run: {preview}{suffix}"
        )
    return image_masks, missing


def _is_masked(coords_for_image, feature_index, mask_array):
    if feature_index >= len(coords_for_image):
        return False
    x, y = coords_for_image[feature_index]
    h, w = mask_array.shape[:2]
    xi, yi = int(round(float(x))), int(round(float(y)))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return False
    return bool(mask_array[yi, xi] == 0)


def prune_masked_correspondences(db_path, image_masks, logger=None):
    """
    Remove ``matches`` / ``two_view_geometries`` rows whose keypoint falls
    inside a masked (dynamic-object) region, IN PLACE on the database at
    db_path.

    ``db_path`` must already be an isolated copy created by ``copy_database``
    — this function does not create one and does not touch any other file.

    ``image_masks``: {image_id: 2D numpy array} using the same convention as
    ``dynamic_mask_generator.generate_frame_mask`` (0 = masked/dynamic,
    255 = keep/static). Images absent from this dict are treated as fully
    static: neither of their keypoints will cause a match row to be dropped.

    Keypoints/descriptors themselves are intentionally left untouched — only
    matches and verified two-view geometries are pruned. A keypoint that no
    longer participates in any surviving match simply contributes no 3D point
    when point_triangulator runs; there is no need to renumber or delete rows
    from the keypoints/descriptors tables, which would otherwise require
    remapping every match's feature indices and materially raise the risk of
    silently corrupting the database.

    Returns a stats dict per table: rows_before / rows_after / rows_removed /
    pairs_touched.
    """
    keypoints = load_keypoint_coordinates(db_path)

    conn = sqlite3.connect(str(db_path))
    stats = {}
    try:
        cur = conn.cursor()
        for table in ("matches", "two_view_geometries"):
            cur.execute(f"SELECT pair_id, rows, cols, data FROM {table}")
            all_rows = cur.fetchall()
            rows_before = 0
            rows_after = 0
            pairs_touched = 0

            for pair_id, rows, cols, data in all_rows:
                if rows == 0 or data is None:
                    continue

                img1, img2 = pair_id_to_image_ids(pair_id)
                mask1 = image_masks.get(img1)
                mask2 = image_masks.get(img2)
                rows_before += rows

                if mask1 is None and mask2 is None:
                    rows_after += rows
                    continue

                kp1 = keypoints.get(img1, np.zeros((0, 2), dtype=np.float32))
                kp2 = keypoints.get(img2, np.zeros((0, 2), dtype=np.float32))
                arr = np.frombuffer(data, dtype=np.uint32).reshape(rows, cols)

                keep = np.ones(rows, dtype=bool)
                for i in range(rows):
                    f1, f2 = int(arr[i, 0]), int(arr[i, 1])
                    if mask1 is not None and _is_masked(kp1, f1, mask1):
                        keep[i] = False
                        continue
                    if mask2 is not None and _is_masked(kp2, f2, mask2):
                        keep[i] = False

                kept_arr = arr[keep]
                rows_after += len(kept_arr)
                if len(kept_arr) != rows:
                    pairs_touched += 1

                new_data = np.ascontiguousarray(kept_arr, dtype=np.uint32).tobytes()
                cur.execute(
                    f"UPDATE {table} SET rows = ?, data = ? WHERE pair_id = ?",
                    (len(kept_arr), new_data, pair_id),
                )

            stats[table] = {
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": rows_before - rows_after,
                "pairs_touched": pairs_touched,
            }
        conn.commit()
    finally:
        conn.close()

    if logger:
        for table, s in stats.items():
            pct = (100.0 * s["rows_removed"] / s["rows_before"]) if s["rows_before"] else 0.0
            logger.info(
                f"Pruned {table}: {s['rows_removed']}/{s['rows_before']} rows removed "
                f"({pct:.1f}%) across {s['pairs_touched']} image pairs."
            )

    return stats
