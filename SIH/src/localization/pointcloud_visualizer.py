import math
from pathlib import Path
import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image, ImageDraw, ImageFont


def get_box_corners_3d(center_3d, dimensions, rotation_matrix):
    l = dimensions["length"]
    w = dimensions["width"]
    h = dimensions["height"]
    R = np.array(rotation_matrix, dtype=np.float64)

    dx = [l / 2.0, l / 2.0, -l / 2.0, -l / 2.0, l / 2.0, l / 2.0, -l / 2.0, -l / 2.0]
    dy = [w / 2.0, -w / 2.0, -w / 2.0, w / 2.0, w / 2.0, -w / 2.0, -w / 2.0, w / 2.0]
    dz = [h / 2.0, h / 2.0, h / 2.0, h / 2.0, -h / 2.0, -h / 2.0, -h / 2.0, -h / 2.0]

    corners = np.vstack([dx, dy, dz])
    rot_corners = np.dot(R, corners)
    rot_corners += np.array(center_3d).reshape((3, 1))

    return rot_corners.T


def generate_wireframe_points(corners_3d, points_per_edge=15, color=(0, 255, 0)):
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    pts = []
    cols = []
    for i1, i2 in edges:
        p1 = corners_3d[i1]
        p2 = corners_3d[i2]
        for step in range(points_per_edge + 1):
            alpha = step / float(points_per_edge)
            pt = (1.0 - alpha) * p1 + alpha * p2
            pts.append(pt)
            cols.append(color)

    return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)


def export_localized_scene_ply(
    base_ply_path, objects_3d, output_ply_path, logger=None
):
    base_p = Path(base_ply_path)
    out_p = Path(output_ply_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    base_pts = np.zeros((0, 3), dtype=np.float32)
    base_colors = np.zeros((0, 3), dtype=np.uint8)

    if base_p.exists():
        try:
            ply = PlyData.read(str(base_p))
            v = ply["vertex"]
            base_pts = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
            if "red" in v.data.dtype.names:
                base_colors = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(
                    np.uint8
                )
            else:
                base_colors = np.full(
                    (len(base_pts), 3), 180, dtype=np.uint8
                )
        except Exception:
            pass

    box_pts_list = []
    box_cols_list = []

    for obj in objects_3d:
        center = obj["position_3d"]
        dims = obj["dimensions"]
        R = obj["orientation"]["rotation_matrix"]
        is_dyn = obj.get("motion", {}).get("is_dynamic", False)

        color = (255, 60, 60) if is_dyn else (60, 255, 60)

        corners = get_box_corners_3d(center, dims, R)
        w_pts, w_cols = generate_wireframe_points(
            corners, points_per_edge=20, color=color
        )
        box_pts_list.append(w_pts)
        box_cols_list.append(w_cols)

    all_pts = [base_pts] + box_pts_list if box_pts_list else [base_pts]
    all_cols = [base_colors] + box_cols_list if box_cols_list else [base_colors]

    combined_pts = np.vstack(all_pts)
    combined_cols = np.vstack(all_cols)

    num_pts = len(combined_pts)
    dtypes = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    data = np.zeros(num_pts, dtype=dtypes)
    data["x"] = combined_pts[:, 0]
    data["y"] = combined_pts[:, 1]
    data["z"] = combined_pts[:, 2]
    data["red"] = combined_cols[:, 0]
    data["green"] = combined_cols[:, 1]
    data["blue"] = combined_cols[:, 2]

    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=False).write(str(out_p))

    if logger:
        logger.info(
            f"Exported combined 3D localized scene to: {out_p} ({num_pts} points)"
        )


def generate_bev_visualization(
    point_cloud_xyz, objects_3d, output_png_path, img_size=(1200, 1200)
):
    out_p = Path(output_png_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", img_size, color=(20, 24, 30))
    draw = ImageDraw.Draw(img)

    all_x = []
    all_y = []
    if len(point_cloud_xyz) > 0:
        pts = np.array(point_cloud_xyz)
        all_x.extend(pts[:, 0])
        all_y.extend(pts[:, 1])

    for obj in objects_3d:
        all_x.append(obj["position_3d"][0])
        all_y.append(obj["position_3d"][1])

    if not all_x or not all_y:
        img.save(str(out_p))
        return

    min_x, max_x = float(np.min(all_x)), float(np.max(all_x))
    min_y, max_y = float(np.min(all_y)), float(np.max(all_y))

    margin_x = max(2.0, (max_x - min_x) * 0.1)
    margin_y = max(2.0, (max_y - min_y) * 0.1)
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    span_x = max(1e-4, max_x - min_x)
    span_y = max(1e-4, max_y - min_y)

    w, h = img_size

    def to_img_coords(x, y):
        px = int(round((x - min_x) / span_x * (w - 100) + 50))
        py = int(round((y - min_y) / span_y * (h - 100) + 50))
        return px, h - py

    if len(point_cloud_xyz) > 0:
        step = max(1, len(point_cloud_xyz) // 8000)
        sub_pts = point_cloud_xyz[::step]
        for pt in sub_pts:
            px, py = to_img_coords(pt[0], pt[1])
            if 0 <= px < w and 0 <= py < h:
                img.putpixel((px, py), (70, 80, 95))

    for obj in objects_3d:
        tid = obj["track_id"]
        cname = obj.get("class_name", "car")
        pos = obj["position_3d"]
        dims = obj["dimensions"]
        R = obj["orientation"]["rotation_matrix"]
        is_dyn = obj.get("motion", {}).get("is_dynamic", False)

        corners_3d = get_box_corners_3d(pos, dims, R)
        ground_corners = corners_3d[:4]
        poly_pts = [to_img_coords(c[0], c[1]) for c in ground_corners]

        fill_col = (220, 60, 60, 100) if is_dyn else (60, 220, 60, 100)
        line_col = (255, 80, 80) if is_dyn else (80, 255, 80)

        draw.polygon(poly_pts, outline=line_col)

        cx, cy = to_img_coords(pos[0], pos[1])
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 255, 255))

        front_mid_x = (ground_corners[0][0] + ground_corners[1][0]) / 2.0
        front_mid_y = (ground_corners[0][1] + ground_corners[1][1]) / 2.0
        fx, fy = to_img_coords(front_mid_x, front_mid_y)
        draw.line([cx, cy, fx, fy], fill=(255, 255, 0), width=2)

        label = f"#{tid} {cname}"
        draw.text((cx + 5, cy - 8), label, fill=(240, 240, 240))

    img.save(str(out_p))
