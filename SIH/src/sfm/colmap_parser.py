import struct
from pathlib import Path
import numpy as np

def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * z - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2]
    ])

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 8, 6: 12, 7: 5, 8: 10}[model_id]
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = {"id": cam_id, "model_id": model_id, "width": w, "height": h, "params": params}
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            
            name_chars = []
            while True:
                char = f.read(1)
                if char == b"\x00":
                    break
                name_chars.append(char)
            name = b"".join(name_chars).decode("utf-8")
            
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points2D)
            
            R = np.array([
                [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[3] * qvec[0], 2 * qvec[1] * qvec[3] + 2 * qvec[2] * qvec[0]],
                [2 * qvec[1] * qvec[2] + 2 * qvec[3] * qvec[0], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[1] * qvec[0]],
                [2 * qvec[1] * qvec[3] - 2 * qvec[2] * qvec[0], 2 * qvec[2] * qvec[3] + 2 * qvec[1] * qvec[0], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
            ])
            pos = -np.dot(R.T, np.array(tvec))

            images[img_id] = {
                "id": img_id,
                "name": name,
                "camera_id": cam_id,
                "qvec": [round(x, 6) for x in qvec],
                "tvec": [round(x, 6) for x in tvec],
                "position_xyz": [round(x, 6) for x in pos],
                "num_points2D": num_points2D,
            }
    return images

def read_points3D_binary(path):
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            pt_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)
            points[pt_id] = {
                "id": pt_id,
                "xyz": [round(x, 6) for x in xyz],
                "rgb": list(rgb),
                "error": round(error, 4),
                "track_len": track_len,
            }
    return points

def parse_cameras_text(path):
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            cameras[cam_id] = {
                "id": cam_id,
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(p) for p in parts[4:]]
            }
    return cameras

def parse_images_text(path):
    images = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if line.startswith("#") or not line:
            continue
        parts = line.split()
        img_id = int(parts[0])
        qvec = [float(x) for x in parts[1:5]]
        tvec = [float(x) for x in parts[5:8]]
        cam_id = int(parts[8])
        name = parts[9]
        if i < len(lines):
            i += 1
            
        R = np.array([
            [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[3] * qvec[0], 2 * qvec[1] * qvec[3] + 2 * qvec[2] * qvec[0]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[3] * qvec[0], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[1] * qvec[0]],
            [2 * qvec[1] * qvec[3] - 2 * qvec[2] * qvec[0], 2 * qvec[2] * qvec[3] + 2 * qvec[1] * qvec[0], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
        ])
        pos = -np.dot(R.T, np.array(tvec))

        images[img_id] = {
            "id": img_id,
            "name": name,
            "camera_id": cam_id,
            "qvec": [round(x, 6) for x in qvec],
            "tvec": [round(x, 6) for x in tvec],
            "position_xyz": [round(x, 6) for x in pos],
        }
    return images

def parse_points3D_text(path):
    points = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            pt_id = int(parts[0])
            points[pt_id] = {
                "id": pt_id,
                "xyz": [round(float(x), 6) for x in parts[1:4]],
                "rgb": [int(x) for x in parts[4:7]],
                "error": round(float(parts[7]), 4),
            }
    return points

def parse_colmap_model(model_dir):
    mdir = Path(model_dir).resolve()
    
    cameras_bin = mdir / "cameras.bin"
    images_bin = mdir / "images.bin"
    points_bin = mdir / "points3D.bin"

    cameras_txt = mdir / "cameras.txt"
    images_txt = mdir / "images.txt"
    points_txt = mdir / "points3D.txt"

    cameras, images, points = {}, {}, {}

    if cameras_bin.exists() and images_bin.exists() and points_bin.exists():
        cameras = read_cameras_binary(cameras_bin)
        images = read_images_binary(images_bin)
        points = read_points3D_binary(points_bin)
    elif cameras_txt.exists() and images_txt.exists() and points_txt.exists():
        cameras = parse_cameras_text(cameras_txt)
        images = parse_images_text(images_txt)
        points = parse_points3D_text(points_txt)

    num_cameras = len(cameras)
    num_registered = len(images)
    num_points = len(points)

    errors = [p["error"] for p in points.values()] if points else []
    mean_reproj_error = round(float(np.mean(errors)), 4) if errors else 0.0

    return {
        "num_cameras": num_cameras,
        "num_registered_images": num_registered,
        "num_sparse_points": num_points,
        "mean_reprojection_error": mean_reproj_error,
        "cameras": list(cameras.values()),
        "registered_images": list(images.values()),
        "sparse_points_sample": list(points.values())[:100],
    }
