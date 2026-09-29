from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from plyfile import PlyData, PlyElement

def read_ply_points(ply_path):
    plydata = PlyData.read(str(ply_path))
    v = plydata['vertex']
    
    pts = np.vstack([v['x'], v['y'], v['z']]).T
    
    colors = None
    if 'red' in v.data.dtype.names:
        colors = np.vstack([v['red'], v['green'], v['blue']]).T
        
    normals = None
    if 'nx' in v.data.dtype.names:
        normals = np.vstack([v['nx'], v['ny'], v['nz']]).T
        
    return pts, colors, normals

def write_ply_points(ply_path, pts, colors=None, normals=None):
    num_pts = len(pts)
    dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    
    if normals is not None and len(normals) == num_pts:
        dtypes.extend([('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
        
    if colors is not None and len(colors) == num_pts:
        dtypes.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        
    data = np.zeros(num_pts, dtype=dtypes)
    data['x'] = pts[:, 0]
    data['y'] = pts[:, 1]
    data['z'] = pts[:, 2]
    
    if normals is not None and len(normals) == num_pts:
        data['nx'] = normals[:, 0]
        data['ny'] = normals[:, 1]
        data['nz'] = normals[:, 2]
        
    if colors is not None and len(colors) == num_pts:
        data['red'] = colors[:, 0].astype(np.uint8)
        data['green'] = colors[:, 1].astype(np.uint8)
        data['blue'] = colors[:, 2].astype(np.uint8)
        
    el = PlyElement.describe(data, 'vertex')
    PlyData([el], text=False).write(str(ply_path))

def clean_point_cloud(input_ply_path, output_ply_path, nb_neighbors=20, std_ratio=2.0):
    in_path = Path(input_ply_path).resolve()
    out_path = Path(output_ply_path).resolve()
    
    if not in_path.exists():
        raise FileNotFoundError(f"Input PLY point cloud not found: {in_path}")
        
    pts, colors, normals = read_ply_points(in_path)
    raw_count = len(pts)
    
    if raw_count == 0:
        write_ply_points(out_path, pts, colors, normals)
        return 0, 0, 0
        
    k = min(nb_neighbors + 1, raw_count)
    tree = KDTree(pts)
    distances, _ = tree.query(pts, k=k)
    
    mean_dists = np.mean(distances[:, 1:], axis=1)
    
    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    
    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_dists <= threshold
    
    clean_pts = pts[inlier_mask]
    clean_colors = colors[inlier_mask] if colors is not None else None
    clean_normals = normals[inlier_mask] if normals is not None else None
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_ply_points(out_path, clean_pts, clean_colors, clean_normals)
    
    clean_count = len(clean_pts)
    removed_count = raw_count - clean_count
    
    return raw_count, clean_count, removed_count
