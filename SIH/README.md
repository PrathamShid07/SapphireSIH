# UAV Digital Twin Pipeline

A modular Python pipeline that converts UAV/drone video into a detailed hybrid 3D digital twin.

## Overview

- Reconstruct static structures (buildings, roads, terrain) from real drone imagery via photogrammetry
- Detect, track, localize, and replace repeated/dynamic objects (vehicles, trees) with predefined 3D assets
- Produce a metrically useful, textured 3D scene
- Export OBJ, PLY, GLB/GLTF formats
- Provide quantitative evaluation

## Repository Structure

```
SIH/
├── main.py                  # Entrypoint — delegates to src/cli/
├── requirements.txt         # Python dependencies
├── pytest.ini               # Test configuration
│
├── src/                     # All source code
│   ├── cli/                 # CLI command routing (main_cli.py)
│   ├── config/              # Config management
│   ├── preprocessing/       # Stage 2: Frame extraction
│   ├── metadata/            # Stage 3: GPS/IMU telemetry
│   ├── sfm/                 # Stage 4: COLMAP sparse reconstruction
│   ├── dense/               # Stage 5: MVS dense reconstruction
│   ├── segmentation/        # Stage 6: UAVid semantic segmentation
│   ├── detection/           # Stage 7: Vehicle detection & tracking
│   ├── masking/             # Stage 8: Dynamic object masking
│   ├── localization/        # Stage 9: 3D object localization
│   ├── assets/              # Stage 10: 3D asset library management
│   ├── fusion/              # Stage 11: Scene fusion & composition
│   ├── texturing/           # Stage 12: Texturing, UV, LOD
│   └── utils/               # Logging, system info, shared helpers
│
├── config/                  # YAML configuration files
├── data/                    # Data directories (NOT tracked in git)
│   ├── input/               # Raw drone videos (.mp4)
│   ├── raw/                 # UAVid dataset images
│   ├── interim/             # Intermediate processing results
│   ├── processed/           # Final processed data
│   └── assets/              # Curated 3D model library (tracked)
│
├── outputs/                 # Pipeline outputs (NOT tracked in git)
├── logs/                    # Runtime logs (NOT tracked in git)
├── tests/                   # Pytest test suite (one file per stage)
├── scripts/                 # Utility scripts
│   ├── run_all_stages.ps1   # Windows: run full pipeline
│   └── cluster/             # SPIT GPU cluster SLURM scripts
└── docs/                    # Design and stage documentation
```

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Place your video

```
data/input/your_video.mp4
```

### 3. Run the full pipeline

```powershell
# Windows (PowerShell)
.\.venv\Scripts\python.exe main.py preprocess-video --video data/input/your_video.mp4
.\.venv\Scripts\python.exe main.py extract-metadata --video data/input/your_video.mp4
.\.venv\Scripts\python.exe main.py reconstruct-sparse --video data/input/your_video.mp4
.\.venv\Scripts\python.exe main.py reconstruct-dense --video data/input/your_video.mp4
.\.venv\Scripts\python.exe main.py track-vehicles --video data/input/your_video.mp4
.\.venv\Scripts\python.exe main.py mask-reconstruct --video data/input/your_video.mp4
.\.venv\Scripts\python.exe main.py localize-3d --video data/input/your_video.mp4
.\.venv\Scripts\python.exe main.py manage-assets
.\.venv\Scripts\python.exe main.py match-assets
.\.venv\Scripts\python.exe main.py fuse-scene
.\.venv\Scripts\python.exe main.py texture-clean-mesh --quality all
```

Or use the convenience script:

```powershell
# Edit VIDEO path in script first
.\scripts\run_all_stages.ps1
```

### 4. Run tests

```bash
pytest tests/ -v
```

## Pipeline Stages

| Stage | Command | Description |
|-------|---------|-------------|
| 2 | `preprocess-video` | Extract frames, compute sharpness, deduplicate |
| 3 | `extract-metadata` | Parse GPS/IMU telemetry from video |
| 4 | `reconstruct-sparse` | COLMAP sparse SfM reconstruction |
| 5 | `reconstruct-dense` | MVS dense point cloud |
| 6 | `train-segmentation` | Train UAVid semantic segmentation model |
| 7 | `track-vehicles` | Detect & track vehicles per frame |
| 8 | `mask-reconstruct` | Generate dynamic masks, re-run clean SfM |
| 9 | `localize-3d` | Lift 2D tracks to 3D positions |
| 10 | `manage-assets` | Normalize & catalog 3D asset library |
| 11 | `fuse-scene` | Compose static mesh + placed assets |
| 12 | `texture-clean-mesh` | UV unwrap, texture, LOD decimation |

## GPU Cluster (SPIT)

See [`scripts/cluster/README.md`](scripts/cluster/README.md) for setup and SLURM job submission.

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Detection & segmentation models |
| `opencv-python` | Frame extraction, image processing |
| `trimesh` | Mesh loading, manipulation, export |
| `pymeshlab` | Robust mesh cleaning and decimation |
| `plyfile` | Point cloud I/O |
| `scipy` | RANSAC, spatial operations |
| `pyyaml` | Configuration files |
| `psutil` | System info and resource monitoring |
| `requests` | Objaverse 3D asset downloading |

## Data

Large data files are **not tracked in git**. Place them manually:

- `data/input/` — drone video files (`.mp4`)
- `data/raw/uavid_train/`, `data/raw/uavid_val/`, `data/raw/uavid_test/` — [UAVid dataset](https://uavid.nl/)
- `data/assets/` — 3D model library (`.glb`, `.obj`) — tracked in git
