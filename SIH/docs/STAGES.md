# Implementation Stages

## Stage 1 — Project Foundation

Implement:

- Project structure
- Virtual environment verification
- Dependency verification
- requirements file
- Configuration system
- Logging
- CLI foundation
- Input/output directories
- System information
- GPU/CUDA detection

Test everything.

Do not implement reconstruction or ML yet.

---

## Stage 2 — Video Preprocessing

Implement:

- Video metadata extraction
- Configurable FPS sampling
- Frame extraction
- Blur detection
- Brightness detection
- Contrast detection
- Near-duplicate detection
- Keyframe selection
- Frame manifest
- Selected-frame directory
- Contact sheet
- Resume-safe processing

Manifest should contain:

- Frame ID
- Timestamp
- Original frame number
- Quality scores
- Selected/not selected
- Rejection reason

---

## Stage 3 — Camera and Flight Metadata

Implement:

- Camera intrinsics
- Image dimensions
- FPS
- Timestamps
- GPS
- Altitude
- Yaw
- Pitch
- Roll
- IMU metadata if available

Normalize metadata to JSON/CSV.

Generate trajectory visualization where valid.

Never invent missing GPS.

---

## Stage 4 — COLMAP SfM

Implement:

- Feature extraction
- Feature matching
- Camera estimation
- Image registration
- Sparse reconstruction
- Camera poses
- Sparse point cloud

Generate diagnostics:

- Registered image count
- Registration percentage
- Number of cameras
- Number of sparse points
- Reconstruction failure reasons

Do not perform dense reconstruction yet.

---

## Stage 5 — Dense Reconstruction

Implement:

- Dense depth
- Dense point cloud
- Point-cloud cleanup
- Mesh reconstruction
- Mesh cleanup
- Initial texture generation
- PLY/OBJ/GLB export where supported

Generate a reconstruction report.

---

## Stage 6 — UAVid Semantic Segmentation

Start with a small development subset.

Implement:

- Dataset loader
- Training/inference pipeline
- Semantic masks
- Visualization overlays
- Per-frame masks
- Evaluation

Evaluate:

- IoU
- mIoU
- Pixel accuracy
- Per-class IoU

Keep model architecture modular.

---

## Stage 7 — Detection, Instance Segmentation and Tracking

Implement:

- Car/vehicle detection
- Object segmentation where useful
- Multi-object tracking
- Persistent object IDs
- Frame-level metadata
- Tracking visualization

Use VisDrone for appropriate training/evaluation.

Do not perform 3D placement yet.

---

## Stage 8 — Dynamic Object Masking

Use detection/segmentation/tracking outputs to create masks.

Implement:

- Dynamic masks
- Mask propagation where appropriate
- Reconstruction with masks
- Comparison against unmasked reconstruction

Measure the effect on reconstruction rather than assuming improvement.

An optional, opt-in optimized path (`masking.reuse_stage4_poses`, default
`false`) reuses Stage 4's COLMAP database and camera poses via
`point_triangulator` instead of re-running feature extraction, matching, and
the mapper a second time. See `docs/STAGE8_OPTIMIZATION.md` for the full
architecture, empirical COLMAP validation, benchmark results, and current
EXPERIMENTAL status.

---

## Stage 9 — 3D Object Localization

For each tracked object estimate:

- 3D position
- Ground contact
- Orientation
- Width
- Length
- Height
- Confidence

Use:

- Camera intrinsics
- Camera poses
- Depth
- Segmentation masks
- Terrain/road geometry
- Multi-frame fusion

Output:

`objects_3d.json`

Visualize localized objects on the reconstructed point cloud.

---

## Stage 10 — 3D Asset Library

Create a small curated library from Objaverse and/or OmniObject3D.

Normalize:

- Units
- Scale
- Coordinate system
- Up axis
- Origin
- Orientation
- Ground-contact plane

Implement candidate matching based on:

- Class
- Estimated dimensions
- Shape/category
- Scene context

Do not place assets yet.

---

## Stage 11 — Asset Placement and Scene Fusion

Place assets using 3D localization.

For cars:

- Road/ground contact
- Correct scale
- Orientation
- Terrain alignment

For trees:

- Ground contact
- Height
- Canopy scale
- Terrain alignment

Implement:

- Collision checks
- Intersection checks
- Floating-object checks
- Terrain penetration checks

Buildings and terrain remain reconstructed geometry.

Export:

- OBJ
- PLY
- GLB/GLTF

where supported.

---

## Stage 12 — Texturing and Mesh Cleanup

Implement:

- Texture blending
- UV processing
- Hole handling
- Mesh cleanup
- Normal correction
- Texture optimization

Support:

- High quality
- Medium quality
- Low quality

Keep a high-quality master output and optimized output where practical.

---

## Stage 13 — Georeferencing

Implement:

- Local/projected coordinate system
- GPS alignment
- RTK/PPK support where available
- Similarity transformation
- Scale correction where justified

Report:

- CRS
- Origin
- Transform
- Scale
- Alignment error
- Positioning source

Never claim centimeter accuracy from ordinary GPS.

---

## Stage 14 — Web 3D Viewer

Use:

- React
- Three.js
- React Three Fiber if useful

Support:

- GLB loading
- Orbit controls
- Zoom
- Pan
- Semantic toggles
- Object selection
- Object metadata
- Confidence
- Camera trajectory
- Original vs hybrid scene comparison

---

## Stage 15 — Evaluation Framework

Evaluate:

### Frame processing
- Selection rate
- Rejection rate
- Processing time

### Camera reconstruction
- Registration percentage
- Pose consistency

### Segmentation
- IoU
- mIoU
- Pixel accuracy

### Detection
- Precision
- Recall
- mAP

### Tracking
- IDF1
- MOTA
- ID switches

### 3D localization
- Position error
- Scale error
- Orientation error

### Reconstruction
- Point-cloud distance
- Chamfer distance where applicable
- Mesh quality

### Final system
- Runtime
- Memory
- GPU usage
- Successfully reconstructed objects
- Successfully replaced objects

Do not fabricate metrics.

---

## Stage 16 — Runtime Optimization

Target:

Less than 15 minutes for a 10-minute video where hardware allows.

Implement profiles:

- FAST
- BALANCED
- HIGH_QUALITY

Use:

- Keyframe selection
- Frame downsampling
- GPU inference
- Batch processing
- Caching
- Parallel processing where safe
- Resume support

Profile each stage and record actual times.

---

## Stage 17 — Final Integration

Create:

`run_pipeline.py`

Target command:

`python run_pipeline.py --video input.mp4`

Optional:

`python run_pipeline.py --video input.mp4 --gps gps.csv`

Requirements:

- Resume-safe
- Run IDs
- Structured logs
- Intermediate caching
- Final JSON report
- Preview outputs
- Exported 3D scene
- Runtime report
- Accuracy report

The pipeline should fail clearly when required data or dependencies are missing.
