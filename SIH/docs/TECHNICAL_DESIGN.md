# Technical Design Notes

## Video Processing

Do not process every frame by default.

Use:

- FPS sampling
- Keyframe selection
- Blur filtering
- Brightness filtering
- Contrast filtering
- Near-duplicate filtering

The objective is to retain frames that provide strong geometric overlap while reducing computation.

## Photogrammetry

COLMAP should be the primary SfM foundation.

Expected flow:

1. Feature extraction
2. Feature matching
3. Sparse mapping
4. Camera poses
5. Sparse cloud
6. Dense reconstruction

High overlap between useful frames is important.

## Dynamic Objects

Moving cars can create ghosting, duplicate geometry, or incorrect camera/point correspondences.

Use dynamic masks to reduce their influence on the static reconstruction.

## 3D Localization

Object localization should not depend on a single frame if multiple observations exist.

Fuse:

- Pixel observations
- Camera poses
- Intrinsics
- Depth
- Ground/terrain geometry

Multi-frame fusion should reduce noise.

## Asset Placement

The asset's coordinate system must be normalized before placement.

For cars:
- Bottom plane = ground contact
- Longitudinal axis = known forward direction

For trees:
- Trunk base = ground contact
- Vertical axis aligned with terrain normal where appropriate

## Geospatial Coordinates

Do not use:

`XYZ = latitude, longitude, altitude`

Instead convert to an appropriate projected/local coordinate system.

If GPS is noisy, report that uncertainty.

## Confidence

Maintain confidence values where practical for:

- Semantic class
- Detection
- Tracking
- Depth
- 3D localization
- Asset match
- Geometry observation

## Failure Handling

Expected failure cases include:

- Too few useful frames
- Excessive blur
- Low overlap
- Poor texture
- Repeated patterns
- GPS noise
- Dynamic-object contamination
- Occlusion
- Missing camera metadata
- Poor calibration
- Insufficient viewpoints

The pipeline should report failures rather than silently producing unreliable outputs.
