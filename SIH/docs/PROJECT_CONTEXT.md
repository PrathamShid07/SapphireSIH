# Project Context

## Project Title

Drone Video to Hybrid 3D Digital Twin

## Objective

Build a real working research prototype that accepts UAV/drone video or images and produces a detailed 3D digital twin of the captured environment.

The final scene should combine:

### Actual reconstructed geometry
- Buildings
- Building facades
- Rooftops
- Roads
- Terrain
- Sidewalks
- Walls
- Poles and fixed infrastructure
- Other static structures

### Predefined 3D assets
- Cars
- Trucks
- Buses
- Trees
- Other repeated/dynamic objects

The predefined objects are detected and localized from the drone imagery, then replaced by suitable normalized 3D assets.

## Hybrid Reconstruction Principle

The system is not intended to photogrammetrically reconstruct every individual car or tree.

Instead:

`Drone imagery -> perception -> 3D localization -> asset selection -> asset placement`

Static scene:

`Drone imagery -> SfM/MVS -> point cloud -> mesh -> texture`

Dynamic/repeated objects:

`Detection -> segmentation -> tracking -> depth/pose -> 3D localization -> asset retrieval -> placement`

## Geometry Confidence

The system must distinguish conceptually between:

- Observed geometry
- Partially observed geometry
- Occluded geometry
- Inferred geometry

Never claim perfect reconstruction for surfaces that were never visible.

## Target Outputs

Where practical, support:

- OBJ
- PLY
- GLB
- GLTF
- FBX
- LAS
- GeoTIFF
- Textured mesh
- Point cloud
- Semantic labels
- Object metadata
- Camera trajectory

Also generate:

- Runtime report
- Accuracy report
- Reconstruction report
- Detection report
- Tracking report
- 3D localization report

## Final User Command

The eventual target is:

`python run_pipeline.py --video input.mp4`

Optional:

`python run_pipeline.py --video input.mp4 --gps gps.csv`

The pipeline should be resume-safe and produce a run-specific output directory.
