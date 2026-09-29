# System Architecture

## End-to-End Pipeline

```text
DRONE VIDEO / IMAGES
        |
        v
1. Video preprocessing
        |
        v
2. Keyframe selection + quality filtering
        |
        v
3. Camera calibration + metadata extraction
        |
        v
4. GPS/IMU trajectory processing
        |
        v
5. COLMAP SfM
        |
        v
6. Camera poses + sparse point cloud
        |
        v
7. Dense reconstruction / MVS
        |
        v
8. Semantic segmentation
        |
        +--------------------------+
        |                          |
        v                          v
9. Dynamic object detection    Static classes
        |                          |
        v                          |
10. Object tracking               |
        |                          |
        v                          |
11. Dynamic-object masking        |
        |                          |
        +------------+-------------+
                     |
                     v
12. Static scene reconstruction
                     |
                     v
13. Semantic 3D fusion
                     |
                     v
14. 3D localization of objects
                     |
                     v
15. Predefined asset retrieval
                     |
                     v
16. Asset normalization + placement
                     |
                     v
17. Scene fusion
                     |
                     v
18. Texturing + mesh cleanup
                     |
                     v
19. Georeferencing
                     |
                     v
20. Final exports
                     |
                     v
21. Web 3D viewer
                     |
                     v
22. Evaluation + runtime report
```

## Static Reconstruction

Use photogrammetry/SfM/MVS for actual static geometry.

Recommended foundation:

- COLMAP for SfM and camera poses
- COLMAP/OpenMVS for dense reconstruction
- Open3D/PyMeshLab or equivalent for processing where useful

## Dynamic Object Pipeline

For each object:

1. Detect
2. Segment
3. Track across frames
4. Estimate depth
5. Use camera intrinsics
6. Use camera pose
7. Estimate ground contact
8. Estimate dimensions
9. Fuse observations across frames
10. Select asset
11. Normalize asset
12. Place asset
13. Validate terrain/road intersection
14. Validate collision/intersection
15. Add metadata and confidence

## Scene Fusion

The final scene should contain:

- Reconstructed terrain
- Reconstructed roads
- Reconstructed buildings
- Reconstructed fixed infrastructure
- Replaced cars
- Replaced trees
- Other selected predefined assets

The original dynamic geometry should be masked or removed where feasible before final static reconstruction.

## Important Experiment

Compare:

### Baseline
Normal photogrammetry

### Pipeline B
Photogrammetry + dynamic-object masking

### Pipeline C
Photogrammetry + dynamic masking + predefined 3D asset replacement

The comparison should use measured results rather than claims.
