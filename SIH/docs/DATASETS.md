# Dataset Strategy

Do not use one dataset for every component. Use a dataset stack.

## 1. UAVid — Primary UAV Semantic Dataset

Official:
https://uavid.nl/

Primary use:

- UAV semantic segmentation
- Building segmentation
- Road segmentation
- Tree segmentation
- Static car segmentation
- Moving car segmentation
- Urban UAV scene understanding

Important classes include:

- Building
- Road
- Static car
- Tree
- Low vegetation
- Human
- Moving car
- Background clutter

Development approach:
Start with a small subset. Verify the entire training/inference/evaluation flow before scaling up.

## 2. UAVid-3D-Scenes — Depth and Camera Pose

Official:
https://huggingface.co/datasets/hrflr/uavid-3d-scenes

Use for:

- Depth experiments
- Metric depth
- Camera intrinsics
- Camera poses
- RGB-depth fusion
- 3D localization validation
- COLMAP-related 3D experiments

## 3. VisDrone — Detection and Tracking

Official:
https://github.com/VisDrone/VisDrone-Dataset

Use for:

- Car detection
- Vehicle detection
- UAV object detection
- Multi-object tracking
- Persistent IDs

Relevant categories include:

- Pedestrian
- People
- Bicycle
- Car
- Van
- Truck
- Tricycle
- Bus
- Motor

Prioritize video/MOT data for tracking experiments.

## 4. Objaverse — Predefined 3D Assets

Official:
https://objaverse.allenai.org/

Use as an asset source, not as the main perception dataset.

Initially curate a small library:

- 5–10 car models
- 5–10 tree models
- A few truck/bus models if required

Record:

- Asset ID
- Source
- License
- Class
- Dimensions
- Units
- Coordinate system
- Up axis
- Origin
- Orientation
- Texture availability

## 5. OmniObject3D — Optional Asset Source

Official:
https://omniobject3d.github.io/

Use only when additional real scanned assets are useful.

## 6. OpenEarthMap — Supplementary Aerial Semantics

Official:
https://open-earth-map.org/overview_oem.html

Useful for:

- Road
- Tree
- Building
- Developed space
- Land-cover semantics

It is supplementary, not the primary UAV-video dataset.

## 7. ETH3D — Reconstruction Evaluation

Official:
https://www.eth3d.net/overview

Use for:

- Multi-view stereo research
- Geometry evaluation
- Point cloud evaluation
- Reconstruction experiments

## Dataset Priority

1. UAVid
2. UAVid-3D-Scenes
3. VisDrone
4. Objaverse
5. OmniObject3D if needed
6. OpenEarthMap if needed
7. ETH3D for evaluation

Do not download everything at once.

Recommended order:

1. Small UAVid subset
2. UAVid-3D-Scenes
3. Small VisDrone subset
4. Curated Objaverse assets
5. Integration
6. Test on real project drone video

## Final Test Dataset

The user's own drone video should be treated as the final real-world test input.

Public datasets are used for training, validation and component evaluation.
