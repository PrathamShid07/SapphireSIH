from .placement_record import (
    PlacementTransform,
    PlacementValidation,
    PlacedAsset,
    FusedSceneManifest,
)
from .terrain_aligner import TerrainElevationModel, TerrainAligner
from .collision_checker import (
    CollisionChecker,
    compute_oriented_box_corners,
    compute_aabb,
    compute_3d_aabb_iou,
)
from .scene_composer import SceneComposer
from .stage11_pipeline import run_stage11

__all__ = [
    "PlacementTransform",
    "PlacementValidation",
    "PlacedAsset",
    "FusedSceneManifest",
    "TerrainElevationModel",
    "TerrainAligner",
    "CollisionChecker",
    "compute_oriented_box_corners",
    "compute_aabb",
    "compute_3d_aabb_iou",
    "SceneComposer",
    "run_stage11",
]
