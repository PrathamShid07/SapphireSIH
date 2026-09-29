from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class PlacementTransform:
    matrix: List[List[float]] = field(default_factory=lambda: np.eye(4).tolist())
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    @classmethod
    def from_components(
        cls,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        scale: np.ndarray,
    ) -> "PlacementTransform":
        pos = np.asarray(position, dtype=np.float64).flatten()[:3]
        rot = np.asarray(rotation_matrix, dtype=np.float64)[:3, :3]
        s = np.asarray(scale, dtype=np.float64).flatten()[:3]

        t_mat = np.eye(4, dtype=np.float64)
        t_mat[:3, :3] = rot * s
        t_mat[:3, 3] = pos

        yaw = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
        pitch = float(np.degrees(np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))))
        roll = float(np.degrees(np.arctan2(rot[2, 1], rot[2, 2])))

        return cls(
            matrix=[[float(v) for v in row] for row in t_mat],
            position=[float(v) for v in pos],
            scale=[float(v) for v in s],
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
        )

    def to_numpy(self) -> np.ndarray:
        return np.array(self.matrix, dtype=np.float64)


@dataclass
class PlacementValidation:
    is_floating: bool = False
    float_distance_m: float = 0.0
    terrain_penetration: bool = False
    penetration_depth_m: float = 0.0
    has_collision: bool = False
    colliding_tracks: List[int] = field(default_factory=list)
    was_adjusted: bool = False
    adjustment_details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_floating": self.is_floating,
            "float_distance_m": round(self.float_distance_m, 4),
            "terrain_penetration": self.terrain_penetration,
            "penetration_depth_m": round(self.penetration_depth_m, 4),
            "has_collision": self.has_collision,
            "colliding_tracks": self.colliding_tracks,
            "was_adjusted": self.was_adjusted,
            "adjustment_details": self.adjustment_details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlacementValidation":
        return cls(
            is_floating=bool(data.get("is_floating", False)),
            float_distance_m=float(data.get("float_distance_m", 0.0)),
            terrain_penetration=bool(data.get("terrain_penetration", False)),
            penetration_depth_m=float(data.get("penetration_depth_m", 0.0)),
            has_collision=bool(data.get("has_collision", False)),
            colliding_tracks=list(data.get("colliding_tracks", [])),
            was_adjusted=bool(data.get("was_adjusted", False)),
            adjustment_details=str(data.get("adjustment_details", "")),
        )


@dataclass
class PlacedAsset:
    track_id: int
    asset_id: str
    class_name: str
    subcategory: str
    is_dynamic: bool
    confidence: float
    mesh_file: str
    transform: PlacementTransform
    ground_contact_3d: List[float]
    bounding_box_3d: Dict[str, Any]
    validation: PlacementValidation = field(default_factory=PlacementValidation)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "asset_id": self.asset_id,
            "class_name": self.class_name,
            "subcategory": self.subcategory,
            "is_dynamic": self.is_dynamic,
            "confidence": round(self.confidence, 4),
            "mesh_file": self.mesh_file,
            "transform": {
                "matrix": self.transform.matrix,
                "position": [round(v, 4) for v in self.transform.position],
                "scale": [round(v, 4) for v in self.transform.scale],
                "yaw_deg": round(self.transform.yaw_deg, 2),
                "pitch_deg": round(self.transform.pitch_deg, 2),
                "roll_deg": round(self.transform.roll_deg, 2),
            },
            "ground_contact_3d": [round(v, 4) for v in self.ground_contact_3d],
            "bounding_box_3d": self.bounding_box_3d,
            "validation": self.validation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlacedAsset":
        t_data = data.get("transform", {})
        t = PlacementTransform(
            matrix=t_data.get("matrix", np.eye(4).tolist()),
            position=t_data.get("position", [0.0, 0.0, 0.0]),
            scale=t_data.get("scale", [1.0, 1.0, 1.0]),
            yaw_deg=t_data.get("yaw_deg", 0.0),
            pitch_deg=t_data.get("pitch_deg", 0.0),
            roll_deg=t_data.get("roll_deg", 0.0),
        )
        val = PlacementValidation.from_dict(data.get("validation", {}))
        return cls(
            track_id=int(data["track_id"]),
            asset_id=str(data["asset_id"]),
            class_name=str(data["class_name"]),
            subcategory=str(data.get("subcategory", "")),
            is_dynamic=bool(data.get("is_dynamic", False)),
            confidence=float(data.get("confidence", 1.0)),
            mesh_file=str(data.get("mesh_file", "")),
            transform=t,
            ground_contact_3d=list(data.get("ground_contact_3d", [0.0, 0.0, 0.0])),
            bounding_box_3d=dict(data.get("bounding_box_3d", {})),
            validation=val,
        )


@dataclass
class FusedSceneManifest:
    timestamp: str
    base_mesh_file: Optional[str]
    total_placed_assets: int
    static_assets: int
    dynamic_assets: int
    floating_detected: int
    penetrations_detected: int
    collisions_resolved: int
    exported_files: Dict[str, str] = field(default_factory=dict)
    placed_assets: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "base_mesh_file": self.base_mesh_file,
            "total_placed_assets": self.total_placed_assets,
            "static_assets": self.static_assets,
            "dynamic_assets": self.dynamic_assets,
            "floating_detected": self.floating_detected,
            "penetrations_detected": self.penetrations_detected,
            "collisions_resolved": self.collisions_resolved,
            "exported_files": self.exported_files,
            "placed_assets": self.placed_assets,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
