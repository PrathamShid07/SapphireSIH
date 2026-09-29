from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AssetDimensions:
    length: float
    width: float
    height: float
    units: str = "meters"

    def aspect_length_width(self) -> float:
        return self.length / max(self.width, 1e-6)

    def aspect_height_width(self) -> float:
        return self.height / max(self.width, 1e-6)

    def to_dict(self) -> dict:
        return {
            "length": round(self.length, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "units": self.units,
        }


REQUIRED_ASSET_FIELDS = {
    "asset_id", "source", "license", "class_name", "subcategory",
    "dimensions", "coordinate_system", "up_axis", "origin_description",
    "forward_axis", "texture_available", "mesh_file",
}


@dataclass
class AssetRecord:
    asset_id: str
    source: str
    license: str
    class_name: str
    subcategory: str
    dimensions: AssetDimensions
    coordinate_system: str
    up_axis: str
    forward_axis: str
    origin_description: str
    texture_available: bool
    mesh_file: str
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def validate(self) -> List[str]:
        errors = []
        if not self.asset_id:
            errors.append("asset_id is empty")
        if not self.class_name:
            errors.append("class_name is empty")
        if self.dimensions.length <= 0:
            errors.append("dimensions.length must be positive")
        if self.dimensions.width <= 0:
            errors.append("dimensions.width must be positive")
        if self.dimensions.height <= 0:
            errors.append("dimensions.height must be positive")
        if self.up_axis not in ("+X", "+Y", "+Z", "-X", "-Y", "-Z"):
            errors.append(f"up_axis '{self.up_axis}' is not a valid axis")
        if self.forward_axis not in ("+X", "+Y", "+Z", "-X", "-Y", "-Z"):
            errors.append(f"forward_axis '{self.forward_axis}' is not a valid axis")
        return errors

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "source": self.source,
            "license": self.license,
            "class_name": self.class_name,
            "subcategory": self.subcategory,
            "dimensions": self.dimensions.to_dict(),
            "coordinate_system": self.coordinate_system,
            "up_axis": self.up_axis,
            "forward_axis": self.forward_axis,
            "origin_description": self.origin_description,
            "texture_available": self.texture_available,
            "mesh_file": self.mesh_file,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssetRecord":
        dims = d["dimensions"]
        return cls(
            asset_id=d["asset_id"],
            source=d["source"],
            license=d["license"],
            class_name=d["class_name"],
            subcategory=d["subcategory"],
            dimensions=AssetDimensions(
                length=dims["length"],
                width=dims["width"],
                height=dims["height"],
                units=dims.get("units", "meters"),
            ),
            coordinate_system=d["coordinate_system"],
            up_axis=d["up_axis"],
            forward_axis=d["forward_axis"],
            origin_description=d["origin_description"],
            texture_available=d.get("texture_available", False),
            mesh_file=d.get("mesh_file", ""),
            tags=d.get("tags", []),
            notes=d.get("notes", ""),
        )


@dataclass
class MatchingScore:
    asset_id: str
    class_name: str
    subcategory: str
    total_score: float
    class_score: float
    dimension_score: float
    shape_score: float
    context_score: float
    suggested_scale: float
    dimension_error_m: dict
    mesh_file: str

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "class_name": self.class_name,
            "subcategory": self.subcategory,
            "total_score": round(self.total_score, 4),
            "class_score": round(self.class_score, 4),
            "dimension_score": round(self.dimension_score, 4),
            "shape_score": round(self.shape_score, 4),
            "context_score": round(self.context_score, 4),
            "suggested_scale": round(self.suggested_scale, 4),
            "dimension_error_m": {k: round(v, 4) for k, v in self.dimension_error_m.items()},
            "mesh_file": self.mesh_file,
        }
