import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .asset_record import AssetDimensions, AssetRecord
from .asset_normalizer import normalize_and_export
from .curated_models import CURATED_GENERATORS, SUBCATEGORY_TO_CLASS

logger = logging.getLogger(__name__)


class LibraryManager:
    def __init__(self, assets_dir: Path, normalized_dir: Optional[Path] = None):
        self.assets_dir = Path(assets_dir)
        self.normalized_dir = Path(normalized_dir) if normalized_dir else self.assets_dir / "normalized"
        self.catalog_path = self.assets_dir / "assets_catalog.json"
        self._catalog: Dict[str, dict] = {}

    def _load_catalog(self) -> None:
        if self.catalog_path.exists():
            with open(self.catalog_path, "r") as f:
                self._catalog = json.load(f)
        else:
            self._catalog = {}

    def _save_catalog(self) -> None:
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        with open(self.catalog_path, "w") as f:
            json.dump(self._catalog, f, indent=2)

    def initialize_curated_library(self, force: bool = False) -> int:
        self._load_catalog()
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_dir.mkdir(parents=True, exist_ok=True)

        added = 0
        for subcategory, (generator_fn, dims) in CURATED_GENERATORS.items():
            asset_id = f"curated_{subcategory}"
            if asset_id in self._catalog and not force:
                glb_path = Path(self._catalog[asset_id].get("mesh_file", ""))
                if glb_path.exists():
                    continue

            logger.info(f"Generating curated asset: {subcategory}")
            try:
                mesh = generator_fn()
                output_glb = self.normalized_dir / f"{asset_id}.glb"
                output_obj = self.normalized_dir / f"{asset_id}.obj"
                diag = normalize_and_export(
                    input_path=None,
                    output_glb=output_glb,
                    output_obj=output_obj,
                    target_length=dims["length"],
                    target_width=dims["width"],
                    target_height=dims["height"],
                    source_up="+Z",
                    source_forward="+X",
                    mesh=mesh,
                )
                class_name = SUBCATEGORY_TO_CLASS.get(subcategory, "unknown")
                record = AssetRecord(
                    asset_id=asset_id,
                    source="curated_procedural",
                    license="CC0",
                    class_name=class_name,
                    subcategory=subcategory,
                    dimensions=AssetDimensions(
                        length=diag["normalized_length"],
                        width=diag["normalized_width"],
                        height=diag["normalized_height"],
                        units="meters",
                    ),
                    coordinate_system="right-handed (+X forward, +Y left, +Z up)",
                    up_axis="+Z",
                    forward_axis="+X",
                    origin_description="ground contact center at (0, 0, 0)",
                    texture_available=False,
                    mesh_file=str(output_glb),
                    tags=[class_name, subcategory, "curated", "procedural"],
                )
                errors = record.validate()
                if errors:
                    logger.warning(f"Validation warnings for {asset_id}: {errors}")
                self._catalog[asset_id] = record.to_dict()
                added += 1
            except Exception as e:
                logger.error(f"Failed to generate {subcategory}: {e}")

        self._save_catalog()
        logger.info(f"Library initialized. {added} new assets generated. Total: {len(self._catalog)}")
        return added

    def register_asset(self, record: AssetRecord) -> None:
        self._load_catalog()
        self._catalog[record.asset_id] = record.to_dict()
        self._save_catalog()

    def get_asset(self, asset_id: str) -> Optional[AssetRecord]:
        self._load_catalog()
        d = self._catalog.get(asset_id)
        if d is None:
            return None
        return AssetRecord.from_dict(d)

    def list_assets(
        self,
        class_name: Optional[str] = None,
        subcategory: Optional[str] = None,
    ) -> List[AssetRecord]:
        self._load_catalog()
        results = []
        for d in self._catalog.values():
            if class_name and d.get("class_name", "").lower() != class_name.lower():
                continue
            if subcategory and d.get("subcategory", "").lower() != subcategory.lower():
                continue
            results.append(AssetRecord.from_dict(d))
        return results

    def verify_library(self) -> dict:
        self._load_catalog()
        total = len(self._catalog)
        issues = []
        missing_files = 0
        invalid_records = 0

        for asset_id, d in self._catalog.items():
            try:
                record = AssetRecord.from_dict(d)
                errors = record.validate()
                if errors:
                    issues.append({"asset_id": asset_id, "errors": errors})
                    invalid_records += 1
                mesh_path = Path(d.get("mesh_file", ""))
                if not mesh_path.exists():
                    issues.append({"asset_id": asset_id, "errors": ["mesh_file not found"]})
                    missing_files += 1
            except Exception as e:
                issues.append({"asset_id": asset_id, "errors": [str(e)]})
                invalid_records += 1

        return {
            "total_assets": total,
            "missing_files": missing_files,
            "invalid_records": invalid_records,
            "issues": issues,
            "status": "OK" if not issues else "ISSUES_FOUND",
        }

    def get_catalog_summary(self) -> dict:
        self._load_catalog()
        class_counts: Dict[str, int] = {}
        for d in self._catalog.values():
            cn = d.get("class_name", "unknown")
            class_counts[cn] = class_counts.get(cn, 0) + 1
        return {
            "total_assets": len(self._catalog),
            "by_class": class_counts,
            "catalog_path": str(self.catalog_path),
        }
