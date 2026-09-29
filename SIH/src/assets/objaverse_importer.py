import json
import logging
from pathlib import Path
from typing import Optional

import trimesh

from .asset_record import AssetDimensions, AssetRecord
from .asset_normalizer import normalize_and_export

logger = logging.getLogger(__name__)


def download_objaverse_glb(uid: str, cache_dir: Path) -> Optional[Path]:
    try:
        import requests
    except ImportError:
        logger.error("requests package not available")
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{uid}.glb"
    if cached.exists():
        logger.info(f"Objaverse asset {uid} already cached: {cached}")
        return cached

    url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/{uid[:2]}/{uid}.glb"
    logger.info(f"Downloading Objaverse asset {uid} from {url}")
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            cached.write_bytes(r.content)
            logger.info(f"Saved {uid}.glb ({len(r.content)} bytes)")
            return cached
        logger.warning(f"HTTP {r.status_code} for Objaverse UID {uid}")
        return None
    except Exception as e:
        logger.warning(f"Failed to download {uid}: {e}")
        return None


def import_external_mesh(
    asset_id: str,
    class_name: str,
    subcategory: str,
    source: str,
    license: str,
    target_length: float,
    target_width: float,
    target_height: float,
    output_dir: Path,
    source_up: str = "+Y",
    source_forward: str = "+Z",
    mesh_path: Optional[Path] = None,
    objaverse_uid: Optional[str] = None,
    objaverse_cache_dir: Optional[Path] = None,
    notes: str = "",
) -> Optional[AssetRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if mesh_path is None or not mesh_path.exists():
        if objaverse_uid:
            cache = objaverse_cache_dir or (output_dir / "objaverse_cache")
            mesh_path = download_objaverse_glb(objaverse_uid, cache)
        if mesh_path is None or not mesh_path.exists():
            logger.error(f"No mesh file found for asset {asset_id}")
            return None

    output_glb = output_dir / f"{asset_id}.glb"
    output_obj = output_dir / f"{asset_id}.obj"

    try:
        diag = normalize_and_export(
            input_path=mesh_path,
            output_glb=output_glb,
            output_obj=output_obj,
            target_length=target_length,
            target_width=target_width,
            target_height=target_height,
            source_up=source_up,
            source_forward=source_forward,
        )
        logger.info(f"Imported and normalized {asset_id}: {diag}")
    except Exception as e:
        logger.error(f"Failed to normalize {asset_id}: {e}")
        return None

    record = AssetRecord(
        asset_id=asset_id,
        source=source,
        license=license,
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
        tags=[class_name, subcategory, source],
        notes=notes,
    )

    errors = record.validate()
    if errors:
        logger.warning(f"AssetRecord validation warnings for {asset_id}: {errors}")

    return record
