import logging
from typing import Dict, List, Optional

from .asset_record import AssetDimensions, AssetRecord, MatchingScore
from .library_manager import LibraryManager

logger = logging.getLogger(__name__)

CLASS_SYNONYMS: Dict[str, List[str]] = {
    "car": ["car", "sedan", "suv", "hatchback", "coupe", "sports_car", "station_wagon", "compact"],
    "van": ["van", "cargo_van"],
    "truck": ["truck", "box_truck", "flatbed_truck", "pickup_truck"],
    "bus": ["bus", "city_bus", "coach_bus"],
    "tree": ["tree", "oak_tree", "pine_tree", "palm_tree", "round_deciduous", "cypress_tree", "birch_tree"],
}

SUBCATEGORY_ASPECT_PROFILES: Dict[str, Dict[str, float]] = {
    "sedan": {"lw": 2.44, "hw": 0.81},
    "suv": {"lw": 2.42, "hw": 0.89},
    "hatchback": {"lw": 2.29, "hw": 0.86},
    "coupe": {"lw": 2.43, "hw": 0.70},
    "sports_car": {"lw": 2.29, "hw": 0.64},
    "station_wagon": {"lw": 2.58, "hw": 0.85},
    "compact": {"lw": 2.26, "hw": 0.88},
    "box_truck": {"lw": 3.20, "hw": 1.28},
    "flatbed_truck": {"lw": 3.40, "hw": 0.72},
    "pickup_truck": {"lw": 2.75, "hw": 0.95},
    "cargo_van": {"lw": 2.60, "hw": 1.10},
    "city_bus": {"lw": 4.04, "hw": 1.23},
    "coach_bus": {"lw": 4.62, "hw": 1.35},
    "oak_tree": {"lw": 1.00, "hw": 1.14},
    "pine_tree": {"lw": 1.00, "hw": 2.40},
    "palm_tree": {"lw": 1.00, "hw": 1.67},
    "round_deciduous": {"lw": 1.00, "hw": 1.17},
    "cypress_tree": {"lw": 1.00, "hw": 4.50},
    "birch_tree": {"lw": 1.00, "hw": 1.80},
}


def _class_score(query_class: str, candidate_class: str, candidate_subcategory: str) -> float:
    qc = query_class.lower()
    cc = candidate_class.lower()
    cs = candidate_subcategory.lower()
    if qc == cc:
        return 1.0
    syns = CLASS_SYNONYMS.get(qc, [])
    if cs in syns or cc in syns:
        return 0.8
    return 0.0


def _dimension_score(
    query: AssetDimensions,
    candidate: AssetDimensions,
) -> tuple:
    eps = 1e-6
    el = abs(query.length - candidate.length) / max(query.length, eps)
    ew = abs(query.width - candidate.width) / max(query.width, eps)
    eh = abs(query.height - candidate.height) / max(query.height, eps)
    score = max(0.0, 1.0 - (0.5 * el + 0.3 * ew + 0.2 * eh))
    error_m = {
        "length_err_m": abs(query.length - candidate.length),
        "width_err_m": abs(query.width - candidate.width),
        "height_err_m": abs(query.height - candidate.height),
    }
    return float(score), error_m


def _shape_score(query: AssetDimensions, candidate_subcategory: str) -> float:
    profile = SUBCATEGORY_ASPECT_PROFILES.get(candidate_subcategory)
    if profile is None:
        return 0.5
    eps = 1e-6
    q_lw = query.length / max(query.width, eps)
    q_hw = query.height / max(query.width, eps)
    c_lw = profile["lw"]
    c_hw = profile["hw"]
    err_lw = abs(q_lw - c_lw) / max(c_lw, eps)
    err_hw = abs(q_hw - c_hw) / max(c_hw, eps)
    score = max(0.0, 1.0 - 0.5 * (err_lw + err_hw))
    return float(score)


def _context_score(
    is_dynamic: bool,
    candidate_class: str,
    candidate_subcategory: str,
    speed_mps: float = 0.0,
) -> float:
    cn = candidate_class.lower()
    if cn == "tree":
        return 0.9 if not is_dynamic else 0.4
    if is_dynamic:
        if speed_mps > 20.0 and candidate_subcategory in ("sports_car", "coupe"):
            return 0.9
        if speed_mps > 5.0 and cn in ("car", "van", "truck", "bus"):
            return 0.85
        return 0.7
    else:
        return 0.8


class AssetMatcher:
    def __init__(
        self,
        library: LibraryManager,
        class_weight: float = 0.5,
        dimension_weight: float = 0.3,
        shape_weight: float = 0.15,
        context_weight: float = 0.05,
        top_k: int = 3,
    ):
        self.library = library
        self.w_class = class_weight
        self.w_dim = dimension_weight
        self.w_shape = shape_weight
        self.w_ctx = context_weight
        self.top_k = top_k

    def match(
        self,
        class_name: str,
        dimensions: AssetDimensions,
        is_dynamic: bool = False,
        speed_mps: float = 0.0,
    ) -> List[MatchingScore]:
        candidates = self.library.list_assets()
        if not candidates:
            logger.warning("Library is empty; no candidates returned")
            return []

        scores: List[MatchingScore] = []
        for record in candidates:
            cs = _class_score(class_name, record.class_name, record.subcategory)
            if cs == 0.0:
                continue

            ds, err_m = _dimension_score(dimensions, record.dimensions)
            ss = _shape_score(dimensions, record.subcategory)
            ctx = _context_score(is_dynamic, record.class_name, record.subcategory, speed_mps)

            total = (
                self.w_class * cs
                + self.w_dim * ds
                + self.w_shape * ss
                + self.w_ctx * ctx
            )

            suggested_scale = dimensions.length / max(record.dimensions.length, 1e-6)

            scores.append(MatchingScore(
                asset_id=record.asset_id,
                class_name=record.class_name,
                subcategory=record.subcategory,
                total_score=total,
                class_score=cs,
                dimension_score=ds,
                shape_score=ss,
                context_score=ctx,
                suggested_scale=float(suggested_scale),
                dimension_error_m=err_m,
                mesh_file=record.mesh_file,
            ))

        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores[: self.top_k]

    def match_object(self, obj: dict) -> dict:
        dims_raw = obj.get("dimensions", {})
        dims = AssetDimensions(
            length=float(dims_raw.get("length", 4.4)),
            width=float(dims_raw.get("width", 1.8)),
            height=float(dims_raw.get("height", 1.5)),
        )
        motion = obj.get("motion", {})
        is_dynamic = bool(motion.get("is_dynamic", False))
        speed_mps = float(motion.get("speed_mps", 0.0))
        class_name = str(obj.get("class_name", "car"))

        candidates = self.match(class_name, dims, is_dynamic, speed_mps)
        return {
            "track_id": obj.get("track_id"),
            "class_name": class_name,
            "query_dimensions": dims.to_dict(),
            "is_dynamic": is_dynamic,
            "speed_mps": speed_mps,
            "top_candidates": [c.to_dict() for c in candidates],
            "best_match": candidates[0].to_dict() if candidates else None,
        }
