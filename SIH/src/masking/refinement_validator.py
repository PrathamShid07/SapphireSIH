"""
Validation / acceptance gate for the optional "limited pose refinement" step
in the optimized Stage 8 path (masking.refine_reused_poses).

Background: the refinement experiment (docs/STAGE8_OPTIMIZATION.md section 9)
showed bundle_adjuster can meaningfully reduce reprojection error after
pruning, but is NOT unconditionally safe - an aggressive exploratory probe
produced Termination: NO_CONVERGENCE with poses diverging by tens of degrees.
This module decides, from evidence gathered so far, whether a given
refinement run's output is trustworthy enough to expose as the Stage 8
result, or whether Stage 8 should fall back to the fixed-pose
(point_triangulator-only) result instead.

Design principle (explicit, not implicit): only two data points currently
exist for pose-change magnitude - a converged, healthy case (~0.02 translation
units, ~0.3 degrees) and a diverged, unhealthy case (~4.0 translation units,
~51 degrees) that also reported NO_CONVERGENCE. Because these are so far
apart, almost any threshold placed between them would "work" for these two
examples without being justified by actual evidence of where a real boundary
lies. Per instructions, this module does NOT invent such a threshold. Pose
magnitude, and the fixed-vs-refined reprojection error delta, are recorded as
DIAGNOSTIC-ONLY fields on every run - always reported, never gating the
accept/reject decision by themselves. Only checks with an objective,
evidence-backed pass/fail condition (COLMAP's own convergence verdict, its
own cost-decrease guarantee for a converged optimization, exact structural
invariants bundle_adjuster must preserve, and basic data validity such as
finiteness and file presence) are allowed to gate the decision. See the
`EVIDENCE_BASED` set below for the definitive list of which checks gate the
decision and which do not.
"""

import math

REFINEMENT_ACCEPTED = "REFINEMENT_ACCEPTED"
REFINEMENT_REJECTED = "REFINEMENT_REJECTED"
REFINEMENT_UNAVAILABLE = "REFINEMENT_UNAVAILABLE"

# Names of checks whose "passed" value is allowed to gate the accept/reject
# decision. Every other check in the returned `checks` dict is diagnostic-only:
# always computed and reported, never causing rejection by itself.
EVIDENCE_BASED_CHECKS = {
    "bundle_adjuster_converged",
    "cost_decreased_or_equal",
    "pose_deltas_finite",
    "reprojection_error_valid",
    "registered_image_count_consistent",
    "sparse_point_count_consistent",
    "model_valid",
    "intrinsics_unchanged",
}

REQUIRED_MODEL_FILES = (
    ("cameras.bin", "cameras.txt"),
    ("images.bin", "images.txt"),
    ("points3D.bin", "points3D.txt"),
)


def _files_present(model_dir):
    from pathlib import Path
    model_dir = Path(model_dir)
    missing = []
    for bin_name, txt_name in REQUIRED_MODEL_FILES:
        if not (model_dir / bin_name).exists() and not (model_dir / txt_name).exists():
            missing.append(f"{bin_name} or {txt_name}")
    return len(missing) == 0, missing


def _quaternions_well_formed(registered_images):
    for img in registered_images:
        qvec = img.get("qvec")
        if not qvec or len(qvec) != 4:
            return False
        if not all(math.isfinite(x) for x in qvec):
            return False
        norm = math.sqrt(sum(x * x for x in qvec))
        # Unit-quaternion validity check (malformed-data detection), not a
        # quality/production threshold: COLMAP always emits normalized
        # quaternions, so a norm far from 1.0 indicates corrupt/invalid data,
        # not "slightly worse" refinement quality.
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-2:
            return False
    return True


def validate_refinement(
    fixed_model_dir,
    refined_model_dir,
    stage4_model_dir,
    ba_diagnostics,
    pose_delta_report,
    bundle_adjust_error=None,
    logger=None,
):
    """
    Decide whether a bundle_adjuster refinement pass should be accepted,
    rejected, or is unavailable, given:

    - fixed_model_dir: the point_triangulator (fixed-pose) model, the
      fallback result if refinement is not accepted.
    - refined_model_dir: bundle_adjuster's output model (may not exist/be
      valid if bundle_adjust_error is set).
    - stage4_model_dir: Stage 4's original model, source of ground-truth
      intrinsics to compare against.
    - ba_diagnostics: dict from ColmapWrapper.bundle_adjust()'s parsed
      "Bundle adjustment report" (residuals, initial/final cost, termination),
      or None if it could not be parsed / bundle_adjuster did not run.
    - pose_delta_report: dict from pose_comparator.compare_poses(stage4_poses,
      refined_poses), or None.
    - bundle_adjust_error: the exception message if the bundle_adjuster
      COLMAP command itself failed (crashed / nonzero exit), else None.

    Returns a dict: {decision, reason, checks, fixed_path_metrics,
    refined_path_metrics}. `checks` maps check name -> {evidence_based, value,
    passed} where `passed` is None for diagnostic-only checks that do not gate
    the decision.
    """
    from src.sfm.colmap_parser import parse_colmap_model

    checks = {}
    reasons = []

    fixed_data = parse_colmap_model(fixed_model_dir)
    fixed_path_metrics = {
        "registered_images": fixed_data.get("num_registered_images", 0),
        "sparse_points": fixed_data.get("num_sparse_points", 0),
        "mean_reprojection_error_px": fixed_data.get("mean_reprojection_error", 0.0),
    }

    # --- bundle_adjuster command itself failed to run ---
    if bundle_adjust_error is not None:
        reasons.append(f"bundle_adjuster command failed: {bundle_adjust_error}")
        result = {
            "decision": REFINEMENT_UNAVAILABLE,
            "reason": "; ".join(reasons),
            "checks": checks,
            "fixed_path_metrics": fixed_path_metrics,
            "refined_path_metrics": None,
        }
        if logger:
            logger.warning(f"[refinement-gate] {result['decision']}: {result['reason']}")
        return result

    # --- Criterion 9: required output files present and readable ---
    files_ok, missing_files = _files_present(refined_model_dir)
    checks["output_files_present"] = {
        "evidence_based": True, "value": {"missing": missing_files}, "passed": files_ok,
    }
    if not files_ok:
        reasons.append(f"Refined model is missing required output files: {missing_files}")
        result = {
            "decision": REFINEMENT_UNAVAILABLE,
            "reason": "; ".join(reasons),
            "checks": checks,
            "fixed_path_metrics": fixed_path_metrics,
            "refined_path_metrics": None,
        }
        if logger:
            logger.warning(f"[refinement-gate] {result['decision']}: {result['reason']}")
        return result

    refined_data = parse_colmap_model(refined_model_dir)
    refined_path_metrics = {
        "registered_images": refined_data.get("num_registered_images", 0),
        "sparse_points": refined_data.get("num_sparse_points", 0),
        "mean_reprojection_error_px": refined_data.get("mean_reprojection_error", 0.0),
    }

    # --- bundle_adjuster diagnostics could not be parsed at all ---
    if ba_diagnostics is None:
        reasons.append(
            "bundle_adjuster's 'Bundle adjustment report' could not be parsed from its "
            "output; convergence status is unknown."
        )
        result = {
            "decision": REFINEMENT_UNAVAILABLE,
            "reason": "; ".join(reasons),
            "checks": checks,
            "fixed_path_metrics": fixed_path_metrics,
            "refined_path_metrics": refined_path_metrics,
        }
        if logger:
            logger.warning(f"[refinement-gate] {result['decision']}: {result['reason']}")
        return result

    # --- Criterion 1: bundle_adjuster termination status (EVIDENCE-BASED) ---
    termination = ba_diagnostics.get("termination")
    converged = termination == "CONVERGENCE"
    checks["bundle_adjuster_converged"] = {
        "evidence_based": True, "value": termination, "passed": converged,
    }
    if not converged:
        reasons.append(f"bundle_adjuster reported termination={termination!r}, not CONVERGENCE.")

    # --- COLMAP's own cost-decrease guarantee for a converged optimization (EVIDENCE-BASED) ---
    # Not an invented threshold: a correctly functioning bundle adjustment
    # minimizes its cost function, so final_cost should never exceed
    # initial_cost. This is a mathematical property of the optimizer, not a
    # quality bar we are choosing.
    initial_cost = ba_diagnostics.get("initial_cost_px")
    final_cost = ba_diagnostics.get("final_cost_px")
    cost_ok = (
        initial_cost is not None and final_cost is not None
        and math.isfinite(initial_cost) and math.isfinite(final_cost)
        and final_cost <= initial_cost
    )
    checks["cost_decreased_or_equal"] = {
        "evidence_based": True,
        "value": {"initial_cost_px": initial_cost, "final_cost_px": final_cost},
        "passed": cost_ok,
    }
    if not cost_ok:
        reasons.append(
            f"bundle_adjuster's own reported cost did not decrease "
            f"(initial={initial_cost}, final={final_cost})."
        )

    # --- Criteria 2 & 3: pose deltas (DIAGNOSTIC-ONLY magnitude; EVIDENCE-BASED finiteness) ---
    if pose_delta_report is not None:
        summary = pose_delta_report["summary"]
        max_t = summary["max_translation_delta"]
        max_r = summary["max_rotation_delta_degrees"]
        checks["max_translation_change"] = {
            "evidence_based": False, "value": max_t, "passed": None,
            "note": "Diagnostic only - no evidence-based numeric threshold has been "
                    "established yet (see docs/STAGE8_OPTIMIZATION.md section 9.2); "
                    "reported for human review, not auto-gated.",
        }
        checks["max_rotation_change_degrees"] = {
            "evidence_based": False, "value": max_r, "passed": None,
            "note": "Diagnostic only - same rationale as max_translation_change.",
        }
        deltas_finite = math.isfinite(max_t) and math.isfinite(max_r)
        checks["pose_deltas_finite"] = {
            "evidence_based": True, "value": deltas_finite, "passed": deltas_finite,
        }
        if not deltas_finite:
            reasons.append("Pose deltas contain non-finite (NaN/Inf) values.")
    else:
        checks["pose_deltas_finite"] = {
            "evidence_based": True, "value": None, "passed": False,
        }
        reasons.append("Pose change vs. Stage 4 could not be computed.")

    # --- Criterion 4: reprojection error before/after (DIAGNOSTIC-ONLY delta; EVIDENCE-BASED validity) ---
    fixed_reproj = fixed_path_metrics["mean_reprojection_error_px"]
    refined_reproj = refined_path_metrics["mean_reprojection_error_px"]
    checks["reprojection_error_before_after"] = {
        "evidence_based": False,
        "value": {
            "fixed_px": fixed_reproj,
            "refined_px": refined_reproj,
            "delta_px": round(refined_reproj - fixed_reproj, 6),
        },
        "passed": None,
        "note": "Diagnostic only - no evidence-based numeric tolerance for an acceptable "
                "increase has been established yet; reported for human review.",
    }
    reproj_valid = (
        isinstance(refined_reproj, (int, float))
        and math.isfinite(refined_reproj)
        and refined_reproj >= 0
    )
    checks["reprojection_error_valid"] = {
        "evidence_based": True, "value": refined_reproj, "passed": reproj_valid,
    }
    if not reproj_valid:
        reasons.append(f"Refined reprojection error is invalid: {refined_reproj}")

    # --- Criteria 5 & 6: structural invariants bundle_adjuster must preserve (EVIDENCE-BASED) ---
    n_fixed = fixed_path_metrics["registered_images"]
    n_refined = refined_path_metrics["registered_images"]
    reg_ok = n_fixed == n_refined
    checks["registered_image_count_consistent"] = {
        "evidence_based": True, "value": {"fixed": n_fixed, "refined": n_refined}, "passed": reg_ok,
    }
    if not reg_ok:
        reasons.append(
            f"Registered image count changed from {n_fixed} (fixed) to {n_refined} "
            f"(refined); bundle_adjuster should only refine existing registrations, "
            f"never add or remove them."
        )

    p_fixed = fixed_path_metrics["sparse_points"]
    p_refined = refined_path_metrics["sparse_points"]
    pts_ok = p_fixed == p_refined
    checks["sparse_point_count_consistent"] = {
        "evidence_based": True, "value": {"fixed": p_fixed, "refined": p_refined}, "passed": pts_ok,
    }
    if not pts_ok:
        reasons.append(
            f"Sparse point count changed from {p_fixed} (fixed) to {p_refined} (refined); "
            f"bundle_adjuster (refine_points3D) should reposition existing points, never "
            f"add or remove them."
        )

    # --- Criterion 7: model validity (EVIDENCE-BASED) ---
    model_ok = (
        n_refined > 0
        and p_refined > 0
        and _quaternions_well_formed(refined_data.get("registered_images", []))
    )
    checks["model_valid"] = {
        "evidence_based": True,
        "value": {"registered_images": n_refined, "sparse_points": p_refined},
        "passed": model_ok,
    }
    if not model_ok:
        reasons.append("Refined model is empty or contains malformed camera poses.")

    # --- Criterion 8: intrinsics unchanged (EVIDENCE-BASED) ---
    stage4_data = parse_colmap_model(stage4_model_dir)
    stage4_intrinsics = (
        tuple(stage4_data["cameras"][0]["params"]) if stage4_data.get("cameras") else None
    )
    refined_intrinsics = (
        tuple(refined_data["cameras"][0]["params"]) if refined_data.get("cameras") else None
    )
    intrinsics_ok = (
        stage4_intrinsics is not None
        and refined_intrinsics is not None
        and stage4_intrinsics == refined_intrinsics
    )
    checks["intrinsics_unchanged"] = {
        "evidence_based": True,
        "value": {"stage4": stage4_intrinsics, "refined": refined_intrinsics},
        "passed": intrinsics_ok,
    }
    if not intrinsics_ok:
        reasons.append(
            f"Camera intrinsics changed during refinement (stage4={stage4_intrinsics}, "
            f"refined={refined_intrinsics}), despite refine_focal_length/principal_point/"
            f"extra_params all being requested as off."
        )

    evidence_based_results = [
        v["passed"] for k, v in checks.items() if k in EVIDENCE_BASED_CHECKS
    ]
    accepted = all(evidence_based_results) if evidence_based_results else False
    decision = REFINEMENT_ACCEPTED if accepted else REFINEMENT_REJECTED

    result = {
        "decision": decision,
        "reason": "; ".join(reasons) if reasons else "All evidence-based checks passed.",
        "checks": checks,
        "fixed_path_metrics": fixed_path_metrics,
        "refined_path_metrics": refined_path_metrics,
    }
    if logger:
        logger.info(f"[refinement-gate] {result['decision']}: {result['reason']}")
    return result
