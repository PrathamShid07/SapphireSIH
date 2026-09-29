# Stage 8 Optimization: Reusing Stage 4's COLMAP Database Instead of Re-running SfM

Status: **EXPERIMENTAL** (see [Verdict](#verdict) below). Default configuration
(`masking.reuse_stage4_poses: false`) is unchanged — this document describes an
opt-in path, not the default pipeline behavior.

## 1. Original Stage 8 Architecture (legacy path, still the default)

Stage 4 already performs the full COLMAP SfM pipeline once:

```
feature_extractor -> sequential/exhaustive_matcher -> mapper
  -> camera poses + sparse point cloud (output_dir/sfm/)
```

The original Stage 8 (`src/masking/stage8_pipeline.py`) generated dynamic-object
masks from Stage 7's tracking output, then ran the **entire SfM pipeline a
second time** on the same images, this time with `--ImageReader.mask_path`
pointing at the dynamic masks:

```
feature_extractor (masked) -> sequential/exhaustive_matcher -> mapper
  -> masked camera poses + masked sparse point cloud (output_dir/sfm_masked/)
```

This duplicates essentially all of Stage 4's computation: feature extraction
runs again on every image, matching runs again over every image pair, and
COLMAP's incremental `mapper` re-solves camera poses from scratch, including
its repeated global bundle-adjustment passes. Nothing from Stage 4's
`database.db`, extracted features, matches, or already-solved camera poses was
reused.

This legacy behavior is **fully preserved, unchanged, and remains the
default** (`masking.reuse_stage4_poses: false`). It is implemented in
`_run_stage8_legacy()` in `src/masking/stage8_pipeline.py`, byte-for-byte the
same COLMAP invocations as before this change, with only additive per-phase
timing instrumentation (feature extraction / matching / mapper / total) so it
can be fairly compared against the optimized path.

## 2. The Optimization

### What is reused

| Artifact | Reused? | How |
|---|---|---|
| Camera intrinsics (focal length, distortion) | Yes | Held fixed via `--refine_intrinsics 0` |
| Camera poses (position + orientation per image) | Yes | Held fixed via `--Mapper.fix_existing_frames 1` |
| Extracted SIFT keypoints for non-dynamic image regions | Yes | Never re-extracted; read directly from a copy of Stage 4's `database.db` |
| Verified feature matches between static regions | Yes | Read from the same database copy; only pruned, never regenerated |

### What is recomputed

| Step | Recomputed? | Why |
|---|---|---|
| `feature_extractor` | **No** — never runs in the optimized path | Features for unmasked pixels are identical whether or not a mask is later applied |
| Any matcher (`sequential_matcher`/`exhaustive_matcher`) | **No** — never runs | Matches for unmasked correspondences are identical; only the ones involving masked pixels need removing |
| Incremental `mapper` (pose solving + repeated global BA) | **No** — never runs | Stage 4 already solved this; re-solving it was the actual waste being eliminated |
| Match/two-view-geometry filtering (new) | **Yes** — pure Python/SQLite, no COLMAP subprocess | Removes correspondences whose keypoint falls inside a dynamic-object mask |
| `point_triangulator` (new usage of an existing COLMAP binary) | **Yes** | Re-triangulates 3D points from the pruned match set, using Stage 4's poses as fixed geometry |

### Optimized flow

```
Stage 4 (unchanged): feature_extractor -> matcher -> mapper
                       -> output_dir/sfm/database.db, output_dir/sfm/sparse/0/
                                          |
                                          | (read-only copy)
                                          v
Stage 8 (optimized):  copy database.db -> output_dir/sfm_masked_optimized/database.db
                       prune matches/two_view_geometries touching dynamic-mask regions
                       point_triangulator (poses fixed from sfm/sparse/0/)
                       -> output_dir/sfm_masked_optimized/sparse/ (new sparse model)
```

### Why `point_triangulator`

COLMAP's CLI (verified against a real local COLMAP 4.2.0 install, see
[Section 4](#4-empirical-colmap-validation)) exposes exactly the primitive
this use case needs:

- `point_triangulator --input_path <existing model> --database_path <db>
  --output_path <out>` re-triangulates 3D points using an **existing**
  model's camera poses and intrinsics, matching images to that model by
  filename, without running incremental registration (`mapper`'s job).
- `--Mapper.fix_existing_frames 1` prevents point_triangulator's internal
  bundle-adjustment pass from moving the input poses at all — verified
  empirically to produce pose components identical to the mapper's own output
  to floating-point noise (~1e-16), see below.
- `--refine_intrinsics 0` (COLMAP's own default) keeps camera intrinsics
  fixed too.

No COLMAP CLI command exists to filter keypoints/matches out of an
already-populated database "in place" — that part (`colmap_db_pruner.py`) is
new code operating directly on COLMAP's documented SQLite schema, verified
against a real database rather than assumed from memory (see below).

### What was NOT changed

- Stage 4 (`src/sfm/stage4_pipeline.py`) — untouched.
- The legacy Stage 8 path — untouched in behavior (only added timing instrumentation).
- Stages 1-7, 9-12 — untouched.
- COLMAP feature count, image resolution, `geom_consistency`, or any dense
  reconstruction (Stage 5) parameter — untouched. This change is scoped
  entirely to Stage 8's redundant SfM re-run.

## 3. Configuration

```yaml
masking:
  reuse_stage4_poses: false  # default; set true to opt into the optimized path
```

Rollback: set `masking.reuse_stage4_poses: false` (the default) at any time.
The legacy code path is fully intact and independently selectable — no code
changes are required to roll back, only the config flag.

The optimized path writes to `output_dir/sfm_masked_optimized/`, which is
**distinct from both** `output_dir/sfm/` (Stage 4, never touched) **and**
`output_dir/sfm_masked/` (the legacy path's output, also never touched).
Running both paths back-to-back on the same `output_dir` produces two
independent, directly comparable outputs plus two independent report files
(`stage8_report.json` vs `stage8_report_optimized.json`,
`masking_comparison_report.json` vs `masking_comparison_report_optimized.json`).

## 4. Empirical COLMAP Validation

Before writing any production code, the exact COLMAP database schema,
`pair_id` encoding, and `point_triangulator` behavior were verified against a
**real COLMAP 4.2.0 installation** (`/opt/homebrew/bin/colmap`, CPU-only
build) on this machine, using real frames extracted from
`data/input/sample_drone.mp4`. This was necessary because COLMAP's exact
binary schema (keypoint blob layout, `pair_id` formula, `point_triangulator`'s
CLI surface) is undocumented at that level of detail in COLMAP's own written
docs and differs across versions.

Findings (all confirmed by direct SQLite inspection of a real generated
database, not assumed from memory):

- Schema (COLMAP 4.2.0, includes newer `rigs`/`frames`/`pose_priors` tables
  not present in older COLMAP versions):
  `keypoints(image_id, rows, cols, data)`,
  `matches(pair_id, rows, cols, data)`,
  `two_view_geometries(pair_id, rows, cols, data, config, F, E, H, ...)`.
- `keypoints.data` is a `(rows, cols)` **float32** array; columns 0-1 are
  always pixel `(x, y)`, regardless of whether `cols` is 4 (scale+orientation)
  or 6 (full affine shape) — confirmed `cols=6` on real SIFT output, and blob
  byte length matches `rows * cols * 4` exactly (float32, not float64).
- `matches`/`two_view_geometries.data` is a `(rows, 2)` **uint32** array of
  0-indexed feature indices — confirmed blob byte length matches
  `rows * cols * 4` exactly (uint32, not uint64).
- `pair_id = image_id1 * 2147483647 + image_id2` with `image_id1 < image_id2`
  — confirmed against 11 real pair_id values from a 6-image test database
  (e.g. images 1&2 -> pair_id 2147483649, images 5&6 -> pair_id 10737418241),
  matching the formula exactly in every case.

`point_triangulator` round-trip test (12→15-image real reconstructions
spread across `sample_drone.mp4`, `exhaustive_matcher`, assumed intrinsics
fx=fy=1500px since the real camera calibration wasn't available for this
probe):

1. Ran real `mapper` to get a baseline model: **15/15 registered, 2937
   points, mean reprojection error 0.3536px**.
2. Ran `point_triangulator` with an **unmodified** copy of the database
   (sanity check, zero pruning) against that same model:
   **15/15 registered, 2990 points, 0.3572px** — consistent with the mapper's
   own output, confirming the round-trip itself introduces no corruption.
   Camera poses compared component-by-component: **max difference
   1.1e-16** (floating-point noise) with `--Mapper.fix_existing_frames 1`.
3. Ran the same test with a synthetic mask covering the right half of every
   image (a deliberately large, controlled pruning case): 68.6% of `matches`
   rows and 68.0% of `two_view_geometries` rows were removed (as expected,
   since a match survives only if **both** endpoints are unmasked -
   `P(both unmasked) ≈ 0.5² = 25%` kept under a uniform feature distribution,
   consistent with the observed removal rate). Result: **15/15 registered
   (unchanged), 695 points (down from 2937/2990), 0.3348px reprojection
   error, pose difference vs. the mapper baseline: 1.1e-16** — again exactly
   preserved.

This validation is what `src/masking/colmap_db_pruner.py` and
`ColmapWrapper.triangulate_points()` are built on, and is also encoded as an
automated regression test in `tests/test_stage8_optimized.py`
(`TestOptimizedPathRealColmapIntegration`), which runs this same recipe
against real COLMAP on every test run (skipped automatically if COLMAP isn't
installed).

**Important caveat on portability**: the `pair_id` formula and blob layout
above are specific to this COLMAP build (4.2.0). `colmap_db_pruner.py` does
not attempt to detect or adapt to a different schema — if you upgrade or
downgrade COLMAP, re-run the validation in this section (or the
`TestOptimizedPathRealColmapIntegration` test class) before trusting the
optimized path again.

## 5. Benchmark Results

### Setup

Since running the real Stage 6/7 models (semantic segmentation, detection)
requires downloading pretrained weights not available in this offline
environment, the benchmark uses a **synthetic Stage 7 tracking output**
(a fixed bounding box marked as a moving object in every frame) as a
stand-in for real detector/tracker output. This is an honest limitation,
called out explicitly: **Stage 4's SfM computation itself is 100% real
COLMAP** run on real frames from `data/input/sample_drone.mp4` (15 frames
spread across the full video, exhaustive matching, assumed intrinsics
fx=fy=1500px, cx=960, cy=540, zero distortion — the real camera calibration
was not available for this benchmark). Only the *input* to the masking step
(which pixels count as "dynamic") is synthetic; the masking, database
pruning, and reconstruction logic exercised is the real production code.

Per the task's instructions, the legacy path was run **twice** on identical
input/config first, to measure natural run-to-run variance, before running
the optimized path once for comparison.

### Results

| Metric | Stage 4 baseline (shared) | Baseline run 1 (legacy) | Baseline run 2 (legacy) | Optimized run |
|---|---|---|---|---|
| Method | `COLMAP mapper` | `legacy_full_colmap_rerun` | `legacy_full_colmap_rerun` | `optimized_point_triangulator_reuse` |
| Registered images | 15/15 | 11/15 | 11/15 | **15/15** |
| Sparse points | 2937 | 1600 | 1562 | 1633 |
| Mean reprojection error (px) | 0.3536 | 0.3599 | 0.3593 | 0.3653 |
| Feature extraction time | 10.28s (whole Stage 4) | 1.045s | 1.059s | **not run (0s)** |
| Matching time | (included above) | 5.978s | 6.354s | **not run (0s)** |
| Mapper time | (included above) | 0.558s | 0.644s | **not run (0s)** |
| Database copy time | n/a | n/a | n/a | 0.003s |
| Mask pruning time | n/a | n/a | n/a | 0.057s |
| Point triangulation time | n/a | n/a | n/a | 0.263s |
| **COLMAP-work total** | — | **7.581s** | **8.057s** | **0.323s** |
| Measured wall-clock (incl. mask generation) | — | 8.657s | 9.16s | 1.419s |
| Dynamic points removed | — | 9 (13.2%) | 37 (54.4%) | 37 (54.4%) |

Natural baseline-vs-baseline variance (the required reference tolerance,
measured before judging the optimized run, per the task's instructions):

- **Reprojection error**: |0.3599 − 0.3593| = **0.0006px**
- **Sparse points**: |1600 − 1562| = **38 points (≈2.4%)**
- **Registered images**: **0 difference** (11 both times)
- **Dynamic points removed**: 9 vs. 37 — a **large** run-to-run difference
  (COLMAP's RANSAC/incremental registration order has real nondeterminism
  even on identical input/config), showing why a single baseline run would
  have been an unreliable reference point.
- **COLMAP-work runtime**: 7.581s vs 8.057s (≈6% variance)

### Applying the acceptance rule

| Metric | Optimized vs. baseline mean | Within measured baseline-vs-baseline variance? |
|---|---|---|
| Registered images | 15 vs. 11 (+4) | N/A — categorically different (see discussion below), not a noise-band question |
| Sparse points | 1633 vs. 1581 (+52, ≈3.3%) | Yes — comparable to the 2.4% baseline noise |
| Reprojection error | 0.3653 vs. 0.3596 (+0.0057px) | **No** — roughly **9.5x** the measured 0.0006px baseline noise |
| COLMAP-work runtime | 0.323s vs. 7.8s | N/A — the entire point of the change; not a quality metric |

## 6. Quality Differences — Honest Discussion

This result is **mixed**, not a clean win, and should not be spun as one:

**In the optimized path's favor**: registration completeness improved
materially (15/15 vs. 11/15). This is a real, reproducible structural
difference (not noise — both legacy runs independently registered exactly 11
images), not a coincidence. It happens because the legacy path re-attempts
full pose estimation from a *feature-reduced* (masked) dataset, and 4 of the
15 images apparently didn't retain enough unmasked, well-distributed features
to re-register reliably from scratch under this benchmark's fairly aggressive
synthetic mask (17.5% of every frame). The optimized path never re-attempts
registration at all — it reuses poses Stage 4 already solved with the full,
unmasked feature set — so it cannot lose images this way.

**Against the optimized path**: this is precisely why that same protection
is a double-edged sword. The reused poses come from Stage 4's *original*
bundle adjustment, which ran on the **unmasked** image set — if a real
dynamic object's feature tracks measurably influenced that original global
optimization (plausible for a large, sustained moving object with many
matched features, less plausible for small/brief ones), the optimized path
has no mechanism to correct for that, whereas the legacy path's full
re-solve, by definition, cannot be influenced by masked-out features at all.
The measured **+0.0057px reprojection error increase, about 9.5x the
baseline noise floor**, is consistent with this — a small, real degradation,
not baseline noise, though numerically modest in absolute terms (0.365px vs.
0.359px).

**Conclusion**: the optimized path traded a small, measured reprojection-error
regression for a real registration-completeness improvement and a large
runtime reduction, on this specific small-scale, synthetic-mask benchmark.
Whether that trade is acceptable depends on which metric matters more for a
given deployment, and **this benchmark used a synthetic mask standing in for
real detection/tracking output** — the actual dynamic-object mask shapes,
sizes, and durations from a real detector will differ from this benchmark's
fixed rectangular region, and should be re-validated before trusting these
exact numbers in production.

## 7. Limitations

- **Small benchmark scale** (15 images). The relative runtime advantage
  should be expected to grow, not shrink, at full-pipeline scale (COLMAP's
  incremental mapper cost grows super-linearly with image count due to
  repeated global bundle adjustment; the optimized path's replacement cost -
  database copy, match pruning, single triangulation pass - is much closer to
  linear). This has not been verified at production scale.
- **Synthetic tracking/mask input**, not real Stage 6/7 model output (see
  Section 5 setup). Real dynamic-object masks will have different shapes,
  sizes, and per-frame variability.
- **No independent per-COLMAP-version validation.** The database schema and
  `pair_id` encoding are verified against this machine's COLMAP 4.2.0 only
  (see Section 4's portability caveat).
- **Reused poses inherit any dynamic-object influence already baked into
  Stage 4's original bundle adjustment.** The optimized path cannot detect or
  correct this; only re-running full SfM on masked-only features (the legacy
  path) can rule it out entirely.
- **Missing per-image masks fail open** (treated as fully static, i.e. no
  pruning for that image), matching COLMAP's own `--ImageReader.mask_path`
  convention, but this means a bug that silently fails to generate a mask for
  some frames would silently under-prune rather than error - `build_image_masks()`
  logs a warning and reports the missing list in the Stage 8 report's
  `recomputed.images_without_mask` field for visibility, but does not hard-fail.
- **The pruning script does not remove/renumber `keypoints`/`descriptors` rows** —
  a deliberate design choice (see `colmap_db_pruner.py` docstring) to avoid the
  much higher corruption risk of reindexing every match's feature indices
  after a row deletion. This means the pruned database is slightly larger on
  disk than a "true" masked re-extraction would produce, though this does not
  affect `point_triangulator`'s output (unreferenced keypoints simply
  contribute no 3D point).

## 8. Rollback Procedure

1. Set `masking.reuse_stage4_poses: false` in `config/default_config.yaml`
   (or per-run config override). This is already the default.
2. No code changes, migrations, or data cleanup are required — the legacy
   `_run_stage8_legacy()` function is untouched and independently selectable.
3. If both `sfm_masked/` (legacy) and `sfm_masked_optimized/` (optimized)
   exist in an output directory from prior experimentation, either can be
   deleted independently without affecting the other or Stage 4's `sfm/`
   directory, since all three are fully isolated.

## 9. Follow-up Experiment: Limited Pose Refinement

The reprojection-error regression in Section 6 prompted one focused follow-up:
can a *bounded* refinement step recover most of that regression without
reintroducing full feature extraction, matching, or the incremental mapper?

### 9.1 What COLMAP 4.2.0 actually supports here (verified empirically)

Two candidate mechanisms were tested against real COLMAP before writing any
production code:

1. **`point_triangulator` with `--Mapper.fix_existing_frames` unset (its own
   documented default of 0, i.e. "not fixed")**: empirically, this produced
   camera poses **byte-identical** to the fully-fixed (`=1`) case in every
   test performed. `point_triangulator`'s internal "Retriangulation and
   Global bundle adjustment" step does not move existing frames' poses in
   this COLMAP build regardless of that flag - it appears to only affect
   points/structure. **This mechanism does not provide pose refinement.**
2. **A separate, explicit `bundle_adjuster` call** on the model
   `point_triangulator` already produced: this genuinely refines poses
   (`--BundleAdjustment.refine_rig_from_world 1`, COLMAP's own flag for the
   camera-pose transform) while `--BundleAdjustment.refine_focal_length 0`,
   `--BundleAdjustment.refine_principal_point 0`, and
   `--BundleAdjustment.refine_extra_params 0` are passed explicitly to
   guarantee intrinsics are never touched unless a caller opts in (verified:
   intrinsics came back byte-identical to Stage 4's in every test run, see
   §9.3). No new COLMAP command was introduced - `bundle_adjuster` is an
   existing, documented COLMAP binary; this only adds one new call to it.

The implemented flow is therefore: **prune → `point_triangulator` (poses
fixed, as in the base optimized path) → `bundle_adjuster` (poses+points
refined, intrinsics fixed)**. This is a reordering of the user's requested
"refine → triangulate" concept into "triangulate → refine," made necessary by
what `point_triangulator` and `bundle_adjuster` actually do: `bundle_adjuster`
needs a model that already has both poses *and* points to refine, so it must
run after triangulation, not before. Running it on Stage 4's *original*
model (before pruning) instead would refine poses against still-contaminated,
un-pruned points - the opposite of the goal. Config: `masking.refine_reused_poses:
false` (default), only consulted when `masking.reuse_stage4_poses: true`.

### 9.2 Safety finding: this is NOT unconditionally safe

Before running the production benchmark, an aggressive exploratory probe
(right half of every image masked - far more aggressive than any realistic
dynamic-object mask) showed `bundle_adjuster` reporting **`Termination:
NO_CONVERGENCE`**, with some poses diverging by up to **51.2 degrees** and 4.0
translation units from Stage 4's originals - a clear sign of an
under-constrained optimization for cameras that lost most of their
observations. This is why `bundle_adjust()` surfaces COLMAP's own convergence
diagnostics (`termination`, `initial_cost_px`, `final_cost_px`, iteration
count) in the Stage 8 report rather than silently trusting the refined poses,
and why no threshold-based auto-accept/reject logic was added (per
instructions) - this is a real capability boundary, not a tuning parameter.

### 9.3 Benchmark: A (legacy) vs B (optimized-fixed) vs C (optimized-refined)

Same dataset, same synthetic mask, same Stage 4 base as Section 5 (15 frames
spread across `sample_drone.mp4`, bbox `[0,0,640,540]` marked dynamic on
every frame, ~17.5% of frame area). Two legacy baseline runs were repeated
first, as before, to measure natural variance.

| Metric | Baseline run 1 (A) | Baseline run 2 (A) | Optimized-fixed (B) | Optimized-refined (C) |
|---|---|---|---|---|
| Registered images | 11/15 | 11/15 | 15/15 | 15/15 |
| Sparse points | 1599 | 1599 | 1633 | 1633 |
| Mean reprojection error (px) | 0.3612 | 0.3612 | 0.3653 | **0.3622** |
| COLMAP-work time | 7.962s | 8.243s | 0.324s | 0.451s |
| Wall clock | 9.079s | 9.335s | 1.431s | 1.569s |
| Pose change vs. Stage 4 (max translation / max rotation) | n/a | n/a | 0.0 / 0.0 (exactly fixed) | 0.0200 / 0.305° |
| bundle_adjuster termination | n/a | n/a | n/a (not run) | **CONVERGENCE** |
| Camera intrinsics vs. Stage 4 | n/a | n/a | identical | **identical** |

Natural baseline-vs-baseline variance in this run pair: **zero** across every
metric (both legacy runs produced identical registration, points, and
reprojection error - note this differs from the Section 5 benchmark's
baseline pair, which showed small nonzero variance; COLMAP's RANSAC/mapper
nondeterminism does not manifest on every run, so two-sample variance
estimates should be treated as a noisy lower bound, not a precise figure).

Reprojection-error regression vs. legacy baseline (0.3612): **B: +0.0041px,
C: +0.0010px**. Refinement recovered **≈76%** of the regression (0.0041 →
0.0010) for an added cost of **0.126s** (bundle_adjuster's own reported time
plus overhead) - COLMAP-work time grew from 0.324s to 0.451s, still **≈18x**
faster than legacy's ~8s (vs. B's ≈25x). Registered image count and sparse
point count were unaffected by refinement (identical to B in both).

### 9.4 Is the refinement worthwhile?

**Yes, directionally** - on this benchmark it recovered most of the
reprojection-error regression identified in Section 6, at a small, one-time
COLMAP-reported cost (0.126s), while preserving the large majority of the
runtime advantage (≈18x vs. legacy, down from ≈25x) and leaving registration
completeness and point count unchanged. Pose changes were small (max 0.02
translation units, max 0.31° rotation) and COLMAP itself reported
`CONVERGENCE`, not the divergence seen in the aggressive-mask probe.

**But not a substitute for validation at scale**: §9.2's `NO_CONVERGENCE`
finding shows this step's safety is directly tied to how much observation
survives pruning - a benchmark with a much larger or more numerous dynamic
mask could plausibly reproduce that divergence. The Stage 8 report now
surfaces `pose_refinement_diagnostics.termination` and
`pose_change_vs_stage4` precisely so this can be checked per-run rather than
assumed.

## Verdict

**EXPERIMENTAL — NOT SAFE FOR PRODUCTION AS THE DEFAULT**, for either
`masking.reuse_stage4_poses` (fixed-pose) or `masking.refine_reused_poses`
(refined). Both remain `false` by default.

The architecture is sound and empirically validated against real COLMAP
(Section 4); the database-pruning and pose-reuse mechanics work correctly,
and the optional refinement step (Section 9) recovered ≈76% of the fixed
path's reprojection-error regression on this benchmark for a small added
cost, while keeping registration completeness and point count unchanged.
This makes optimized-refined the better-validated of the two optimized
variants so far.

It is still not production-safe because:

1. Both variants' benchmarks used a synthetic mask, not real Stage 6/7 model
   output (pretrained weights unavailable in this environment) - real mask
   shapes/sizes/durations will differ.
2. Both were only benchmarked at a small scale (15 images).
3. The refinement step's safety is demonstrably conditional, not universal:
   an aggressive exploratory probe (Section 9.2) reproduced
   `Termination: NO_CONVERGENCE` with poses diverging by tens of degrees when
   too much observation was pruned. The production benchmark's realistic
   mask converged cleanly, but this must be re-checked (via the
   `pose_refinement_diagnostics.termination` field the code now surfaces) on
   any new dataset before trusting the refined output, not assumed safe by
   default.

`masking.reuse_stage4_poses` and `masking.refine_reused_poses` should remain
`false` until:

1. The benchmark is re-run with real Stage 6/7 model output rather than a
   synthetic mask, and
2. The benchmark is re-run at a scale closer to a production keyframe count,
   and
3. `bundle_adjuster`'s convergence status is confirmed `CONVERGENCE` (not
   `NO_CONVERGENCE`) across that realistic benchmark, and
4. Any residual reprojection-error gap is judged acceptable against the
   registration-completeness improvement and runtime reduction for the
   specific deployment's priorities.
