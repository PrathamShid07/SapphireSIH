# Antigravity Agent Instructions

## Project
This project is a Major Project for converting UAV/drone video or images into a detailed hybrid 3D digital twin.

The core idea is:

- Reconstruct static structures from real drone imagery.
- Detect, track, localize, and replace repeated/dynamic objects with predefined 3D assets.
- Produce a metrically useful, textured 3D scene.
- Export OBJ, PLY, GLB/GLTF and other formats where practical.
- Provide a 3D web viewer and quantitative evaluation.

## Critical Rules

1. Work one stage at a time.
2. Inspect the existing project before changing anything.
3. Preserve useful existing work.
4. Never create fake outputs, fake metrics, fake GPS, fake detections, or placeholder implementations presented as completed functionality.
5. Actually run and test code after implementation.
6. Do not silently skip failed stages.
7. Do not claim geometry is accurate where the source imagery does not observe the surface.
8. Do not use raw latitude/longitude directly as arbitrary XYZ coordinates.
9. Ordinary GPS must not be presented as centimeter-level positioning.
10. Moving objects must be handled so they do not unnecessarily corrupt static photogrammetric reconstruction.
11. Do not reconstruct every car/tree individually when a predefined asset can be used.
12. Buildings, roads, terrain and fixed infrastructure should remain based on the actual reconstructed environment.
13. Do not implement later stages before the current stage is tested.
14. At the end of every stage, report what changed, what was tested, and what remains.
15. Stop after the requested stage and wait for `NEXT`.
16. Do not add comments to code unless explicitly requested by the user.

## Development Environment

Primary target:
- Windows
- VS Code / Antigravity
- Python
- GPU acceleration where available

Prefer:
- Modular Python code
- Clear configuration
- CLI execution
- Reproducible outputs
- Resume-safe processing
- Cached intermediate results
- Structured JSON/CSV metadata
- Separate data, source, output and evaluation directories

## Current First Task

Before implementing the full system:
1. Inspect the current project.
2. Detect the Python environment.
3. Detect installed dependencies.
4. Detect CUDA/GPU availability.
5. Inspect existing datasets and scripts.
6. Identify reusable work.
7. Implement only Stage 1 from `docs/STAGES.md`.
8. Run real tests.
9. Stop.
