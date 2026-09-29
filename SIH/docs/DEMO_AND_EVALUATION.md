# Final Demo and Evaluation Plan

## Demo Sequence

Show the system progressively:

1. Original drone video
2. Extracted frames
3. Selected keyframes
4. Camera trajectory
5. Sparse point cloud
6. Dense point cloud
7. Reconstructed mesh
8. Semantic segmentation
9. Building/road/tree masks
10. Car detection
11. Car tracking
12. Dynamic-object masks
13. Masked reconstruction
14. 3D car localization
15. 3D tree localization
16. Asset candidates
17. Selected assets
18. Asset placement
19. Hybrid scene
20. Textured scene
21. GLB/OBJ export
22. Web viewer
23. Baseline vs masked vs hybrid comparison
24. Runtime report
25. Accuracy report

## Main Research Comparison

### Baseline

Normal photogrammetry.

### Improved Reconstruction

Photogrammetry with dynamic-object masking.

### Proposed Hybrid

Photogrammetry with dynamic masking plus predefined 3D asset replacement.

The comparison should focus on:

- Static geometry quality
- Dynamic-object artifacts
- Object placement quality
- Visual quality
- Runtime
- Quantitative reconstruction metrics

## Do Not

- Invent accuracy numbers
- Invent runtime
- Claim all hidden surfaces are reconstructed
- Claim ordinary GPS provides centimeter-level accuracy
- Present a visualization as proof of metric accuracy

## Final Evidence

Every major claim should have measurable evidence or a clearly stated limitation.
