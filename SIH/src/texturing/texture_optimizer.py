from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import trimesh


class TextureOptimizer:
    def __init__(
        self,
        default_color: Tuple[int, int, int] = (160, 160, 160),
    ):
        self.default_color = default_color

    def bake_vertex_colors(
        self,
        mesh: trimesh.Trimesh,
        texture_size: int = 2048,
    ) -> Image.Image:
        w, h = texture_size, texture_size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        vertex_colors = None
        if (
            hasattr(mesh, "visual")
            and hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            and len(mesh.visual.vertex_colors) == len(mesh.vertices)
        ):
            vertex_colors = np.asarray(mesh.visual.vertex_colors)
        elif (
            hasattr(mesh, "vertex_attributes")
            and mesh.vertex_attributes.get("orig_vertex_colors") is not None
        ):
            candidate = np.asarray(mesh.vertex_attributes["orig_vertex_colors"])
            if len(candidate) == len(mesh.vertices):
                vertex_colors = candidate

        has_uv = (
            hasattr(mesh, "visual")
            and hasattr(mesh.visual, "uv")
            and mesh.visual.uv is not None
            and len(mesh.visual.uv) == len(mesh.vertices)
        )

        if vertex_colors is None or not has_uv or len(mesh.vertices) == 0:
            img[:] = self.default_color
            return Image.fromarray(img, mode="RGB")

        uvs = np.asarray(mesh.visual.uv, dtype=np.float64)
        colors = vertex_colors[:, :3].astype(np.uint8)

        px = np.clip((uvs[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
        py = np.clip(((1.0 - uvs[:, 1]) * (h - 1)).astype(np.int32), 0, h - 1)

        img[py, px] = colors

        unmapped = (img.sum(axis=2) == 0).astype(np.uint8)
        if np.any(unmapped):
            kernel = np.ones((5, 5), dtype=np.uint8)
            dilated = cv2.dilate(img, kernel, iterations=3)
            img[unmapped > 0] = dilated[unmapped > 0]

            still_empty = (img.sum(axis=2) == 0)
            if np.any(still_empty):
                img[still_empty] = self.default_color

        return Image.fromarray(img, mode="RGB")

    def resize_texture(
        self,
        image: Image.Image,
        target_size: int = 2048,
    ) -> Image.Image:
        if image.size == (target_size, target_size):
            return image
        return image.resize((target_size, target_size), resample=Image.Resampling.LANCZOS)

    def blend_multi_textures(
        self,
        images: List[Image.Image],
        weights: Optional[List[float]] = None,
    ) -> Image.Image:
        if not images:
            return Image.new("RGB", (1024, 1024), color=self.default_color)
        if len(images) == 1:
            return images[0]

        target_size = images[0].size
        resized = [img.resize(target_size, resample=Image.Resampling.BILINEAR) for img in images]

        n = len(resized)
        if weights is None or len(weights) != n:
            w_arr = np.ones(n, dtype=np.float32) / n
        else:
            w_arr = np.array(weights, dtype=np.float32)
            total = np.sum(w_arr)
            if total > 0:
                w_arr /= total
            else:
                w_arr = np.ones(n, dtype=np.float32) / n

        accum = np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)
        for i, img in enumerate(resized):
            accum += np.asarray(img, dtype=np.float32) * w_arr[i]

        blended = np.clip(accum, 0, 255).astype(np.uint8)
        return Image.fromarray(blended, mode="RGB")
