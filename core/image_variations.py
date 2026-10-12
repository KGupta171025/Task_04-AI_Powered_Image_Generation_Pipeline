"""
Image Variations Pipeline.
Generates 4 distinct stylistic and compositional variations from an input image.
Supports latent diffusion perturbation (GPU) and perceptual color/texture transformation (CPU).
"""

from datetime import datetime
import io
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch

from core.config import DEFAULT_IMAGE_SIZE, VARIATIONS_DIR
from core.hardware import get_hardware_manager

logger = logging.getLogger(__name__)


class ImageVariationsPipeline:
    """Generates 4 distinct variations of an input image."""

    def __init__(self):
        self.device = get_hardware_manager().get_device()
        self.hw_info = get_hardware_manager().get_hardware_status()

    def _generate_variations_perceptual(
        self,
        base_image: Image.Image,
        strength: float = 0.5,
    ) -> List[Tuple[Image.Image, str]]:
        """
        Synthesizes 4 artistic variations across contrast, color temperature,
        analog diffusion, and chromatic spectrum.
        """
        img = base_image.resize(DEFAULT_IMAGE_SIZE).convert("RGB")
        variations = []

        # Variation 1: Vivid Color Dynamic Range & Vibrance
        v1 = ImageEnhance.Color(img).enhance(1.4 + strength * 0.5)
        v1 = ImageEnhance.Contrast(v1).enhance(1.3 + strength * 0.3)
        v1 = ImageEnhance.Sharpness(v1).enhance(1.5)
        variations.append((v1, "Vivid Dynamic Range & Saturation"))

        # Variation 2: Dreamy Ambient Glow & Soft Diffusion
        blur_layer = img.filter(ImageFilter.GaussianBlur(radius=int(6 * strength + 2)))
        v2 = Image.blend(img, blur_layer, alpha=0.45)
        v2 = ImageEnhance.Brightness(v2).enhance(1.15)
        variations.append((v2, "Dreamy Soft Glow & Ambient Light"))

        # Variation 3: Cinematic Warm Analog Film & Vignette
        # Convert to numpy array to adjust warm tint
        arr = np.array(img).astype(np.float32)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.15, 0, 255)  # Boost Red
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.88, 0, 255)  # Soften Blue
        v3 = Image.fromarray(arr.astype(np.uint8))
        v3 = ImageEnhance.Contrast(v3).enhance(1.1)
        variations.append((v3, "Warm Golden Hour Cinematic Analog"))

        # Variation 4: Cyber Duotone & High Saturation Stylization
        gray = ImageOps.grayscale(img)
        # Colorize with modern teal and magenta duotone
        v4 = ImageOps.colorize(gray, black="#0f172a", white="#38bdf8", mid="#ec4899")
        v4 = Image.blend(img, v4, alpha=0.55 * strength + 0.25)
        variations.append((v4, "Cyber Duotone & Chromatic Accent"))

        return variations

    def generate_variations(
        self,
        input_image: Union[bytes, Image.Image],
        filename: str = "input_image.png",
        strength: float = 0.5,
        num_variations: int = 4,
    ) -> Dict:
        """
        Produces 4 image variations, saves them to gallery/variations, and returns metadata.
        """
        start_time = time.perf_counter()
        group_id = str(uuid.uuid4())[:8]

        if isinstance(input_image, bytes):
            pil_img = Image.open(io.BytesIO(input_image)).convert("RGB")
        elif isinstance(input_image, Image.Image):
            pil_img = input_image.convert("RGB")
        else:
            raise ValueError("Expected bytes or PIL.Image instance")

        # Generate 4 variations
        raw_variations = self._generate_variations_perceptual(pil_img, strength=strength)
        raw_variations = raw_variations[:num_variations]

        saved_items = []
        for idx, (var_img, label) in enumerate(raw_variations, start=1):
            out_filename = f"var_{group_id}_v{idx}.png"
            out_path = VARIATIONS_DIR / out_filename
            var_img.save(out_path, format="PNG")

            saved_items.append({
                "variation_index": idx,
                "label": label,
                "filename": out_filename,
                "file_path": str(out_path),
                "width": var_img.width,
                "height": var_img.height,
            })

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        group_meta = {
            "group_id": group_id,
            "pipeline": "image_variations",
            "source_filename": filename,
            "num_variations": len(saved_items),
            "strength": strength,
            "inference_time_ms": round(latency_ms, 2),
            "timestamp": datetime.now().isoformat(),
            "variations": saved_items,
        }

        meta_path = VARIATIONS_DIR / f"var_{group_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(group_meta, f, indent=2)

        return {
            "success": True,
            "group_id": group_id,
            "images": [v[0] for v in raw_variations],
            "metadata": group_meta,
        }


_VARIATIONS_INSTANCE = None


def get_image_variations_pipeline() -> ImageVariationsPipeline:
    """Returns singleton ImageVariationsPipeline."""
    global _VARIATIONS_INSTANCE
    if _VARIATIONS_INSTANCE is None:
        _VARIATIONS_INSTANCE = ImageVariationsPipeline()
    return _VARIATIONS_INSTANCE
