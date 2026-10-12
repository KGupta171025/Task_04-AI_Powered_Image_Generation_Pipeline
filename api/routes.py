"""
FastAPI Route Handlers for Image Generation Pipeline.
Endpoints for Text-to-Image, VGG19 Style Transfer, Variations, and Gallery inspection.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image

from api.schemas import (
    ErrorResponse,
    GalleryItem,
    GalleryResponse,
    HardwareResponse,
    HealthResponse,
    ImageGenerationResponse,
    TextToImageRequest,
    VariationItem,
    VariationsResponse,
)
from core.config import (
    API_VERSION,
    GALLERY_DIR,
    MAX_FILE_SIZE_BYTES,
    PRESET_STYLES,
    PRESET_STYLES_DIR,
    STYLE_TRANSFER_DIR,
    TEXT_TO_IMAGE_DIR,
    VARIATIONS_DIR,
)
from core.hardware import get_hardware_manager
from core.image_variations import get_image_variations_pipeline
from core.style_transfer import get_style_transfer_pipeline
from core.text_to_image import get_text_to_image_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Image Generation Pipeline"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & Readiness Check",
    description="Returns service uptime, active compute hardware, and total gallery count.",
)
async def health_check() -> HealthResponse:
    hw = get_hardware_manager().get_hardware_status()
    total_imgs = len(list(GALLERY_DIR.glob("**/*.png")))
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        active_device=hw["device"],
        hardware_mode=hw["recommended_mode"],
        gallery_count=total_imgs,
    )


@router.get(
    "/hardware",
    response_model=HardwareResponse,
    summary="Hardware Diagnostics & Routing Policy",
    description="Inspects GPU VRAM, CPU cores, system RAM, and active generation routing tier.",
)
async def get_hardware() -> HardwareResponse:
    hw = get_hardware_manager().get_hardware_status()
    return HardwareResponse(**hw)


@router.get(
    "/preset-styles",
    summary="List Built-in Art Styles",
    description="Returns list of curated artistic styles available for Neural Style Transfer.",
)
async def list_preset_styles():
    return {"preset_styles": PRESET_STYLES}


@router.post(
    "/generate/text-to-image",
    response_model=ImageGenerationResponse,
    summary="Generate 512x512 Image from Text Prompt",
    description="Generates an image from natural language prompt with seed and step controls.",
)
async def generate_text_to_image(request: TextToImageRequest) -> ImageGenerationResponse:
    t2i = get_text_to_image_pipeline()
    try:
        res = t2i.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            width=request.width,
            height=request.height,
        )
    except Exception as e:
        logger.exception("Text-to-Image failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text-to-Image generation failed: {str(e)}",
        )

    meta = res["metadata"]
    return ImageGenerationResponse(
        success=True,
        image_id=meta["image_id"],
        image_url=f"/api/gallery/text_to_image/{meta['filename']}",
        pipeline="text_to_image",
        inference_time_ms=meta["inference_time_ms"],
        metadata=meta,
    )


@router.post(
    "/generate/style-transfer",
    response_model=ImageGenerationResponse,
    summary="Neural Style Transfer (VGG19)",
    description="Combines content photo and style artwork using VGG19 feature extraction.",
)
async def generate_style_transfer(
    content_file: UploadFile = File(..., description="Content image file"),
    style_file: Optional[UploadFile] = File(None, description="Custom style image file"),
    preset_style: Optional[str] = Form(None, description="Preset style key (e.g. 'starry_night')"),
    content_weight: float = Form(1.0, description="Weight for content preservation"),
    style_weight: float = Form(100000.0, description="Weight for style application"),
) -> ImageGenerationResponse:
    # 1. Read content image
    content_bytes = await content_file.read()
    if len(content_bytes) == 0:
        raise HTTPException(status_code=400, detail="Content image file is empty.")

    # 2. Determine style image
    style_bytes = None
    style_label = "custom_style"
    if style_file is not None:
        style_bytes = await style_file.read()
        style_label = style_file.filename or "custom_style.jpg"
    elif preset_style and preset_style in PRESET_STYLES:
        preset_path = PRESET_STYLES_DIR / PRESET_STYLES[preset_style]["style_file"]
        if preset_path.exists():
            style_bytes = preset_path.read_bytes()
            style_label = PRESET_STYLES[preset_style]["name"]

    if not style_bytes:
        # Default to starry_night if none provided
        default_p = PRESET_STYLES_DIR / "starry_night.jpg"
        if default_p.exists():
            style_bytes = default_p.read_bytes()
            style_label = "Van Gogh Starry Night (Default)"
        else:
            raise HTTPException(status_code=400, detail="Must provide either style_file or valid preset_style.")

    # 3. Execute style transfer
    st_pipe = get_style_transfer_pipeline()
    try:
        res = st_pipe.transfer(
            content_image=content_bytes,
            style_image=style_bytes,
            content_name=content_file.filename or "content.jpg",
            style_name=style_label,
            content_weight=content_weight,
            style_weight=style_weight,
        )
    except Exception as e:
        logger.exception("Style transfer error: %s", e)
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {str(e)}")

    meta = res["metadata"]
    return ImageGenerationResponse(
        success=True,
        image_id=meta["image_id"],
        image_url=f"/api/gallery/style_transfer/{meta['filename']}",
        pipeline="style_transfer",
        inference_time_ms=meta["inference_time_ms"],
        metadata=meta,
    )


@router.post(
    "/generate/variations",
    response_model=VariationsResponse,
    summary="Generate 4 Variations of Input Image",
    description="Takes an input image and synthesizes 4 distinct stylistic and compositional variations.",
)
async def generate_variations(
    image_file: UploadFile = File(..., description="Base input image"),
    strength: float = Form(0.5, ge=0.1, le=1.0, description="Variation perturbation strength"),
    num_variations: int = Form(4, ge=1, le=4, description="Number of variations (default: 4)"),
) -> VariationsResponse:
    img_bytes = await image_file.read()
    if len(img_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    var_pipe = get_image_variations_pipeline()
    try:
        res = var_pipe.generate_variations(
            input_image=img_bytes,
            filename=image_file.filename or "input.png",
            strength=strength,
            num_variations=num_variations,
        )
    except Exception as e:
        logger.exception("Image variations failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Variations generation failed: {str(e)}")

    meta = res["metadata"]
    items = [
        VariationItem(
            variation_index=v["variation_index"],
            label=v["label"],
            filename=v["filename"],
            image_url=f"/api/gallery/variations/{v['filename']}",
        )
        for v in meta["variations"]
    ]

    return VariationsResponse(
        success=True,
        group_id=meta["group_id"],
        source_filename=meta["source_filename"],
        inference_time_ms=meta["inference_time_ms"],
        variations=items,
        metadata=meta,
    )


@router.get(
    "/gallery",
    response_model=GalleryResponse,
    summary="Explore Generated Image Gallery",
    description="Lists all persisted images across text-to-image, style transfer, and variations.",
)
async def list_gallery():
    items = []
    counts = {"text_to_image": 0, "style_transfer": 0, "variations": 0}

    # Iterate over meta JSON files
    for cat, d in [("text_to_image", TEXT_TO_IMAGE_DIR), ("style_transfer", STYLE_TRANSFER_DIR), ("variations", VARIATIONS_DIR)]:
        for meta_file in sorted(d.glob("*.json"), reverse=True):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                img_id = m.get("image_id") or m.get("group_id", meta_file.stem)
                fn = m.get("filename") or (m.get("variations")[0]["filename"] if "variations" in m else "")
                items.append(
                    GalleryItem(
                        image_id=img_id,
                        pipeline=cat,
                        filename=fn,
                        image_url=f"/api/gallery/{cat}/{fn}",
                        timestamp=m.get("timestamp", ""),
                        metadata=m,
                    )
                )
                counts[cat] += 1
            except Exception:
                continue

    return GalleryResponse(total_images=len(items), categories=counts, items=items)


@router.get(
    "/gallery/{category}/{filename}",
    summary="Download or View Gallery Image",
    description="Serves PNG/JPG file from the gallery.",
)
async def get_gallery_image(category: str, filename: str):
    target_dir = {
        "text_to_image": TEXT_TO_IMAGE_DIR,
        "style_transfer": STYLE_TRANSFER_DIR,
        "variations": VARIATIONS_DIR,
        "preset_styles": PRESET_STYLES_DIR,
    }.get(category)

    if not target_dir:
        raise HTTPException(status_code=404, detail="Invalid gallery category.")

    file_path = target_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found.")

    return FileResponse(file_path, media_type="image/png" if filename.endswith(".png") else "image/jpeg")
