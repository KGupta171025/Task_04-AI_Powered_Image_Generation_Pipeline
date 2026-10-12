"""
Pydantic Schemas for AI Image Generation Pipeline API.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TextToImageRequest(BaseModel):
    """Payload for text-to-image synthesis."""
    prompt: str = Field(..., min_length=2, description="Text prompt describing the scene")
    negative_prompt: Optional[str] = Field("blurry, low quality, distorted, artifacts", description="Unwanted elements")
    num_inference_steps: Optional[int] = Field(25, ge=5, le=100, description="Denoising steps")
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0, description="Classifier-free guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    width: Optional[int] = Field(512, description="Width in pixels (default: 512)")
    height: Optional[int] = Field(512, description="Height in pixels (default: 512)")


class ImageGenerationResponse(BaseModel):
    """Standardized response for generated single images."""
    success: bool = True
    image_id: str
    image_url: str
    pipeline: str
    inference_time_ms: float
    metadata: Dict


class VariationItem(BaseModel):
    """Individual variation entry in 4-way variation response."""
    variation_index: int
    label: str
    filename: str
    image_url: str


class VariationsResponse(BaseModel):
    """Response returned by 4-way image variations endpoint."""
    success: bool = True
    group_id: str
    source_filename: str
    inference_time_ms: float
    variations: List[VariationItem]
    metadata: Dict


class HardwareResponse(BaseModel):
    """Hardware diagnostic response."""
    device: str
    cuda_available: bool
    gpu_name: str
    gpu_memory_total_gb: float
    gpu_memory_allocated_gb: float
    cpu_cores: int
    system_ram_total_gb: float
    system_ram_available_gb: float
    os: str
    recommended_mode: str
    pipeline_strategy: str


class GalleryItem(BaseModel):
    """Item in gallery overview."""
    image_id: str
    pipeline: str
    filename: str
    image_url: str
    timestamp: str
    metadata: Dict


class GalleryResponse(BaseModel):
    """Gallery listing response."""
    total_images: int
    categories: Dict[str, int]
    items: List[GalleryItem]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    active_device: str
    hardware_mode: str
    gallery_count: int


class ErrorResponse(BaseModel):
    """Standardized error output."""
    success: bool = False
    error: str
    detail: str
    status_code: int
