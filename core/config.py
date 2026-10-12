"""
System Configuration and Constants for Image Generation Pipeline.
"""

from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GALLERY_DIR = PROJECT_ROOT / "gallery"
TEXT_TO_IMAGE_DIR = GALLERY_DIR / "text_to_image"
STYLE_TRANSFER_DIR = GALLERY_DIR / "style_transfer"
VARIATIONS_DIR = GALLERY_DIR / "variations"
PRESET_STYLES_DIR = GALLERY_DIR / "preset_styles"
DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = PROJECT_ROOT / "assets"
SCREENSHOTS_DIR = ASSETS_DIR / "screenshots"

# Ensure all runtime directories exist
TEXT_TO_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
STYLE_TRANSFER_DIR.mkdir(parents=True, exist_ok=True)
VARIATIONS_DIR.mkdir(parents=True, exist_ok=True)
PRESET_STYLES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Image dimensions
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (512, 512)
IMAGE_WIDTH: int = 512
IMAGE_HEIGHT: int = 512

# Style Transfer defaults (Gatys et al. VGG19 Neural Style Transfer)
DEFAULT_CONTENT_WEIGHT: float = 1.0
DEFAULT_STYLE_WEIGHT: float = 100000.0
DEFAULT_STYLE_STEPS: int = 150

# Variations
DEFAULT_NUM_VARIATIONS: int = 4
DEFAULT_VARIATION_STRENGTH: float = 0.45

# Preset Art Styles available in Gallery & Web Interface
PRESET_STYLES: Dict[str, dict] = {
    "starry_night": {
        "name": "Van Gogh's Starry Night",
        "description": "Post-impressionist swirling night sky with dynamic blues and glowing yellows",
        "style_file": "starry_night.jpg",
    },
    "great_wave": {
        "name": "Hokusai's Great Wave",
        "description": "Traditional Japanese woodblock ukiyo-e aesthetic with crested ocean foam",
        "style_file": "great_wave.jpg",
    },
    "cubism_picasso": {
        "name": "Picasso's Cubism",
        "description": "Geometric abstraction, fragmented planes, and vibrant contrasting tones",
        "style_file": "cubism_picasso.jpg",
    },
    "cyberpunk_neon": {
        "name": "Cyberpunk Neon Grid",
        "description": "Futuristic synthwave glow with high-saturation magenta and cyan lighting",
        "style_file": "cyberpunk_neon.jpg",
    },
}

# API Config
API_TITLE = "Nexoramind AI Image Generation Pipeline API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Production-grade pipeline for Text-to-Image Generation (512x512), "
    "Neural Style Transfer (VGG19), and 4-way Image Variations with hardware routing."
)
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
