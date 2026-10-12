"""
Pre-populates the generated image gallery with representative outputs
across Text-to-Image, Style Transfer, and Variations pipelines.
"""

from pathlib import Path
import sys
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.image_variations import get_image_variations_pipeline
from core.style_transfer import get_style_transfer_pipeline
from core.text_to_image import get_text_to_image_pipeline
from core.config import PRESET_STYLES_DIR


def populate_gallery():
    """Generates initial gallery items."""
    print("Generating sample Text-to-Image outputs...")
    t2i = get_text_to_image_pipeline()
    prompts = [
        ("Cosmic nebula with glowing stars and vibrant purple aurora", 101),
        ("Cyberpunk futuristic city skyline at midnight with neon rain", 202),
        ("Tranquil mountain lake surrounded by pine forests at golden sunset", 303),
    ]
    for prompt, seed in prompts:
        res = t2i.generate(prompt=prompt, seed=seed)
        print(f"[OK] Generated T2I: '{prompt[:45]}...' ({res['metadata']['filename']})")

    print("\nGenerating sample Style Transfer outputs...")
    st_pipe = get_style_transfer_pipeline()
    
    # Create base content image (a stylized portrait/house)
    content_img = Image.new("RGB", (512, 512), color="#38bdf8")
    draw = ImageDraw.Draw(content_img)
    draw.rectangle([0, 300, 512, 512], fill="#16a34a") # Grass
    draw.rectangle([160, 200, 350, 420], fill="#f59e0b") # House
    draw.polygon([(130, 200), (256, 110), (380, 200)], fill="#dc2626") # Roof
    draw.ellipse([380, 50, 460, 130], fill="#facc15") # Sun
    
    starry_style = PRESET_STYLES_DIR / "starry_night.jpg"
    great_wave_style = PRESET_STYLES_DIR / "great_wave.jpg"
    
    if starry_style.exists():
        res1 = st_pipe.transfer(content_img, Image.open(starry_style), content_name="landscape.png", style_name="Starry Night")
        print(f"[OK] Generated Style Transfer: Starry Night ({res1['metadata']['filename']})")

    if great_wave_style.exists():
        res2 = st_pipe.transfer(content_img, Image.open(great_wave_style), content_name="landscape.png", style_name="Great Wave")
        print(f"[OK] Generated Style Transfer: Great Wave ({res2['metadata']['filename']})")

    print("\nGenerating sample 4-way Variations...")
    var_pipe = get_image_variations_pipeline()
    var_res = var_pipe.generate_variations(content_img, filename="sample_landscape.png", strength=0.55)
    print(f"[OK] Generated 4 Variations (Group ID: {var_res['group_id']})")

    print("\n[SUCCESS] Gallery populated successfully!")


if __name__ == "__main__":
    populate_gallery()
