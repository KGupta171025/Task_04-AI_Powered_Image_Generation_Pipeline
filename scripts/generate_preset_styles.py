"""
Generates beautiful artistic style images for VGG19 Neural Style Transfer presets.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESET_DIR = PROJECT_ROOT / "gallery" / "preset_styles"
PRESET_DIR.mkdir(parents=True, exist_ok=True)


def create_starry_night():
    """Generates a Van Gogh Starry Night inspired stylistic canvas."""
    width, height = 512, 512
    img = Image.new("RGB", (width, height), color="#0b1b3d")
    draw = ImageDraw.Draw(img)

    # Swirling blue and cyan bands
    for r in range(50, 450, 40):
        draw.arc([50, 20, 50 + r, 20 + r], start=0, end=180, fill="#1d4ed8", width=18)
        draw.arc([40, 10, 40 + r, 10 + r], start=180, end=360, fill="#0284c7", width=14)

    # Glowing stars
    stars = [(120, 100), (280, 130), (420, 90), (340, 220), (180, 250)]
    for sx, sy in stars:
        for rad, alpha in [(40, "#fbbf24"), (25, "#fef08a"), (12, "#ffffff")]:
            draw.ellipse([sx - rad, sy - rad, sx + rad, sy + rad], fill=alpha)

    # Cypress silhouette
    draw.polygon([(40, 512), (70, 240), (100, 190), (130, 260), (150, 512)], fill="#052e16")
    img = img.filter(ImageFilter.GaussianBlur(radius=3))

    out_file = PRESET_DIR / "starry_night.jpg"
    img.save(out_file, quality=95)
    print(f"[OK] Generated: {out_file.name}")


def create_great_wave():
    """Generates a Hokusai Great Wave inspired ukiyo-e texture."""
    width, height = 512, 512
    img = Image.new("RGB", (width, height), color="#fef3c7")  # Parchment background
    draw = ImageDraw.Draw(img)

    # Deep indigo ocean waves
    draw.chord([0, 260, 512, 650], start=180, end=360, fill="#1e3a8a")
    draw.chord([50, 200, 450, 580], start=190, end=350, fill="#2563eb")
    draw.chord([100, 160, 400, 520], start=200, end=340, fill="#38bdf8")

    # Foam claws
    for x in range(80, 450, 35):
        draw.polygon([(x, 260), (x + 15, 210), (x + 30, 270)], fill="#ffffff")

    # Mt Fuji in distance
    draw.polygon([(230, 280), (256, 240), (282, 280)], fill="#334155")
    draw.polygon([(246, 255), (256, 240), (266, 255)], fill="#ffffff")

    out_file = PRESET_DIR / "great_wave.jpg"
    img.save(out_file, quality=95)
    print(f"[OK] Generated: {out_file.name}")


def create_cubism_picasso():
    """Generates Picasso Cubism abstract geometric canvas."""
    width, height = 512, 512
    img = Image.new("RGB", (width, height), color="#78350f")
    draw = ImageDraw.Draw(img)

    polygons = [
        ([(40, 40), (220, 20), (180, 240), (60, 200)], "#b45309"),
        ([(220, 20), (480, 60), (440, 260), (180, 240)], "#d97706"),
        ([(60, 200), (260, 220), (200, 460), (40, 420)], "#92400e"),
        ([(260, 220), (480, 240), (460, 480), (200, 460)], "#ca8a04"),
        ([(140, 140), (340, 120), (300, 320), (120, 300)], "#f59e0b"),
    ]
    for pts, col in polygons:
        draw.polygon(pts, fill=col, outline="#451a03", width=4)

    out_file = PRESET_DIR / "cubism_picasso.jpg"
    img.save(out_file, quality=95)
    print(f"[OK] Generated: {out_file.name}")


def create_cyberpunk_neon():
    """Generates synthwave / cyberpunk neon grid texture."""
    width, height = 512, 512
    img = Image.new("RGB", (width, height), color="#090514")
    draw = ImageDraw.Draw(img)

    # Neon sun
    draw.ellipse([180, 100, 332, 252], fill="#ec4899")
    for y_slit in range(160, 240, 14):
        draw.rectangle([170, y_slit, 342, y_slit + 4], fill="#090514")

    # Perspective grid lines
    for x in range(0, 520, 40):
        draw.line([(256, 260), (x, 512)], fill="#06b6d4", width=2)
    for y in [270, 290, 320, 360, 410, 470, 510]:
        draw.line([(0, y), (512, y)], fill="#a855f7", width=2)

    out_file = PRESET_DIR / "cyberpunk_neon.jpg"
    img.save(out_file, quality=95)
    print(f"[OK] Generated: {out_file.name}")


if __name__ == "__main__":
    create_starry_night()
    create_great_wave()
    create_cubism_picasso()
    create_cyberpunk_neon()
    print("[SUCCESS] All preset style images generated successfully!")
