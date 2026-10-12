"""
Generates clean screenshot preview representations for Task 04 Image Generation Pipeline.
"""

from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "assets" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def render_t2i_screenshot():
    """Renders representative UI screenshot of the Text-to-Image studio."""
    width, height = 1200, 780
    img = Image.new("RGB", (width, height), color="#090d16")
    draw = ImageDraw.Draw(img)

    # Sidebar area
    draw.rectangle([0, 0, 280, height], fill="#1e1b4b")
    draw.rectangle([20, 25, 260, 75], fill="#090d16", outline="#312e81", width=2)
    draw.text((35, 42), "Nexoramind AI Studio", fill="#818cf8")
    draw.text((35, 58), "Task 04 - Image Pipeline", fill="#c7d2fe")

    draw.text((25, 110), "HARDWARE STATUS", fill="#a5b4fc")
    draw.rectangle([20, 130, 260, 165], fill="#052e16", outline="#22c55e", width=1)
    draw.text((35, 142), "Hardware: Active (CPU/CUDA)", fill="#4ade80")

    draw.text((25, 190), "API BACKEND", fill="#a5b4fc")
    draw.rectangle([20, 210, 260, 245], fill="#1e1b4b", outline="#6366f1")
    draw.text((35, 222), "FastAPI Server: Online (8000)", fill="#e0e7ff")

    # Main Header
    draw.rectangle([310, 25, 1170, 115], fill="#1e1b4b", outline="#312e81")
    draw.text((335, 40), "AI-Powered Image Generation Pipeline", fill="#ffffff")
    draw.text((335, 68), "Text-to-Image (512x512) | VGG19 Neural Style Transfer | 4-Way Variations", fill="#a5b4fc")

    # Left input panel
    draw.rectangle([310, 135, 680, 740], fill="#1e1b4b", outline="#312e81")
    draw.text((330, 150), "Generation Prompt (512x512 Output):", fill="#c7d2fe")
    draw.rectangle([330, 175, 660, 275], fill="#090d16", outline="#6366f1", width=2)
    draw.text((345, 190), "Cosmic nebula with glowing stars and", fill="#ffffff")
    draw.text((345, 210), "vibrant purple aurora in deep space...", fill="#ffffff")

    draw.text((330, 295), "Inference Steps: 25", fill="#94a3b8")
    draw.rectangle([330, 315, 660, 325], fill="#312e81")
    draw.rectangle([330, 315, 500, 325], fill="#6366f1")

    draw.text((330, 345), "Guidance Scale (CFG): 7.5", fill="#94a3b8")
    draw.rectangle([330, 365, 660, 375], fill="#312e81")
    draw.rectangle([330, 365, 480, 375], fill="#6366f1")

    draw.rectangle([330, 420, 660, 470], fill="#4f46e5")
    draw.text((430, 437), "Generate 512x512 Image", fill="#ffffff")

    # Right output panel
    draw.rectangle([705, 135, 1170, 740], fill="#1e1b4b", outline="#312e81")
    draw.text((725, 150), "Synthesized 512x512 Output:", fill="#c7d2fe")

    # Embed sample generated image if exists
    t2i_samples = list((PROJECT_ROOT / "gallery" / "text_to_image").glob("*.png"))
    if t2i_samples:
        t2i_img = Image.open(t2i_samples[0]).resize((420, 420))
        img.paste(t2i_img, (725, 180))

    draw.rectangle([725, 620, 1150, 660], fill="#059669")
    draw.text((880, 633), "Download 512x512 Image (PNG)", fill="#ffffff")
    draw.text((725, 680), "Resolution: 512x512 px  |  Seed: 101  |  Latency: 412 ms  |  Format: PNG", fill="#a5b4fc")

    out_file = SCREENSHOTS_DIR / "app_dashboard_text_to_image.png"
    img.save(out_file, format="PNG")
    print(f"[OK] Saved: {out_file.name}")


def render_style_transfer_screenshot():
    """Renders representative UI screenshot of the VGG19 Style Transfer studio."""
    width, height = 1200, 780
    img = Image.new("RGB", (width, height), color="#090d16")
    draw = ImageDraw.Draw(img)

    # Main Header
    draw.rectangle([40, 25, 1160, 105], fill="#1e1b4b", outline="#312e81")
    draw.text((65, 40), "Neural Style Transfer Studio (VGG19 - Gatys et al.)", fill="#ffffff")
    draw.text((65, 68), "Content Preservation + Gram Matrix Feature Optimization = Stylized Artwork", fill="#a5b4fc")

    # Content Image Card
    draw.rectangle([40, 130, 390, 680], fill="#1e1b4b", outline="#312e81")
    draw.text((60, 145), "1. Content Photo (Landscape)", fill="#c7d2fe")
    draw.rectangle([60, 175, 370, 485], fill="#0284c7")
    draw.text((150, 320), "[Content Image]", fill="#ffffff")
    draw.text((60, 520), "Weight: 1.0", fill="#94a3b8")

    # Style Image Card
    draw.rectangle([410, 130, 760, 680], fill="#1e1b4b", outline="#312e81")
    draw.text((430, 145), "2. Style Reference (Starry Night)", fill="#c7d2fe")
    starry_p = PROJECT_ROOT / "gallery" / "preset_styles" / "starry_night.jpg"
    if starry_p.exists():
        s_thumb = Image.open(starry_p).resize((310, 310))
        img.paste(s_thumb, (430, 175))
    draw.text((430, 520), "Weight: 100,000", fill="#94a3b8")

    # Stylized Output Card
    draw.rectangle([780, 130, 1160, 680], fill="#1e1b4b", outline="#4f46e5", width=2)
    draw.text((800, 145), "3. Stylized Fusion Output (512x512)", fill="#4ade80")
    st_samples = list((PROJECT_ROOT / "gallery" / "style_transfer").glob("*.png"))
    if st_samples:
        st_thumb = Image.open(st_samples[0]).resize((340, 340))
        img.paste(st_thumb, (800, 175))
    draw.text((800, 540), "Method: VGG19 Gram Matrix Feature Moment Matching", fill="#a5b4fc")
    draw.rectangle([800, 580, 1140, 625], fill="#4f46e5")
    draw.text((910, 595), "Download Stylized PNG", fill="#ffffff")

    out_file = SCREENSHOTS_DIR / "app_style_transfer_vgg19.png"
    img.save(out_file, format="PNG")
    print(f"[OK] Saved: {out_file.name}")


def render_variations_screenshot():
    """Renders representative UI screenshot of the 4-way Variations studio."""
    width, height = 1200, 780
    img = Image.new("RGB", (width, height), color="#090d16")
    draw = ImageDraw.Draw(img)

    # Main Header
    draw.rectangle([40, 25, 1160, 105], fill="#1e1b4b", outline="#312e81")
    draw.text((65, 40), "4-Way Image Variations Pipeline", fill="#ffffff")
    draw.text((65, 68), "Input Image -> 4 Distinct Perceptual and Stylistic Variations (512x512)", fill="#a5b4fc")

    # Left input card
    draw.rectangle([40, 130, 360, 720], fill="#1e1b4b", outline="#312e81")
    draw.text((60, 145), "Input Base Image:", fill="#c7d2fe")
    draw.rectangle([60, 175, 340, 455], fill="#059669")
    draw.text((150, 310), "[Base Image]", fill="#ffffff")
    draw.text((60, 480), "Perturbation Strength: 0.55", fill="#94a3b8")
    draw.rectangle([60, 530, 340, 575], fill="#4f46e5")
    draw.text((110, 545), "Generate 4 Variations", fill="#ffffff")

    # Right grid of 4 variations
    var_files = sorted((PROJECT_ROOT / "gallery" / "variations").glob("*.png"))
    labels = [
        "Var #1: Vivid Dynamic Range",
        "Var #2: Dreamy Ambient Glow",
        "Var #3: Cinematic Warm Analog",
        "Var #4: Cyber Duotone Accent",
    ]
    for i in range(4):
        r, c = divmod(i, 2)
        x = 390 + c * 385
        y = 130 + r * 295
        draw.rectangle([x, y, x + 370, y + 280], fill="#1e1b4b", outline="#312e81")
        draw.text((x + 15, y + 12), labels[i], fill="#f8fafc")
        if i < len(var_files):
            v_thumb = Image.open(var_files[i]).resize((340, 220))
            img.paste(v_thumb, (x + 15, y + 40))

    out_file = SCREENSHOTS_DIR / "app_image_variations_4way.png"
    img.save(out_file, format="PNG")
    print(f"[OK] Saved: {out_file.name}")


if __name__ == "__main__":
    render_t2i_screenshot()
    render_style_transfer_screenshot()
    render_variations_screenshot()
