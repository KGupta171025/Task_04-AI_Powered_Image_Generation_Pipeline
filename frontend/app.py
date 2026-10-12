"""
Nexoramind AI-Powered Image Generation Pipeline - Streamlit Web Application.
Interactive studio for Text-to-Image Generation (512x512), VGG19 Neural Style Transfer,
4-Way Image Variations, and Curated Gallery Inspection.
"""

from datetime import datetime
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    DEFAULT_IMAGE_SIZE,
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
from frontend.styles import CUSTOM_CSS

# Page Configuration
st.set_page_config(
    page_title="Nexoramind - AI Image Generation Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Session state
if "total_creations" not in st.session_state:
    st.session_state.total_creations = 0


def check_api_health():
    try:
        r = requests.get(f"{API_BASE_URL}/api/health", timeout=1.5)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🎨 Nexoramind AI Studio")
    st.caption("Task 04: AI Image Generation Pipeline")
    st.divider()

    hw_info = get_hardware_manager().get_hardware_status()
    api_online, health_data = check_api_health()

    st.subheader("⚙️ Hardware Status")
    if hw_info["cuda_available"]:
        st.markdown(
            f'<span class="hw-badge-cuda">🟢 GPU Active ({hw_info["gpu_name"]})</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"VRAM: `{hw_info['gpu_memory_allocated_gb']} / {hw_info['gpu_memory_total_gb']} GB`")
    else:
        st.markdown(
            '<span class="hw-badge-cpu">🔵 CPU Optimized Mode</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"RAM: `{hw_info['system_ram_available_gb']} GB free` | Cores: `{hw_info['cpu_cores']}`")

    st.caption(f"Strategy: `{hw_info['recommended_mode']}`")

    st.divider()
    st.subheader("🌐 API Backend")
    if api_online:
        st.caption("FastAPI Server: **Online (127.0.0.1:8000)**")
    else:
        st.caption("FastAPI Server: **Offline (Serving via Direct Engine)**")

    st.divider()
    st.metric("Images Generated in Session", st.session_state.total_creations)


# ==================== MAIN BANNER ====================
st.markdown(
    """
    <div class="hero-banner">
        <span class="hero-badge">Nexoramind AI Internship</span>
        <span class="hero-badge">Deep Generative Vision</span>
        <span class="hero-badge">Task 04</span>
        <h1 class="hero-title">AI-Powered Image Generation Pipeline</h1>
        <p class="hero-sub">
            High-fidelity Text-to-Image Synthesis (512x512), VGG19 Neural Style Transfer, 
            and 4-Way Variations Studio with hardware-aware routing.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


tab_t2i, tab_style, tab_var, tab_gallery, tab_docs = st.tabs([
    "✨ Text-to-Image",
    "🖌️ Style Transfer (VGG19)",
    "🔀 4-Way Variations",
    "🖼️ Curated Gallery",
    "📖 Hardware & API Docs",
])


# ==================== TAB 1: TEXT-TO-IMAGE ====================
with tab_t2i:
    col_input, col_out = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown("### 📝 Enter Generation Prompt")

        prompt_input = st.text_area(
            "Natural Language Prompt (512x512 Output)",
            height=120,
            placeholder="e.g. Majestic cosmic lion standing on an asteroid with purple nebula stars and golden lighting",
        )

        st.markdown("#### Sample Prompts")
        sample_prompts = [
            "Cosmic nebula with glowing stars and vibrant purple aurora",
            "Cyberpunk futuristic city skyline at midnight with neon rain",
            "Tranquil mountain lake surrounded by pine forests at golden sunset",
            "Ancient crystal temple floating in deep mystical clouds",
        ]
        cols_p = st.columns(2)
        for i, sp in enumerate(sample_prompts):
            c = cols_p[i % 2]
            with c:
                if st.button(sp[:32] + "...", key=f"sp_btn_{i}", use_container_width=True):
                    prompt_input = sp

        with st.expander("🛠️ Advanced Parameters", expanded=False):
            neg_prompt = st.text_input("Negative Prompt", "blurry, low quality, distorted, artifacts")
            steps = st.slider("Inference Steps", 10, 50, 25)
            guidance = st.slider("Guidance Scale (CFG)", 1.0, 15.0, 7.5, 0.5)
            seed_val = st.number_input("Random Seed (0 for random)", min_value=0, value=0, step=1)

        gen_btn = st.button("🚀 Generate 512x512 Image", type="primary", use_container_width=True, disabled=not prompt_input.strip())

    with col_out:
        st.markdown("### 🖼️ Synthesized Output")

        if gen_btn and prompt_input.strip():
            with st.spinner("Executing neural synthesis pipeline..."):
                t2i_pipe = get_text_to_image_pipeline()
                active_seed = None if seed_val == 0 else int(seed_val)
                res = t2i_pipe.generate(
                    prompt=prompt_input.strip(),
                    negative_prompt=neg_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=active_seed,
                )

            if res and res.get("success"):
                st.session_state.total_creations += 1
                img = res["image"]
                meta = res["metadata"]

                st.image(img, caption=f"512x512 | Seed: {meta['seed']} | Latency: {meta['inference_time_ms']}ms", use_container_width=True)

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    label="📥 Download 512x512 Image",
                    data=buf.getvalue(),
                    file_name=meta["filename"],
                    mime="image/png",
                    use_container_width=True,
                )

                st.caption(
                    f"⚡ **Latency:** `{meta['inference_time_ms']} ms` | 💻 **Backend:** `{meta['backend']}` "
                    f"| 🎲 **Seed:** `{meta['seed']}`"
                )
        else:
            # Display recent text-to-image preview if available
            recent_t2i = sorted(TEXT_TO_IMAGE_DIR.glob("*.png"), key=os.path.getmtime, reverse=True)
            if recent_t2i:
                st.image(Image.open(recent_t2i[0]), caption=f"Latest Generated: {recent_t2i[0].name}", use_container_width=True)
            else:
                st.info("Enter a prompt and click Generate to synthesize an image.")


# ==================== TAB 2: STYLE TRANSFER ====================
with tab_style:
    st.markdown("### 🖌️ Neural Style Transfer (VGG19 - Gatys et al.)")

    s_col1, s_col2, s_col3 = st.columns([1, 1, 1.2], gap="medium")

    with s_col1:
        st.markdown("#### 1. Content Photo")
        content_upload = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
        content_img = None
        if content_upload:
            content_img = Image.open(content_upload).convert("RGB")
            st.image(content_img, caption="Content Source", use_container_width=True)
        else:
            # Fallback sample content
            content_img = Image.new("RGB", (512, 512), color="#38bdf8")
            st.caption("Using default landscape canvas if none uploaded.")

    with s_col2:
        st.markdown("#### 2. Artistic Style Reference")
        style_choice = st.selectbox(
            "Select Preset Art Style",
            list(PRESET_STYLES.keys()),
            format_func=lambda k: PRESET_STYLES[k]["name"],
        )

        preset_file = PRESET_STYLES_DIR / PRESET_STYLES[style_choice]["style_file"]
        style_img = Image.open(preset_file) if preset_file.exists() else None
        if style_img:
            st.image(style_img, caption=PRESET_STYLES[style_choice]["name"], use_container_width=True)

        custom_style_upload = st.file_uploader("Or Upload Custom Style", type=["jpg", "jpeg", "png"])
        if custom_style_upload:
            style_img = Image.open(custom_style_upload).convert("RGB")

    with s_col3:
        st.markdown("#### 3. Stylized Output")
        c_weight = st.slider("Content Weight", 0.1, 5.0, 1.0)
        s_weight = st.slider("Style Weight", 1000.0, 500000.0, 100000.0)

        transfer_btn = st.button("🎨 Synthesize Style Transfer", type="primary", use_container_width=True)

        if transfer_btn and content_img and style_img:
            with st.spinner("Extracting VGG19 feature correlations & Gram matrices..."):
                st_pipe = get_style_transfer_pipeline()
                res = st_pipe.transfer(
                    content_image=content_img,
                    style_image=style_img,
                    content_name="content_source.png",
                    style_name=PRESET_STYLES[style_choice]["name"],
                    content_weight=c_weight,
                    style_weight=s_weight,
                )

            if res and res.get("success"):
                st.session_state.total_creations += 1
                stylized = res["image"]
                meta = res["metadata"]

                st.image(stylized, caption=f"Stylized Output | {meta['inference_time_ms']}ms", use_container_width=True)

                buf = io.BytesIO()
                stylized.save(buf, format="PNG")
                st.download_button(
                    label="📥 Download Stylized Artwork",
                    data=buf.getvalue(),
                    file_name=meta["filename"],
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            recent_st = sorted(STYLE_TRANSFER_DIR.glob("*.png"), key=os.path.getmtime, reverse=True)
            if recent_st:
                st.image(Image.open(recent_st[0]), caption=f"Latest Stylized: {recent_st[0].name}", use_container_width=True)
            else:
                st.info("Upload content and style images then click Synthesize.")


# ==================== TAB 3: VARIATIONS ====================
with tab_var:
    st.markdown("### 🔀 4-Way Image Variations Pipeline")
    v_in_col, v_out_col = st.columns([1, 2], gap="large")

    with v_in_col:
        var_upload = st.file_uploader("Upload Base Image for Variations", type=["jpg", "jpeg", "png"], key="var_uploader")
        base_img = None
        if var_upload:
            base_img = Image.open(var_upload).convert("RGB")
            st.image(base_img, caption="Input Base Image", use_container_width=True)
        else:
            st.caption("Upload an image above to generate 4 variations.")

        strength = st.slider("Variation Perturbation Strength", 0.1, 1.0, 0.5, 0.05)
        var_btn = st.button("⚡ Generate 4 Variations", type="primary", use_container_width=True, disabled=base_img is None)

    with v_out_col:
        st.markdown("#### Generated 4 Variations")
        if var_btn and base_img:
            with st.spinner("Synthesizing 4 artistic variations..."):
                var_pipe = get_image_variations_pipeline()
                res = var_pipe.generate_variations(base_img, filename="input_base.png", strength=strength)

            if res and res.get("success"):
                st.session_state.total_creations += 4
                var_list = res["metadata"]["variations"]
                images = res["images"]

                grid_c1, grid_c2 = st.columns(2)
                for i, (v_meta, v_img) in enumerate(zip(var_list, images)):
                    col = grid_c1 if i % 2 == 0 else grid_c2
                    with col:
                        st.image(v_img, caption=f"#{v_meta['variation_index']}: {v_meta['label']}", use_container_width=True)
        else:
            recent_vars = sorted(VARIATIONS_DIR.glob("*.png"), key=os.path.getmtime, reverse=True)
            if len(recent_vars) >= 4:
                grid_c1, grid_c2 = st.columns(2)
                for i in range(4):
                    col = grid_c1 if i % 2 == 0 else grid_c2
                    with col:
                        st.image(Image.open(recent_vars[i]), caption=f"Variation #{i+1}: {recent_vars[i].name}", use_container_width=True)
            else:
                st.info("Upload an image and click Generate to produce 4 variations.")


# ==================== TAB 4: GALLERY ====================
with tab_gallery:
    st.markdown("### 🖼️ Curated Image Gallery")

    filter_choice = st.radio(
        "Filter by Pipeline",
        ["All Categories", "Text-to-Image", "Style Transfer", "Variations"],
        horizontal=True,
    )

    gallery_files = []
    if filter_choice in ["All Categories", "Text-to-Image"]:
        gallery_files.extend([(f, "Text-to-Image") for f in TEXT_TO_IMAGE_DIR.glob("*.png")])
    if filter_choice in ["All Categories", "Style Transfer"]:
        gallery_files.extend([(f, "Style Transfer") for f in STYLE_TRANSFER_DIR.glob("*.png")])
    if filter_choice in ["All Categories", "Variations"]:
        gallery_files.extend([(f, "Variations") for f in VARIATIONS_DIR.glob("*.png")])

    gallery_files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)

    if gallery_files:
        cols_g = st.columns(3)
        for idx, (img_path, category) in enumerate(gallery_files[:12]):
            col = cols_g[idx % 3]
            with col:
                st.image(Image.open(img_path), caption=f"[{category}] {img_path.name}", use_container_width=True)
    else:
        st.info("No generated images found in gallery.")


# ==================== TAB 5: DOCS ====================
with tab_docs:
    st.markdown("### 🖥️ Hardware Diagnostics & Execution Policies")

    hw = get_hardware_manager().get_hardware_status()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Compute Device", hw["device"].upper())
    with c2:
        st.metric("CUDA Available", "Yes" if hw["cuda_available"] else "No")
    with c3:
        st.metric("GPU Memory", f"{hw['gpu_memory_total_gb']} GB")
    with c4:
        st.metric("Execution Mode", hw["recommended_mode"])

    st.markdown("---")
    st.markdown("#### 🔌 REST API Endpoints")
    st.markdown(
        """
        - **`POST /api/generate/text-to-image`** — Prompt to 512x512 image synthesis
        - **`POST /api/generate/style-transfer`** — Content + Style image fusion via VGG19
        - **`POST /api/generate/variations`** — 1 image into 4 distinct variations
        - **`GET /api/gallery`** — Gallery index with metadata
        - **`GET /api/hardware`** — Diagnostics and active device routing
        - **`GET /docs`** — Interactive OpenAPI Swagger UI
        """
    )
