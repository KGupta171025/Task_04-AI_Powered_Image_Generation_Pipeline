"""
Unit and Integration Tests for AI Image Generation Pipeline.
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.app import app
from core.hardware import get_hardware_manager
from core.image_variations import get_image_variations_pipeline
from core.style_transfer import get_style_transfer_pipeline
from core.text_to_image import get_text_to_image_pipeline

client = TestClient(app)


def make_dummy_image(width: int = 128, height: int = 128, color: str = "blue") -> bytes:
    """Helper to create dummy in-memory image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestHardware:
    def test_hardware_status_keys(self):
        hw = get_hardware_manager().get_hardware_status()
        assert "device" in hw
        assert "cuda_available" in hw
        assert "recommended_mode" in hw
        assert "cpu_cores" in hw
        assert hw["cpu_cores"] > 0

    def test_get_device(self):
        dev = get_hardware_manager().get_device()
        assert dev.type in ["cuda", "cpu"]


class TestTextToImage:
    def test_t2i_generates_512x512(self):
        pipe = get_text_to_image_pipeline()
        res = pipe.generate(prompt="A mystical crystal dragon over mountain peaks", seed=42)
        assert res["success"] is True
        img = res["image"]
        assert img.size == (512, 512)
        assert res["metadata"]["seed"] == 42
        assert "inference_time_ms" in res["metadata"]

    def test_t2i_negative_prompt_and_metadata(self):
        pipe = get_text_to_image_pipeline()
        res = pipe.generate(prompt="Cyberpunk street in neon rain", negative_prompt="blurry", seed=99)
        assert res["metadata"]["negative_prompt"] == "blurry"
        assert res["metadata"]["width"] == 512
        assert res["metadata"]["height"] == 512


class TestStyleTransfer:
    def test_style_transfer_produces_image(self):
        pipe = get_style_transfer_pipeline()
        c_img = Image.new("RGB", (256, 256), color="red")
        s_img = Image.new("RGB", (256, 256), color="yellow")
        res = pipe.transfer(c_img, s_img, content_name="test_c.png", style_name="test_s.png", fast_mode=True)
        assert res["success"] is True
        assert res["image"].size == (512, 512)
        assert "inference_time_ms" in res["metadata"]


class TestImageVariations:
    def test_variations_generates_4_images(self):
        pipe = get_image_variations_pipeline()
        base_img = Image.new("RGB", (256, 256), color="green")
        res = pipe.generate_variations(base_img, filename="base.png", strength=0.5, num_variations=4)
        assert res["success"] is True
        assert len(res["images"]) == 4
        assert len(res["metadata"]["variations"]) == 4
        for img in res["images"]:
            assert img.size == (512, 512)


class TestAPIEndpoints:
    def test_root_endpoint(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "endpoints" in data

    def test_health_endpoint(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_hardware_endpoint(self):
        resp = client.get("/api/hardware")
        assert resp.status_code == 200
        assert "device" in resp.json()

    def test_preset_styles_endpoint(self):
        resp = client.get("/api/preset-styles")
        assert resp.status_code == 200
        assert "preset_styles" in resp.json()
        assert "starry_night" in resp.json()["preset_styles"]

    def test_text_to_image_endpoint(self):
        payload = {"prompt": "Sunset over futuristic metropolis", "seed": 123}
        resp = client.post("/api/generate/text-to-image", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "image_url" in data

    def test_style_transfer_endpoint(self):
        content_bytes = make_dummy_image(128, 128, "blue")
        files = {"content_file": ("c.png", content_bytes, "image/png")}
        data = {"preset_style": "starry_night"}
        resp = client.post("/api/generate/style-transfer", files=files, data=data)
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_variations_endpoint(self):
        img_bytes = make_dummy_image(128, 128, "purple")
        files = {"image_file": ("base.png", img_bytes, "image/png")}
        data = {"strength": "0.5", "num_variations": "4"}
        resp = client.post("/api/generate/variations", files=files, data=data)
        assert resp.status_code == 200
        assert len(resp.json()["variations"]) == 4

    def test_gallery_endpoint(self):
        resp = client.get("/api/gallery")
        assert resp.status_code == 200
        assert resp.json()["total_images"] >= 1
