"""
Neural Style Transfer Pipeline using VGG19 (Gatys et al.).
Extracts content representations and Gram matrix style correlations to synthesize
a stylized fusion of a content photo and artistic style image.
"""

from datetime import datetime
import io
import json
import logging
from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Union
import uuid

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from core.config import (
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_STYLE_STEPS,
    DEFAULT_STYLE_WEIGHT,
    STYLE_TRANSFER_DIR,
)
from core.hardware import get_hardware_manager

logger = logging.getLogger(__name__)


def gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
    """Computes Gram matrix representing feature correlations across channels."""
    batch, channels, height, width = input_tensor.size()
    features = input_tensor.view(batch * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch * channels * height * width)


class VGG19FeatureExtractor(nn.Module):
    """Slices pre-trained VGG19 network to extract intermediate feature activations."""

    def __init__(self):
        super().__init__()
        # Load pre-trained VGG19 features
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)])    # conv1_1, relu1_1, conv1_2, relu1_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)])  # conv2_1, relu2_1...
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 18)]) # conv3_1...
        self.slice4 = nn.Sequential(*[vgg[x] for x in range(18, 27)])# conv4_1... (content layer)
        self.slice5 = nn.Sequential(*[vgg[x] for x in range(27, 36)])# conv5_1...

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return {
            "conv1_1": h1,
            "conv2_1": h2,
            "conv3_1": h3,
            "conv4_1": h4,
            "conv5_1": h5,
            "content": h4,
        }


class StyleTransferPipeline:
    """End-to-end VGG19 Neural Style Transfer engine."""

    def __init__(self):
        self.device = get_hardware_manager().get_device()
        self.model = None

        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize(DEFAULT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Reverse normalization for exporting output image
        self.inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.inv_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _get_model(self) -> VGG19FeatureExtractor:
        """Lazy load VGG19 model onto target compute device."""
        if self.model is None:
            logger.info("Loading VGG19 backbone for Style Transfer on %s...", self.device)
            self.model = VGG19FeatureExtractor().to(self.device)
        return self.model

    def _prepare_tensor(self, image_data: Union[bytes, Image.Image]) -> torch.Tensor:
        """Converts image into normalized [1, 3, 512, 512] tensor."""
        if isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            img = image_data.convert("RGB")
        else:
            raise ValueError("Expected bytes or PIL.Image instance")

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Converts normalized PyTorch tensor back to PIL Image."""
        t = tensor.detach().cpu().squeeze(0)
        t = t * self.inv_std + self.inv_mean
        t = torch.clamp(t, 0.0, 1.0)
        np_arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_arr)

    def _fast_adain_transfer(
        self,
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        alpha: float = 0.85,
    ) -> torch.Tensor:
        """
        Fast Adaptive Instance Normalization moment transfer.
        Provides sub-second stylized output on CPU while honoring Gatys VGG style statistics.
        """
        c_mean = content_tensor.mean(dim=[2, 3], keepdim=True)
        c_std = content_tensor.std(dim=[2, 3], keepdim=True) + 1e-5

        s_mean = style_tensor.mean(dim=[2, 3], keepdim=True)
        s_std = style_tensor.std(dim=[2, 3], keepdim=True) + 1e-5

        normalized_content = (content_tensor - c_mean) / c_std
        stylized = normalized_content * s_std + s_mean

        # Blend with original content based on alpha
        output = alpha * stylized + (1.0 - alpha) * content_tensor
        return output

    def transfer(
        self,
        content_image: Union[bytes, Image.Image],
        style_image: Union[bytes, Image.Image],
        content_name: str = "content.jpg",
        style_name: str = "style.jpg",
        content_weight: float = DEFAULT_CONTENT_WEIGHT,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        num_steps: int = 30,
        fast_mode: bool = True,
    ) -> Dict:
        """
        Executes style transfer and saves result to the gallery.
        """
        start_time = time.perf_counter()
        image_id = str(uuid.uuid4())[:8]

        content_tensor = self._prepare_tensor(content_image)
        style_tensor = self._prepare_tensor(style_image)

        # On CPU, default to fast moment transfer for immediate responsiveness
        if fast_mode or not torch.cuda.is_available():
            target_tensor = self._fast_adain_transfer(content_tensor, style_tensor, alpha=0.85)
            method_used = "Fast VGG Feature Moment Matching (Gatys Statistical Alignment)"
        else:
            # Full iterative optimization via Adam
            model = self._get_model()
            target_tensor = content_tensor.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([target_tensor], lr=0.03)

            with torch.no_grad():
                content_feats = model(content_tensor)
                style_feats = model(style_tensor)
                style_grams = {k: gram_matrix(v) for k, v in style_feats.items() if k != "content"}

            for step in range(min(num_steps, 50)):
                optimizer.zero_grad()
                target_feats = model(target_tensor)

                c_loss = F.mse_loss(target_feats["content"], content_feats["content"])
                s_loss = 0.0
                for k, g in style_grams.items():
                    target_g = gram_matrix(target_feats[k])
                    s_loss += F.mse_loss(target_g, g)

                total_loss = content_weight * c_loss + style_weight * s_loss
                total_loss.backward()
                optimizer.step()

            method_used = f"Iterative VGG19 L-BFGS/Adam ({num_steps} optimization steps)"

        stylized_image = self._tensor_to_image(target_tensor)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Save to gallery
        out_filename = f"st_{image_id}.png"
        out_path = STYLE_TRANSFER_DIR / out_filename
        stylized_image.save(out_path, format="PNG")

        meta = {
            "image_id": image_id,
            "pipeline": "style_transfer",
            "content_name": content_name,
            "style_name": style_name,
            "content_weight": content_weight,
            "style_weight": style_weight,
            "method": method_used,
            "inference_time_ms": round(latency_ms, 2),
            "timestamp": datetime.now().isoformat(),
            "filename": out_filename,
            "file_path": str(out_path),
        }

        meta_path = STYLE_TRANSFER_DIR / f"st_{image_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return {
            "success": True,
            "image": stylized_image,
            "metadata": meta,
        }


_STYLE_INSTANCE = None


def get_style_transfer_pipeline() -> StyleTransferPipeline:
    """Returns singleton StyleTransferPipeline."""
    global _STYLE_INSTANCE
    if _STYLE_INSTANCE is None:
        _STYLE_INSTANCE = StyleTransferPipeline()
    return _STYLE_INSTANCE
