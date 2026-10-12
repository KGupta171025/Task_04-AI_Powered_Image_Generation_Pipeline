"""
Hardware Inspection & Resource Routing Engine.
Detects GPU/CUDA capabilities, memory limits, and routes image pipelines between
full GPU Stable Diffusion and optimized CPU / Hugging Face Inference API fallbacks.
"""

import logging
import os
import platform
import psutil
import torch

logger = logging.getLogger(__name__)


class HardwareManager:
    """Manages compute device detection, memory tracking, and routing policies."""

    @staticmethod
    def get_hardware_status() -> dict:
        """Collects detailed hardware diagnostics and active execution tier."""
        cuda_available = torch.cuda.is_available()
        device_name = "cpu"
        gpu_name = "None"
        gpu_memory_total_gb = 0.0
        gpu_memory_allocated_gb = 0.0

        if cuda_available:
            device_name = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_total_gb = round(total_bytes / (1024**3), 2)
            alloc_bytes = torch.cuda.memory_allocated(0)
            gpu_memory_allocated_gb = round(alloc_bytes / (1024**3), 2)

        # System RAM
        mem = psutil.virtual_memory()
        system_ram_total_gb = round(mem.total / (1024**3), 2)
        system_ram_available_gb = round(mem.available / (1024**3), 2)

        # Recommended pipeline mode
        if cuda_available and gpu_memory_total_gb >= 6.0:
            recommended_mode = "GPU_STABLE_DIFFUSION_FP16"
            pipeline_strategy = "Full Stable Diffusion with CUDA acceleration and FP16 precision."
        elif cuda_available:
            recommended_mode = "GPU_STABLE_DIFFUSION_LOW_VRAM"
            pipeline_strategy = "Sequential CPU offload with attention slicing enabled."
        else:
            has_hf_token = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"))
            if has_hf_token:
                recommended_mode = "HUGGINGFACE_INFERENCE_API"
                pipeline_strategy = "Cloud GPU inference via Hugging Face Inference API."
            else:
                recommended_mode = "CPU_OPTIMIZED_FALLBACK"
                pipeline_strategy = "Lightweight procedural neural synthesis & latent perturbation (CPU safe)."

        return {
            "device": device_name,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "gpu_memory_total_gb": gpu_memory_total_gb,
            "gpu_memory_allocated_gb": gpu_memory_allocated_gb,
            "cpu_cores": psutil.cpu_count(logical=True),
            "system_ram_total_gb": system_ram_total_gb,
            "system_ram_available_gb": system_ram_available_gb,
            "os": f"{platform.system()} {platform.release()}",
            "recommended_mode": recommended_mode,
            "pipeline_strategy": pipeline_strategy,
        }

    @staticmethod
    def get_device() -> torch.device:
        """Returns the appropriate PyTorch device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


_HARDWARE_INSTANCE = HardwareManager()


def get_hardware_manager() -> HardwareManager:
    """Returns the HardwareManager singleton."""
    return _HARDWARE_INSTANCE
