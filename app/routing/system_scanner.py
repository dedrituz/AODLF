"""
Hardware specification scanner for local backend optimization.
Scans CPU, RAM, and GPU/VRAM to advise on model parameter limits and smart routing.
"""

import os
import platform
import psutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SystemHardwareSpecs:
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    cuda_available: bool = False
    recommended_max_param_size: str = "7B"
    summary: str = ""


class SystemScanner:
    """Scans hardware specifications on Windows/Linux to assist in local model tiering."""

    @staticmethod
    def scan() -> SystemHardwareSpecs:
        cpu_model = platform.processor() or platform.machine()
        cpu_physical = psutil.cpu_count(logical=False) or 4
        cpu_logical = psutil.cpu_count(logical=True) or 8

        mem = psutil.virtual_memory()
        ram_total = round(mem.total / (1024 ** 3), 2)
        ram_avail = round(mem.available / (1024 ** 3), 2)

        gpu_name = None
        vram_total = None
        vram_free = None
        cuda_avail = False

        # Try scanning GPU via torch if available
        try:
            import torch
            if torch.cuda.is_available():
                cuda_avail = True
                gpu_name = torch.cuda.get_device_name(0)
                vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
                vram_free = round(torch.cuda.mem_get_info()[0] / (1024 ** 3), 2)
        except Exception:
            pass

        # If torch did not find CUDA, try nvidia-smi command
        if not cuda_avail:
            try:
                import subprocess
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if res.returncode == 0 and res.stdout.strip():
                    parts = [p.strip() for p in res.stdout.strip().split(",")]
                    if len(parts) >= 3:
                        gpu_name = parts[0]
                        vram_total = round(float(parts[1]) / 1024, 2)
                        vram_free = round(float(parts[2]) / 1024, 2)
                        cuda_avail = True
            except Exception:
                pass

        # Recommend parameter limit based on RAM / VRAM
        effective_mem = vram_total if vram_total else ram_avail
        if effective_mem >= 32:
            rec_limit = "70B+"
        elif effective_mem >= 20:
            rec_limit = "27B-35B"
        elif effective_mem >= 12:
            rec_limit = "14B-20B"
        elif effective_mem >= 6:
            rec_limit = "7B-9B"
        else:
            rec_limit = "1B-4B"

        summary = (
            f"CPU: {cpu_physical}C/{cpu_logical}T | "
            f"RAM: {ram_avail:.1f}GB free / {ram_total:.1f}GB total | "
            f"GPU: {gpu_name or 'Integrated/CPU'} "
            f"({f'{vram_total:.1f}GB VRAM' if vram_total else 'No Dedicated VRAM'}) | "
            f"Recommended Max Local Model: {rec_limit}"
        )

        return SystemHardwareSpecs(
            cpu_model=cpu_model,
            cpu_cores_physical=cpu_physical,
            cpu_cores_logical=cpu_logical,
            ram_total_gb=ram_total,
            ram_available_gb=ram_avail,
            gpu_name=gpu_name,
            vram_total_gb=vram_total,
            vram_free_gb=vram_free,
            cuda_available=cuda_avail,
            recommended_max_param_size=rec_limit,
            summary=summary
        )
