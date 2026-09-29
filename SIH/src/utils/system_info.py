import os
import sys
import platform
import shutil
import subprocess
from pathlib import Path
import psutil

def check_command_available(cmd_name):
    path = shutil.which(cmd_name)
    if path:
        return True, path
    return False, None

def check_colmap_installation():
    path = shutil.which("colmap") or shutil.which("COLMAP.bat")
    if not path:
        candidates = [
            Path("C:/COLMAP/COLMAP.bat"),
            Path("C:/COLMAP/bin/colmap.exe"),
            Path("C:/Program Files/COLMAP/COLMAP.bat"),
            Path.home() / "colmap_env" / "bin" / "colmap",
        ]
        for c in candidates:
            if c.exists():
                path = str(c)
                break
                
    if not path:
        return {"available": False, "path": None, "version": None, "cuda_support": False}

    out = ""
    version = None
    try:
        res = subprocess.run([path, "-h"], capture_output=True, text=True, timeout=15)
        out = (res.stdout or "") + "\n" + (res.stderr or "")
    except Exception:
        try:
            res = subprocess.run([path, "help"], capture_output=True, text=True, timeout=15)
            out = (res.stdout or "") + "\n" + (res.stderr or "")
        except Exception:
            out = ""

    out_lower = out.lower()
    if "without cuda" in out_lower or "without gpu support" in out_lower:
        cuda_support = False
    else:
        cuda_support = "cuda" in out_lower

    for line in out.splitlines():
        line_s = line.strip()
        if "COLMAP" in line_s and ("version" in line_s.lower() or "commit" in line_s.lower() or "structure" in line_s.lower() or "with cuda" in line_s.lower()):
            version = line_s
            break

    return {
        "available": True,
        "path": path,
        "version": version or "COLMAP (CUDA Enabled)" if cuda_support else (version or "COLMAP Detected"),
        "cuda_support": cuda_support
    }

def get_nvidia_smi_info():
    available, path = check_command_available("nvidia-smi")
    if not available:
        return {"available": False, "gpu_name": None, "vram_mb": None, "driver_version": None, "cuda_version": None}
    
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        output = res.stdout.strip()
        if output:
            parts = [p.strip() for p in output.split(",")]
            name = parts[0] if len(parts) > 0 else "Unknown"
            vram = parts[1] if len(parts) > 1 else "Unknown"
            driver = parts[2] if len(parts) > 2 else "Unknown"
            return {
                "available": True,
                "gpu_name": name,
                "vram_mb": vram,
                "driver_version": driver,
            }
    except Exception:
        pass
    return {"available": True, "gpu_name": "NVIDIA GPU Detected", "vram_mb": None, "driver_version": None}

def get_pytorch_gpu_info():
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_avail else 0
        device_name = torch.cuda.get_device_name(0) if cuda_avail and device_count > 0 else None
        return {
            "torch_installed": True,
            "torch_version": torch.__version__,
            "cuda_available": cuda_avail,
            "device_count": device_count,
            "device_name": device_name,
        }
    except ImportError:
        return {
            "torch_installed": False,
            "torch_version": None,
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
        }

def get_system_info():
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.getcwd())
    
    ffmpeg_avail, ffmpeg_path = check_command_available("ffmpeg")
    colmap_info = check_colmap_installation()
    smi_info = get_nvidia_smi_info()
    torch_info = get_pytorch_gpu_info()
    
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_available_gb": round(mem.available / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "pytorch": torch_info,
        "nvidia_smi": smi_info,
        "colmap": colmap_info,
        "ffmpeg": {"available": ffmpeg_avail, "path": ffmpeg_path},
    }

def print_system_report(info=None):
    if info is None:
        info = get_system_info()
        
    lines = [
        "=" * 60,
        "       SYSTEM & ENVIRONMENT DIAGNOSTIC REPORT",
        "=" * 60,
        f"OS:                 {info['os']}",
        f"Python Executable:  {info['python_executable']}",
        f"Python Version:     {info['python_version']}",
        f"CPU Cores:          {info['cpu_count_physical']} Physical / {info['cpu_count_logical']} Logical",
        f"System RAM:         {info['ram_total_gb']} GB total ({info['ram_available_gb']} GB available)",
        "-" * 60,
        f"PyTorch Installed:  {info['pytorch']['torch_installed']} (Version: {info['pytorch']['torch_version']})",
        f"PyTorch CUDA:       {info['pytorch']['cuda_available']}",
        f"CUDA Device Name:   {info['pytorch']['device_name']}",
        f"CUDA Device Count:  {info['pytorch']['device_count']}",
        f"NVIDIA GPU (smi):   {info['nvidia_smi']['available']}",
        f"GPU Model:          {info['nvidia_smi']['gpu_name']}",
        f"GPU VRAM:           {info['nvidia_smi']['vram_mb']} MB",
        f"Driver Version:     {info['nvidia_smi']['driver_version']}",
        "-" * 60,
        f"COLMAP Available:   {info['colmap']['available']} ({info['colmap']['path']})",
        f"COLMAP Version:     {info['colmap']['version']}",
        f"COLMAP CUDA:        {info['colmap']['cuda_support']}",
        f"FFmpeg Available:   {info['ffmpeg']['available']} ({info['ffmpeg']['path']})",
        "=" * 60,
    ]
    return "\n".join(lines)
