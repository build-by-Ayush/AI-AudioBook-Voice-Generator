'''
install_torch_cuda.py
Purpose:
- Detect NVIDIA GPU/driver availability and current PyTorch build.
- Recommend a compatible CUDA-enabled torch/torchaudio wheel for Windows.
- Optionally install it in the ACTIVE virtual environment with a single confirmation.

Warnings:
- Close Python processes using torch before installation.
- Matches common, well-supported combos (CUDA 12.1 or 11.8); you can override manually.
- Requires internet access to fetch wheels from the official PyTorch CUDA indices.
- This script does not install the NVIDIA driver; install/update the driver separately via NVIDIA.

References:
- Official PyTorch install selectors and CUDA wheel indices provide the correct commands. 
'''

import subprocess
import sys
import shutil

def run(cmd):
    print(f"\n$ {cmd}")
    return subprocess.run(cmd, shell=True, text=True)

def which(cmd):
    return shutil.which(cmd)

def nvidia_info():
    if not which("nvidia-smi"):
        print("nvidia-smi not found. NVIDIA driver not detected in PATH.")
        return None
    out = subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", 
                         shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        print("nvidia-smi query failed.")
        return None
    line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    if not line:
        return None
    try:
        name, driver = [x.strip() for x in line.split(",")]
        return {"name": name, "driver": driver}
    except Exception:
        print("Could not parse nvidia-smi output.")
        return None

def torch_info():
    try:
        import torch
        ver = torch.__version__
        has_cuda = torch.cuda.is_available()
        cuda_ver = getattr(torch.version, "cuda", None)
        device = torch.cuda.get_device_name(0) if has_cuda else "CPU"
        return {"torch": ver, "cuda_available": has_cuda, "cuda_version": cuda_ver, "device": device}
    except Exception as e:
        return {"error": str(e)}

def pick_command(pref="auto"):
    # Default to CUDA 12.1 wheels for recent drivers; fallback to 11.8 if you prefer.
    # You can change these to exact versions if needed.
    cu121 = 'pip install --index-url https://download.pytorch.org/whl/cu121 torch torchaudio'
    cu118 = 'pip install --index-url https://download.pytorch.org/whl/cu118 torch torchaudio'
    cpu   = 'pip install torch torchaudio'
    if pref == "cu121":
        return cu121
    if pref == "cu118":
        return cu118
    # auto: prefer cu121 for recent drivers; otherwise suggest cu118
    info = nvidia_info()
    if info:
        print(f"Detected GPU: {info['name']} | Driver: {info['driver']}")
        # Heuristic: newer Windows drivers (>= 531) generally fine with cu121
        try:
            major = int(info['driver'].split(".")[0])
        except Exception:
            major = 0
        if major >= 531:
            return cu121
        else:
            return cu118
    else:
        # No GPU/driver visible; propose CPU wheels
        return cpu

def main():
    print("=== Torch/CUDA Environment Detector ===")
    nv = nvidia_info()
    if nv:
        print(f"- NVIDIA GPU detected: {nv['name']} (Driver {nv['driver']})")
    else:
        print("- No NVIDIA GPU driver detected via nvidia-smi.")

    ti = torch_info()
    if "error" in ti:
        print(f"- PyTorch not importable: {ti['error']}")
    else:
        print(f"- torch: {ti['torch']}")
        print(f"- torch.cuda.is_available(): {ti['cuda_available']}")
        print(f"- torch.version.cuda: {ti['cuda_version']}")
        print(f"- device: {ti['device']}")

    print("\nSelecting an installation command...")
    cmd = pick_command(pref="auto")
    print(f"Suggested command:\n  {cmd}")

    print("\nNotes:")
    print("- This installs matching torch and torchaudio wheels for the selected CUDA/CPU.")
    print("- Ensure this is run inside the correct virtual environment.")
    print("- If you prefer a different CUDA level, re-run and force pick_command('cu118' or 'cu121').")

    choice = input("\nProceed with installation? [y/N]: ").strip().lower()
    if choice != "y":
        print("Aborted by user.")
        sys.exit(0)

    # Uninstall existing torch/torchaudio to avoid conflicts
    print("\nUninstalling existing torch/torchaudio (if present)...")
    run("pip uninstall -y torch torchaudio")

    print("\nInstalling suggested wheels...")
    res = run(cmd)
    if res.returncode != 0:
        print("\n❌ Installation failed. You may try the alternate CUDA index or check your internet/permissions.")
        print("Alternate commands you can try manually:")
        print("  CUDA 12.1: pip install --index-url https://download.pytorch.org/whl/cu121 torch torchaudio")
        print("  CUDA 11.8: pip install --index-url https://download.pytorch.org/whl/cu118 torch torchaudio")
        print("  CPU only:  pip install torch torchaudio")
        sys.exit(1)

    # Post-check
    print("\nVerifying installation...")
    ti2 = torch_info()
    if "error" in ti2:
        print(f"⚠️ torch import failed after install: {ti2['error']}")
        sys.exit(1)
    print(f"- torch: {ti2['torch']}")
    print(f"- torch.cuda.is_available(): {ti2['cuda_available']}")
    print(f"- torch.version.cuda: {ti2['cuda_version']}")
    print(f"- device: {ti2['device']}")
    if ti2["cuda_available"]:
        print("\n✅ CUDA is enabled. You can now run XTTS on GPU.")
    else:
        print("\nℹ️ CUDA not enabled. You can still run on CPU or try the other CUDA command.")

if __name__ == "__main__":
    main()
