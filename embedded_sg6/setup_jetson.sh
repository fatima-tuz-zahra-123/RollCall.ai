#!/usr/bin/env bash
# =============================================================================
# setup_jetson.sh: one-time software setup ON THE JETSON, after flashing.
#
# Installs: CUDA/TensorRT (JetPack), Python tools, OpenCV, jtop, 8 GB swap,
#           a Python env at ~/venvs/rollcall with Ultralytics (YOLOv12) and
#           the GPU builds of PyTorch / torchvision / ONNX Runtime for Jetson.
#
# Needs internet on the Jetson (Ethernet cable or Wi-Fi), not just USB-C.
#
# Usage (on the Jetson, from the repo folder):
#   bash embedded_sg6/setup_jetson.sh
# Takes 20-40 minutes. Safe to run again if it stops halfway.
# =============================================================================
set -euo pipefail

VENV="${HOME}/venvs/rollcall"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQS="${HERE}/requirements-jetson.txt"

# GPU builds for JetPack 6 (Python 3.10). Source: Ultralytics NVIDIA Jetson guide.
WHEELS="https://github.com/ultralytics/assets/releases/download/v0.0.0"
TORCH_WHL="${WHEELS}/torch-2.10.0-cp310-cp310-linux_aarch64.whl"
TORCHVISION_WHL="${WHEELS}/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl"
ORT_GPU_WHL="${WHEELS}/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl"
CUDSS_DEB_URL="https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb"

say()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33mWARNING: $*\033[0m"; }
die()  { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

# ---- 0. checks ---------------------------------------------------------------
say "1/7 Checking the board"
[[ -f /etc/nv_tegra_release ]] || die "This is not a Jetson. Run this script ON the Jetson."
head -1 /etc/nv_tegra_release
grep -q "R36" /etc/nv_tegra_release || die "Expected Jetson Linux R36 (JetPack 6). Re-flash with flash/flash_jetson.sh."
[[ -f "$REQS" ]] || die "Missing $REQS. Run this from the cloned repo."
ping -c1 -W3 pypi.org >/dev/null 2>&1 || die "No internet. Connect Ethernet or Wi-Fi first (see SETUP.md)."

# ---- 1. system packages ------------------------------------------------------
say "2/7 Installing system packages (CUDA, TensorRT, tools)"
sudo apt-get update
sudo apt-get install -y nvidia-jetpack python3-pip python3-venv python3-dev \
  git v4l-utils libopenblas-dev wget

# CUDA on PATH for every login
if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
  {
    echo 'export PATH=/usr/local/cuda/bin:/usr/src/tensorrt/bin:$PATH'
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}'
  } >> ~/.bashrc
fi
export PATH=/usr/local/cuda/bin:/usr/src/tensorrt/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# ---- 2. jtop -----------------------------------------------------------------
say "3/7 Installing jtop (shows CPU/GPU/RAM/power: run 'jtop')"
sudo pip3 install -U jetson-stats

# ---- 3. swap -----------------------------------------------------------------
say "4/7 Adding 8 GB swap (the 8 GB RAM is shared by CPU and GPU)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q "^/swapfile" /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
else
  echo "swap already on"
fi

# ---- 4. max performance mode -------------------------------------------------
say "5/7 Setting maximum power mode"
MAXN_ID=$(grep -oP 'POWER_MODEL ID=\K[0-9]+(?= NAME=MAXN_SUPER)' /etc/nvpmodel.conf 2>/dev/null || true)
[[ -n "$MAXN_ID" ]] || MAXN_ID=$(grep -oP 'POWER_MODEL ID=\K[0-9]+(?= NAME=MAXN)' /etc/nvpmodel.conf | head -1 || true)
if [[ -n "$MAXN_ID" ]]; then
  echo "y" | sudo nvpmodel -m "$MAXN_ID" || warn "nvpmodel may need a reboot to apply."
fi
sudo nvpmodel -q || true

# ---- 5. Python environment ---------------------------------------------------
say "6/7 Creating Python env at ${VENV} and installing packages"
# --system-site-packages lets the env see JetPack's TensorRT
[[ -d "$VENV" ]] || python3 -m venv "$VENV" --system-site-packages
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U pip wheel

# Install the Jetson GPU builds FIRST, so pip does not download the normal
# PyTorch (which cannot use the Jetson GPU) when installing ultralytics.
if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.10') else 1)" 2>/dev/null; then
  pip uninstall -y torch torchvision onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  pip install "$TORCH_WHL" "$TORCHVISION_WHL" "$ORT_GPU_WHL"
fi
pip install -r "$REQS"
python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.10') else 1)" \
  || die "pip replaced the Jetson PyTorch build. Check requirements-jetson.txt for a package that pins torch."

# cuDSS: a library the Jetson PyTorch build needs
if ! dpkg -s cudss >/dev/null 2>&1; then
  TMP_DEB="/tmp/$(basename "$CUDSS_DEB_URL")"
  wget -q --show-progress -O "$TMP_DEB" "$CUDSS_DEB_URL"
  sudo dpkg -i "$TMP_DEB"
  sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get install -y cudss
fi

# Auto-activate the env on login
grep -q "venvs/rollcall/bin/activate" ~/.bashrc || echo "source ${VENV}/bin/activate" >> ~/.bashrc

# ---- 6. verify ---------------------------------------------------------------
say "7/7 Checking everything works"
python - <<'EOF'
import sys
ok = True
def check(name, fn):
    global ok
    try:
        print(f"  [OK]   {name}: {fn()}")
    except Exception as e:
        ok = False
        print(f"  [FAIL] {name}: {e}")

check("Python", lambda: sys.version.split()[0])
check("OpenCV", lambda: __import__("cv2").__version__)
check("NumPy", lambda: __import__("numpy").__version__)
check("TensorRT", lambda: __import__("tensorrt").__version__)
check("ONNX Runtime providers", lambda: __import__("onnxruntime").get_available_providers())
def torch_gpu():
    import torch
    assert torch.cuda.is_available(), f"torch {torch.__version__} cannot see the GPU"
    return f"{torch.__version__}, GPU = {torch.cuda.get_device_name(0)}"
check("PyTorch + GPU", torch_gpu)
check("torchvision", lambda: __import__("torchvision").__version__)
check("Ultralytics", lambda: __import__("ultralytics").__version__)
print("\nALL GOOD" if ok else "\nSOME CHECKS FAILED: copy this output into a GitHub issue (label: Integration)")
EOF

cat <<'EOF'

Setup finished. Next:
  - Reboot once:              sudo reboot
  - Before every speed test:  sudo jetson_clocks
  - Watch the board:          jtop
  - Laptop camera stream:     see SETUP.md, step 4
EOF
