#!/usr/bin/env bash
# =============================================================================
# flash_jetson.sh: install the operating system (JetPack 6.2.1 / Jetson Linux
# 36.4.4) onto a Jetson Orin Nano 8GB Developer Kit that has NO microSD card.
#
# Runs on: an Ubuntu 22.04 (or 20.04) x86_64 PC. Not a VM, not WSL.
# Writes to: the SSD inside the Jetson (NVMe, default) or a USB drive (--usb).
# WARNING: everything on that SSD / USB drive is erased.
#
# Usage:
#   chmod +x flash_jetson.sh
#   ./flash_jetson.sh                      # NVMe SSD, asks for username/password
#   ./flash_jetson.sh --usb                # USB SSD / USB stick instead of NVMe
#   ./flash_jetson.sh --user sg6 --hostname rollcall-jetson
#   ./flash_jetson.sh --no-super           # older (non-"Super") power settings
#
# See README.md in this folder for the full walkthrough.
# =============================================================================
set -euo pipefail

# ---- settings (change only if the team agrees on a different release) -------
L4T_VERSION="36.4.4"                     # JetPack 6.2.1
L4T_URL_DIR="https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/release"
BSP_FILE="Jetson_Linux_r${L4T_VERSION}_aarch64.tbz2"
ROOTFS_FILE="Tegra_Linux_Sample-Root-Filesystem_r${L4T_VERSION}_aarch64.tbz2"

WORK_DIR="${HOME}/jetson_flash"
STORAGE="nvme"                           # nvme | usb
BOARD="jetson-orin-nano-devkit-super"
JETSON_USER=""
JETSON_HOSTNAME="rollcall-jetson"

# ---- helpers -----------------------------------------------------------------
say()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33mWARNING: $*\033[0m"; }
die()  { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

usage() { sed -n '2,20p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --usb)        STORAGE="usb"; shift ;;
    --nvme)       STORAGE="nvme"; shift ;;
    --no-super)   BOARD="jetson-orin-nano-devkit"; shift ;;
    --user)       JETSON_USER="$2"; shift 2 ;;
    --hostname)   JETSON_HOSTNAME="$2"; shift 2 ;;
    --workdir)    WORK_DIR="$2"; shift 2 ;;
    -h|--help)    usage ;;
    *)            die "Unknown option: $1 (use --help)" ;;
  esac
done

# ---- 0. check this PC --------------------------------------------------------
say "Step 0/5: checking this PC"
[[ "$(uname -m)" == "x86_64" ]] || die "This PC is $(uname -m). Flashing needs an x86_64 (Intel/AMD) PC."
if grep -qi microsoft /proc/version 2>/dev/null; then
  die "This is WSL. Flashing does not work from WSL. Use a real Ubuntu install (dual boot or lab PC)."
fi
if systemd-detect-virt -q 2>/dev/null; then
  warn "This looks like a virtual machine. Flashing from a VM often fails (USB disconnects). A real Ubuntu PC is strongly recommended."
  read -r -p "Continue anyway? [y/N] " a; [[ "$a" =~ ^[Yy]$ ]] || exit 1
fi
. /etc/os-release
case "${VERSION_ID:-}" in
  22.04|20.04) echo "Ubuntu ${VERSION_ID}: OK" ;;
  *) warn "Ubuntu ${VERSION_ID:-unknown} is not officially supported (use 22.04 or 20.04)." ;;
esac
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
(( FREE_GB >= 30 )) || die "Only ${FREE_GB} GB free in $HOME. Need at least 30 GB."
sudo -v || die "This script needs sudo."

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# ---- 1. download -------------------------------------------------------------
say "Step 1/5: downloading Jetson Linux ${L4T_VERSION} (about 2 GB, resumes if interrupted)"
for f in "$BSP_FILE" "$ROOTFS_FILE"; do
  wget -c --show-progress "${L4T_URL_DIR}/${f}" -O "$f"
done

# ---- 2. unpack + prepare -----------------------------------------------------
L4T_DIR="${WORK_DIR}/Linux_for_Tegra"
if [[ ! -f "${L4T_DIR}/.prepared_${L4T_VERSION}" ]]; then
  say "Step 2/5: unpacking and preparing (10-20 min)"
  sudo rm -rf "$L4T_DIR"
  tar xf "$BSP_FILE"
  sudo tar xpf "$ROOTFS_FILE" -C "${L4T_DIR}/rootfs/"
  cd "$L4T_DIR"
  sudo ./tools/l4t_flash_prerequisites.sh
  sudo ./apply_binaries.sh
  # Pre-create the login user (skips the first-boot setup screen, so no monitor needed).
  if [[ -z "$JETSON_USER" ]]; then
    read -r -p "Username to create on the Jetson [sg6]: " JETSON_USER
    JETSON_USER="${JETSON_USER:-sg6}"
  fi
  while true; do
    read -r -s -p "Password for '${JETSON_USER}' on the Jetson: " JETSON_PASS; echo
    read -r -s -p "Type it again: " P2; echo
    [[ -n "$JETSON_PASS" && "$JETSON_PASS" == "$P2" ]] && break
    echo "Passwords empty or do not match, try again."
  done
  sudo ./tools/l4t_create_default_user.sh -u "$JETSON_USER" -p "$JETSON_PASS" \
    -n "$JETSON_HOSTNAME" -a --accept-license || die "Could not create the default user."
  unset JETSON_PASS P2
  echo "$JETSON_USER" > "${L4T_DIR}/.jetson_user"
  sudo touch "${L4T_DIR}/.prepared_${L4T_VERSION}"
else
  say "Step 2/5: already prepared, skipping (to change the username, delete ${WORK_DIR}/Linux_for_Tegra and re-run)"
  JETSON_USER="$(cat "${L4T_DIR}/.jetson_user" 2>/dev/null || echo '<username>')"
fi
cd "$L4T_DIR"


# ---- 3. wait for the Jetson in recovery mode ---------------------------------
say "Step 3/5: put the Jetson in RECOVERY MODE now"
cat <<'EOF'
  1. Unplug the Jetson's power.
  2. Put a jumper (or a wire) between the FC REC and GND pins on the button
     header underneath the module (pins 9 and 10 on the Orin Nano dev kit).
  3. Connect the USB-C port of the Jetson to this PC (directly, no hub).
  4. Plug the Jetson's power back in. (Screen stays black, that is normal.)
EOF
echo -n "Waiting for the Jetson to appear on USB (0955:7523) "
for _ in $(seq 1 120); do
  if lsusb | grep -q "0955:7523"; then echo " found!"; break; fi
  if lsusb | grep -q "0955:7623"; then
    echo; warn "Found an Orin Nano 4GB (0955:7623), not the 8GB. Continuing."; break
  fi
  echo -n "."; sleep 5
done
lsusb | grep -qE "0955:7(5|6)23" || die "Jetson not found after 10 minutes. See Troubleshooting in README.md."

# Avoid USB power-saving dropping the connection mid-flash.
echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend >/dev/null

# ---- 4. flash ----------------------------------------------------------------
if [[ "$STORAGE" == "nvme" ]]; then DEVICE="nvme0n1p1"; else DEVICE="sda1"; fi
say "Step 4/5: flashing to ${STORAGE^^} (${DEVICE}) with board config ${BOARD} (20-40 min)"
warn "Everything on the Jetson's ${STORAGE^^} drive will be erased. Do NOT unplug anything."
read -r -p "Type YES to start flashing: " a; [[ "$a" == "YES" ]] || die "Cancelled."

LOG="${WORK_DIR}/flash_$(date +%Y%m%d_%H%M%S).log"
if ! sudo ./tools/kernel_flash/l4t_initrd_flash.sh \
  --external-device "$DEVICE" \
  -c tools/kernel_flash/flash_l4t_t234_nvme.xml \
  -p "-c bootloader/generic/cfg/flash_t234_qspi.xml" \
  --showlogs --network usb0 --erase-all \
  "$BOARD" internal 2>&1 | tee "$LOG"; then
  die "Flashing failed. Full log: $LOG"
fi

# ---- 5. done -----------------------------------------------------------------
say "Step 5/5: DONE. Flash log saved to $LOG"
cat <<EOF
  Next:
  1. Unplug the Jetson's power and REMOVE the recovery jumper.
  2. Plug power back in, keep the USB-C cable connected, wait ~1 minute.
  3. From this PC or your laptop:   ssh ${JETSON_USER}@192.168.55.1
  4. Then run setup_jetson.sh on the Jetson (see SETUP.md).
EOF
