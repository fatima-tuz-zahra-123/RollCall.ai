# Jetson Setup (SG6)

**Board:** Jetson **Orin** Nano 8GB ·  **Camera:** the laptop webcam, sent to the Jetson over USB-C



## Files in this folder

| File | Runs on | What it does |
|---|---|---|
| `flash/flash_jetson.sh` (+ `flash/README.md`) | Ubuntu PC | Installs the OS on the Jetson's SSD (one time) |
| `setup_jetson.sh` | Jetson | Installs CUDA, Python, OpenCV, YOLOv12 (one time) |
| `requirements-jetson.txt` | Jetson | Python packages for the Jetson |
| `requirements-dev.txt` | Laptops | Python packages for everyone's laptop |
| `camera/send_camera.py` | Laptop | Sends webcam frames to the Jetson |
| `camera/receive_camera.py` | Jetson | Receives the frames |

---

## Pre-Install, check: 

- [ ] Confirm it's an **Orin** Nano 8GB.
- [ ] Find an **SSD**: an M.2 NVMe SSD for the slot under the board (best), or a USB SSD / USB stick.
- [ ] Find a PC with **Ubuntu 22.04** installed for real (not a VM, not WSL).
- [ ] Find a USB-C **data** cable and a jumper wire.
- [ ] Everyone installs Python 3.10 and `pip install -r embedded_sg6/requirements-dev.txt` on their laptop.
- [ ] Test the camera scripts.

---

## Step 1: Install the OS (one time, from the Ubuntu PC)

Follow [`flash/README.md`](flash/README.md). In short: `./flash_jetson.sh`, put the board in recovery mode when asked, and wait about 30 minutes.

## Step 2: Log in from your laptop

1. Power on the Jetson and connect USB-C to your laptop. Wait 1 minute.
2. In PowerShell / Terminal:
   ```bash
   ssh <username>@192.168.55.1
   ```
3. Give the Jetson internet. Plug in an Ethernet cable, or connect to Wi-Fi:
   ```bash
   sudo nmcli device wifi connect "<WiFi name>" password "<password>"
   ```

## Step 3: Install the software (one time, on the Jetson)

```bash
git clone https://github.com/fatima-tuz-zahra-123/RollCall.ai.git
cd RollCall.ai
bash embedded_sg6/setup_jetson.sh
sudo reboot
```
It takes 20 to 40 minutes and ends with a checklist. Everything should say `[OK]`, including **PyTorch + GPU**.

## Step 4: Laptop webcam → Jetson

On the **Jetson**:
```bash
python embedded_sg6/camera/receive_camera.py
```
On the **laptop**:
```bash
pip install opencv-python
python embedded_sg6/camera/send_camera.py
```
The Jetson prints the frames per second it receives.

- **Repeatable tests:** record a clip once (`receive_camera.py --save test.mp4`), then replay it with `send_camera.py --video test.mp4`.
- **Test without a Jetson:** run both scripts on your laptop and add `--host 127.0.0.1` to the sender.

## Step 5: Every time you measure speed

```bash
sudo jetson_clocks     # lock maximum speed
jtop                   # watch CPU / GPU / memory
```
Always write down: power mode, model, input size, FP32/FP16.

---

## Rules

- Python **3.10** everywhere.
- Never `pip install torch` on the Jetson. `setup_jetson.sh` installs the special GPU version.
- No passwords, model weights or face photos on GitHub.
- New Python package? Add it to **both** requirements files in a PR and tell SG6.
- Problems go in a GitHub issue with the label **Integration**.
