# Flashing the Jetson (one-time)

`flash_jetson.sh` installs the operating system on the Jetson Orin Nano 8GB.
We need this because the board has **no microSD card** and nothing installed.
The OS goes onto an **SSD inside the Jetson** instead.

You only do this **once**. After that, go back to [`../SETUP.md`](../SETUP.md).

## What you need

| Item | Notes |
|---|---|
| Jetson Orin Nano 8GB Developer Kit + its power adapter | USB-C does **not** power the board |
| **An SSD for the Jetson** | Best: an **M.2 NVMe SSD** (2280 size, 128 GB or more) in the slot **under** the board. If there is none: a **USB SSD or USB stick** (64 GB or more) works too, but is slower. |
| A PC with **Ubuntu 22.04** installed for real | Not a virtual machine, not WSL. Intel/AMD, 30 GB free space, internet |
| A USB-C **data** cable | Some cheap cables only charge. Those won't work. |
| A jumper cap or small wire | To put the board in recovery mode |

**Check the SSD first:** turn the Jetson upside down and look at the M.2 slot. If there's an SSD screwed in, you're fine. If it's empty, ask the lab for an NVMe SSD, or use a USB drive with the `--usb` option.

## Steps

1. Copy this `flash` folder to the Ubuntu PC.
2. Run:
   ```bash
   chmod +x flash_jetson.sh
   ./flash_jetson.sh            # SSD inside the Jetson (NVMe)
   ./flash_jetson.sh --usb      # or: USB SSD / USB stick plugged into the Jetson
   ```
3. The script asks for a **username and password** for the Jetson. Choose them and share them privately with the team. **Never put them on GitHub.**
4. The script downloads and prepares everything (about 30 min the first time).
5. When it says **"put the Jetson in RECOVERY MODE"**:
   - unplug the Jetson's power
   - put the jumper on the **FC REC** and **GND** pins (button header under the module, pins 9 and 10)
   - connect the Jetson's USB-C port to the PC
   - plug the power back in
6. The script finds the board on its own. Type `YES` to start flashing. This takes 20 to 40 minutes. **Don't unplug anything.**
7. When it says DONE: unplug power, **remove the jumper**, and plug power back in.
8. Test the login: `ssh <username>@192.168.55.1`

## Troubleshooting

| Problem | Fix |
|---|---|
| "Jetson not found" | Check the jumper is on the right pins. Try another USB-C cable (a data cable) and a USB port directly on the PC, not a hub. Then power-cycle the Jetson. |
| Flash stops with a timeout / USB error | Re-run the script (it skips the download). Try a different USB port or cable. Close any file-manager popups. |
| Board doesn't start after flashing | The recovery jumper is still on. Remove it and power-cycle. |
| `ssh` doesn't connect | Wait 2 minutes after power-on. Make sure the USB-C cable is still connected. |
| Want to start over | Just run the script again. It erases and reinstalls. |

The full flash log is saved as `~/jetson_flash/flash_<date>.log`. Attach it to a GitHub issue if something fails.

**Version:** JetPack 6.2.1 (Jetson Linux 36.4.4, Ubuntu 22.04, CUDA 12.6, Python 3.10).
