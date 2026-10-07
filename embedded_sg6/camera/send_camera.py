"""send_camera.py: runs on the LAPTOP.

Grabs frames from the laptop webcam (or a video file) and sends them to the
Jetson over the USB-C cable (the Jetson is at 192.168.55.1).

    pip install opencv-python
    python send_camera.py                               # webcam -> Jetson
    python send_camera.py --video classroom_test.mp4    # video file -> Jetson (repeatable tests)
    python send_camera.py --host 127.0.0.1              # test on your own laptop, no Jetson needed

Start receive_camera.py on the Jetson first. Stop with Ctrl+C.

Message format (one per frame): 4-byte length (uint32) + 8-byte timestamp
(float64, seconds since the stream started) + JPEG bytes. Big-endian.
"""
import argparse
import platform
import socket
import struct
import time

import cv2

HEADER = struct.Struct(">Id")   # jpeg length, timestamp


def open_source(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
    elif platform.system() == "Windows":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)   # opens faster on Windows
    else:
        cap = cv2.VideoCapture(args.camera)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {'video ' + args.video if args.video else f'camera {args.camera}'}")
    return cap


def connect(host, port):
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(None)
            print(f"Connected to {host}:{port}")
            return sock
        except OSError as e:
            print(f"Waiting for receiver at {host}:{port} ({e}); is receive_camera.py running?")
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description="Send laptop webcam / video frames to the Jetson.")
    ap.add_argument("--host", default="192.168.55.1", help="Jetson address (USB-C link default)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (0 = built-in)")
    ap.add_argument("--video", help="send a video file instead of the webcam")
    ap.add_argument("--loop", action="store_true", help="repeat the video file forever")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=15, help="max frames per second to send")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    ap.add_argument("--preview", action="store_true", help="show the frames on the laptop too")
    args = ap.parse_args()

    cap = open_source(args)
    sock = connect(args.host, args.port)
    start = time.time()
    sent, last_report, kbytes = 0, start, 0.0
    period = 1.0 / args.fps if args.fps > 0 else 0

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                if args.video and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("No more frames.")
                break

            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                continue
            data = jpg.tobytes()
            try:
                sock.sendall(HEADER.pack(len(data), t0 - start) + data)
            except OSError:
                print("Connection lost, reconnecting...")
                sock.close()
                sock = connect(args.host, args.port)
                continue

            sent += 1
            kbytes += len(data) / 1024
            if args.preview:
                cv2.imshow("sending (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if t0 - last_report >= 2:
                el = t0 - last_report
                print(f"sent {sent} frames | {frame.shape[1]}x{frame.shape[0]} | "
                      f"{sent / (t0 - start):.1f} FPS avg | {kbytes / el:.0f} KB/s")
                last_report, kbytes = t0, 0.0

            wait = period - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        sock.close()
        cv2.destroyAllWindows()
        print(f"Stopped after {sent} frames.")


if __name__ == "__main__":
    main()
