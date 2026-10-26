"""receive_camera.py: runs on the JETSON.

Receives frames sent by send_camera.py from the laptop over USB-C.

    python receive_camera.py                       # print FPS for each incoming frame
    python receive_camera.py --save test.mp4       # also record what arrives (for repeatable tests)
    python receive_camera.py --show                # show the frames (only if a monitor is attached)

Start this first, then start send_camera.py on the laptop. Stop with Ctrl+C.

Use from the pipeline:
    from receive_camera import receive_frames
    for frame_id, timestamp, frame in receive_frames(port=5000):
        result = detector.detect(frame, frame_id, timestamp)    # SG-1 V1 interface
"""
import argparse
import socket
import struct
import time

import cv2
import numpy as np

HEADER = struct.Struct(">Id")   # jpeg length, timestamp (seconds since stream start)
MAX_JPEG_BYTES = 20 * 1024 * 1024


def _recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def receive_frames(port=5000, host="0.0.0.0"):
    """Yield (frame_id, timestamp, frame) forever. Waits for a new sender if one disconnects.

    frame: numpy uint8 BGR HxWx3. frame_id: 0, 1, 2, ... timestamp: seconds (sender clock).
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    frame_id = 0
    try:
        while True:
            print(f"Waiting for the laptop on port {port}...")
            conn, addr = server.accept()
            print(f"Laptop connected from {addr[0]}")
            with conn:
                while True:
                    header = _recv_exact(conn, HEADER.size)
                    if header is None:
                        print("Laptop disconnected.")
                        break
                    size, timestamp = HEADER.unpack(header)
                    if not 0 < size <= MAX_JPEG_BYTES:
                        print(f"Bad frame size {size}; dropping connection.")
                        break
                    data = _recv_exact(conn, size)
                    if data is None:
                        print("Laptop disconnected.")
                        break
                    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    yield frame_id, timestamp, frame
                    frame_id += 1
    finally:
        server.close()


def main():
    ap = argparse.ArgumentParser(description="Receive laptop camera frames on the Jetson.")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--save", help="record received frames to this .mp4 file")
    ap.add_argument("--save-fps", type=float, default=15)
    ap.add_argument("--show", action="store_true", help="display frames (needs a monitor)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = never)")
    args = ap.parse_args()

    writer = None
    start = last = time.time()
    count = 0
    try:
        for frame_id, ts, frame in receive_frames(args.port):
            if count == 0:
                start = last = time.time()
            # >>> pipeline goes here: e.g. result = detector.detect(frame, frame_id, ts) <<<
            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), args.save_fps, (w, h))
                writer.write(frame)
            if args.show:
                cv2.imshow("from laptop (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            count += 1
            now = time.time()
            if now - last >= 2:
                print(f"frame {frame_id} | {frame.shape[1]}x{frame.shape[0]} | "
                      f"{count / (now - start):.1f} FPS received | stream time {ts:.1f}s")
                last = now
            if args.max_frames and count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved {args.save}")
        cv2.destroyAllWindows()
        print(f"Received {count} frames.")


if __name__ == "__main__":
    main()
