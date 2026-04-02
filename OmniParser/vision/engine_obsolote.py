from __future__ import annotations
import argparse
import asyncio
import queue
import sys
import threading
import time
import hashlib
import cv2
import msgpack
import mss
import numpy as np
try:
    from WDS.wehgp import process as wehgp_process
except ImportError:
    from wehgp import process as wehgp_process


try:
    from ws_server import WsServer
    from semantic import SemanticBrain
except ImportError:
    class WsServer: 
        def __init__(self, **kwargs): pass
        async def serve_forever(self): pass
        async def broadcast(self, p): pass
    class SemanticBrain:
        def __init__(self): self.memory = {}
        def process_semantic_snapshot(self, f, b): pass

parser = argparse.ArgumentParser(description="Wavelet Vision Engine")
parser.add_argument("--monitor", type=int, default=1)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=9001)
parser.add_argument("--fps", type=int, default=60)
args = parser.parse_args()

capture_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
broadcast_queue: queue.Queue[bytes] = queue.Queue(maxsize=1)
running = threading.Event()
running.set()

manual_parse_event = threading.Event()

_state_lock = threading.Lock()
_current_band = "HH"
_current_monitor = args.monitor

semantic_queue: queue.Queue[tuple[np.ndarray, list[dict]]] = queue.Queue(maxsize=1)

server = WsServer(host=args.host, port=args.port)
brain = SemanticBrain()


class FPSTracker:
    def __init__(self, alpha: float = 0.15) -> None:
        self._alpha = alpha
        self.fps: float = 0.0
        self._last = time.monotonic()

    def tick(self) -> float:
        now = time.monotonic()
        instant = 1.0 / max(now - self._last, 1e-9)
        self.fps = self._alpha * instant + (1 - self._alpha) * self.fps
        self._last = now
        return self.fps


def _capture_loop() -> None:
    with mss.mss() as sct:
        while running.is_set():
            if not manual_parse_event.wait(timeout=0.5):
                continue
            manual_parse_event.clear()
            
            t0 = time.monotonic()
            with _state_lock:
                mon_idx = _current_monitor
            try:
                monitor = sct.monitors[mon_idx]
                raw = sct.grab(monitor)
                frame = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
                try:
                    capture_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        capture_queue.get_nowait()
                        capture_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            except Exception as exc:
                print(f"[capture] {exc}", file=sys.stderr)



def _semantic_loop() -> None:
    while running.is_set():
        try:
            frame, boxes = semantic_queue.get(timeout=0.5)
            brain.process_semantic_snapshot(frame, boxes)
                        
        except queue.Empty:
            continue
        except Exception as exc:
            print(f"[semantic] {exc}", file=sys.stderr)
        
        time.sleep(1.0)


def _process_loop() -> None:
    """DWT + detection + JPEG encode + msgpack pack → broadcast_queue."""
    target_interval = 1.0 / args.fps
    fps_tracker = FPSTracker()

    while running.is_set():
        t0 = time.monotonic()
        try:
            frame = capture_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        with _state_lock:
            band = _current_band

        try:
            t_proc = time.monotonic()
            wavelet_img, boxes, mask_img = wehgp_process(frame, band=band)
            
            try:
                semantic_queue.put_nowait((frame.copy(), [b.copy() for b in boxes]))
            except queue.Full:
                pass
                
            for b in boxes:
                x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                px, py = max(0, x - 2), max(0, y - 2)
                pw = min(frame.shape[1] - px, w + 4)
                ph = min(frame.shape[0] - py, h + 4)
                crop_bgr = frame[py:py+ph, px:px+pw]
                if crop_bgr.size > 0:
                    chash = hashlib.md5(crop_bgr.tobytes()).hexdigest()
                    val = brain.memory.get(chash)
                    if val:
                        b["semantic_value"] = val
            _, raw_jpg = cv2.imencode(".jpg", mask_img, [cv2.IMWRITE_JPEG_QUALITY, 55])
            _, wav_jpg = cv2.imencode(".jpg", wavelet_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _, scr_jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]) # REAL SCREENSHOT

            latency_ms = (time.monotonic() - t_proc) * 1000
            fps = fps_tracker.tick()
            h, w = frame.shape[:2]

            payload = msgpack.packb(
                {
                    "type": "frame",
                    "fps": round(fps, 1),
                    "latency_ms": round(latency_ms, 1),
                    "raw": raw_jpg.tobytes(),
                    "wavelet": wav_jpg.tobytes(),
                    "original": scr_jpg.tobytes(),
                    "boxes": boxes,
                    "band": band,
                    "element_count": len(boxes),
                    "source_w": w,
                    "source_h": h,
                },
                use_bin_type=True,
            )


            try:
                broadcast_queue.put_nowait(payload)
            except queue.Full:
                try:
                    broadcast_queue.get_nowait()
                    broadcast_queue.put_nowait(payload)
                except queue.Empty:
                    pass

        except Exception as exc:
            print(f"[process] {exc}", file=sys.stderr)

        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, target_interval - elapsed))


async def _broadcast_loop() -> None:
    loop = asyncio.get_event_loop()
    while running.is_set():
        try:
            payload = await loop.run_in_executor(
                None, lambda: broadcast_queue.get(timeout=0.1)
            )
            await server.broadcast(payload)
        except queue.Empty:
            pass
        except Exception as exc:
            print(f"[broadcast] {exc}", file=sys.stderr)


def _on_band(band: str) -> None:
    global _current_band
    if band in ("LH", "HL", "HH"):
        with _state_lock:
            _current_band = band
        print(f"[engine] band → {band}", file=sys.stderr)


def _on_monitor(idx: int) -> None:
    global _current_monitor
    with mss.mss() as sct:
        if 1 <= idx < len(sct.monitors):
            with _state_lock:
                _current_monitor = idx
            print(f"[engine] monitor → {idx}", file=sys.stderr)


def _on_parse() -> None:
    print(f"[engine] Manual parse triggered", file=sys.stderr)
    manual_parse_event.set()

async def main() -> None:
    with mss.mss() as sct:
        server.monitors_info = [
            {
                "index": i,
                "name": f"Monitor {i}",
                "width": m["width"],
                "height": m["height"],
            }
            for i, m in enumerate(sct.monitors)
            if i > 0  # index 0 = combined virtual monitor
        ]

    server.band_callback = _on_band
    server.monitor_callback = _on_monitor
    server.parse_callback = _on_parse

    # Start CPU-bound threads
    threading.Thread(target=_capture_loop, daemon=True, name="CaptureThread").start()
    threading.Thread(target=_process_loop, daemon=True, name="ProcessThread").start()
    threading.Thread(target=_semantic_loop, daemon=True, name="SemanticThread").start()

    print(
        f"[engine] ws://{args.host}:{args.port}  monitor={_current_monitor}  target={args.fps}fps",
        file=sys.stderr,
    )

    await asyncio.gather(server.serve_forever(), _broadcast_loop())


if __name__ == "__main__":
    try:
        import cv2, mss, msgpack, pywt
    except ImportError as _e:
        print(
            f"[engine] FATAL: {_e}\n"
            f"  Run vision/setup.bat first, or launch via Tauri's Start Engine button.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        running.clear()
