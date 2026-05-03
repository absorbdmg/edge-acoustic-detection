"""
server.py — FastAPI websocket server for the HELLBAT dashboard.

ARCHITECTURE
============
One canonical message shape, built in exactly one place: build_message().
Three input modes feed it:
  - run_live:     real ESP32 + GPU inference, sliding window
  - run_playback: replay a pre-recorded JSON file
  - run_mock:     synthetic data for UI development

All shared constants come from config.py — change once there, every entry
point picks it up.

Frontend contract (every websocket message has this shape):
{
    "ts":         float,
    "idx":        int,
    "llr":        float,
    "decision":   "drone"|"clear",
    "threshold":  float,
    "doa": {
        "angle_deg":      float,   # -90 (left) to +90 (right), 0 = front
        "confidence":     float,   # 0 to 1
        "delay_samples":  float
    },
    "rsf": {
        "data":         [60 floats],     # flattened row-major (6 x 10)
        "shape":        [6, 10],         # [n_scales, n_rates]
        "rates":        [-32,...,32],
        "scales":       [0.25,...,8],
        "midline_col":  5
    },
    "telemetry": {
        "chunk_ms":         float,
        "rt_factor":        float,
        "drift_pct":        float,
        "samples_dropped":  int,
        "buffer_fill_pct":  float
    }
}

UPDATE RATE
===========
1-sec audio WINDOW size is fixed (model expects 16,000 samples). What
changes is the HOP between windows, controlled by --hop-hz {1,2,4}.

Usage:
    poetry run python scripts/server.py                            # live, default Hz
    poetry run python scripts/server.py --hop-hz 4                 # stretch
    poetry run python scripts/server.py --mock                     # synthetic
    poetry run python scripts/server.py --playback demo.json --loop
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import pickle
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DEFAULT_HOP_HZ,
    DEFAULT_HTTP_PORT,
    DEFAULT_SERIAL_PORT,
    DEFAULT_SERVER_HOST,
    DEFAULT_THRESHOLD,
    DEFAULT_WARMUP_FRAMES,
    DOA_BANDPASS_ENABLED,
    DOA_MIN_CONFIDENCE,
    DOA_SMOOTHING_ENABLED,
    MIC_BASELINE_M,
    MIC_LEFT_CH,
    MIC_RIGHT_CH,
    MIDLINE_COL,
    N_RATES,
    N_SCALES,
    RATES,
    SAMPLE_RATE,
    SCALES,
    TARGET_RMS,
    WINDOW_SAMPLES,
    doa_alpha_for_hop,
    hop_samples,
)


# ── Shared state ──────────────────────────────────────────────────────────────

_latest: Optional[dict] = None
_lock = threading.Lock()
_stop = threading.Event()
_MODE = "?"


def publish(msg: dict) -> None:
    with _lock:
        global _latest
        _latest = msg


def get_latest() -> Optional[dict]:
    with _lock:
        return _latest


# ── Single source of truth for message shape ─────────────────────────────────

@dataclass
class InferenceFrame:
    """Raw inference outputs from any mode. build_message() turns this into
    the canonical websocket message."""
    idx: int
    llr: float
    threshold: float

    angle_deg: float
    doa_confidence: float
    delay_samples: float

    rate_scale: np.ndarray   # shape (n_scales=6, n_rates=10), float

    chunk_ms: float
    rt_factor: float
    drift_pct: float
    samples_dropped: int
    buffer_fill_pct: float


def build_message(f: InferenceFrame) -> dict:
    """The ONLY place that constructs the websocket dict."""
    rs = f.rate_scale
    if rs.ndim != 2 or rs.shape != (N_SCALES, N_RATES):
        raise ValueError(
            f"rate_scale must be ({N_SCALES}, {N_RATES}); got {rs.shape}"
        )

    return {
        "ts":        time.time(),
        "idx":       int(f.idx),
        "llr":       float(f.llr),
        "decision":  "drone" if f.llr > f.threshold else "clear",
        "threshold": float(f.threshold),
        "doa": {
            "angle_deg":     float(f.angle_deg),
            "confidence":    float(f.doa_confidence),
            "delay_samples": float(f.delay_samples),
        },
        "rsf": {
            "data":        rs.flatten().astype(float).tolist(),
            "shape":       [N_SCALES, N_RATES],
            "rates":       RATES,
            "scales":      SCALES,
            "midline_col": MIDLINE_COL,
        },
        "telemetry": {
            "chunk_ms":         float(f.chunk_ms),
            "rt_factor":        float(f.rt_factor),
            "drift_pct":        float(f.drift_pct),
            "samples_dropped":  int(f.samples_dropped),
            "buffer_fill_pct":  float(f.buffer_fill_pct),
        },
    }


def reduce_rsf(rsf_full: np.ndarray) -> np.ndarray:
    """(n_rates=10, n_scales=6, n_freq=128) → (n_scales=6, n_rates=10)"""
    if rsf_full.ndim != 3:
        raise ValueError(f"rsf_full must be 3D; got shape {rsf_full.shape}")
    return rsf_full.mean(axis=-1).T.astype(np.float32)


def normalize_chunk(chunk: np.ndarray) -> np.ndarray:
    """Mean-center, RMS-normalize, peak-limit. Matches dataset builder."""
    chunk = chunk.astype(np.float32)
    chunk = chunk - chunk.mean()
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    if rms < 1e-8:
        return chunk
    chunk = chunk * (TARGET_RMS / rms)
    peak = float(np.abs(chunk).max())
    if peak > 0.95:
        chunk = chunk * (0.95 / peak)
    return chunk


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"service": "hellbat", "mode": _MODE, "ws": "/ws/state"}


@app.websocket("/ws/state")
async def websocket_state(ws: WebSocket):
    await ws.accept()
    last_idx = -1
    try:
        while not _stop.is_set():
            msg = get_latest()
            if msg is not None and msg["idx"] != last_idx:
                await ws.send_json(msg)
                last_idx = msg["idx"]
            # Poll faster than the highest hop rate (4 Hz = 250 ms).
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] error: {e}")


# ── Mode 1: Live inference (sliding window) ──────────────────────────────────

def run_live(args):
    """Real ESP32 + GPU inference with sliding-window updates."""
    import cupy as cp

    from capture import CaptureConfig, StereoCapture
    from doa import SmoothedDoa, gcc_phat
    from features import Config, PyJetsonSTM

    if not (args.models / "tsvd.pkl").exists():
        sys.exit(f"models not found in {args.models}/")

    HOP = hop_samples(args.hop_hz)

    print(f"[live] hop = {HOP} samples = {HOP/SAMPLE_RATE:.3f}s "
          f"({args.hop_hz} Hz)")
    print("[live] loading models...")
    with open(args.models / "tsvd.pkl", "rb") as f:
        tsvd = pickle.load(f)
    with open(args.models / "gmm.pkl", "rb") as f:
        gmm = pickle.load(f)

    print("[live] building feature pipeline...")
    pipeline = PyJetsonSTM(Config(), chunk_samples=WINDOW_SAMPLES)
    dummy = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    _ = pipeline.compute_device(dummy)
    cp.cuda.Stream.null.synchronize()
    print("[live] pipeline ready.")

    print(f"[live] opening {args.port}...")
    cap = StereoCapture(CaptureConfig(port=args.port))
    cap.start()

    # DoA smoother — alpha scales with hop rate, can be disabled in config
    if DOA_SMOOTHING_ENABLED:
        smoother = SmoothedDoa(
            alpha=doa_alpha_for_hop(args.hop_hz),
            min_confidence=DOA_MIN_CONFIDENCE,
        )
    else:
        smoother = None
        print("[live] DoA smoothing DISABLED via config")

    if not DOA_BANDPASS_ENABLED:
        print("[live] DoA bandpass DISABLED via config")

    # Prime buffer with one full window
    try:
        buffer = cap.get_frame(WINDOW_SAMPLES, timeout=2.0)
    except TimeoutError as e:
        cap.stop()
        sys.exit(f"[live] initial buffer timeout ({e}); is ESP32 streaming?")

    idx = 0
    try:
        while not _stop.is_set():
            try:
                new_samples = cap.get_frame(HOP, timeout=2.0)
            except TimeoutError:
                continue

            # Slide window
            buffer = np.concatenate([buffer[HOP:], new_samples], axis=0)

            left  = buffer[:, MIC_LEFT_CH ].astype(np.float32)
            right = buffer[:, MIC_RIGHT_CH].astype(np.float32)

            t0 = time.monotonic()

            doa_raw = gcc_phat(left, right, sr=SAMPLE_RATE,
                               baseline_m=args.baseline,
                               bandpass=DOA_BANDPASS_ENABLED)
            if smoother is not None:
                smooth_angle, smooth_conf = smoother.update(
                    doa_raw.angle_deg, doa_raw.confidence
                )
            else:
                smooth_angle, smooth_conf = doa_raw.angle_deg, doa_raw.confidence

            mono = (left + right) * 0.5
            audio_n = normalize_chunk(mono)
            rsf_dev = pipeline.compute_device(audio_n)
            rsf_full = cp.asnumpy(rsf_dev.mean(axis=0)).astype(np.float32)
            V = tsvd.transform(rsf_full)
            llr = float(gmm.llr(V[None, :])[0])
            chunk_time = time.monotonic() - t0

            if idx < args.warmup_frames:
                idx += 1
                continue

            stats = cap.stats()
            buf_fill, buf_total = stats.get("buffer_fill", (0, 1))
            buf_fill_pct = (buf_fill / buf_total * 100.0) if buf_total else 0.0

            audio_per_step = HOP / SAMPLE_RATE
            rt_factor = audio_per_step / chunk_time if chunk_time > 0 else 0.0

            inf = InferenceFrame(
                idx=idx,
                llr=llr,
                threshold=args.threshold,
                angle_deg=smooth_angle,
                doa_confidence=smooth_conf,
                delay_samples=getattr(doa_raw, "delay_samples", 0.0),
                rate_scale=reduce_rsf(rsf_full),
                chunk_ms=chunk_time * 1000.0,
                rt_factor=rt_factor,
                drift_pct=stats.get("drift_pct", 0.0),
                samples_dropped=stats.get("samples_dropped", 0),
                buffer_fill_pct=buf_fill_pct,
            )
            publish(build_message(inf))
            idx += 1
    finally:
        cap.stop()


# ── Mode 2: Playback ─────────────────────────────────────────────────────────

def run_playback(args):
    """Replay a pre-recorded canonical-message JSON list at the configured rate."""
    print(f"[playback] loading {args.playback}...")
    with open(args.playback) as f:
        records = json.load(f)
    print(f"[playback] {len(records)} records, looping={args.loop}")

    period = 1.0 / args.hop_hz

    idx = 0
    while not _stop.is_set():
        for rec in records:
            if _stop.is_set():
                break
            msg = dict(rec)
            msg["idx"] = idx
            msg["ts"] = time.time()
            publish(msg)
            idx += 1
            time.sleep(period)
        if not args.loop:
            break


# ── Mode 3: Mock ─────────────────────────────────────────────────────────────

def run_mock(args):
    """Synthetic data so the frontend can be developed without hardware.
    30-sec cycle: 0-15s clear, 15-30s drone."""
    period = 1.0 / args.hop_hz
    cycle_steps = 30 * args.hop_hz   # 30 sec cycle regardless of rate

    idx = 0
    while not _stop.is_set():
        t = (idx % cycle_steps) / args.hop_hz   # seconds within cycle
        is_drone = t >= 15

        if is_drone:
            phase = (t - 15) / 15.0
            llr = 4.0 + random.random() * 8.0 + 1.5 * math.sin(phase * 2 * math.pi)
            angle = -30.0 + phase * 70.0 + (random.random() - 0.5) * 4.0
            conf = 0.65 + random.random() * 0.3
            delay = angle / 30.0
        else:
            llr = -8.0 + random.random() * 5.0
            angle = (random.random() - 0.5) * 180.0
            conf = random.random() * 0.25
            delay = 0.0

        # Synthesize a plausible (n_scales, n_rates) rate-scale matrix.
        # Drone: bilateral peaks at high |rate|, mid scales. Clear: noise.
        rate_scale = np.zeros((N_SCALES, N_RATES), dtype=np.float32)
        for s in range(N_SCALES):
            for r in range(N_RATES):
                noise = random.random() * 0.2
                if is_drone:
                    abs_r_idx = abs(r - (N_RATES - 1) / 2.0)
                    blob = (
                        math.exp(-((abs_r_idx - 3.5) ** 2) / 4.0)
                        * math.exp(-((s - 3) ** 2) / 4.0)
                    )
                    rate_scale[s, r] = noise + blob * (0.6 + random.random() * 0.4)
                else:
                    rate_scale[s, r] = noise

        inf = InferenceFrame(
            idx=idx,
            llr=llr,
            threshold=args.threshold,
            angle_deg=angle,
            doa_confidence=max(0.0, min(1.0, conf)),
            delay_samples=delay,
            rate_scale=rate_scale,
            chunk_ms=180 + random.random() * 30,
            rt_factor=_mock_rt_factor(args.hop_hz),
            drift_pct=0.02 + random.random() * 0.03,
            samples_dropped=0,
            buffer_fill_pct=4.0 + random.random() * 2.0,
        )
        publish(build_message(inf))
        idx += 1
        time.sleep(period)


def _mock_rt_factor(hop_hz: int) -> float:
    """Plausible RT factor for mock telemetry, rate-dependent."""
    base = {1: 7.0, 2: 3.5, 4: 1.8}.get(hop_hz, 5.0)
    return base + random.random() * 0.3


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _MODE

    p = argparse.ArgumentParser()
    p.add_argument("--port",     default=DEFAULT_SERIAL_PORT)
    p.add_argument("--models",   type=Path, default=Path("models"))
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--baseline", type=float, default=MIC_BASELINE_M)
    p.add_argument("--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES)
    p.add_argument("--host",     default=DEFAULT_SERVER_HOST)
    p.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT)
    p.add_argument("--hop-hz",   type=int, default=DEFAULT_HOP_HZ, choices=[1, 2, 4],
                   help=f"updates per second (default: {DEFAULT_HOP_HZ})")
    p.add_argument("--mock",     action="store_true")
    p.add_argument("--playback", type=Path, default=None)
    p.add_argument("--loop",     action="store_true")
    args = p.parse_args()

    if args.mock:
        target = run_mock
        _MODE = f"MOCK@{args.hop_hz}Hz"
    elif args.playback is not None:
        target = run_playback
        _MODE = f"PLAYBACK({args.playback.name})@{args.hop_hz}Hz"
    else:
        target = run_live
        _MODE = f"LIVE@{args.hop_hz}Hz"

    print(f"=== HELLBAT server [{_MODE}] ===")
    print(f"  http://{args.host}:{args.http_port}/")
    print(f"  ws://{args.host}:{args.http_port}/ws/state")

    def handle_sigint(signum, frame):
        print("\n[server] stopping...")
        _stop.set()
    signal.signal(signal.SIGINT, handle_sigint)

    def _wrapped_target(args):
        try:
            target(args)
        except SystemExit as e:
            print(f"[inference] exited: {e}", flush=True)
            _stop.set()
        except Exception:
            import traceback
            print("[inference] CRASHED:", flush=True)
            traceback.print_exc()
            _stop.set()
        else:
            print("[inference] thread exited normally", flush=True)

    inf_thread = threading.Thread(target=_wrapped_target, args=(args,), daemon=True)
    inf_thread.start()

    config = uvicorn.Config(app, host=args.host, port=args.http_port,
                            log_level="warning")
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()


if __name__ == "__main__":
    main()