"""
test_mics.py — verify L/R mic channels are correctly mapped.

Walks you through a guided finger-snap test:
    1. SILENCE     (2 sec)  — baseline ambient
    2. SNAP LEFT   (3 sec)  — snap fingers next to LEFT mic
    3. SILENCE     (2 sec)  — gap
    4. SNAP RIGHT  (3 sec)  — snap fingers next to RIGHT mic
    5. SILENCE     (2 sec)  — final ambient

Then computes per-channel RMS during each phase and checks:
    - During "snap left":  config-mapped LEFT  channel should be louder
    - During "snap right": config-mapped RIGHT channel should be louder
    - Both channels should be alive (RMS well above ambient floor)

Verdict at the end: PASS, REVERSED (swap MIC_LEFT_CH/MIC_RIGHT_CH in config),
or DEAD (one channel silent — physical wiring problem).

Usage:
    poetry run python scripts/test_mics.py
    poetry run python scripts/test_mics.py --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture import CaptureConfig, StereoCapture
from config import DEFAULT_SERIAL_PORT, MIC_LEFT_CH, MIC_RIGHT_CH, SAMPLE_RATE


PHASES = [
    ("silence_pre",  2.0, "Be quiet — measuring ambient",          False),
    ("snap_left",    3.0, "SNAP FINGERS next to the LEFT mic",     True),
    ("silence_mid",  2.0, "Stop. Be quiet.",                       False),
    ("snap_right",   3.0, "SNAP FINGERS next to the RIGHT mic",    True),
    ("silence_post", 2.0, "Stop. Be quiet.",                       False),
]


def rms(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(x ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--ratio-threshold", type=float, default=2.0,
                        help="dominant-channel RMS must exceed quiet-channel "
                             "RMS by at least this ratio (default: 2.0)")
    parser.add_argument("--alive-ratio", type=float, default=2.0,
                        help="snap-phase RMS must exceed silence-phase RMS by "
                             "at least this ratio to count as 'mic alive' "
                             "(default: 2.0)")
    args = parser.parse_args()

    print(f"Opening {args.port}...")
    cap = StereoCapture(CaptureConfig(port=args.port))
    try:
        cap.start()
    except Exception as e:
        sys.exit(f"ERROR: failed to open {args.port}: {e}")

    print(f"\nConfig says: LEFT  mic = channel {MIC_LEFT_CH}")
    print(f"             RIGHT mic = channel {MIC_RIGHT_CH}")
    print(f"\nThis test will tell you whether that mapping matches reality.\n")

    # Phase results: name -> (ch0_rms, ch1_rms)
    results: dict[str, tuple[float, float]] = {}

    for name, secs, prompt, is_active in PHASES:
        # Countdown so user has a chance to read the prompt
        if is_active:
            print(f"\n>>> {prompt}")
            for i in (3, 2, 1):
                print(f"    starting in {i}...", end="\r", flush=True)
                time.sleep(1)
            print(f"    GO! (for {secs:.0f} seconds)" + " " * 20)
        else:
            print(f"\n[{prompt}]")

        # Capture the phase audio
        n_samples = int(secs * SAMPLE_RATE)
        try:
            frame = cap.get_frame(n_samples, timeout=secs + 3.0)
        except TimeoutError as e:
            cap.stop()
            sys.exit(f"\nERROR: capture timeout during {name}: {e}")

        ch0_rms = rms(frame[:, 0])
        ch1_rms = rms(frame[:, 1])
        results[name] = (ch0_rms, ch1_rms)

        if is_active:
            print(f"    ch0 RMS = {ch0_rms:>7.0f}    ch1 RMS = {ch1_rms:>7.0f}")
        else:
            print(f"    [silence]  ch0 = {ch0_rms:>7.0f}, ch1 = {ch1_rms:>7.0f}")

    cap.stop()

    # ── Analysis ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Average ambient floor across all silence phases
    silences = [results[k] for k in results if k.startswith("silence")]
    ambient_ch0 = float(np.mean([s[0] for s in silences]))
    ambient_ch1 = float(np.mean([s[1] for s in silences]))
    print(f"\nAmbient floor:   ch0 = {ambient_ch0:>7.0f}    ch1 = {ambient_ch1:>7.0f}")

    snap_left_ch0,  snap_left_ch1  = results["snap_left"]
    snap_right_ch0, snap_right_ch1 = results["snap_right"]

    print(f"Snap near LEFT:  ch0 = {snap_left_ch0:>7.0f}    ch1 = {snap_left_ch1:>7.0f}")
    print(f"Snap near RIGHT: ch0 = {snap_right_ch0:>7.0f}    ch1 = {snap_right_ch1:>7.0f}")

    # Aliveness: did each channel actually rise above ambient during snaps?
    ch0_max_snap = max(snap_left_ch0, snap_right_ch0)
    ch1_max_snap = max(snap_left_ch1, snap_right_ch1)
    ch0_alive = ch0_max_snap > ambient_ch0 * args.alive_ratio
    ch1_alive = ch1_max_snap > ambient_ch1 * args.alive_ratio

    print(f"\nch0 alive: {'YES' if ch0_alive else 'NO'}  "
          f"(peak snap RMS {ch0_max_snap:.0f} vs ambient {ambient_ch0:.0f})")
    print(f"ch1 alive: {'YES' if ch1_alive else 'NO'}  "
          f"(peak snap RMS {ch1_max_snap:.0f} vs ambient {ambient_ch1:.0f})")

    if not (ch0_alive and ch1_alive):
        print("\n" + "=" * 60)
        print("VERDICT: DEAD CHANNEL")
        print("=" * 60)
        if not ch0_alive:
            print("  Channel 0 didn't respond to either snap.")
        if not ch1_alive:
            print("  Channel 1 didn't respond to either snap.")
        print("\nThis is a physical wiring problem, not a config problem.")
        print("Things to check:")
        print("  - Loose jumper wire on the breadboard (BCLK/WS/SD/VDD/GND)")
        print("  - L/R select pin: one mic should be GND, the other VDD")
        print("  - SD line: both mics share it; one wired wrong = silent half")
        print("  - Mic itself: try swapping in a spare INMP441 if you have one")
        sys.exit(2)

    # Direction check — which channel was louder during each snap?
    left_louder_on_left  = snap_left_ch0  > snap_left_ch1
    right_louder_on_right = snap_right_ch1 > snap_right_ch0

    # Margins
    left_margin  = snap_left_ch0  / max(snap_left_ch1,  1.0) if left_louder_on_left  else snap_left_ch1  / max(snap_left_ch0,  1.0)
    right_margin = snap_right_ch1 / max(snap_right_ch0, 1.0) if right_louder_on_right else snap_right_ch0 / max(snap_right_ch1, 1.0)

    print(f"\nDuring 'snap LEFT':  louder channel = ch{0 if left_louder_on_left else 1}  "
          f"(margin {left_margin:.2f}x)")
    print(f"During 'snap RIGHT': louder channel = ch{1 if right_louder_on_right else 0}  "
          f"(margin {right_margin:.2f}x)")

    # Map "physical loudness winner" to "config-mapped name"
    physical_left_ch  = 0 if left_louder_on_left  else 1
    physical_right_ch = 1 if right_louder_on_right else 0

    print("\n" + "=" * 60)

    # Margins below threshold = ambiguous, ask user to retry
    if left_margin < args.ratio_threshold or right_margin < args.ratio_threshold:
        print("VERDICT: AMBIGUOUS")
        print("=" * 60)
        print(f"Margins ({left_margin:.2f}x, {right_margin:.2f}x) below threshold "
              f"({args.ratio_threshold}x).")
        print("Possible causes:")
        print("  - Snaps weren't loud enough or not close enough to one mic")
        print("  - Mics too close together — try snapping right at one mic")
        print("  - Room reverb is washing out the directionality")
        print("\nRe-run the test, snapping closer to each mic.")
        sys.exit(3)

    # Consistent? (e.g. both said channel 0 = left → consistent)
    if physical_left_ch == physical_right_ch:
        print("VERDICT: INCONSISTENT")
        print("=" * 60)
        print("Both snap directions said the same channel was louder.")
        print("Either the snaps weren't actually on different sides, or")
        print("one channel is much louder than the other regardless of source.")
        print("Re-run, and make sure to clearly snap on opposite sides of the array.")
        sys.exit(4)

    # OK — we have a coherent (left_ch, right_ch) mapping. Does it match config?
    if physical_left_ch == MIC_LEFT_CH and physical_right_ch == MIC_RIGHT_CH:
        print("VERDICT: PASS ✓")
        print("=" * 60)
        print(f"Physical L/R matches config.")
        print(f"  config.MIC_LEFT_CH  = {MIC_LEFT_CH}  ✓")
        print(f"  config.MIC_RIGHT_CH = {MIC_RIGHT_CH}  ✓")
        print(f"\nDoA sign convention will be correct:")
        print(f"  source on LEFT  → positive angle")
        print(f"  source on RIGHT → negative angle")
        sys.exit(0)
    else:
        print("VERDICT: REVERSED")
        print("=" * 60)
        print(f"Physical wiring says:")
        print(f"  LEFT  mic = channel {physical_left_ch}")
        print(f"  RIGHT mic = channel {physical_right_ch}")
        print(f"But config.py says:")
        print(f"  MIC_LEFT_CH  = {MIC_LEFT_CH}")
        print(f"  MIC_RIGHT_CH = {MIC_RIGHT_CH}")
        print(f"\nFix: edit config.py to:")
        print(f"  MIC_LEFT_CH  = {physical_left_ch}")
        print(f"  MIC_RIGHT_CH = {physical_right_ch}")
        print(f"\nOr physically swap the two mic SD lines on the breadboard.")
        sys.exit(1)


if __name__ == "__main__":
    main()