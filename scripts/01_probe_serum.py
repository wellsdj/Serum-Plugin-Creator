#!/usr/bin/env python3
"""Probe Serum: dump its parameter map and measure render throughput.

This produces the two facts the whole project is planned around:

  * out/serum_parameters.json - every automatable parameter, its name and
    current value. This is the search space the model will learn to predict.
  * a renders-per-second number - decides how big a dataset is realistic
    on this laptop, and therefore how ambitious the model can be.

Usage:
    python3 scripts/01_probe_serum.py --plugin "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"

If --plugin is omitted the script reuses whatever 00_doctor.py found.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import soundfile as sf
from mido import Message
from pedalboard import load_plugin

SAMPLE_RATE = 44100
NOTE = 48          # C3 - low enough to expose filter movement and harmonics
VELOCITY = 100
HOLD = 2.0         # seconds the note is held
TAIL = 1.0         # release tail, so the amp envelope's release is audible
DURATION = HOLD + TAIL

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")


def autodetect():
    for pattern in (
        "/Library/Audio/Plug-Ins/VST3/*erum*.vst3",
        os.path.expanduser("~/Library/Audio/Plug-Ins/VST3/*erum*.vst3"),
        "/Library/Audio/Plug-Ins/Components/*erum*.component",
    ):
        hits = sorted(glob.glob(pattern))
        if hits:
            # Prefer a v2 build when both are installed.
            v2 = [h for h in hits if "2" in os.path.basename(h)]
            return (v2 or hits)[0]
    raise SystemExit("No Serum plug-in found. Pass --plugin explicitly.")


def render(plug, duration=DURATION):
    """Render one held note through the instrument and return mono audio."""
    messages = [
        Message("note_on", note=NOTE, velocity=VELOCITY, time=0.0),
        Message("note_off", note=NOTE, time=HOLD),
    ]
    audio = plug(messages, duration=duration, sample_rate=SAMPLE_RATE)
    # pedalboard returns (channels, samples); collapse to mono for analysis.
    return audio.mean(axis=0) if audio.ndim > 1 else audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default=None)
    ap.add_argument("--trials", type=int, default=10,
                    help="renders to time for the throughput estimate")
    args = ap.parse_args()

    path = args.plugin or autodetect()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {path} ...")
    t0 = time.time()
    plug = load_plugin(path)
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    # --- Parameter map -----------------------------------------------------
    params = plug.parameters
    records = []
    for key, p in params.items():
        rec = {"key": key, "name": getattr(p, "name", key)}
        # Discrete params expose valid_values; continuous ones expose a range.
        for attr in ("min_value", "max_value", "step_size", "units"):
            val = getattr(p, attr, None)
            if val is not None:
                rec[attr] = val
        valid = getattr(p, "valid_values", None)
        if valid:
            rec["valid_values"] = list(valid)
        try:
            rec["default"] = getattr(plug, key)
        except Exception:  # noqa: BLE001 - some params refuse reads
            rec["default"] = None
        records.append(rec)

    param_path = os.path.join(OUT_DIR, "serum_parameters.json")
    with open(param_path, "w") as fh:
        json.dump(records, fh, indent=2, default=str)

    discrete = [r for r in records if "valid_values" in r]
    print(f"{len(records)} automatable parameters "
          f"({len(discrete)} discrete, {len(records) - len(discrete)} continuous)")
    print(f"Written to {os.path.relpath(param_path)}\n")

    # --- Reference render --------------------------------------------------
    print("Rendering reference note (init patch)...")
    audio = render(plug)
    wav_path = os.path.join(OUT_DIR, "reference_C3.wav")
    sf.write(wav_path, audio, SAMPLE_RATE)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f"  {len(audio) / SAMPLE_RATE:.2f}s, peak {peak:.3f} -> "
          f"{os.path.relpath(wav_path)}")
    if peak < 1e-4:
        print("  !! Silence. Serum may be unauthorised or awaiting its GUI.")
        print("     Open Serum once in a DAW to activate, then retry.")

    # --- Throughput --------------------------------------------------------
    print(f"\nTiming {args.trials} renders...")
    t0 = time.time()
    for _ in range(args.trials):
        render(plug)
    per = (time.time() - t0) / args.trials

    print(f"  {per:.3f}s per {DURATION:.0f}s render "
          f"({DURATION / per:.1f}x realtime)")
    print("\nDataset feasibility on this machine (single core):")
    for n in (10_000, 100_000, 1_000_000):
        hrs = n * per / 3600
        print(f"  {n:>9,} examples  ->  {hrs:8.1f} core-hours "
              f"({hrs / 6:6.1f} h across 6 workers)")
    print("\nPick the largest tier you can render in ~2 nights, and start there.")


if __name__ == "__main__":
    main()
