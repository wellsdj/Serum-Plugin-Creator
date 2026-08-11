#!/usr/bin/env python3
"""Environment check. Run this first.

Answers the three questions that decide whether this project is viable on your
machine, and what the rest of the pipeline has to look like:

  1. Can this laptop render audio at a useful rate?
  2. Is Serum installed, and is it Serum 1 or Serum 2?
  3. Does Pedalboard load it?

Usage:  python3 scripts/00_doctor.py
"""

import glob
import os
import platform
import shutil
import subprocess
import sys

# Standard macOS plug-in locations, system then per-user.
PLUGIN_DIRS = [
    "/Library/Audio/Plug-Ins/VST3",
    "/Library/Audio/Plug-Ins/Components",
    "/Library/Audio/Plug-Ins/VST",
    os.path.expanduser("~/Library/Audio/Plug-Ins/VST3"),
    os.path.expanduser("~/Library/Audio/Plug-Ins/Components"),
    os.path.expanduser("~/Library/Audio/Plug-Ins/VST"),
]


def hr(title):
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def check_machine():
    hr("1. MACHINE")
    print(f"Platform      : {platform.system()} {platform.release()}")
    print(f"Architecture  : {platform.machine()}")
    print(f"Python        : {sys.version.split()[0]} ({sys.executable})")

    if platform.system() == "Darwin":
        mac_ver = platform.mac_ver()[0]
        print(f"macOS         : {mac_ver}")
        try:
            major = int(mac_ver.split(".")[0])
            if major < 11:
                print("  !! Pedalboard needs macOS 11+. You will need to upgrade.")
        except ValueError:
            pass

    is_intel_mac = (platform.system() == "Darwin"
                    and platform.machine() in ("x86_64", "i386"))
    if is_intel_mac:
        print("\n  Intel Mac detected. Consequences for this project:")
        print("   - No CUDA and no Metal/MPS. PyTorch here is CPU-only.")
        print("   - Do NOT plan to train models locally. Render data here,")
        print("     train on Kaggle's free GPU (30 h/week).")
        print("   - Expect thermal throttling on long renders. Prefer running")
        print("     overnight, elevated, and leave one core free.")

    try:
        import psutil

        phys = psutil.cpu_count(logical=False) or 1
        gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"\nPhysical cores: {phys}")
        print(f"RAM           : {gb:.1f} GB")
        print(f"Suggested render workers: {max(1, phys - 1)}")
        if gb < 8:
            print("  !! Under 8 GB. Render in small batches to avoid swapping.")
    except ImportError:
        print("\n(psutil not installed - skipping CPU/RAM report)")

    free_gb = shutil.disk_usage(os.path.expanduser("~")).free / (1024 ** 3)
    print(f"Free disk     : {free_gb:.1f} GB")
    if free_gb < 60:
        print("  !! Tight. A 100k-example dataset of mel-spectrograms is ~20-40 GB.")
        print("     Store mel features, never raw WAV, and use float16.")


def find_serum():
    hr("2. SERUM INSTALLATION")
    hits = []
    for d in PLUGIN_DIRS:
        if not os.path.isdir(d):
            continue
        for pattern in ("*erum*", "*Xfer*"):
            hits.extend(glob.glob(os.path.join(d, pattern)))
    hits = sorted(set(hits))

    if not hits:
        print("No Serum plug-in found in the standard macOS locations.")
        print("Searched:")
        for d in PLUGIN_DIRS:
            print(f"  {d}  {'' if os.path.isdir(d) else '(missing)'}")
        print("\nThis project cannot render audio without Serum installed.")
        return []

    print("Found:")
    for h in hits:
        print(f"  {h}")

    # Serum 2 ships as 'Serum2.vst3' / 'Serum2.component'; v1 has no digit.
    v2 = [h for h in hits if "2" in os.path.basename(h)]
    print("\nVersion assessment:")
    if v2:
        print("  Serum 2 detected. This is the version you want:")
        print("  its .SerumPreset format is reverse-engineered, so the")
        print("  pipeline can WRITE preset files Serum can open.")
    else:
        print("  Looks like Serum 1 only.")
        print("  Serum 1's .fxp is an opaque zlib chunk with no public packer.")
        print("  You can still render and search sounds, but you cannot emit a")
        print("  loadable preset file. Serum 2 is effectively required.")
    return hits


def find_preset_dirs():
    hr("3. PRESET FOLDERS")
    roots = [
        os.path.expanduser("~/Documents/Xfer"),
        os.path.expanduser("~/Library/Audio/Presets/Xfer Records"),
    ]
    found = False
    for r in roots:
        if os.path.isdir(r):
            found = True
            print(f"  {r}")
            for sub in sorted(os.listdir(r))[:10]:
                print(f"    - {sub}")
    if not found:
        print("  No Xfer preset folder found (normal if Serum has never run).")


def check_pedalboard(plugin_paths):
    hr("4. PEDALBOARD")
    try:
        import pedalboard

        print(f"pedalboard {pedalboard.__version__} imported OK")
    except ImportError:
        print("pedalboard not installed.  pip install -r requirements.txt")
        return

    if not plugin_paths:
        print("No plug-in to test against.")
        return

    # Prefer VST3: parameter naming is more consistent than AU here.
    candidates = [p for p in plugin_paths if p.endswith(".vst3")] or plugin_paths
    target = candidates[0]
    print(f"\nAttempting to load: {target}")
    print("(first load can take 30-60 s while the plug-in scans wavetables)")
    try:
        from pedalboard import load_plugin

        plug = load_plugin(target)
        params = list(plug.parameters.keys())
        print(f"  Loaded. {len(params)} automatable parameters exposed.")
        print("  First 25:")
        for name in params[:25]:
            print(f"    {name}")
        print("\n  Next step: python3 scripts/01_probe_serum.py")
    except Exception as exc:  # noqa: BLE001 - surface whatever the host throws
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        print("  Common causes: plug-in is Apple-Silicon-only, unsigned/quarantined,")
        print("  or Serum is not authorised on this machine (open it in a DAW once).")


def main():
    print("Serum-Plugin-Creator :: environment doctor")
    check_machine()
    plugins = find_serum()
    find_preset_dirs()
    check_pedalboard(plugins)
    hr("DONE")
    print("Paste this whole output back into the chat.")


if __name__ == "__main__":
    main()
