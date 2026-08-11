# Serum-Plugin-Creator

Train a local model that generates **Xfer Serum 2** presets from a text prompt
or a reference audio clip — on free hardware.

## Hardware reality (Intel MacBook Pro, 2017–2019)

| Job | Where |
|---|---|
| Rendering the training data | This laptop, CPU, overnight |
| Training the model | Kaggle free GPU (30 h/week) or Colab free T4 |
| Running the finished model | This laptop, CPU |

An Intel Mac has no CUDA and no Metal/MPS backend, so PyTorch is CPU-only here.
That rules out local training but **not** local data generation, which is the
part that actually takes wall-clock time.

Tooling note: this project uses **Spotify Pedalboard**, not DawDreamer.
DawDreamer's current macOS wheels are Apple-Silicon only; Pedalboard ships
Intel-macOS wheels and hosts Serum as a VST3/AU instrument with full parameter
access and MIDI rendering.

## Why Serum 2 specifically

Serum 1's `.fxp` is an opaque zlib-compressed chunk with no public packer — you
can render sounds but cannot write a preset file the synth will open. Serum 2's
`.SerumPreset` (CBOR) has been reverse-engineered by
[serum-preset-packager](https://github.com/KennethWussmann/serum-preset-packager),
which unpacks to JSON and repacks losslessly. **Serum 2 is required** for the
pipeline to emit a usable preset.

## The core idea

You do not need to collect a preset library. You generate one:

```
random parameters -> render audio in Serum -> (audio, parameters) pair
```

Infinite perfectly-labelled training data, free, with no scraping and no
licensing exposure.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/00_doctor.py          # environment + Serum detection
python3 scripts/01_probe_serum.py     # parameter map + render throughput
```

`01_probe_serum.py` writes `out/serum_parameters.json` (the search space) and
prints a renders-per-second figure that determines how large a dataset is
realistic on this machine.

## Roadmap

- **Phase 0** — host Serum headlessly, render a note from Python. *(scripts here)*
- **Phase 1** — CLAP + CMA-ES search. Text prompt → optimised parameters, no
  training required. A working tool on CPU alone.
- **Phase 2** — render 100k examples over a restricted parameter set
  (2 osc, 1 filter, 2 envelopes), train an audio→parameters model on free GPU.
- **Phase 3** — widen the parameter space by curriculum; use the model's
  prediction to seed the Phase 1 search so it converges an order faster.
- **Phase 4** — differentiable neural proxy for Serum; macro assignment and
  mod-matrix cleanup so the output is a preset a human wants to open.

## Status

Phase 0, unverified. The scripts are syntax-checked but have not been run
against a real Serum installation — that requires the laptop.
