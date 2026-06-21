# Auragroove

A self-contained, portable **local AI music generator** for Windows — a slim,
fast UI on top of [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5),
tuned for 8 GB GPUs.

It exists to fix two pain points of the stock ACE-Step Gradio app:
1. **The memory leak** ([ACE-Step-1.5 #142](https://github.com/ace-step/ACE-Step-1.5/issues/142)) that crashes the machine after a few generations.
2. **Slow reloads** — the model reloading every run.

Auragroove runs generations through a **persistent worker process** that keeps the
(INT8-quantized) model **resident in VRAM** — so repeat runs are fast — while a
**memory watchdog** recycles the worker before the leak can pile up. One-shot
"Think" runs fall back to a safe subprocess.

## Features
- 🎛️ Clean Gradio UI with all the useful DiT/LM controls (caption, lyrics, duration,
  steps, seed, BPM/key/time-sig, guidance, sampler, velocity smoothing, Think/LM, …)
- ⚡ **Resident model** (INT8) → near-instant repeat generations on 8 GB
- 🧠 **Resident "Think"** via a small 0.6B LM (optional)
- 🩹 **Leak-bounded** via a growth-based worker watchdog
- ⏹️ **Stop** button, **batch** (sequential) outputs, **save/load settings**
- 🗂️ Flat outputs: `auragroove_outputs/audio/` + `auragroove_outputs/settings/`
- 🟣 Dark purple→red "aura" theme
- 📦 **Portable**: self-installs its own Python env + downloads models on first run

## Requirements
- Windows 10/11, 64-bit
- **NVIDIA GPU** + recent driver (8 GB VRAM is the tuned target; more is fine).
  CUDA is *not* a separate install — it ships inside PyTorch.
- ~25 GB free disk; internet for the first install
  (~4–5 GB PyTorch + ~12 GB models)

## Install & run
```text
1. git clone <this repo>   (or download the ZIP)
2. Double-click  run_auragroove.bat
   → first run auto-runs install.bat: builds the venv (uv) + downloads models
   → UI opens at http://127.0.0.1:7861
3. After that, run_auragroove.bat just launches.
```
You can also run `install.bat` first by itself.

## How it works
```
run_auragroove.bat  → launches the UI (auto-installs if needed)
install.bat         → builds venv from uv.lock + runs fetch_models.py
fetch_models.py     → downloads weights into acestep_engine/checkpoints
auragroove.py       → the Gradio UI (orchestration layer)
worker.py           → persistent generation worker (imports acestep, holds model)
acestep_engine/     → vendored ACE-Step 1.5 (engine) + patched cli.py
make_release.ps1    → stage a clean copy (no venv) to move to another PC
```

## Tuning (top of `auragroove.py`)
- `WORKER_OFFLOAD_TO_CPU` — keep `False` to stay resident; set `True` if you OOM.
- `WORKER_QUANTIZATION` — `int8_weight_only` (default) fits 8 GB; `none` on big GPUs.
- `WORKER_LM_MODEL` — resident Think LM (`acestep-5Hz-lm-0.6B`); `none` to disable.
- `RECYCLE_GROWTH_GB` — how much the worker may grow before it's recycled.

## Credits & licenses
- **Auragroove** wrapper code: MIT (see `LICENSE`).
- **Engine:** [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) by ACE Studio &
  StepFun — Apache-2.0 (`acestep_engine/LICENSE`).
- **Models** are downloaded separately and carry their own licenses (e.g. the DiT
  is Apache-2.0; ScragVAE and others have their own terms — check before commercial use).

Not affiliated with ACE Studio / StepFun. This is a community wrapper.
