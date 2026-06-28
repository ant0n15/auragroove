# Auragroove

A self-contained, portable **local AI music generator** for Windows — a slim,
fast UI on top of [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5),
tuned for 8 GB GPUs.

It exists to fix two pain points of the stock ACE-Step Gradio app:
1. **The memory leak** ([ACE-Step-1.5 #142](https://github.com/ace-step/ACE-Step-1.5/issues/142)) that crashes the machine after a few generations.
2. **Slow reloads** — the model reloading every run.

Auragroove runs generations through a **persistent worker process** that keeps the
(INT8-quantized) model **resident in VRAM** — so repeat runs are fast — while a
**memory watchdog** recycles the worker before the leak can pile up.

## Features
- **Text → music** with full control (caption, lyrics, BPM, key, steps, seed, …).
- **Genre templates** — one-click presets (Brazilian phonk, lo-fi, trap, techno, …); save your own.
- **Remix from reference audio** — drop a track to generate a cover/remix of it.
- **Think (LM)** — optional 5Hz LM to reason about structure/metadata.
- **Fine-tune your own style** — train a LoRA on your tracks (see below).
- **Up to 10 outputs** per run, real progress in the status box, settings embedded in each file.

## Requirements
- Windows 10/11, 64-bit
- **NVIDIA GPU** (8 GB VRAM is the tuned target; more is fine).
- ~25 GB free disk; internet for the first install

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
train_lora.bat      → fine-tune a LoRA on your own tracks (see finetune/)
finetune/           → dataset prep + LoRA training + merge scripts
make_release.ps1    → stage a clean copy (no venv) to move to another PC
```

## Fine-tune on your own music (LoRA)
Train a small adapter so generations lean toward your sound, then run it on the
fast turbo path. Drop your tracks in `finetune/dataset/` and double-click
**`train_lora.bat`** (it scaffolds a dataset file, then trains + bakes a ready
checkpoint you select in the **DiT Model** dropdown). Full guide: [`finetune/README.md`](finetune/README.md).

## Credits & licenses
- **Auragroove** wrapper code: MIT (see `LICENSE`).
- **Engine:** [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) by ACE Studio &
  StepFun — Apache-2.0 (`acestep_engine/LICENSE`).
- **Models** are downloaded separately and carry their own licenses (e.g. the DiT
  is Apache-2.0; ScragVAE and others have their own terms — check before commercial use).

Not affiliated with ACE Studio / StepFun. This is a community wrapper.
