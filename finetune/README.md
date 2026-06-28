# Fine-tuning AuraGroove (LoRA)

Train a small **LoRA adapter** on your own tracks so generations lean toward your
sound (e.g. your phonk style). LoRA keeps the base model frozen and trains a tiny
add-on — this fits an **8 GB GPU**; full fine-tuning does not.

> Expectations: ~20 tracks gives a **style** adapter (timbre/drums/vibe), not a
> model that reproduces specific songs. 50–100+ similar tracks works better. Keep
> rank low and epochs modest to avoid overfitting.

## 1. Put your tracks in `finetune/dataset/`
Drop your `.mp3/.wav/.flac` files there. Any sample rate is fine (auto-resampled).

For **vocals**, add text sidecars next to each track (same name):

```
mytrack.mp3
mytrack.txt           # lyrics for this track
mytrack.caption.txt   # optional: style description (else a default phonk caption)
```

## 2. Build the dataset file
Double-click **`train_lora.bat`** once. It scans the folder and writes
`finetune/dataset.json`, then stops so you can review/edit it. Or run manually:

```bat
acestep_engine\.venv\Scripts\python.exe finetune\prepare_dataset.py ^
    --audio-dir finetune\dataset --trigger agphonk
```

`--trigger agphonk` is your **style token**: it's prepended to every caption, so
later you can put `agphonk` in a prompt to invoke the style.

## 3. Train
Run **`train_lora.bat`** again (or directly):

```bat
acestep_engine\.venv\Scripts\python.exe finetune\run_finetune.py ^
    --dataset-json finetune\dataset.json --name myphonk ^
    --variant base --rank 16 --epochs 30
```

It will: download the **base** model the first time (several GB) → preprocess your
audio into tensors → train the LoRA. The adapter is saved to
`finetune/loras/myphonk/`.

## 4. Use it
Restart `run_auragroove.bat`, open **🎛 LoRA**, pick `myphonk`, set the strength,
and generate. (Selecting/changing a LoRA reloads the worker once.)

## Tuning & 8 GB tips
- **Out of memory?** Lower `--max-duration` (e.g. 30), keep `--rank 16`, base model.
- **Overfitting** (everything sounds identical / artefacts): fewer `--epochs`
  (15–25), lower `--rank` (8–16), or generate with a lower LoRA strength.
- **Underfitting** (no effect): more epochs, higher strength, or `--rank 32`.
- `--variant base` (50-step, higher quality) vs `--variant turbo` (8-step, fast).
- Re-run with `--skip-preprocess` to reuse cached tensors and just retrain.

## Notes
- Training runs on the **base** model; you can still generate with turbo at
  inference — but a LoRA trained on base is intended to be used on base.
- `finetune/dataset/`, `finetune/cache/`, and `finetune/loras/` are git-ignored.
