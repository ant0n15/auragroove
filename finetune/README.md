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
    --dataset-json finetune\dataset.json --name myphonk_turbo ^
    --variant turbo --tensors finetune\cache\tensors_turbo ^
    --rank 16 --epochs 30 --max-duration 30
```

It preprocesses your audio into tensors → trains the LoRA on **turbo** (the
reliable, fast path on 8 GB). The adapter is saved to
`finetune/loras/myphonk_turbo/`.

> Why turbo, not base? On 8 GB the base model must be INT8-quantized to fit, and
> INT8 corrupts the non-distilled base (garbled output). Turbo is built for INT8,
> and runtime LoRAs apply cleanly on top of it.

## 4. Use it
Restart `run_auragroove.bat`. Keep **DiT Model = `acestep-v15-turbo`** (stock),
open **🎛 LoRA**, pick **`myphonk_turbo`**, and set the strength slider. Put your
trigger word (`agphonk`) in the caption and generate. (Changing the LoRA reloads
the worker once.) Runtime LoRAs apply fine on the fast INT8 turbo path.

### Optional: bake it into a standalone model
If you'd rather have a single model with the style baked in (e.g. to share/ship),
merge the adapter into a checkpoint:

```bat
acestep_engine\.venv\Scripts\python.exe finetune\merge_lora.py ^
    --lora finetune\loras\myphonk_turbo\final --src-variant turbo ^
    --out-name acestep-v15-turbo-myphonk
```

Then pick `acestep-v15-turbo-myphonk` in **DiT Model** with **LoRA = none**.
(Don't use both at once or the style is applied twice.)

## Tuning & 8 GB tips
- **Out of memory?** Lower `--max-duration` (e.g. 20), keep `--rank 16`.
- **Overfitting** (everything sounds identical / artefacts): fewer `--epochs`
  (15–25), lower `--rank` (8–16), or drop the LoRA **strength** at generation.
- **Underfitting** (no effect): more epochs, higher strength, or `--rank 32`.
- Re-run with `--skip-preprocess` to reuse cached tensors and just retrain.

## Notes
- Train on **turbo** and use it via the **LoRA dropdown** on the stock turbo
  model — runtime LoRAs apply fine on turbo's INT8 path (fast).
- Base (`--variant base`) is higher quality in theory but on 8 GB its required
  INT8 quantization corrupts output — avoid it on this card.
- `finetune/dataset/`, `finetune/cache/`, and `finetune/loras/` are git-ignored.
