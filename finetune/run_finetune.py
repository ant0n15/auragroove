"""
LoRA fine-tune ACE-Step on your own tracks (Side-Step pipeline), end to end.

    ensure base model -> preprocess audio to tensors -> train LoRA adapter

Run via train_lora.bat, or directly:
    acestep_engine/.venv/Scripts/python.exe finetune/run_finetune.py \
        --dataset-json finetune/dataset.json --name myphonk --epochs 30 --rank 16

The trained adapter lands in finetune/loras/<name>/ and can be loaded in the
AuraGroove UI (LoRA dropdown).
"""
import os
import sys
import gc
import argparse
from pathlib import Path

for _s in (sys.stdout, sys.stderr):           # avoid charmap crashes on cp1253 consoles
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

APP_DIR = Path(__file__).resolve().parent.parent
ACESTEP_DIR = APP_DIR / "acestep_engine"
CKPT = ACESTEP_DIR / "checkpoints"
sys.path.insert(0, str(ACESTEP_DIR))
# Keep caches/estimates next to the project rather than the cwd.
os.environ.setdefault("ACESTEP_PROJECT_ROOT", str(ACESTEP_DIR))

_VARIANT_DIR = {"base": "acestep-v15-base", "turbo": "acestep-v15-turbo", "sft": "acestep-v15-sft"}


def _ensure_model(variant):
    name = _VARIANT_DIR[variant]
    if (CKPT / name).exists():
        print(f"[model] {name} present.")
        return
    print(f"[model] {name} missing -> downloading (several GB, first time only)...")
    from acestep.model_downloader import download_submodel, ensure_main_model
    if variant == "turbo":
        ok, msg = ensure_main_model(checkpoints_dir=CKPT)
    else:
        ok, msg = download_submodel(name, checkpoints_dir=CKPT)
    print(f"[model] {msg}")
    if not ok:
        raise SystemExit(f"Could not download {name}. Check your connection and retry.")


def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tune ACE-Step (Side-Step)")
    ap.add_argument("--dataset-json", default="finetune/dataset.json",
                    help="dataset.json from prepare_dataset.py")
    ap.add_argument("--audio-dir", default=None,
                    help="Alternative to --dataset-json: a folder of audio (filename=caption)")
    ap.add_argument("--name", default="myphonk_turbo", help="Name for the output LoRA")
    ap.add_argument("--variant", default="turbo", choices=["base", "turbo", "sft"],
                    help="Which model to train on (turbo recommended for 8GB; base "
                         "is corrupted by the INT8 it needs to fit)")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank (16-32; lower = less overfit)")
    ap.add_argument("--alpha", type=int, default=16, help="LoRA alpha (scaling = alpha/rank)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--epochs", type=int, default=30, help="Max epochs")
    ap.add_argument("--save-every", type=int, default=10, help="Checkpoint every N epochs")
    ap.add_argument("--max-duration", type=float, default=30.0,
                    help="Max seconds per clip in preprocessing (lower if you OOM)")
    ap.add_argument("--tensors", default="finetune/cache/tensors_turbo", help="Preprocessed tensor dir")
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="Reuse existing tensors in --tensors (skip the encode step)")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] No CUDA GPU detected — training on CPU will be extremely slow.")

    dataset_json = str(Path(args.dataset_json).resolve()) if args.dataset_json and not args.audio_dir else None
    audio_dir = str(Path(args.audio_dir).resolve()) if args.audio_dir else None
    tensors_dir = Path(args.tensors).resolve()
    out_dir = (APP_DIR / "finetune" / "loras" / args.name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_model(args.variant)

    # ---- Pass 1+2: preprocess audio -> .pt tensors ----
    if args.skip_preprocess and tensors_dir.exists() and any(tensors_dir.glob("*.pt")):
        print(f"[preprocess] reusing tensors in {tensors_dir}")
    else:
        from acestep.training_v2.preprocess import preprocess_audio_files
        print(f"[preprocess] encoding audio -> {tensors_dir} (variant={args.variant})")
        res = preprocess_audio_files(
            audio_dir=audio_dir,
            output_dir=str(tensors_dir),
            checkpoint_dir=str(CKPT),
            variant=args.variant,
            max_duration=args.max_duration,
            dataset_json=dataset_json,
            device=device,
            precision="auto",
            progress_callback=lambda c, t, m: print(f"[preprocess {c}/{t}] {m}"),
        )
        print(f"[preprocess] {res}")
        if not res or res.get("processed", 0) == 0:
            raise SystemExit("Preprocessing produced no tensors — check your dataset/audio paths.")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Train the LoRA adapter ----
    from acestep.training_v2.model_loader import load_decoder_for_training
    from acestep.training_v2.configs import LoRAConfigV2, TrainingConfigV2
    from acestep.training_v2.trainer_fixed import FixedLoRATrainer

    print(f"[train] loading {args.variant} model for training on {device}...")
    model = load_decoder_for_training(str(CKPT), variant=args.variant, device=device, precision="bf16")

    lora_cfg = LoRAConfigV2(r=args.rank, alpha=args.alpha, dropout=args.dropout)
    train_cfg = TrainingConfigV2(
        learning_rate=args.lr,
        batch_size=1,
        max_epochs=args.epochs,
        save_every_n_epochs=args.save_every,
        output_dir=str(out_dir),
        dataset_dir=str(tensors_dir),
        device=device,
        adapter_type="lora",
        model_variant=args.variant,
        gradient_checkpointing=True,
        seed=42,
        num_workers=0,
    )

    print(f"[train] rank={args.rank} alpha={args.alpha} lr={args.lr} epochs={args.epochs}")
    trainer = FixedLoRATrainer(model, lora_cfg, train_cfg)
    for upd in trainer.train():
        # updates are (global_step, loss, message)-like with a .status_message
        msg = getattr(upd, "status_message", None) or getattr(upd, "message", None) or str(upd)
        print(f"[train] {msg}")

    print(f"\n[DONE] LoRA adapter saved under: {out_dir}")
    print("   Load it in AuraGroove via the LoRA dropdown (restart the app to pick it up).")


if __name__ == "__main__":
    main()
