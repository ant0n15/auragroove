"""
Bake a trained LoRA into a standalone checkpoint so it runs on the normal
(fast, proven) model path -- no runtime LoRA, no quant/offload surprises.

    python finetune/merge_lora.py --lora finetune/loras/myphonk_turbo/final \
        --src-variant turbo --out-name acestep-v15-turbo-myphonk

Result: checkpoints/<out-name>/ -> appears in the AuraGroove "DiT Model" dropdown.
Because the name contains "turbo", the worker runs it with the same INT8 path as
stock turbo (8 steps), with your style merged into the weights.
"""
import os
import sys
import shutil
import argparse
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

APP_DIR = Path(__file__).resolve().parent.parent
ACESTEP_DIR = APP_DIR / "acestep_engine"
CKPT = ACESTEP_DIR / "checkpoints"
sys.path.insert(0, str(ACESTEP_DIR))
os.environ.setdefault("ACESTEP_PROJECT_ROOT", str(ACESTEP_DIR))

_VARIANT_DIR = {"turbo": "acestep-v15-turbo", "base": "acestep-v15-base", "sft": "acestep-v15-sft"}
# Files copied verbatim from the source checkpoint (everything except the weights).
_COPY = ["config.json", "configuration_acestep_v15.py", "silence_latent.pt"]


def main():
    ap = argparse.ArgumentParser(description="Merge a LoRA into a standalone checkpoint")
    ap.add_argument("--lora", required=True, help="PEFT adapter dir (…/<name>/final)")
    ap.add_argument("--src-variant", default="turbo", choices=["turbo", "base", "sft"])
    ap.add_argument("--out-name", default="acestep-v15-turbo-myphonk",
                    help="New checkpoint dir name (keep 'turbo' in it for the fast path)")
    args = ap.parse_args()

    src_dir = CKPT / _VARIANT_DIR[args.src_variant]
    if not src_dir.exists():
        raise SystemExit(f"Source model not found: {src_dir}")
    lora_dir = Path(args.lora).resolve()
    if not lora_dir.exists():
        raise SystemExit(f"LoRA adapter not found: {lora_dir}")
    out_dir = CKPT / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from acestep.training_v2.model_loader import load_decoder_for_training
    from acestep.training.lora_checkpoint import load_lora_weights
    from acestep.training.lora_utils import merge_lora_weights
    from safetensors.torch import save_model

    print(f"[merge] loading {args.src_variant} on CPU (bf16)...")
    model = load_decoder_for_training(str(CKPT), variant=args.src_variant,
                                      device="cpu", precision="bf16")
    print(f"[merge] applying LoRA: {lora_dir}")
    model = load_lora_weights(model, str(lora_dir))
    print("[merge] merging LoRA into base weights...")
    model = merge_lora_weights(model)

    # copy non-weight files (config, code, silence latent) + the variant's modeling file
    for fn in _COPY:
        src = src_dir / fn
        if src.exists():
            shutil.copy2(src, out_dir / fn)
    for mod in src_dir.glob("modeling_*.py"):
        shutil.copy2(mod, out_dir / mod.name)

    out_weights = out_dir / "model.safetensors"
    print(f"[merge] saving merged weights -> {out_weights}")
    save_model(model, str(out_weights))

    print(f"\n[DONE] Merged checkpoint: {out_dir}")
    print(f"   Pick '{args.out_name}' in the AuraGroove DiT Model dropdown "
          f"(LoRA = none). Restart the app to see it.")


if __name__ == "__main__":
    main()
