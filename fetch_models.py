"""
Download the model weights Auragroove needs (idempotent — skips what's present).

Pulls into acestep_engine/checkpoints:
  - main model: DiT turbo + VAE + Qwen3 text encoder
  - 5Hz LM 0.6B (resident "Think")
  - ScragVAE (default VAE)

Run automatically by install.bat; safe to run again any time.
"""
import os
import sys
from pathlib import Path

ACESTEP_DIR = Path(__file__).resolve().parent / "acestep_engine"
sys.path.insert(0, str(ACESTEP_DIR))
CKPT = ACESTEP_DIR / "checkpoints"

from acestep.model_downloader import (  # noqa: E402
    ensure_main_model, ensure_lm_model, ensure_vae_model,
)


def _step(label, fn, *args):
    print(f"\n==> {label}")
    try:
        ok, msg = fn(*args, checkpoints_dir=CKPT)
        print(("   OK: " if ok else "   FAILED: ") + str(msg))
        return ok
    except Exception as e:
        print(f"   ERROR: {e!r}")
        return False


def main():
    CKPT.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints dir: {CKPT}")
    ok = True
    ok &= _step("Main model (DiT turbo + VAE + text encoder)", ensure_main_model)
    ok &= _step("5Hz LM 0.6B", ensure_lm_model, "acestep-5Hz-lm-0.6B")
    ok &= _step("ScragVAE", ensure_vae_model, "scragvae")
    print("\nAll models present." if ok else "\nSome downloads failed — see above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
