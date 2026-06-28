"""
Auto-caption your training tracks with the ACE-Step LM, then write a rich
dataset.json (distinct caption + lyrics + bpm/key per track). Varied, accurate
labels are what make a LoRA learn a *recognizable* style instead of an average.

    python finetune/autocaption.py --audio-dir finetune/dataset --trigger agphonk

For each track: audio -> semantic codes (DiT/VAE) -> understanding (LM) ->
caption/lyrics/bpm/keyscale/language/timesignature. A same-name .caption.txt or
.txt sidecar, if present, overrides the auto value. Run this BEFORE train_lora.bat.
"""
import os
import sys
import json
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

AUDIO_EXTS = (".mp3", ".wav", ".flac")
DEFAULT_CAPTION = ("brazilian phonk, aggressive bass-boosted funk, distorted 808 bass, "
                   "cowbell melody, chopped vocals, punchy drums")


def _sidecar(p):
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(description="LM auto-caption tracks -> dataset.json")
    ap.add_argument("--audio-dir", default="finetune/dataset")
    ap.add_argument("--out", default="finetune/dataset.json")
    ap.add_argument("--trigger", default="agphonk")
    ap.add_argument("--genre", default="phonk")
    ap.add_argument("--model", default="acestep-v15-turbo")
    ap.add_argument("--lm", default="acestep-5Hz-lm-0.6B")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise SystemExit(f"No audio in {audio_dir}")

    import torch
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import understand_music
    from acestep.gpu_config import get_gpu_config, set_global_gpu_config

    set_global_gpu_config(get_gpu_config())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[autocaption] loading {args.model} + LM {args.lm} ...")
    dit = AceStepHandler()
    dit.initialize_service(
        project_root=str(ACESTEP_DIR), config_path=args.model, device=device,
        use_flash_attention=dit.is_flash_attention_available(device),
        offload_to_cpu=False, offload_dit_to_cpu=False,
        quantization="int8_weight_only", vae_checkpoint=None,
    )
    llm = LLMHandler()
    llm.initialize(checkpoint_dir=str(CKPT), lm_model_path=args.lm, backend="pt",
                   device=device, offload_to_cpu=False, dtype=None)

    samples = []
    for i, af in enumerate(files, 1):
        cap_override = _sidecar(af.with_suffix(".caption.txt"))
        lyr_override = _sidecar(af.with_suffix(".txt"))
        caption, lyrics, bpm, keyscale, lang, tsig = "", "", None, "", "", ""
        try:
            codes = dit.convert_src_audio_to_codes(str(af))
            if isinstance(codes, str) and codes.startswith("❌"):
                print(f"  [{i}/{len(files)}] {af.name}: codes failed -> default caption")
            else:
                r = understand_music(llm, codes)
                if getattr(r, "success", False):
                    caption = (r.caption or "").strip()
                    lyrics = (r.lyrics or "").strip()
                    bpm = r.bpm
                    keyscale = (r.keyscale or "").strip()
                    lang = (r.language or "").strip()
                    tsig = (r.timesignature or "").strip()
        except Exception as e:
            print(f"  [{i}/{len(files)}] {af.name}: understand error: {e!r}")

        caption = cap_override or caption or DEFAULT_CAPTION
        lyrics = lyr_override or lyrics or "[Verse]\nla la la, la la la"
        samples.append({
            "audio_path": str(af), "filename": af.name,
            "caption": caption, "lyrics": lyrics, "genre": args.genre,
            "bpm": bpm, "keyscale": keyscale,
            "language": lang, "timesignature": tsig,
        })
        print(f"  [{i}/{len(files)}] {af.name}: {caption[:70]}")

    dataset = {
        "custom_tag": args.trigger, "tag_position": "prepend",
        "genre_ratio": 0, "samples": samples,
    }
    out = Path(args.out).resolve()
    out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] Wrote {out}  ({len(samples)} samples with LM captions). "
          f"Now run train_lora.bat to train on the richer labels.")


if __name__ == "__main__":
    main()
