"""
Build a Side-Step `dataset.json` from a folder of training tracks.

Usage:
    python finetune/prepare_dataset.py --audio-dir finetune/dataset --trigger agphonk

For each audio file (.mp3/.wav/.flac) in --audio-dir it creates one sample.
Per-track text can be supplied via sidecar files next to the audio (same stem):

    mytrack.mp3
    mytrack.txt           -> lyrics for this track (recommended for vocals)
    mytrack.caption.txt   -> caption (style description) for this track

If a sidecar is missing, sensible defaults are used (and a note is printed).
A dataset-level `custom_tag` (trigger word) is prepended to every caption so you
can later invoke the style by putting that word in your prompt.
"""
import argparse
import json
from pathlib import Path

AUDIO_EXTS = (".mp3", ".wav", ".flac")

DEFAULT_CAPTION = ("brazilian phonk, aggressive bass-boosted funk, distorted 808 bass, "
                   "cowbell melody, chopped vocals, punchy drums")

# Generic vocal placeholder used when a track has no <stem>.txt lyrics. Marks the
# sample as "has vocals" (rather than instrumental) without inventing real words.
DEFAULT_VOCAL_LYRICS = "[Verse]\nla la la, la la la\nyeah, uh, la la la"


def _read_sidecar(path: Path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(description="Build a Side-Step dataset.json")
    ap.add_argument("--audio-dir", default="finetune/dataset",
                    help="Folder containing the training audio (and optional sidecars)")
    ap.add_argument("--out", default="finetune/dataset.json", help="Output JSON path")
    ap.add_argument("--trigger", default="agphonk",
                    help="Trigger word prepended to every caption (your style token)")
    ap.add_argument("--genre", default="phonk", help="Genre tag stored per sample")
    ap.add_argument("--default-lyrics", default=DEFAULT_VOCAL_LYRICS,
                    help="Lyrics used when a track has no <stem>.txt sidecar "
                         "(default: a generic vocal placeholder; pass '[Instrumental]' for no vocals)")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    if not audio_dir.is_dir():
        raise SystemExit(f"Audio dir not found: {audio_dir}")

    files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise SystemExit(f"No audio files ({', '.join(AUDIO_EXTS)}) found in {audio_dir}")

    samples, no_lyrics = [], 0
    for af in files:
        cap = _read_sidecar(af.with_suffix(".caption.txt")) or DEFAULT_CAPTION
        lyr = _read_sidecar(af.with_suffix(".txt"))
        if not lyr:
            lyr = args.default_lyrics
            no_lyrics += 1
        samples.append({
            "audio_path": str(af),
            "filename": af.name,
            "caption": cap,
            "lyrics": lyr,
            "genre": args.genre,
            "bpm": None,
            "keyscale": "",
        })

    dataset = {
        "custom_tag": args.trigger,   # prepended to every caption -> your style token
        "tag_position": "prepend",
        "genre_ratio": 0,
        "samples": samples,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {out}  ({len(samples)} samples, trigger='{args.trigger}')")
    if no_lyrics:
        print(f"  NOTE: {no_lyrics}/{len(samples)} tracks had no <stem>.txt lyrics "
              f"-> defaulted to '{args.default_lyrics}'.")
        print("  For vocals, add a <stem>.txt next to each track with its lyrics, then re-run.")


if __name__ == "__main__":
    main()
