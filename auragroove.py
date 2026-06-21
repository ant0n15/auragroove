"""
Auragroove — local ACE-Step 1.5 music generator (persistent-worker edition).

Generations run through a PERSISTENT worker (worker.py) that loads the model
ONCE and stays resident -> no reload between runs (fast). Think ON additionally
keeps the selected 5Hz LM resident. A watchdog recycles the worker (kill +
respawn) when its RAM grows too far or system free RAM gets low, to bound the
known memory leak (#142).

Outputs go to `auragroove_outputs/audio/`. Each track's settings are embedded
directly in its file metadata (ID3 for mp3, tags for wav/flac) -- no sidecar.
The Load button reads them back from any generated track. Timings are shown
in the status box after each run.

Run:
    .venv/Scripts/python.exe auragroove.py
(or double-click run_auragroove.bat)
"""

import os
import sys
import gc
import glob
import json
import time
import shutil
import atexit
import threading
import subprocess
from pathlib import Path

import gradio as gr

try:
    import toml
except ImportError:
    toml = None

try:
    import psutil
except ImportError:
    psutil = None

APP_DIR = Path(__file__).resolve().parent                       # portable home for this UI
# Self-contained engine bundled in this project: its own venv, the `acestep`
# package, the model checkpoints, and the patched cli.py. No Pinokio needed.
ACESTEP_DIR = APP_DIR / "acestep_engine"
CKPT_DIR = ACESTEP_DIR / "checkpoints"
VENV_PY = ACESTEP_DIR / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable
WORKER = str(APP_DIR / "worker.py")                             # worker lives here, runs with the bundled venv
OUT_ROOT = APP_DIR / "auragroove_outputs"                        # outputs land here
OUT_ROOT.mkdir(exist_ok=True)

AUDIO_EXTS = ("*.mp3", "*.wav", "*.flac")

# --- Watchdog tuning (persistent worker) -------------------------------------
# Growth-based recycling: the FIRST generation on a worker only MEASURES its
# resident memory (no release). From the 2nd generation on, the worker is
# recycled only once its RAM has grown more than this many GB ABOVE that
# first-gen baseline (i.e. the leak has actually accumulated meaningfully).
RECYCLE_GROWTH_GB = 6.0
# Hard safety net: recycle if system free RAM gets critically low regardless.
EMERGENCY_FREE_RAM_GB = 1.2
# Safety cap: recycle after this many generations no matter what.
RECYCLE_AFTER_N_GENS = 50

# Keep the model fully resident instead of CPU-offloading it.
# offload_to_cpu=True makes the handler load/free model components per
# generation ("one-at-a-time"), which RELEASES memory after each run and
# defeats keeping the model resident. With offload OFF the model stays loaded
# on the GPU so reuse is instant -- but it must fit in VRAM (~6.7 GB for turbo
# + activations). If you hit CUDA out-of-memory (esp. long durations or
# batch > 1), set this back to True (you'll trade resident speed for fit).
WORKER_OFFLOAD_TO_CPU = False

# Quantize the resident model so it leaves enough VRAM to actually generate.
# Full bf16 resident is ~7.3 GB on 8 GB -> only ~0.8 GB free -> generations fail
# the VRAM pre-flight. INT8 drops it to ~5.6 GB resident (~2.3 GB free), which
# fits comfortably AND loads faster. Options: "none", "int8_weight_only",
# "fp8_weight_only", "w8a8_dynamic". Use "none" only if you turn offload back on.
WORKER_QUANTIZATION = "int8_weight_only"

# Resident LM for "Think" so it works through the fast worker instead of the slow
# one-shot path. The 0.6B LM fits alongside the INT8 DiT on 8GB (it's tight ~1.2GB
# free, relies on tiled VAE decode). Think runs then stay resident: ~10x faster on
# repeat runs vs the one-shot path. Set to "none" to disable resident Think (Think
# falls back to the one-shot subprocess, freeing max VRAM for pure-DiT runs).
# The 1.7B LM is too big to fit resident with the DiT -- use 0.6B here.
WORKER_LM_MODEL = "acestep-5Hz-lm-0.6B"

_N_SETTINGS = 33


# ── helpers ──────────────────────────────────────────────────────────────────

def _available_models():
    ckpt = CKPT_DIR
    found = []
    if ckpt.exists():
        for p in sorted(ckpt.iterdir()):
            if p.is_dir() and p.name.lower().startswith("acestep-v15"):
                found.append(p.name)
    if not found:
        found = ["acestep-v15-turbo"]
    found.sort(key=lambda n: (0 if "turbo" in n.lower() else 1, n))
    return found


def _available_lm_models():
    ckpt = CKPT_DIR
    found = []
    if ckpt.exists():
        for p in sorted(ckpt.iterdir()):
            if p.is_dir() and "lm" in p.name.lower():
                found.append(p.name)
    return found or ["acestep-5Hz-lm-1.7B"]


def _available_vaes():
    """'official' (= checkpoints/vae) plus any extra VAE dirs (e.g. scragvae)."""
    ckpt = CKPT_DIR
    extra = []
    if ckpt.exists():
        for p in sorted(ckpt.iterdir()):
            if p.is_dir() and "vae" in p.name.lower() and p.name.lower() != "vae":
                extra.append(p.name)
    return ["official"] + extra


def _write_config(cfg: dict, path: Path):
    if toml is not None:
        with open(path, "w", encoding="utf-8") as f:
            toml.dump(cfg, f)
        return

    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'

    with open(path, "w", encoding="utf-8") as f:
        for k, v in cfg.items():
            if v is None:
                continue
            f.write(f"{k} = {fmt(v)}\n")


def _all_audio(folder: Path):
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(str(folder / "**" / ext), recursive=True))
    return sorted(files, key=os.path.getmtime)


def _read_seed(folder: Path):
    for m in glob.glob(str(folder / "**" / "*.json"), recursive=True):
        try:
            with open(m, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in ("seed", "seeds", "actual_seed"):
                if key in data and data[key] not in (None, "", -1):
                    return data[key]
        except Exception:
            continue
    return None


def _free_ram_gb():
    if psutil is None:
        return None
    try:
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return None


def _embed_settings(audio_path, cfg):
    """Embed the (trimmed) settings JSON into the audio file's own metadata."""
    try:
        payload = json.dumps(cfg, ensure_ascii=False)
        ext = Path(audio_path).suffix.lower()
        if ext == ".mp3":
            from mutagen.id3 import ID3, TXXX, ID3NoHeaderError
            try:
                tags = ID3(audio_path)
            except ID3NoHeaderError:
                tags = ID3()
            tags.delall("TXXX:auragroove")
            tags.add(TXXX(encoding=3, desc="auragroove", text=payload))
            tags.save(audio_path)
        elif ext == ".flac":
            from mutagen.flac import FLAC
            f = FLAC(audio_path)
            f["auragroove"] = payload
            f.save()
        elif ext == ".wav":
            from mutagen.wave import WAVE
            from mutagen.id3 import TXXX
            f = WAVE(audio_path)
            if f.tags is None:
                f.add_tags()
            f.tags.delall("TXXX:auragroove")
            f.tags.add(TXXX(encoding=3, desc="auragroove", text=payload))
            f.save()
    except Exception:
        pass  # never fail a generation over tagging


def _read_embedded_settings(path):
    """Read settings JSON embedded in an audio file; return dict or None."""
    ext = Path(path).suffix.lower()
    try:
        if ext == ".mp3":
            from mutagen.id3 import ID3
            for fr in ID3(path).getall("TXXX"):
                if fr.desc == "auragroove":
                    return json.loads(str(fr.text[0]))
        elif ext == ".flac":
            from mutagen.flac import FLAC
            f = FLAC(path)
            if "auragroove" in f:
                return json.loads(f["auragroove"][0])
        elif ext == ".wav":
            from mutagen.wave import WAVE
            f = WAVE(path)
            if f.tags is not None:
                for fr in f.tags.getall("TXXX"):
                    if fr.desc == "auragroove":
                        return json.loads(str(fr.text[0]))
    except Exception:
        return None
    return None


def _open_outputs_folder():
    OUT_ROOT.mkdir(exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(str(OUT_ROOT))  # noqa: S606 (local, trusted path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(OUT_ROOT)])
        else:
            subprocess.Popen(["xdg-open", str(OUT_ROOT)])
    except Exception:
        pass


# ── persistent worker manager ────────────────────────────────────────────────

_worker = None
_worker_model = None
_worker_offload = None
_worker_vae = None
_worker_lm = None
_worker_gens = 0
_worker_baseline_rss = None  # worker RSS (GB) measured after its first generation
_worker_lock = threading.Lock()

# Tracks the subprocess currently generating (worker or one-shot) so the Stop
# button can terminate it. Guarded by its own lock so Stop never has to wait on
# _worker_lock (which the running generation holds).
_active_proc = None
_active_lock = threading.Lock()
_stop_requested = False


def _set_active(proc):
    global _active_proc
    with _active_lock:
        _active_proc = proc


def stop_generation():
    """Interrupt the in-flight generation by killing its process."""
    global _stop_requested
    _stop_requested = True
    with _active_lock:
        p = _active_proc
    if p is None or p.poll() is not None:
        return "Nothing is generating right now."
    try:
        p.kill()
    except Exception:
        pass
    return "🛑 Stop requested — terminating the current generation…"


def _worker_rss_gb():
    if psutil is None or _worker is None:
        return None
    try:
        return psutil.Process(_worker.pid).memory_info().rss / 1e9
    except Exception:
        return None


def _read_event(proc):
    """Read worker stdout until a @@-prefixed JSON protocol line; return dict."""
    while True:
        line = proc.stdout.readline()
        if line == "":
            return {"event": "error", "msg": "worker exited unexpectedly"}
        line = line.strip()
        if line.startswith("@@"):
            try:
                return json.loads(line[2:])
            except Exception:
                return {"event": "error", "msg": "bad protocol line"}
        # ignore stray stdout


def _kill_worker():
    global _worker, _worker_model, _worker_offload, _worker_vae, _worker_lm, _worker_gens, _worker_baseline_rss
    if _worker is not None:
        try:
            if _worker.poll() is None:
                try:
                    _worker.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    _worker.stdin.flush()
                except Exception:
                    pass
                _worker.terminate()
                try:
                    _worker.wait(timeout=10)
                except Exception:
                    _worker.kill()
        except Exception:
            pass
    _worker = None
    _worker_model = None
    _worker_offload = None
    _worker_vae = None
    _worker_lm = None
    _worker_gens = 0
    _worker_baseline_rss = None


atexit.register(_kill_worker)


def _ensure_worker(model, vae="official", lm="none", offload=True):
    """Spawn/respawn the worker if needed. Returns load_time if a (re)spawn
    happened this call, else None (worker was already resident)."""
    global _worker, _worker_model, _worker_offload, _worker_vae, _worker_lm, _worker_gens, _worker_baseline_rss
    need = (
        _worker is None
        or _worker.poll() is not None
        or _worker_model != model
        or _worker_offload != offload
        or _worker_vae != vae
        or _worker_lm != lm
    )
    if not need:
        return None
    _kill_worker()
    cmd = [PYTHON, WORKER, "--model", model, "--offload", "1" if offload else "0",
           "--quant", WORKER_QUANTIZATION, "--lm", lm or "none",
           "--vae", vae or "official"]
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(ACESTEP_DIR),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=None, text=True, bufsize=1,
    )
    ev = _read_event(proc)
    if ev.get("event") != "ready":
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError(f"worker failed to load: {ev.get('msg')}")
    _worker = proc
    _worker_model = model
    _worker_offload = offload
    _worker_vae = vae
    _worker_lm = lm
    _worker_gens = 0
    _worker_baseline_rss = None
    return ev.get("load_time", time.time() - t0)


def _worker_generate(cfg, model, vae="official", lm="none"):
    global _worker, _worker_gens, _worker_baseline_rss
    with _worker_lock:
        try:
            load_time = _ensure_worker(model, vae=vae, lm=lm, offload=WORKER_OFFLOAD_TO_CPU)
        except Exception as e:
            return {"ok": False, "msg": str(e)}
        try:
            _worker.stdin.write(json.dumps(cfg) + "\n")
            _worker.stdin.flush()
        except Exception as e:
            _kill_worker()
            return {"ok": False, "msg": f"failed to send request: {e}"}

        _set_active(_worker)
        ev = _read_event(_worker)
        _set_active(None)
        _worker_gens += 1
        rss = _worker_rss_gb()
        free = _free_ram_gb()

        if ev.get("event") != "result":
            _kill_worker()  # recycle on error
            return {"ok": False, "msg": ev.get("msg", "worker error"),
                    "trace": ev.get("trace"), "load_time": load_time}

        # First generation on this worker: only MEASURE the baseline, never release.
        if _worker_baseline_rss is None:
            _worker_baseline_rss = rss

        growth = None
        if rss is not None and _worker_baseline_rss is not None:
            growth = rss - _worker_baseline_rss

        recycled = False
        reason = ""
        # NEVER release on the first generation -- it only establishes the
        # baseline. All release checks apply from the 2nd generation onward.
        if _worker_gens >= 2:
            if growth is not None and growth > RECYCLE_GROWTH_GB:
                recycled, reason = True, f"growth +{growth:.1f}GB > {RECYCLE_GROWTH_GB}GB"
            elif free is not None and free < EMERGENCY_FREE_RAM_GB:
                recycled, reason = True, f"free {free:.1f}GB < {EMERGENCY_FREE_RAM_GB}GB (emergency)"
            elif _worker_gens >= RECYCLE_AFTER_N_GENS:
                recycled, reason = True, f"{_worker_gens} gens (safety cap)"

        gens_done = _worker_gens
        baseline = _worker_baseline_rss
        if recycled:
            _kill_worker()

        return {
            "ok": True,
            "files": ev.get("files", []),
            "gen_time": ev.get("gen_time"),
            "load_time": load_time,
            "rss": rss,
            "free": free,
            "baseline": baseline,
            "growth": growth,
            "recycled": recycled,
            "reason": reason,
            "gens": gens_done,
        }


def _worker_generate_n(cfg, model, vae, lm, n, run_dir):
    """Run N sequential single-item generations (batch>1 won't fit in VRAM next
    to the resident model on 8GB, so we loop instead). Aggregates the results."""
    files, load_time, gen_time, last = [], None, 0.0, None
    for k in range(n):
        sub = dict(cfg)
        sub["batch_size"] = 1
        sub["save_dir"] = str(run_dir / f"item_{k}")
        r = _worker_generate(sub, model, vae=vae, lm=lm)
        if not r.get("ok"):
            if files:
                break  # partial success: keep what we have
            return r
        files.extend(r.get("files", []))
        if load_time is None:
            load_time = r.get("load_time")
        gen_time += (r.get("gen_time") or 0.0)
        last = r
        if _stop_requested:
            break
    out = dict(last or {})
    out.update({"ok": bool(files), "files": files,
                "load_time": load_time, "gen_time": gen_time})
    return out


def reset_worker():
    """Manually recycle the worker to free its RAM immediately."""
    with _worker_lock:
        _kill_worker()
    free = _free_ram_gb()
    extra = f" System free RAM: {free:.1f} GB." if free is not None else ""
    return f"♻️ Worker stopped. Next generation will load the model fresh.{extra}"


def _on_generate_start():
    """Disable the button while a generation runs (progress bar shows status)."""
    return gr.update(value="Generating…", interactive=False)


# ── main generate ────────────────────────────────────────────────────────────

MAX_OUTPUTS = 10


def _audio_outputs(files):
    """Build value/visibility updates for the result players (one per output)."""
    ups = []
    for i in range(MAX_OUTPUTS):
        if i < len(files):
            ups.append(gr.update(value=files[i], visible=True))
        else:
            ups.append(gr.update(value=None, visible=(i == 0)))
    return ups


def _build_cfg(
    caption, lyrics, instrumental, duration, auto_duration, steps, seed, batch_size,
    audio_format, model, bpm, keyscale, timesignature, vocal_language,
    think, lm_model, lm_temperature, lm_top_k, lm_top_p, lm_cfg_scale,
    lm_negative_prompt, guidance_scale, shift, infer_method, sampler_mode,
    use_adg, cfg_interval_start, cfg_interval_end,
    velocity_norm_threshold, velocity_ema_factor,
    cot_metas, cot_language, vae,
):
    """Build the settings dict from the UI values (no save_dir)."""
    caption = (caption or "").strip()
    lyrics = (lyrics or "").strip()
    if instrumental:
        lyrics = "[Instrumental]"
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = -1
    cfg = {
        "task_type": "text2music",
        "config_path": model,
        "caption": caption,
        "lyrics": lyrics if lyrics else None,
        "instrumental": bool(instrumental),
        "duration": -1.0 if auto_duration else float(duration),
        "inference_steps": int(steps),
        "seed": seed,
        "use_random_seed": seed < 0,
        "batch_size": int(batch_size),
        "thinking": bool(think),
        "use_cot_metas": bool(think and cot_metas),
        "use_cot_caption": False,
        "use_cot_lyrics": False,
        "use_cot_language": bool(think and cot_language),
        "use_constrained_decoding": False,
        "sample_mode": False,
        "use_format": False,
        "lm_temperature": float(lm_temperature),
        "lm_top_k": int(lm_top_k),
        "lm_top_p": float(lm_top_p),
        "lm_cfg_scale": float(lm_cfg_scale),
        "guidance_scale": float(guidance_scale),
        "shift": float(shift),
        "infer_method": (infer_method or "ode").strip(),
        "sampler_mode": (sampler_mode or "euler").strip(),
        "use_adg": bool(use_adg),
        "cfg_interval_start": float(cfg_interval_start),
        "cfg_interval_end": float(cfg_interval_end),
        "velocity_norm_threshold": float(velocity_norm_threshold),
        "velocity_ema_factor": float(velocity_ema_factor),
        "offload_to_cpu": True,
        "offload_dit_to_cpu": False,
        "device": "auto",
        "backend": "pt",
        "audio_format": audio_format,
        "vae": vae,
    }
    if think:
        cfg["lm_model_path"] = lm_model
        if (lm_negative_prompt or "").strip():
            cfg["lm_negative_prompt"] = lm_negative_prompt.strip()
    if bpm and int(bpm) > 0:
        cfg["bpm"] = int(bpm)
    if (keyscale or "").strip():
        cfg["keyscale"] = keyscale.strip()
    if (timesignature or "").strip():
        cfg["timesignature"] = timesignature.strip()
    if vocal_language and not instrumental:
        cfg["vocal_language"] = vocal_language.strip()
    return cfg


def save_template(name, *values):
    """Save the current UI values as a named template; refresh the dropdown."""
    name = (name or "").strip()
    if not name:
        return gr.update(), "❌ Enter a template name first. Use an existing name to overwrite."
    safe = "".join(c for c in name if c.isalnum() or c in " -_").strip()
    if not safe:
        return gr.update(), "❌ Invalid template name."
    cfg = _build_cfg(*values)
    cfg.pop("save_dir", None)
    existed = (TEMPLATES_DIR / f"{safe}.toml").exists()
    try:
        _write_config(cfg, TEMPLATES_DIR / f"{safe}.toml")
    except Exception as e:
        return gr.update(), f"❌ Could not save: {e}"
    msg = (f"💾 Updated existing template '{safe}'." if existed
           else f"💾 Saved new template '{safe}'.")
    # Refresh choices only (don't change the selected value) so this doesn't
    # trigger the dropdown's change handler (which would clear the name field).
    return gr.update(choices=_list_templates()), msg


def generate(
    caption, lyrics, instrumental, duration, auto_duration, steps, seed, batch_size,
    audio_format, model, bpm, keyscale, timesignature, vocal_language,
    think, lm_model, lm_temperature, lm_top_k, lm_top_p, lm_cfg_scale,
    lm_negative_prompt, guidance_scale, shift, infer_method, sampler_mode,
    use_adg, cfg_interval_start, cfg_interval_end,
    velocity_norm_threshold, velocity_ema_factor,
    cot_metas, cot_language, vae,
    progress=gr.Progress(),
):
    global _stop_requested
    _stop_requested = False
    caption = (caption or "").strip()
    lyrics = (lyrics or "").strip()
    if instrumental:
        lyrics = "[Instrumental]"
    if not caption and (not lyrics or lyrics == "[Instrumental]"):
        return (*_audio_outputs([]), "❌ Please enter a caption (and/or lyrics).")

    run_dir = OUT_ROOT / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.toml"

    cfg = _build_cfg(
        caption, lyrics, instrumental, duration, auto_duration, steps, seed, batch_size,
        audio_format, model, bpm, keyscale, timesignature, vocal_language,
        think, lm_model, lm_temperature, lm_top_k, lm_top_p, lm_cfg_scale,
        lm_negative_prompt, guidance_scale, shift, infer_method, sampler_mode,
        use_adg, cfg_interval_start, cfg_interval_end,
        velocity_norm_threshold, velocity_ema_factor,
        cot_metas, cot_language, vae,
    )
    cfg["save_dir"] = str(run_dir)

    _write_config(cfg, cfg_path)

    # Think uses the selected LM (resident); Think off loads no LM (more VRAM).
    lm_sel = lm_model if think else "none"
    n_out = max(1, min(int(batch_size), MAX_OUTPUTS))
    fresh_load = (_worker is None or _worker.poll() is not None
                  or _worker_model != model or _worker_vae != vae or _worker_lm != lm_sel)
    if fresh_load:
        desc = "Model was not loaded on GPU, this may take longer…"
    elif n_out > 1:
        desc = f"Generating {n_out} outputs…"
    else:
        desc = "Generating…"
    progress(0.05, desc=desc)
    res = _worker_generate_n(cfg, model, vae, lm_sel, n_out, run_dir)

    if not res.get("ok"):
        shutil.rmtree(run_dir, ignore_errors=True)
        if _stop_requested:
            return (*_audio_outputs([]), "🛑 Generation stopped. (Worker was terminated; next run reloads the model.)")
        tail = res.get("trace") or res.get("log_tail") or ""
        return (*_audio_outputs([]), f"❌ {res.get('msg', 'no audio produced')}\n\n{tail[-1500:]}")

    raw_files = res.get("files") or _all_audio(run_dir)
    if not raw_files:
        shutil.rmtree(run_dir, ignore_errors=True)
        return (*_audio_outputs([]), "❌ No audio produced.")

    # Flat layout: all audio in audio/, with the settings embedded in each file's
    # own metadata (no sidecar). The per-run temp folder is then removed.
    audio_dir = OUT_ROOT / "audio"
    audio_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    files = []
    used_seed = None
    for i, f in enumerate(raw_files):
        src = Path(f)
        base = f"{stamp}-{i + 1}" if len(raw_files) > 1 else stamp
        meta = src.with_suffix(".json")  # pipeline metadata (same stem as audio)
        if meta.exists():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                s = data.get("seed")
                if used_seed is None and s not in (None, "", -1):
                    used_seed = s
            except Exception:
                pass
        dest_audio = audio_dir / f"{base}{src.suffix}"
        try:
            shutil.move(str(src), str(dest_audio))
        except Exception:
            dest_audio = src
        _embed_settings(str(dest_audio), cfg)  # settings live inside the file
        files.append(str(dest_audio))

    # Drop the temp run folder (leftover item subdirs, intermediates, config.toml).
    shutil.rmtree(run_dir, ignore_errors=True)

    newest = files[-1]
    seed_note = f"  |  seed: {used_seed}" if used_seed is not None else ""

    lt, gt = res.get("load_time"), res.get("gen_time")
    if lt is not None:
        load_disp = f"{lt:.1f}s (fresh load)"
    else:
        load_disp = "0s (model already resident ✔)" if not think else "n/a"
    gen_disp = f"{gt:.1f}s" if gt is not None else "n/a"
    timing = f"⏱ load: {load_disp}  |  generate: {gen_disp}"

    mem = ""
    if not think:
        rss, free = res.get("rss"), res.get("free")
        baseline, growth = res.get("baseline"), res.get("growth")
        bits = []
        if rss is not None:
            bits.append(f"worker RAM {rss:.1f}GB")
        if baseline is not None and growth is not None:
            if res.get("gens") == 1:
                bits.append(f"baseline set {baseline:.1f}GB")
            else:
                bits.append(f"growth +{growth:.1f}GB vs baseline {baseline:.1f}GB")
        if free is not None:
            bits.append(f"system free {free:.1f}GB")
        bits.append(f"gens: {res.get('gens')}")
        mem = "\n🧠 " + " | ".join(bits)
        if res.get("recycled"):
            mem += f"  →  recycled ({res.get('reason')}; next run reloads)"

    status = (
        f"✅ Done{seed_note}  ({len(files)} output{'s' if len(files) != 1 else ''})\n"
        f"{timing}{mem}"
    )
    return (*_audio_outputs(files), status)


# ── load settings ────────────────────────────────────────────────────────────

MODELS = _available_models()
LM_MODELS = _available_lm_models()
VAES = _available_vaes()


def _cfg_to_updates(cfg):
    """Map a settings dict to the ordered list of SETTING_INPUTS values."""
    def g(k, d=None):
        v = cfg.get(k, d)
        return d if v is None else v

    lyr = str(g("lyrics", "") or "")
    if lyr.strip() == "[Instrumental]":
        lyr = ""
    dur = cfg.get("duration", 60)
    auto = dur is None or float(dur) <= 0
    dur_val = 60.0 if auto else float(dur)

    model_val = g("config_path", MODELS[0])
    if model_val not in MODELS:
        model_val = MODELS[0]
    lm_val = g("lm_model_path", LM_MODELS[0])
    if lm_val not in LM_MODELS:
        lm_val = LM_MODELS[0]

    return [
        str(g("caption", "")),
        lyr,
        bool(g("instrumental", True)),
        dur_val,
        bool(auto),
        int(g("inference_steps", 8)),
        int(g("seed", -1)),
        int(g("batch_size", 1)),
        str(g("audio_format", "mp3")),
        model_val,
        int(g("bpm", 0)),
        str(g("keyscale", "")),
        str(g("timesignature", "")),
        str(g("vocal_language", "en")),
        bool(g("thinking", False)),
        lm_val,
        float(g("lm_temperature", 0.85)),
        int(g("lm_top_k", 50)),
        float(g("lm_top_p", 0.95)),
        float(g("lm_cfg_scale", 2.0)),
        str(g("lm_negative_prompt", "")),
        float(g("guidance_scale", 7.0)),
        float(g("shift", 3.0)),
        str(g("infer_method", "ode")),
        str(g("sampler_mode", "euler")),
        bool(g("use_adg", False)),
        float(g("cfg_interval_start", 0.0)),
        float(g("cfg_interval_end", 1.0)),
        float(g("velocity_norm_threshold", 0.0)),
        float(g("velocity_ema_factor", 0.0)),
        bool(g("use_cot_metas", True)),          # cot_metas
        bool(g("use_cot_language", False)),       # cot_language
        (g("vae", "official") if g("vae", "official") in VAES else "official"),  # vae
    ]


def load_settings(file):
    """Load settings from a generated track (mp3/wav/flac) or an old .toml."""
    if file is None:
        return [gr.update()] * _N_SETTINGS
    path = file.name if hasattr(file, "name") else file
    ext = Path(path).suffix.lower()
    try:
        if ext in (".mp3", ".wav", ".flac"):
            cfg = _read_embedded_settings(path)
        elif ext == ".toml":
            with open(path, "r", encoding="utf-8") as f:
                cfg = toml.load(f)
        else:
            cfg = None
    except Exception:
        cfg = None
    if not cfg:
        return [gr.update()] * _N_SETTINGS
    return _cfg_to_updates(cfg)


def load_audio(file):
    """Apply a track's embedded settings AND load it into the player to play."""
    settings = load_settings(file)
    players = [gr.update()] * MAX_OUTPUTS
    if file is not None:
        path = file.name if hasattr(file, "name") else file
        if path and Path(path).suffix.lower() in (".mp3", ".wav", ".flac"):
            players = _audio_outputs([path])
    return [*settings, *players]


# ── templates (named presets: caption, lyrics, and all params) ────────────────

TEMPLATES_DIR = APP_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)


def _list_templates():
    return sorted(p.stem for p in TEMPLATES_DIR.glob("*.toml"))


def load_template(name):
    """Apply a saved template's settings to the UI."""
    if not name:
        return [gr.update()] * _N_SETTINGS
    path = TEMPLATES_DIR / f"{name}.toml"
    if not path.exists():
        return [gr.update()] * _N_SETTINGS
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = toml.load(f)
    except Exception:
        return [gr.update()] * _N_SETTINGS
    return _cfg_to_updates(cfg)


# Template applied to the form on startup (keeps defaults in sync with the file).
DEFAULT_TEMPLATE = "Brazilian phonk"


def _load_default_template():
    if DEFAULT_TEMPLATE in _list_templates():
        return load_template(DEFAULT_TEMPLATE)
    return [gr.update()] * _N_SETTINGS


def _on_infer_method_change(infer_method, sampler_mode):
    """SDE can't use Heun (engine falls back to Euler), so restrict the choices."""
    if infer_method == "sde":
        return gr.update(choices=["euler"], value="euler")
    keep = sampler_mode if sampler_mode in ("euler", "heun") else "heun"
    return gr.update(choices=["euler", "heun"], value=keep)


def _on_lm_change(lm):
    """Warn when a bigger LM is selected (it needs much more VRAM); clear otherwise."""
    if lm and ("1.7b" in lm.lower() or "4b" in lm.lower()):
        return ("⚠️ Bigger LM selected — keep it only if your GPU has more than 12GB VRAM "
                "(on 8 GB use the 0.6B, or it will run out of memory).")
    return ""


# ── UI ───────────────────────────────────────────────────────────────────────

# Dark black/violet + purple-magenta "phonk" theme.
AG_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.fuchsia,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0a0a0d",
    body_background_fill_dark="#0a0a0d",
    body_text_color="#f5f5f7",
    body_text_color_dark="#f5f5f7",
    background_fill_primary="#16161c",
    background_fill_primary_dark="#16161c",
    background_fill_secondary="#1c1c24",
    background_fill_secondary_dark="#1c1c24",
    block_background_fill="#16161c",
    block_background_fill_dark="#16161c",
    block_border_color="#2a2a33",
    block_border_color_dark="#2a2a33",
    block_label_text_color="#cf9bff",
    block_title_text_color="#cf9bff",
    input_background_fill="#1c1c24",
    input_background_fill_dark="#1c1c24",
    button_primary_background_fill="linear-gradient(90deg, #a22ff0, #e11d48)",
    button_primary_background_fill_dark="linear-gradient(90deg, #a22ff0, #e11d48)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#24242d",
    button_secondary_background_fill_dark="#24242d",
    button_secondary_text_color="#f5f5f7",
)

AG_CSS = """
.gradio-container {
  background: radial-gradient(1100px 520px at 50% -8%, #3a0a30 0%, #2a0712 38%, #0a0a0d 100%) !important;
}
#ag-title h1 {
  font-weight: 900; font-size: 2.6rem; letter-spacing: .5px;
  text-shadow: 0 0 24px rgba(200,55,120,.45);
}
#ag-go {
  background: linear-gradient(90deg, #a22ff0, #e11d48) !important;
  color: #fff !important; border: none !important; font-weight: 700 !important;
  box-shadow: 0 0 22px rgba(200,55,120,.5) !important;
}
#ag-go:hover { filter: brightness(1.08); box-shadow: 0 0 32px rgba(200,55,120,.75) !important; }
#ag-go:disabled, #ag-go[disabled] {
  filter: grayscale(.5) brightness(.65) !important;
  box-shadow: none !important; opacity: .6 !important; cursor: not-allowed !important;
}
/* template row: bigger gap below it, and align the 💾 with the input fields */
#ag-tpl-row { margin-bottom: 22px; }
#ag-save { align-self: flex-end; }
/* purple text selection to match the theme */
::selection { background: #8b2fd6; color: #ffffff; }
::-moz-selection { background: #8b2fd6; color: #ffffff; }
"""

with gr.Blocks(title="Auragroove", theme=AG_THEME, css=AG_CSS) as demo:
    gr.Markdown(
        '# <span style="color:#ffffff">Aura</span>'
        '<span style="background:linear-gradient(90deg,#a22ff0,#e11d48);'
        '-webkit-background-clip:text;background-clip:text;color:transparent">Groove</span>',
        elem_id="ag-title",
    )
    with gr.Row():
        with gr.Column(scale=3):
            caption = gr.Textbox(
                label="Music Caption",
                value="Aggressive brazilian phonk driven by brazilian baile funk, hollering, and chopped vocals. The overall production is bass boosted and punchy, designed for maximum impact.",
                placeholder="Brazilian phonk, aggressive bass-boosted funk, heavy distorted 808, cowbell melody, 130 BPM",
                lines=3,
            )
            instrumental = gr.Checkbox(label="Instrumental (no vocals)", value=False)
            lyrics = gr.Textbox(
                label="Lyrics (optional; ignored if Instrumental is checked)",
                value="yy, kaa, koo,\ntung, tung, tung.",
                placeholder="[verse]\n...",
                lines=4,
            )
            with gr.Row():
                duration = gr.Slider(10, 240, value=60, step=5, label="Duration (s)")
                steps = gr.Slider(4, 60, value=8, step=1, label="Steps (8 = turbo)")
            auto_duration = gr.Checkbox(label="Auto duration (-1): let the model decide (ignores the slider)", value=False)
            with gr.Row():
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                batch_size = gr.Slider(1, 10, value=1, step=1, label="# Outputs")
            with gr.Row():
                bpm = gr.Number(value=140, label="BPM (0 = auto)", precision=0)
                keyscale = gr.Textbox(value="B Major", label="Key (e.g. C Major; empty = auto)")
                timesignature = gr.Textbox(value="4/4", label="Time sig (e.g. 4/4; empty = auto)")
            with gr.Row():
                model = gr.Dropdown(choices=MODELS, value=MODELS[0], label="DiT Model")
                vae = gr.Dropdown(choices=VAES, value=("scragvae" if "scragvae" in VAES else VAES[0]), label="VAE (changing reloads worker)")
                audio_format = gr.Dropdown(choices=["mp3", "wav", "flac"], value="mp3", label="Format")
                vocal_language = gr.Textbox(value="fi", label="Vocal lang")

            with gr.Accordion("🧠 Think (LM)", open=False):
                think = gr.Checkbox(
                    label="Enable Think (loads the 5Hz LM to reason about structure/metadata)",
                    value=True,
                )
                lm_model = gr.Dropdown(
                    choices=LM_MODELS, label="LM model (Think)",
                    value=(WORKER_LM_MODEL if WORKER_LM_MODEL in LM_MODELS else LM_MODELS[0]),
                )
                with gr.Row():
                    lm_temperature = gr.Slider(0.0, 2.0, value=0.85, step=0.05, label="LM temperature (0.85 = recommended)")
                    lm_cfg_scale = gr.Slider(1.0, 10.0, value=2.8, step=0.1, label="LM CFG scale")
                with gr.Row():
                    lm_top_k = gr.Number(value=0, label="LM top-k (0 = off)", precision=0)
                    lm_top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="LM top-p")
                lm_negative_prompt = gr.Textbox(value="low quality, noise", label="LM negative prompt (optional)")
                with gr.Row():
                    cot_metas = gr.Checkbox(value=True, label="CoT Metas (LM reasons out BPM/key/structure first)")
                    cot_language = gr.Checkbox(value=False, label="CoT Language Detection (LM detects vocal language)")

            with gr.Accordion("⚙️ Advanced DiT (mostly for base models; turbo ignores guidance)", open=False):
                with gr.Row():
                    guidance_scale = gr.Slider(1.0, 15.0, value=7.0, step=0.5, label="Guidance / CFG (base only)")
                    shift = gr.Slider(1.0, 5.0, value=3.0, step=0.1, label="Timestep shift")
                with gr.Row():
                    infer_method = gr.Dropdown(choices=["ode", "sde"], value="ode", label="Infer method")
                    sampler_mode = gr.Dropdown(choices=["euler", "heun"], value="heun", label="Sampler (heun = slower/smoother)")
                    use_adg = gr.Checkbox(label="Adaptive Dual Guidance (base only)", value=False)
                with gr.Row():
                    cfg_interval_start = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="CFG interval start")
                    cfg_interval_end = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="CFG interval end")
                with gr.Row():
                    velocity_norm_threshold = gr.Slider(0.0, 5.0, value=1.6, step=0.1, label="Velocity Norm Threshold (0 = off)")
                    velocity_ema_factor = gr.Slider(0.0, 1.0, value=0.05, step=0.05, label="Velocity EMA Factor (0 = off)")

        with gr.Column(scale=2):
            with gr.Row():
                go = gr.Button("Generate", variant="primary", elem_id="ag-go")
                stop_btn = gr.Button("🛑 Stop", variant="stop")
            out_audios = [
                gr.Audio(label=f"Result {i + 1}", type="filepath", visible=(i == 0))
                for i in range(MAX_OUTPUTS)
            ]
            status = gr.Textbox(label="Status", lines=12, interactive=False)
            with gr.Row():
                load_btn = gr.UploadButton(
                    "📂 Load audio",
                    file_types=[".mp3", ".wav", ".flac", ".toml"], file_count="single",
                    variant="secondary",
                )
                open_btn = gr.Button("📁 Open outputs folder", variant="secondary")
                reset_btn = gr.Button("♻️ Reset worker", variant="secondary")
            with gr.Row(elem_id="ag-tpl-row"):
                template_dd = gr.Dropdown(
                    choices=_list_templates(), label="Load template", scale=4,
                    value=("Brazilian phonk" if "Brazilian phonk" in _list_templates() else None),
                )
                tpl_name = gr.Textbox(label="Save as template", placeholder="e.g. brazilian phonk", scale=4)
                save_tpl_btn = gr.Button("💾", variant="secondary", scale=0, min_width=46, elem_id="ag-save")

    SETTING_INPUTS = [
        caption, lyrics, instrumental, duration, auto_duration, steps, seed, batch_size,
        audio_format, model, bpm, keyscale, timesignature, vocal_language,
        think, lm_model, lm_temperature, lm_top_k, lm_top_p, lm_cfg_scale,
        lm_negative_prompt, guidance_scale, shift, infer_method, sampler_mode,
        use_adg, cfg_interval_start, cfg_interval_end,
        velocity_norm_threshold, velocity_ema_factor,
        cot_metas, cot_language, vae,
    ]

    _busy = go.click(_on_generate_start, inputs=None, outputs=go, queue=False)
    gen_event = _busy.then(generate, inputs=SETTING_INPUTS, outputs=[*out_audios, status])
    gen_event.then(lambda: gr.update(value="Generate", interactive=True),
                   inputs=None, outputs=go, queue=False)
    stop_btn.click(
        stop_generation, inputs=None, outputs=status, queue=False, cancels=[gen_event]
    ).then(lambda: gr.update(value="Generate", interactive=True),
           inputs=None, outputs=go, queue=False)
    load_btn.upload(load_audio, inputs=load_btn, outputs=[*SETTING_INPUTS, *out_audios])
    open_btn.click(_open_outputs_folder, inputs=None, outputs=None)
    reset_btn.click(reset_worker, inputs=None, outputs=status)
    template_dd.change(load_template, inputs=template_dd, outputs=SETTING_INPUTS).then(
        lambda: gr.update(value=""), inputs=None, outputs=tpl_name)
    save_tpl_btn.click(save_template, inputs=[tpl_name, *SETTING_INPUTS],
                       outputs=[template_dd, status])
    infer_method.change(_on_infer_method_change,
                        inputs=[infer_method, sampler_mode], outputs=sampler_mode)
    lm_model.change(_on_lm_change, inputs=lm_model, outputs=status)
    demo.load(_load_default_template, inputs=None, outputs=SETTING_INPUTS)

if __name__ == "__main__":
    demo.queue()  # required for the Stop button's cancel + concurrent events
    _favicon = APP_DIR / "favicon.ico"
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True,
                favicon_path=str(_favicon) if _favicon.exists() else None)
