"""
Auragroove — local ACE-Step 1.5 music generator (persistent-worker edition).

Two generation paths:
  * Pure-DiT (Think OFF): handled by a PERSISTENT worker (worker.py) that loads
    the model ONCE and serves many requests -> no reload between runs (fast).
    A watchdog recycles the worker (kill + respawn) when system free RAM drops
    too low or after N generations, to bound the known memory leak (#142).
  * Think ON: handled by a one-shot subprocess (cli.py), which exits and frees
    all RAM afterward (the LM path is heavier and stays on the safe route).

Outputs are organized flatly: audio in `auragroove_outputs/audio/`, config in
`auragroove_outputs/settings/` (matching names). Settings reload via the Load
button. Timings (load vs generate) are shown and logged.

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
CLI = str(ACESTEP_DIR / "cli.py")
WORKER = str(APP_DIR / "worker.py")                             # worker lives here, runs with the bundled venv
OUT_ROOT = APP_DIR / "auragroove_outputs"                        # outputs land here
OUT_ROOT.mkdir(exist_ok=True)

AUDIO_EXTS = ("*.mp3", "*.wav", "*.flac")
STDIN_AUTOCONFIRM = "\n" * 64

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

_N_SETTINGS = 34


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
    global _worker, _worker_model, _worker_offload, _worker_vae, _worker_gens, _worker_baseline_rss
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
    _worker_gens = 0
    _worker_baseline_rss = None


atexit.register(_kill_worker)


def _ensure_worker(model, vae="official", offload=True):
    """Spawn/respawn the worker if needed. Returns load_time if a (re)spawn
    happened this call, else None (worker was already resident)."""
    global _worker, _worker_model, _worker_offload, _worker_vae, _worker_gens, _worker_baseline_rss
    need = (
        _worker is None
        or _worker.poll() is not None
        or _worker_model != model
        or _worker_offload != offload
        or _worker_vae != vae
    )
    if not need:
        return None
    _kill_worker()
    cmd = [PYTHON, WORKER, "--model", model, "--offload", "1" if offload else "0",
           "--quant", WORKER_QUANTIZATION, "--lm", WORKER_LM_MODEL or "none",
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
    _worker_gens = 0
    _worker_baseline_rss = None
    return ev.get("load_time", time.time() - t0)


def _worker_generate(cfg, model, vae="official"):
    global _worker, _worker_gens, _worker_baseline_rss
    with _worker_lock:
        try:
            load_time = _ensure_worker(model, vae=vae, offload=WORKER_OFFLOAD_TO_CPU)
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


def _worker_generate_n(cfg, model, vae, n, run_dir):
    """Run N sequential single-item generations (batch>1 won't fit in VRAM next
    to the resident model on 8GB, so we loop instead). Aggregates the results."""
    files, load_time, gen_time, last = [], None, 0.0, None
    for k in range(n):
        sub = dict(cfg)
        sub["batch_size"] = 1
        sub["save_dir"] = str(run_dir / f"item_{k}")
        r = _worker_generate(sub, model, vae=vae)
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


# ── one-shot path (Think ON) ─────────────────────────────────────────────────

def _oneshot_generate(cfg, cfg_path, run_dir):
    try:
        (ACESTEP_DIR / "instruction.txt").unlink()
    except FileNotFoundError:
        pass

    cmd = [PYTHON, CLI, "-c", str(cfg_path), "--log-level", "INFO"]
    t0 = time.time()
    load_done = gen_start = None
    out_lines = []
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(ACESTEP_DIR),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
    except Exception as e:
        return {"ok": False, "msg": f"failed to launch generator: {e}"}
    try:
        proc.stdin.write(STDIN_AUTOCONFIRM)
        proc.stdin.flush()
        proc.stdin.close()
    except Exception:
        pass

    _set_active(proc)
    for line in proc.stdout:
        out_lines.append(line)
        now = time.time()
        if load_done is None and "Handlers initialized." in line:
            load_done = now
        if gen_start is None and "Starting Generation" in line:
            gen_start = now
        if now - t0 > 3600:
            proc.kill()
            break
    proc.wait()
    _set_active(None)
    t_end = time.time()

    files = _all_audio(run_dir)
    gen_anchor = gen_start or load_done
    return {
        "ok": bool(files),
        "files": files,
        "load_time": (load_done - t0) if load_done else None,
        "gen_time": (t_end - gen_anchor) if gen_anchor else None,
        "log_tail": "".join(out_lines[-30:]),
    }


# ── main generate ────────────────────────────────────────────────────────────

MAX_OUTPUTS = 4


def _audio_outputs(files):
    """Build value/visibility updates for the result players (one per output)."""
    ups = []
    for i in range(MAX_OUTPUTS):
        if i < len(files):
            ups.append(gr.update(value=files[i], visible=True))
        else:
            ups.append(gr.update(value=None, visible=(i == 0)))
    return ups


def generate(
    caption, lyrics, instrumental, duration, auto_duration, steps, seed, batch_size,
    audio_format, model, bpm, keyscale, timesignature, vocal_language,
    think, lm_model, lm_temperature, lm_top_k, lm_top_p, lm_cfg_scale,
    lm_negative_prompt, guidance_scale, shift, infer_method, sampler_mode,
    use_adg, cfg_interval_start, cfg_interval_end,
    velocity_norm_threshold, velocity_ema_factor,
    timesig_auto, cot_metas, cot_language, vae,
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
        "save_dir": str(run_dir),
    }
    if think:
        cfg["lm_model_path"] = lm_model
        if (lm_negative_prompt or "").strip():
            cfg["lm_negative_prompt"] = lm_negative_prompt.strip()
    if bpm and int(bpm) > 0:
        cfg["bpm"] = int(bpm)
    if (keyscale or "").strip():
        cfg["keyscale"] = keyscale.strip()
    if not timesig_auto and (timesignature or "").strip():
        cfg["timesignature"] = timesignature.strip()
    if vocal_language and not instrumental:
        cfg["vocal_language"] = vocal_language.strip()

    _write_config(cfg, cfg_path)

    resident_think = WORKER_LM_MODEL not in (None, "none", "")
    if think and not resident_think:
        progress(0.05, desc="Think ON: one-shot run (loads LM + model, frees after)...")
        res = _oneshot_generate(cfg, cfg_path, run_dir)
    else:
        n_out = max(1, min(int(batch_size), MAX_OUTPUTS))
        desc = (f"Persistent worker (resident model"
                + (" + LM" if resident_think else "")
                + (f"), generating {n_out} sequentially..." if n_out > 1 else ")..."))
        progress(0.05, desc=desc)
        res = _worker_generate_n(cfg, model, vae, n_out, run_dir)

    if not res.get("ok"):
        if _stop_requested:
            return (*_audio_outputs([]), "🛑 Generation stopped. (Worker was terminated; next run reloads the model.)")
        tail = res.get("trace") or res.get("log_tail") or ""
        return (*_audio_outputs([]), f"❌ {res.get('msg', 'no audio produced')}\n\n{tail[-1500:]}")

    raw_files = res.get("files") or _all_audio(run_dir)
    if not raw_files:
        return (*_audio_outputs([]), "❌ No audio produced.")

    # Reorganize into a flat layout: all audio in audio/, all config in settings/,
    # with matching timestamped names. The per-run temp folder is then removed.
    audio_dir = OUT_ROOT / "audio"
    settings_dir = OUT_ROOT / "settings"
    audio_dir.mkdir(exist_ok=True)
    settings_dir.mkdir(exist_ok=True)
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
        files.append(str(dest_audio))
        # settings: our request config (.toml, loadable) + pipeline metadata (.json)
        try:
            shutil.copyfile(cfg_path, str(settings_dir / f"{base}.toml"))
        except Exception:
            pass
        if meta.exists():
            try:
                shutil.move(str(meta), str(settings_dir / f"{base}.json"))
            except Exception:
                pass

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

    try:
        with open(OUT_ROOT / "timings.log", "a", encoding="utf-8") as lf:
            lf.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{timing}\t"
                f"path={'think/oneshot' if think else 'worker'}\t"
                f"steps={int(steps)} dur={cfg['duration']}\t{caption[:60]}\n"
            )
    except Exception:
        pass

    status = (
        f"✅ Done{seed_note}  ({len(files)} output{'s' if len(files) != 1 else ''})\n"
        f"{timing}{mem}"
    )
    return (*_audio_outputs(files), status)


# ── load settings ────────────────────────────────────────────────────────────

MODELS = _available_models()
LM_MODELS = _available_lm_models()
VAES = _available_vaes()


def load_settings(file):
    if file is None:
        return [gr.update()] * _N_SETTINGS
    path = file.name if hasattr(file, "name") else file
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = toml.load(f)
    except Exception:
        return [gr.update()] * _N_SETTINGS

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
        (not bool(cfg.get("timesignature"))),    # timesig_auto
        bool(g("use_cot_metas", True)),          # cot_metas
        bool(g("use_cot_language", False)),       # cot_language
        (g("vae", "official") if g("vae", "official") in VAES else "official"),  # vae
    ]


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
"""

with gr.Blocks(title="Auragroove", theme=AG_THEME, css=AG_CSS) as demo:
    gr.Markdown(
        '# <span style="color:#ffffff">Aura</span>'
        '<span style="background:linear-gradient(90deg,#a22ff0,#e11d48);'
        '-webkit-background-clip:text;background-clip:text;color:transparent">Groove</span>',
        elem_id="ag-title",
    )
    with gr.Row():
        load_btn = gr.UploadButton(
            "📂 Load settings (.toml)", file_types=[".toml"], file_count="single",
            variant="secondary",
        )
        open_btn = gr.Button("📁 Open outputs folder", variant="secondary")
        reset_btn = gr.Button("♻️ Reset worker (free RAM)", variant="secondary")
    with gr.Row():
        with gr.Column(scale=3):
            caption = gr.Textbox(
                label="Music Caption",
                value="brazilian phonk, brazilian baile funk, aggressive, dark, vocal chops, female vocals",
                placeholder="Brazilian phonk, aggressive bass-boosted funk, heavy distorted 808, cowbell melody, 130 BPM",
                lines=3,
            )
            instrumental = gr.Checkbox(label="Instrumental (no vocals)", value=False)
            lyrics = gr.Textbox(
                label="Lyrics (optional; ignored if Instrumental is checked)",
                value="boom, boom,\nboom, boom",
                placeholder="[verse]\n...",
                lines=4,
            )
            with gr.Row():
                duration = gr.Slider(10, 240, value=75, step=5, label="Duration (s)")
                steps = gr.Slider(4, 60, value=8, step=1, label="Steps (8 = turbo)")
            auto_duration = gr.Checkbox(label="Auto duration (-1): let the model decide (ignores the slider)", value=False)
            with gr.Row():
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                batch_size = gr.Slider(1, 4, value=1, step=1, label="# Outputs")
            with gr.Row():
                bpm = gr.Number(value=140, label="BPM (0 = auto)", precision=0)
                keyscale = gr.Textbox(value="B Major", label="Key (e.g. C Major; empty = auto)")
                timesignature = gr.Textbox(value="4/4", label="Time sig (e.g. 4/4)")
            timesig_auto = gr.Checkbox(value=False, label="TimeSig Auto (let model choose; ignores the Time sig field)")
            with gr.Row():
                model = gr.Dropdown(choices=MODELS, value=MODELS[0], label="DiT Model")
                vae = gr.Dropdown(choices=VAES, value=("scragvae" if "scragvae" in VAES else VAES[0]), label="VAE (changing reloads worker)")
                audio_format = gr.Dropdown(choices=["mp3", "wav", "flac"], value="mp3", label="Format")
                vocal_language = gr.Textbox(value="en", label="Vocal lang")

            with gr.Accordion("🧠 Think (LM) — one-shot path; slower + more RAM each run", open=False):
                think = gr.Checkbox(
                    label="Enable Think (loads the 5Hz LM to reason about structure/metadata)",
                    value=True,
                )
                lm_model = gr.Dropdown(choices=LM_MODELS, value=LM_MODELS[0], label="LM model")
                with gr.Row():
                    lm_temperature = gr.Slider(0.0, 2.0, value=0.85, step=0.05, label="LM temperature (0.85 = recommended)")
                    lm_cfg_scale = gr.Slider(1.0, 10.0, value=2.8, step=0.1, label="LM CFG scale")
                with gr.Row():
                    lm_top_k = gr.Number(value=0, label="LM top-k (0 = off)", precision=0)
                    lm_top_p = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="LM top-p")
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

    SETTING_INPUTS = [
        caption, lyrics, instrumental, duration, auto_duration, steps, seed, batch_size,
        audio_format, model, bpm, keyscale, timesignature, vocal_language,
        think, lm_model, lm_temperature, lm_top_k, lm_top_p, lm_cfg_scale,
        lm_negative_prompt, guidance_scale, shift, infer_method, sampler_mode,
        use_adg, cfg_interval_start, cfg_interval_end,
        velocity_norm_threshold, velocity_ema_factor,
        timesig_auto, cot_metas, cot_language, vae,
    ]

    _busy = go.click(lambda: gr.update(value="Generating…", interactive=False),
                     inputs=None, outputs=go, queue=False)
    gen_event = _busy.then(generate, inputs=SETTING_INPUTS, outputs=[*out_audios, status])
    gen_event.then(lambda: gr.update(value="Generate", interactive=True),
                   inputs=None, outputs=go, queue=False)
    stop_btn.click(
        stop_generation, inputs=None, outputs=status, queue=False, cancels=[gen_event]
    ).then(lambda: gr.update(value="Generate", interactive=True),
           inputs=None, outputs=go, queue=False)
    load_btn.upload(load_settings, inputs=load_btn, outputs=SETTING_INPUTS)
    open_btn.click(_open_outputs_folder, inputs=None, outputs=None)
    reset_btn.click(reset_worker, inputs=None, outputs=status)

if __name__ == "__main__":
    demo.queue()  # required for the Stop button's cancel + concurrent events
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
