"""
Persistent generation worker for Auragroove.

Loads the ACE-Step DiT model ONCE, then serves many generation requests over a
simple line-based JSON protocol on stdin/stdout. Auragroove keeps this
process alive so the model isn't reloaded every run (fast), and recycles it
(kills + respawns) when memory grows too far -- bounding the known leak.

Protocol (one JSON object per line):
  UI -> worker (stdin):   {"caption": ..., "duration": ..., "save_dir": ...}
                          {"cmd": "quit"}
  worker -> UI (stdout):  lines prefixed with "@@" then JSON, e.g.
                          @@{"event": "ready", "load_time": 41.2}
                          @@{"event": "result", "files": [...], "gen_time": 18.1}
                          @@{"event": "error", "msg": "..."}

All library/log output is routed to stderr so stdout carries only the protocol.
This worker handles pure-DiT (no LM / Think off) requests only.
"""

import sys
import os
import gc
import glob
import json
import time
import threading
import argparse

# Route any library prints to stderr so stdout is a clean protocol channel.
_real_stdout = sys.stdout
sys.stdout = sys.stderr
# Force UTF-8 on the streams so emoji/Unicode in engine logs (e.g. the "✅ LoRA
# loaded" message) don't crash on legacy console code pages (e.g. cp1253).
for _s in (_real_stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Self-contained engine bundled in this project (acestep package + checkpoints).
# Put it on the import path so `import acestep` resolves locally. No Pinokio.
ACESTEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acestep_engine")
if ACESTEP_DIR not in sys.path:
    sys.path.insert(0, ACESTEP_DIR)


_emit_lock = threading.Lock()


def emit(obj):
    with _emit_lock:
        _real_stdout.write("@@" + json.dumps(obj) + "\n")
        _real_stdout.flush()


def _progress(value, desc=None):
    """Forward the engine's progress (ratio, desc) to the UI as a protocol event.
    Called from the engine's diffusion-progress thread, ~every 0.5s."""
    try:
        emit({"event": "progress", "ratio": float(value),
              "desc": desc if isinstance(desc, str) else None})
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--offload", default="1")
    ap.add_argument("--quant", default="none",
                    help="quantization: none | int8_weight_only | fp8_weight_only | w8a8_dynamic")
    ap.add_argument("--lm", default="none",
                    help="resident LM model name for Think, or 'none' to disable")
    ap.add_argument("--vae", default="official",
                    help="VAE variant: official | scragvae | <path>")
    ap.add_argument("--lora", default="none",
                    help="path to a trained LoRA adapter dir/file, or 'none'")
    args = ap.parse_args()
    offload = str(args.offload).lower() in ("1", "true", "yes")
    quantization = None if str(args.quant).lower() in ("none", "", "0") else args.quant
    lm_model = None if str(args.lm).lower() in ("none", "", "0") else args.lm
    vae_checkpoint = None if str(args.vae).lower() in ("official", "none", "") else args.vae
    lora_path = None if str(args.lora).lower() in ("none", "", "0") else args.lora

    t0 = time.time()
    try:
        import torch
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.inference import generate_music, GenerationParams, GenerationConfig
        from acestep.gpu_config import get_gpu_config, set_global_gpu_config

        set_global_gpu_config(get_gpu_config())
        project_root = ACESTEP_DIR
        device = "cuda" if torch.cuda.is_available() else "cpu"

        dit = AceStepHandler()
        dit.initialize_service(
            project_root=project_root,
            config_path=args.model,
            device=device,
            use_flash_attention=dit.is_flash_attention_available(device),
            offload_to_cpu=offload,
            offload_dit_to_cpu=False,
            quantization=quantization,
            vae_checkpoint=vae_checkpoint,
        )
        if lora_path:
            msg = dit.load_lora(lora_path)
            print(f"[worker] load_lora: {msg}", file=sys.stderr)

        llm = LLMHandler()
        if lm_model:
            llm.initialize(checkpoint_dir=os.path.join(ACESTEP_DIR, "checkpoints"),
                           lm_model_path=lm_model, backend="pt",
                           device=device, offload_to_cpu=offload, dtype=None)
    except Exception as e:
        emit({"event": "error", "stage": "load", "msg": repr(e)})
        return

    emit({"event": "ready", "load_time": time.time() - t0,
          "lm": bool(getattr(llm, "llm_initialized", False)),
          "lora": bool(lora_path)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            emit({"event": "error", "msg": "bad request json"})
            continue
        if req.get("cmd") == "quit":
            break

        try:
            instrumental = bool(req.get("instrumental", False))
            lyrics = req.get("lyrics") or ""
            if instrumental:
                lyrics = "[Instrumental]"

            # Think only if requested AND the LM is actually resident.
            think = bool(req.get("thinking", False)) and getattr(llm, "llm_initialized", False)
            seed = int(req.get("seed", -1))
            params = GenerationParams(
                task_type=req.get("task_type", "text2music"),
                src_audio=req.get("src_audio") or None,
                audio_cover_strength=float(req.get("audio_cover_strength", 1.0)),
                cover_noise_strength=float(req.get("cover_noise_strength", 0.0)),
                caption=req.get("caption", ""),
                lyrics=lyrics,
                instrumental=instrumental,
                vocal_language=req.get("vocal_language", "unknown"),
                bpm=req.get("bpm"),
                keyscale=req.get("keyscale", "") or "",
                timesignature=req.get("timesignature", "") or "",
                duration=float(req.get("duration", -1)),
                inference_steps=int(req.get("inference_steps", 8)),
                seed=seed,
                guidance_scale=float(req.get("guidance_scale", 7.0)),
                use_adg=bool(req.get("use_adg", False)),
                cfg_interval_start=float(req.get("cfg_interval_start", 0.0)),
                cfg_interval_end=float(req.get("cfg_interval_end", 1.0)),
                shift=float(req.get("shift", 3.0)),
                infer_method=req.get("infer_method", "ode"),
                sampler_mode=req.get("sampler_mode", "euler"),
                velocity_norm_threshold=float(req.get("velocity_norm_threshold", 0.0)),
                velocity_ema_factor=float(req.get("velocity_ema_factor", 0.0)),
                thinking=think,
                use_cot_metas=bool(req.get("use_cot_metas", False)) and think,
                use_cot_caption=bool(req.get("use_cot_caption", False)) and think,
                use_cot_lyrics=bool(req.get("use_cot_lyrics", False)) and think,
                use_cot_language=bool(req.get("use_cot_language", False)) and think,
                use_constrained_decoding=False,
                lm_temperature=float(req.get("lm_temperature", 0.85)),
                lm_top_k=int(req.get("lm_top_k", 50)),
                lm_top_p=float(req.get("lm_top_p", 0.95)),
                lm_cfg_scale=float(req.get("lm_cfg_scale", 2.0)),
                lm_negative_prompt=req.get("lm_negative_prompt") or "NO USER INPUT",
            )
            config = GenerationConfig(
                batch_size=int(req.get("batch_size", 1)),
                use_random_seed=bool(req.get("use_random_seed", seed < 0)),
                seeds=None,
                audio_format=req.get("audio_format", "mp3"),
            )
            save_dir = req.get("save_dir")

            if lora_path:
                try:
                    dit.set_lora_scale(float(req.get("lora_scale", 1.0)))
                except Exception:
                    pass

            gt0 = time.time()
            generate_music(dit, llm, params, config, save_dir=save_dir, progress=_progress)
            gen_time = time.time() - gt0

            files = []
            for ext in ("*.mp3", "*.wav", "*.flac"):
                files += glob.glob(os.path.join(save_dir, "**", ext), recursive=True)
            files.sort(key=os.path.getmtime)
            emit({"event": "result", "files": files, "gen_time": gen_time})
        except Exception as e:
            import traceback
            emit({"event": "error", "msg": repr(e), "trace": traceback.format_exc()[-1500:]})
        finally:
            try:
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()
