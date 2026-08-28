#!/usr/bin/env python3
"""Single-GPU DiTraj runner with CPU model offload and measured memory.

Use the author's patched Diffusers 0.33.1 transformer. Select exactly one
physical GPU with CUDA_VISIBLE_DEVICES before starting Python. Unlike newer
Wan VAE versions, AutoencoderKLWan in 0.33.1 has no enable_tiling method.

This runner preserves the model, scheduler, precision, guidance, seed, and
trajectory defaults of upstream run.py. It does not change the DiTraj method.
The initial short run is a smoke test, not a paper-quality reproduction.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt-json", type=Path)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=25234)
    parser.add_argument("--mask-step", type=int, default=30)
    parser.add_argument("--fix-rope-step", type=int, default=5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.num_frames < 5 or (args.num_frames - 1) % 4:
        parser.error("--num-frames must be at least 5 and of the form 4*k+1")
    if min(args.height, args.width) < 64 or args.height % 16 or args.width % 16:
        parser.error("height and width must be multiples of 16, at least 64")
    if args.steps < 1 or args.fps < 1 or min(args.mask_step, args.fix_rope_step) < 0:
        parser.error("steps/fps must be positive; control step cutoffs cannot be negative")
    if args.prompt_index < 0:
        parser.error("prompt index cannot be negative")
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must have an .mp4 extension")
    args.repo_root = args.repo_root.resolve()
    args.prompt_json = (args.prompt_json or args.repo_root / "demo/test_prompts_refined.json").resolve()
    args.output = args.output.resolve()
    return args


def negative_prompt_from_source(source):
    """Read an upstream literal without executing its top-level inference code."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "negative_prompt"
            for target in node.targets
        ):
            result = ast.literal_eval(node.value)
            if isinstance(result, str) and result:
                return result
    raise ValueError("Cannot find the upstream negative_prompt string")


def trajectory_for_frames(num_frames):
    # Cover the complete video to avoid upstream plan_path extrapolation bugs.
    return [[0, 0.3, 0.7, 0.1, 0.4], [num_frames - 1, 0.3, 0.7, 0.7, 1.0]]


def load_prompt(path, index):
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or index >= len(prompts):
        raise ValueError("Prompt JSON must contain a list with the requested index")
    prompt = prompts[index]
    for key in ("base_prompt", "bg_prompt", "fg_prompt"):
        if not isinstance(prompt.get(key), str) or not prompt[key].strip():
            raise ValueError(f"Missing nonempty {key} in prompt JSON")
    return prompt


def source_fingerprints(repo_root):
    names = ("run.py", "module/pipe.py", "module/attention_processor.py", "module/transformer_wan.py")
    return {name: hashlib.sha256((repo_root / name).read_bytes()).hexdigest() for name in names}


def draw_boxes(frames, frame_boxes):
    """Annotate only a copy of float32 [0,1] frames; generated frames stay intact."""
    boxed = frames.copy()
    height, width = boxed.shape[1:3]
    for frame, (y1, y2, x1, x2) in zip(boxed, frame_boxes):
        top, bottom = max(0, int(y1 * height)), min(height - 1, int(y2 * height))
        left, right = max(0, int(x1 * width)), min(width - 1, int(x2 * width))
        color = (127 / 255, 1.0, 127 / 255)
        frame[top : min(top + 2, height), left : right + 1] = color
        frame[max(top, bottom - 1) : bottom + 1, left : right + 1] = color
        frame[top : bottom + 1, left : min(left + 2, width)] = color
        frame[top : bottom + 1, max(left, right - 1) : right + 1] = color
    return boxed


def main(argv=None):
    args = parse_args(argv)
    prompt = load_prompt(args.prompt_json, args.prompt_index)
    negative_prompt = negative_prompt_from_source((args.repo_root / "run.py").read_text(encoding="utf-8"))
    bboxs = trajectory_for_frames(args.num_frames)
    latent_frames = (args.num_frames - 1) // 4 + 1
    bbox_height, bbox_width = args.height // 16, args.width // 16
    tokens = latent_frames * bbox_height * bbox_width
    box_output = args.output.with_name(args.output.stem + "_box.mp4")
    metrics_path = args.output.with_suffix(".metrics.json")
    for path in (args.output, box_output, metrics_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    metrics = {
        "model": args.model,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "seed": args.seed,
        "mask_step": args.mask_step,
        "fix_rope_step": args.fix_rope_step,
        "fps": args.fps,
        "bboxs": bboxs,
        "video_tokens": tokens,
        "persistent_bool_mask_gib_estimate": tokens * tokens / 1024**3,
        "cpu_model_offload": True,
        "vae_precision": "float32",
        "vae_tiling": False,
        "source_sha256": source_fingerprints(args.repo_root),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "output": str(args.output),
        "output_with_boxes": str(box_output),
    }
    if args.dry_run:
        print(json.dumps(metrics, indent=2), flush=True)
        return

    sys.path.insert(0, str(args.repo_root))
    import numpy as np
    import torch
    import diffusers
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    from module.attention_processor import MyWanAttnProcessor2_0
    from module.pipe import myWanPipeline
    from utils import plan_path

    if diffusers.__version__ != "0.33.1":
        raise RuntimeError(f"Expected author's diffusers 0.33.1; found {diffusers.__version__}")
    if "is_fixRope_step" not in inspect.getsource(WanTransformer3DModel.forward):
        raise RuntimeError("Apply the author's module/transformer_wan.py replacement before running")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one NVIDIA GPU with CUDA_VISIBLE_DEVICES before starting Python")
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    metrics.update(torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                   diffusers_version=diffusers.__version__, gpu=torch.cuda.get_device_name(0))
    print(json.dumps({"event": "start", **metrics}), flush=True)
    load_start = time.perf_counter()
    load_kwargs = {"local_files_only": args.local_files_only}
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32, **load_kwargs)
    pipe = myWanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16, **load_kwargs)
    pipe.transformer.set_attn_processor({name: MyWanAttnProcessor2_0() for name in pipe.transformer.attn_processors})
    pipe.scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True,
        num_train_timesteps=1000, flow_shift=3.0,
    )
    # This REPLACES pipe.to('cuda'). Do not add that call later.
    # v0.33.1 Wan VAE already streams temporal chunks but has NO spatial tiling API.
    pipe.enable_model_cpu_offload(gpu_id=0, device="cuda")
    metrics["model_load_seconds"] = time.perf_counter() - load_start

    # Model construction can consume global RNG state. Match upstream run.py by
    # seeding after loading; the separate CPU noise Generator below is unchanged.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    frame_boxes = plan_path(bboxs, video_length=args.num_frames)
    if len(frame_boxes) != args.num_frames:
        raise ValueError("Trajectory length differs from requested video length")
    bbox_mask = torch.zeros((latent_frames, 1, bbox_height, bbox_width), dtype=torch.float32)
    for index, (y1, y2, x1, x2) in enumerate(frame_boxes[::4]):
        top, bottom, left, right = int(y1 * bbox_height), int(y2 * bbox_height), int(x1 * bbox_width), int(x2 * bbox_width)
        if bottom - top < 2 or right - left < 2:
            raise ValueError("Foreground box is too small on the token grid for original RoPE manipulation")
        bbox_mask[index, :, top:bottom, left:right] = 1
    bbox_mask = bbox_mask.to(device)
    encoder_mask = torch.cat((torch.zeros(512, dtype=torch.bool), torch.ones(512, dtype=torch.bool))).to(device)

    def report_step(_pipe, step, _timestep, callback_kwargs):
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(json.dumps({"event": "step", "step": step + 1,
                              "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
        return callback_kwargs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    inference_start = time.perf_counter()
    try:
        frames = pipe(
            prompt=prompt["base_prompt"], negative_prompt=negative_prompt,
            height=args.height, width=args.width, num_frames=args.num_frames,
            num_inference_steps=args.steps, guidance_scale=5.0,
            generator=torch.Generator(device="cpu").manual_seed(args.seed),
            attention_kwargs={"bbox_mask": bbox_mask, "encoder_attention_mask": encoder_mask,
                              "bg_prompt": prompt["bg_prompt"], "fg_prompt": prompt["fg_prompt"],
                              "fixRope_step": args.fix_rope_step, "mask_step": args.mask_step},
            callback_on_step_end=report_step,
            output_type="np",
        ).frames[0]
        torch.cuda.synchronize()
        metrics["inference_seconds"] = time.perf_counter() - inference_start
        expected_shape = (args.num_frames, args.height, args.width, 3)
        if frames.shape != expected_shape or not np.isfinite(frames).all():
            raise ValueError(f"Invalid video frames: {frames.shape}; expected {expected_shape}")
        export_to_video(frames, str(args.output), fps=args.fps)
        export_to_video(draw_boxes(frames, frame_boxes), str(box_output), fps=args.fps)
        metrics.update(status="completed", frame_shape=list(frames.shape), frame_mean=float(frames.mean()),
                       frame_std=float(frames.std()))
    except Exception as exc:
        metrics.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                       elapsed_seconds=time.perf_counter() - inference_start)
        raise
    finally:
        metrics["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        metrics["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        if sys.platform.startswith("linux"):
            import resource
            metrics["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "result", **metrics}), flush=True)


if __name__ == "__main__":
    main()
