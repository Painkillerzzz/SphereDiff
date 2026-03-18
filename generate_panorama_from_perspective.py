"""
Generate a 360° panoramic image from a single perspective image + text prompt.

Usage examples:

  # From existing image
  python generate_panorama_from_perspective.py \
    --prompt_txt data/prompts/forest.txt \
    --input_image my_photo.jpg \
    --save_path ./outputs/panorama

  # Auto-generate initial image from prompt, then expand to panorama
  python generate_panorama_from_perspective.py \
    --prompt_txt data/prompts/forest.txt \
    --save_path ./outputs/panorama

  # Specify camera direction and FOV
  python generate_panorama_from_perspective.py \
    --prompt_txt data/prompts/ocean.txt \
    --input_image my_photo.jpg \
    --input_view_theta 0 \
    --input_view_phi 0 \
    --input_fov 80 80

  # Fast test with fewer sphere points
  python generate_panorama_from_perspective.py \
    --prompt_txt data/prompts/city.txt \
    --n_spherical_points 2600 \
    --num_inference_steps 20 \
    --save_path ./outputs/test

Prompt txt format:
  The file is read line by line; non-empty lines are joined with ", " to form
  the final prompt.  The same prompt is used both for the optional initial
  image generation and for the panorama outpainting.
"""
import argparse
import os
from datetime import datetime

import torch
from PIL import Image

from pipelines_ours import PerspectiveToPanoramaFluxPipeline


def load_prompt(txt_path: str) -> str:
    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    lines = [l for l in lines if l]
    return ", ".join(lines)


def generate_initial_image(
    pipe: PerspectiveToPanoramaFluxPipeline,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    num_steps: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    save_path_prefix: str,
) -> Image.Image:
    """
    Use the underlying Flux pipeline to generate a single 1024×1024 perspective image.
    Reuses the already-loaded model weights from pipe.
    """
    from diffusers.utils.torch_utils import randn_tensor
    from pipelines_ours.pipeline_spherical_flux import retrieve_timesteps, calculate_shift
    import numpy as np

    print("[InitGen] Generating initial perspective image with Flux ...", flush=True)

    generator = torch.Generator(device=device).manual_seed(seed)

    # Use the base FluxPipeline.__call__ via super(), passing standard arguments.
    # pipe IS a FluxPipeline so we can call it directly with standard kwargs.
    result = pipe.flux_generate(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        height=1024,
        width=1024,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pil",
    )
    init_image = result.images[0]

    # Optionally save the generated init image alongside the panorama
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    init_save = f"{save_path_prefix}_init_{ts}.png"
    os.makedirs(os.path.dirname(init_save) or ".", exist_ok=True)
    init_image.save(init_save)
    print(f"[InitGen] Saved initial image to: {init_save}", flush=True)

    return init_image


def parse_args():
    parser = argparse.ArgumentParser(description="Perspective → Panorama with Flux")

    # Input
    parser.add_argument("--prompt_txt", type=str, required=True,
                        help="Path to txt file containing the prompt (one concept per line, joined with ', ')")
    parser.add_argument("--negative_prompt_txt", type=str, default="",
                        help="Path to txt file for negative prompt (optional)")
    parser.add_argument("--input_image", type=str, default=None,
                        help="Path to the input perspective image. "
                             "If omitted, an image is auto-generated from the prompt.")

    # Camera / FOV (only relevant when --input_image is given or for generated image placement)
    parser.add_argument("--input_view_theta", type=float, default=0.0,
                        help="Camera azimuth in degrees (left/right). Default: 0 (front)")
    parser.add_argument("--input_view_phi", type=float, default=0.0,
                        help="Camera elevation in degrees (up/down). Default: 0 (horizon)")
    parser.add_argument("--input_fov", type=float, nargs=2, default=[80.0, 80.0],
                        metavar=("FOV_H", "FOV_W"),
                        help="Field of view in degrees (height width). Default: 80 80")

    # Model
    parser.add_argument("--model_path", type=str,
                        default="black-forest-labs/FLUX.1-dev",
                        help="HuggingFace model ID or local path for FLUX.1-dev")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["fp16", "bf16", "fp32"])

    # Generation parameters
    parser.add_argument("--n_spherical_points", type=int, default=26500,
                        help="Fibonacci sphere point count. Lower = faster, higher = better quality")
    parser.add_argument("--num_inference_steps", type=int, default=28,
                        help="Denoising steps per view (also used for initial image generation)")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--min_overlap_ratio", type=float, default=0.25,
                        help="Minimum overlap ratio with known region when selecting next view")
    parser.add_argument("--repaint_strength", type=float, default=1.0,
                        help="Fraction of steps to reinject known latents (0=free, 1=strict)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Gaussian blending temperature for panorama assembly")
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--erp_height", type=int, default=2048)
    parser.add_argument("--erp_width", type=int, default=4096)
    parser.add_argument("--save_path", type=str, default="./outputs/panorama",
                        help="Output file path prefix (timestamp + .png appended automatically)")

    # Memory options
    parser.add_argument("--enable_model_cpu_offload", action="store_true",
                        help="Offload model weights to CPU between uses (saves VRAM)")
    parser.add_argument("--enable_vae_tiling", action="store_true",
                        help="Enable VAE tiling for large output resolutions")

    return parser.parse_args()


def main():
    args = parse_args()

    assert os.path.exists(args.prompt_txt), f"Prompt file not found: {args.prompt_txt}"
    if args.input_image:
        assert os.path.exists(args.input_image), f"Input image not found: {args.input_image}"
    assert torch.cuda.is_available(), "CUDA not available. GPU required."

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.mixed_precision]
    device = torch.device("cuda")

    # Load prompt
    prompt = load_prompt(args.prompt_txt)
    negative_prompt = load_prompt(args.negative_prompt_txt) if args.negative_prompt_txt else ""
    print(f"Prompt: {prompt}", flush=True)

    # Load pipeline (single model load for both optional init-gen and panorama)
    print(f"Loading model from: {args.model_path}", flush=True)
    pipe = PerspectiveToPanoramaFluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    )

    if args.enable_vae_tiling:
        pipe.vae.enable_tiling()
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device, dtype=dtype)

    # Obtain the initial perspective image
    if args.input_image:
        input_image = Image.open(args.input_image).convert("RGB")
        print(f"Input image: {input_image.size} ({args.input_image})", flush=True)
    else:
        print("No input image provided — generating one from the prompt.", flush=True)
        input_image = _generate_initial_image(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_inference_steps,
            seed=args.seed,
            device=device,
            save_path_prefix=args.save_path,
        )

    # Panorama generation
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(
        f"Camera: theta={args.input_view_theta}°, phi={args.input_view_phi}°, fov={args.input_fov}",
        flush=True,
    )
    print(
        f"Sphere pts: {args.n_spherical_points}, steps/view: {args.num_inference_steps}",
        flush=True,
    )

    result = pipe(
        input_image=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_fov=tuple(args.input_fov),
        input_view_theta=args.input_view_theta,
        input_view_phi=args.input_view_phi,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        n_spherical_points=args.n_spherical_points,
        min_overlap_ratio=args.min_overlap_ratio,
        repaint_strength=args.repaint_strength,
        weighted_average_temperature=args.temperature,
        erp_height=args.erp_height,
        erp_width=args.erp_width,
    )

    panorama = result.images[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = f"{args.save_path}_{timestamp}.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    panorama.save(out_path)
    print(f"\nSaved panorama to: {out_path}", flush=True)


def _generate_initial_image(
    pipe: PerspectiveToPanoramaFluxPipeline,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    num_steps: int,
    seed: int,
    device: torch.device,
    save_path_prefix: str,
) -> Image.Image:
    """
    Generate a 1024×1024 perspective image using the already-loaded Flux pipeline.
    Calls the base FluxPipeline.__call__ directly to avoid the spherical overhead.
    """
    from diffusers.pipelines.flux import FluxPipeline

    generator = torch.Generator(device=device).manual_seed(seed)

    print("[InitGen] Running Flux text-to-image (1024×1024) ...", flush=True)
    result = FluxPipeline.__call__(
        pipe,
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        height=1024,
        width=1024,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pil",
        max_sequence_length=512,
    )
    init_image = result.images[0]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    init_save = f"{save_path_prefix}_init_{ts}.png"
    os.makedirs(os.path.dirname(init_save) or ".", exist_ok=True)
    init_image.save(init_save)
    print(f"[InitGen] Saved initial image: {init_save}", flush=True)
    return init_image


if __name__ == "__main__":
    main()
