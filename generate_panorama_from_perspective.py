"""
Generate a 360° panoramic image from a single perspective image + text prompt.

Usage examples:

  # Basic usage
  python generate_panorama_from_perspective.py \
    --input_image my_photo.jpg \
    --prompt "A breathtaking mountain landscape, 360 panorama, high quality" \
    --save_path ./outputs/panorama

  # Specify camera direction and FOV
  python generate_panorama_from_perspective.py \
    --input_image my_photo.jpg \
    --prompt "A serene forest scene" \
    --input_view_theta 0 \
    --input_view_phi 0 \
    --input_fov 80 80 \
    --save_path ./outputs/panorama

  # Use fewer sphere points for faster testing
  python generate_panorama_from_perspective.py \
    --input_image my_photo.jpg \
    --prompt "A beautiful sunset" \
    --n_spherical_points 2600 \
    --num_inference_steps 20 \
    --save_path ./outputs/test
"""
import argparse
import os
from datetime import datetime

import torch
from PIL import Image

import pipelines_ours
from pipelines_ours import PerspectiveToPanoramaFluxPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Perspective → Panorama with Flux")

    # Input
    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to the input perspective image")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt describing the panorama")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt (optional)")

    # Camera / FOV
    parser.add_argument("--input_view_theta", type=float, default=0.0,
                        help="Camera azimuth angle in degrees (left/right, 0=front)")
    parser.add_argument("--input_view_phi", type=float, default=0.0,
                        help="Camera elevation angle in degrees (up/down, 0=horizon)")
    parser.add_argument("--input_fov", type=float, nargs=2, default=[80.0, 80.0],
                        metavar=("FOV_H", "FOV_W"),
                        help="Field of view (height, width) in degrees. Default: 80 80")

    # Model
    parser.add_argument("--model_path", type=str,
                        default="black-forest-labs/FLUX.1-dev",
                        help="Hugging Face model path or local path")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["fp16", "bf16", "fp32"])

    # Generation parameters
    parser.add_argument("--n_spherical_points", type=int, default=26500,
                        help="Number of Fibonacci sphere points. Lower = faster but lower quality")
    parser.add_argument("--num_inference_steps", type=int, default=28,
                        help="Number of denoising steps per view")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--min_overlap_ratio", type=float, default=0.25,
                        help="Minimum overlap fraction between new view and already-covered region")
    parser.add_argument("--repaint_strength", type=float, default=1.0,
                        help="Fraction of steps to reinject known latents (0=free, 1=strict)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Gaussian blending temperature for panorama assembly")
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--erp_height", type=int, default=2048)
    parser.add_argument("--erp_width", type=int, default=4096)
    parser.add_argument("--save_path", type=str, default="./outputs/panorama",
                        help="Output path prefix (timestamp + .png appended)")

    # Memory options
    parser.add_argument("--enable_model_cpu_offload", action="store_true",
                        help="Enable CPU offloading to reduce GPU memory usage")
    parser.add_argument("--enable_vae_tiling", action="store_true",
                        help="Enable VAE tiling for large outputs")

    return parser.parse_args()


def main():
    args = parse_args()

    assert os.path.exists(args.input_image), f"Input image not found: {args.input_image}"
    assert torch.cuda.is_available(), "CUDA not available. GPU required."

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.mixed_precision]
    device = torch.device("cuda")

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

    # Load input image
    input_image = Image.open(args.input_image).convert("RGB")
    print(f"Input image: {input_image.size} ({args.input_image})", flush=True)

    # Reproducibility
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"Prompt: {args.prompt}", flush=True)
    print(f"Camera: theta={args.input_view_theta}°, phi={args.input_view_phi}°, "
          f"fov={args.input_fov}", flush=True)
    print(f"Sphere points: {args.n_spherical_points}, "
          f"Steps/view: {args.num_inference_steps}", flush=True)

    result = pipe(
        input_image=input_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
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


if __name__ == "__main__":
    main()
