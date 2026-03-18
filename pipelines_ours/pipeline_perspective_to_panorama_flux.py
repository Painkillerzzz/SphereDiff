"""
Perspective-to-Panorama Pipeline based on Flux.1-dev

Given a single perspective image + prompt, progressively generates a full 360° panorama
by iteratively outpainting into uncovered sphere regions using a greedy coverage strategy.

Key differences from SphericalFluxPipeline:
  - Starts from a real perspective image (not random noise)
  - Greedy view selection: maximise new coverage while keeping overlap with known region
  - Per-view RePaint: reinject already-known sphere latents to maintain consistency
  - Only new (unknown) sphere points are updated after each view's denoising
"""
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.flux import FluxPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from PIL import Image

from .pipeline_spherical_flux import SphericalFluxPipeline, calculate_shift, retrieve_timesteps
from .spherical_functions import SphericalFunctions

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _pil_to_tensor(image: Image.Image, device, dtype) -> torch.Tensor:
    """PIL → (1, 3, H, W) float tensor in [-1, 1]."""
    import numpy as np
    arr = np.array(image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=dtype)


def _get_visible_mask(
    sph_pts: torch.Tensor,    # (1, N, 3)
    view_dir: torch.Tensor,   # (1, 3)
    fov: Tuple[float, float],
) -> torch.Tensor:
    """
    Returns (N,) boolean mask: True if sphere point is visible from view_dir within FOV.
    Points outside FOV or behind the camera get -100 in world_to_perspective, so we
    filter on |u| <= 1 and |v| <= 1.
    """
    coord = SphericalFunctions.world_to_perspective(
        sph_pts.to(torch.float32), view_dir.to(torch.float32), fov=fov
    )  # (N, 2)
    return (coord[:, 0].abs() <= 1.0) & (coord[:, 1].abs() <= 1.0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PerspectiveToPanoramaFluxPipeline(SphericalFluxPipeline):
    """
    Generates a 360° equirectangular panorama from a single perspective image + text prompt.

    Usage::

        from pipelines_ours import PerspectiveToPanoramaFluxPipeline
        from PIL import Image

        pipe = PerspectiveToPanoramaFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
        ).to("cuda")

        init_image = Image.open("my_photo.jpg")
        result = pipe(
            input_image=init_image,
            prompt="A breathtaking mountain landscape, 360 panorama",
            input_fov=(80.0, 80.0),
            input_view_theta=0.0,
            input_view_phi=0.0,
        )
        result.images[0].save("panorama.png")
    """

    # ------------------------------------------------------------------
    # Greedy view-order computation
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _greedy_view_order(
        all_view_dirs: torch.Tensor,      # (V, 3)
        sph_pts_1n3: torch.Tensor,        # (1, N, 3)
        initial_covered: torch.Tensor,    # (N,) bool
        fov: Tuple[float, float],
        min_overlap_ratio: float,
    ) -> List[int]:
        """
        Greedy ordering of view directions to cover the full sphere.
        At each step: select unused view with:
          - overlap with covered region >= min_overlap_ratio  (boundary consistency)
          - maximum new sphere points covered
        Falls back to max-new-coverage if no view meets overlap threshold.
        """
        device = sph_pts_1n3.device
        V = all_view_dirs.shape[0]
        N = sph_pts_1n3.shape[1]

        # Precompute visibility masks: (V, N) bool
        print(f"[PerspToPano] Pre-computing visibility for {V} views × {N} pts ...", flush=True)
        vis = torch.zeros(V, N, dtype=torch.bool, device=device)
        for vi in range(V):
            vis[vi] = _get_visible_mask(sph_pts_1n3, all_view_dirs[vi:vi+1], fov)
        print(f"[PerspToPano] Visibility precomputed.", flush=True)

        covered = initial_covered.clone()
        used = torch.zeros(V, dtype=torch.bool, device=device)
        order: List[int] = []

        while True:
            cov_pct = covered.float().mean().item()
            if cov_pct >= 0.99:
                break

            vis_cnt = vis.sum(dim=1).float().clamp(min=1)               # (V,)
            new_cnt = (vis & ~covered.unsqueeze(0)).sum(dim=1).float()   # (V,)
            ovl_cnt = (vis & covered.unsqueeze(0)).sum(dim=1).float()    # (V,)
            overlap_ratio = ovl_cnt / vis_cnt                             # (V,)

            # Mask exhausted views and views with zero new coverage
            valid = (~used) & (new_cnt > 0)
            if not valid.any():
                break

            eligible = valid & (overlap_ratio >= min_overlap_ratio)

            if eligible.any():
                scores = new_cnt.clone()
                scores[~eligible] = -1.0
                best_vi = int(scores.argmax().item())
            else:
                # Relax overlap constraint — just pick max new coverage
                scores = new_cnt.clone()
                scores[~valid] = -1.0
                best_vi = int(scores.argmax().item())

            order.append(best_vi)
            used[best_vi] = True
            covered |= vis[best_vi]

            print(
                f"  [greedy] step {len(order):3d}: view {best_vi:3d}, "
                f"overlap={overlap_ratio[best_vi]:.2f}, "
                f"new_pts={int(new_cnt[best_vi]):5d}, "
                f"coverage={covered.float().mean().item()*100:.1f}%",
                flush=True,
            )

        print(
            f"[PerspToPano] Greedy done: {len(order)} views, "
            f"coverage={covered.float().mean().item()*100:.1f}%",
            flush=True,
        )
        return order

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        # ---- Input image ----
        input_image: Image.Image,
        input_fov: Tuple[float, float] = (80.0, 80.0),
        input_view_theta: float = 0.0,   # azimuth  in degrees (left/right)
        input_view_phi: float = 0.0,     # elevation in degrees (up/down)
        # ---- Text conditioning ----
        prompt: str = "",
        negative_prompt: str = "",
        max_sequence_length: int = 512,
        # ---- Diffusion parameters ----
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        sigmas: Optional[List[float]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # ---- Sphere / coverage ----
        n_spherical_points: int = 26500,
        min_overlap_ratio: float = 0.25,
        weighted_average_temperature: float = 0.1,
        # ---- RePaint strength ----
        # Fraction of denoising steps during which the known region is re-injected.
        # 1.0 = re-inject at every step (maximum consistency).
        repaint_strength: float = 1.0,
        # ---- Output ----
        erp_height: int = 2048,
        erp_width: int = 4096,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable] = None,
    ):
        device = self._execution_device
        dtype = self.dtype

        # ----------------------------------------------------------------
        # 1. Encode text prompt
        # ----------------------------------------------------------------
        (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )

        # ----------------------------------------------------------------
        # 2. Initialise spherical latent representation
        # ----------------------------------------------------------------
        # Flux transformer: in_channels = 64 = C_vae(16) × 4 (2×2 patchify)
        C_vae = self.transformer.config.in_channels // 4   # 16
        C_packed = C_vae * 4                                # 64  (= in_channels)

        sph_pts_flat = SphericalFunctions.fibonacci_sphere(N=n_spherical_points).to(device, dtype)
        N = sph_pts_flat.shape[0]

        # sphere_latents stores PACKED latents: (1, C_packed, 1, N)
        # Each sphere point = one 2×2 patch in VAE latent space
        sphere_latents = torch.zeros(1, C_packed, 1, N, device=device, dtype=dtype)
        sphere_covered = torch.zeros(N, dtype=torch.bool, device=device)

        # shape needed by SphericalFunctions: (B, F, N, 3)
        sph_pts_B1N3 = sph_pts_flat.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 3)
        # shape (1, N, 3) for _get_visible_mask
        sph_pts_1N3 = sph_pts_flat.unsqueeze(0)                # (1, N, 3)

        # ----------------------------------------------------------------
        # 3. Encode input perspective image → project onto sphere
        # ----------------------------------------------------------------
        fov = input_fov  # same FOV used for all views

        input_view_dir = SphericalFunctions.spherical_to_cartesian(
            torch.tensor([math.radians(input_view_theta)], device=device, dtype=dtype),
            torch.tensor([math.radians(input_view_phi)], device=device, dtype=dtype),
        )  # (1, 3)

        # Determine latent spatial size for the initial view
        lat_h0, lat_w0 = SphericalFunctions.get_height_width_from_fov(fov, n_spherical_points)
        # VAE latent for this view: (2*lat_h0, 2*lat_w0, C_vae)
        # Pixel image for this view: (2*lat_h0 * vae_scale_factor) × (2*lat_w0 * vae_scale_factor)
        px_h0 = 2 * lat_h0 * self.vae_scale_factor
        px_w0 = 2 * lat_w0 * self.vae_scale_factor

        # Resize and encode
        img_resized = input_image.resize((px_w0, px_h0), Image.LANCZOS)
        img_tensor = _pil_to_tensor(img_resized, device, self.vae.dtype)   # (1,3,H,W) ∈ [-1,1]

        vae_latent0 = self.vae.encode(img_tensor).latent_dist.sample(generator)
        vae_latent0 = vae_latent0 * self.vae.config.scaling_factor         # (1, C_vae, lat_h0*2, lat_w0*2)

        # Interpolate to exact expected size if VAE changes dimensions
        vae_h_expected, vae_w_expected = 2 * lat_h0, 2 * lat_w0
        if vae_latent0.shape[-2] != vae_h_expected or vae_latent0.shape[-1] != vae_w_expected:
            vae_latent0 = F.interpolate(
                vae_latent0.float(), size=(vae_h_expected, vae_w_expected),
                mode='bilinear', align_corners=False,
            ).to(dtype)

        # 2×2 patchify: (1, C_vae, 2H, 2W) → (1, H*W, C_packed)
        packed0 = self._pack_latents(vae_latent0.to(dtype), 1, C_vae, 2 * lat_h0, 2 * lat_w0)
        # packed0: (1, M0, C_packed) where M0 = lat_h0 * lat_w0

        # Get ordered sphere indices for the initial view
        indices_init, _ = SphericalFunctions.dynamic_laetent_sampling(
            sph_pts_B1N3, input_view_dir, n_spherical_points, fov,
            temperature=weighted_average_temperature, center_first=False,
        )  # (M0,) — sphere point indices for this view, in perspective-grid order

        M0 = indices_init.shape[0]
        if M0 != packed0.shape[1]:
            # Mismatch: resize packed latent to match sphere point count
            lat_h_actual = round(M0 ** 0.5)
            vae_latent0_r = F.interpolate(
                vae_latent0.float(), size=(2 * lat_h_actual, 2 * lat_h_actual),
                mode='bilinear', align_corners=False,
            ).to(dtype)
            packed0 = self._pack_latents(vae_latent0_r, 1, C_vae, 2 * lat_h_actual, 2 * lat_h_actual)

        # Store in sphere: sphere_latents (1, C_packed, 1, N)
        # packed0: (1, M0, C_packed) → permute → (1, C_packed, M0) → store at indices
        sphere_latents[0, :, 0, indices_init] = packed0[0].T   # (C_packed, M0)
        sphere_covered[indices_init] = True

        init_cov = sphere_covered.float().mean().item() * 100
        print(
            f"[PerspToPano] Init: {sphere_covered.sum().item()} / {N} pts covered ({init_cov:.1f}%)",
            flush=True,
        )

        # ----------------------------------------------------------------
        # 4. Greedy view-order computation
        # ----------------------------------------------------------------
        all_view_dirs = SphericalFunctions.horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator(
            fov_single=fov[0]
        ).to(device, dtype)   # (V, 3)

        view_order = self._greedy_view_order(
            all_view_dirs, sph_pts_1N3, sphere_covered, fov, min_overlap_ratio,
        )

        # ----------------------------------------------------------------
        # 5. Setup timesteps (shared across all views)
        # ----------------------------------------------------------------
        # Use a representative latent to compute mu for shift schedule
        lat_h_ref, lat_w_ref = SphericalFunctions.get_height_width_from_fov(fov, n_spherical_points)
        M_ref = lat_h_ref * lat_w_ref  # number of patches per view

        sigmas_arr = (
            np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
            if sigmas is None else sigmas
        )
        mu = calculate_shift(
            M_ref,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas_arr, mu=mu,
        )

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        else:
            guidance = None

        repaint_steps = max(1, int(repaint_strength * num_inference_steps))

        # ----------------------------------------------------------------
        # 6. Progressive outpainting loop
        # ----------------------------------------------------------------
        total_views = len(view_order)
        print(f"[PerspToPano] Starting progressive generation over {total_views} views ...", flush=True)

        for step_idx, vi in enumerate(view_order):
            cur_view_dir = all_view_dirs[vi:vi+1]   # (1, 3)

            # Which sphere points are visible from this view?
            vis_mask = _get_visible_mask(sph_pts_1N3, cur_view_dir, fov)   # (N,) bool
            known_mask   = vis_mask & sphere_covered        # already generated
            unknown_mask = vis_mask & (~sphere_covered)     # need to generate

            # Dynamic latent sampling: ordered sphere indices for this view
            indices_v, _ = SphericalFunctions.dynamic_laetent_sampling(
                sph_pts_B1N3, cur_view_dir, n_spherical_points, fov,
                temperature=weighted_average_temperature, center_first=False,
            )   # (M,) where M = lat_h * lat_w

            M = indices_v.shape[0]
            lat_h = round(M ** 0.5)
            lat_w = lat_h

            # Which of the M patches are known / unknown?
            known_in_view   = known_mask[indices_v]   # (M,) bool
            unknown_in_view = unknown_mask[indices_v]  # (M,) bool

            # ---- Build initial latent for this view ----
            # Known positions: use sphere latents (they are "clean" encoded values)
            # Unknown positions: pure noise
            # Shape needed by transformer: (1, M, C_packed)

            init_latent = randn_tensor(
                (1, M, C_packed), generator=generator, device=device, dtype=dtype,
            )  # all-noise baseline

            # Place known sphere latents (unnoised) at known positions
            # We re-noise them to t=timesteps[0] level below in the RePaint loop,
            # but initialise here with clean content so step 0 has context.
            if known_in_view.any():
                k_pos = torch.where(known_in_view)[0]  # positions in M-list
                # sphere_latents: (1, C_packed, 1, N) → select → (C_packed, K) → T → (K, C_packed)
                clean_k = sphere_latents[0, :, 0, indices_v[k_pos]].T  # (K, C_packed)
                sigma0 = timesteps[0].item() / 1000.0
                noise_k = torch.randn_like(clean_k)
                # Add noise at first timestep level
                init_latent[0, k_pos] = (
                    (1.0 - sigma0) * clean_k + sigma0 * noise_k
                ).to(dtype)

            view_latent = init_latent   # (1, M, C_packed)

            # Prepare positional IDs for this view
            latent_image_ids = self._prepare_latent_image_ids(
                1, lat_h, lat_w, device, dtype
            )

            # ---- Denoising loop ----
            self.scheduler._step_index = None

            for t_idx, t in enumerate(timesteps):
                timestep = t.expand(1).to(dtype)

                noise_pred = self.transformer(
                    hidden_states=view_latent,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

                self.scheduler._step_index = None
                view_latent = self.scheduler.step(
                    noise_pred, t, view_latent, return_dict=False
                )[0]   # (1, M, C_packed)

                # RePaint: reinject known region at the noise level of the NEXT timestep
                if t_idx < repaint_steps - 1 and known_in_view.any():
                    sigma_next = timesteps[min(t_idx + 1, len(timesteps) - 1)].item() / 1000.0
                    k_pos = torch.where(known_in_view)[0]
                    clean_k = sphere_latents[0, :, 0, indices_v[k_pos]].T  # (K, C_packed)
                    noise_k = torch.randn_like(clean_k)
                    reinjected = ((1.0 - sigma_next) * clean_k + sigma_next * noise_k).to(dtype)
                    view_latent[0, k_pos] = reinjected

            # ---- Update sphere with newly generated points ----
            unk_pos = torch.where(unknown_in_view)[0]   # positions in M-list
            if unk_pos.numel() > 0:
                # view_latent: (1, M, C_packed) → select → (K_unk, C_packed) → T → (C_packed, K_unk)
                new_vals = view_latent[0, unk_pos].T   # (C_packed, K_unk)
                sphere_latents[0, :, 0, indices_v[unk_pos]] = new_vals.to(sphere_latents.dtype)
                sphere_covered[indices_v[unk_pos]] = True

            cov_pct = sphere_covered.float().mean().item() * 100
            print(
                f"[PerspToPano] View {step_idx+1}/{total_views} (idx {vi}): "
                f"+{unk_pos.numel()} new pts → coverage {cov_pct:.1f}%",
                flush=True,
            )

            if callback_on_step_end is not None:
                callback_on_step_end(self, step_idx, vi, {"sphere_covered": sphere_covered})

        # ----------------------------------------------------------------
        # 7. Decode sphere latents → ERP panorama
        # ----------------------------------------------------------------
        print("[PerspToPano] Decoding sphere to ERP panorama ...", flush=True)

        decode_view_dirs = SphericalFunctions.horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator(
            fov_single=fov[0]
        ).to(device, dtype)

        wb     = torch.zeros(1, 3, 1, erp_height, erp_width, device=device, dtype=torch.float32)
        wb_cnt = torch.zeros_like(wb)

        with self.progress_bar(total=len(decode_view_dirs)) as pbar:
            for vi in range(len(decode_view_dirs)):
                cur_view_dir = decode_view_dirs[vi:vi+1]

                indices_v, weight_v = SphericalFunctions.dynamic_laetent_sampling(
                    sph_pts_B1N3, cur_view_dir, n_spherical_points, fov,
                    temperature=weighted_average_temperature, center_first=False,
                )
                M = indices_v.shape[0]
                lat_h = round(M ** 0.5)

                # Check coverage
                if not sphere_covered[indices_v].any():
                    pbar.update()
                    continue

                # Retrieve packed latents: (1, C_packed, 1, M) → (1, M, C_packed)
                packed_v = sphere_latents[0, :, 0, indices_v].T.unsqueeze(0)  # (1, M, C_packed)

                # Unpack → VAE latent: (1, C_vae, 2*lat_h, 2*lat_w)
                vae_lat = self._unpack_latents(packed_v, lat_h * 2, lat_h * 2, 1)

                # VAE decode
                vae_lat = vae_lat.to(self.vae.dtype)
                img_v = self.vae.decode(
                    vae_lat / self.vae.config.scaling_factor, return_dict=False
                )[0]   # (1, 3, H_px, W_px)

                img_v = img_v.unsqueeze(2)  # (1, 3, 1, H_px, W_px) for paste function

                wb, wb_cnt = SphericalFunctions.paste_perspective_to_erp_rectangle(
                    wb, img_v.to(wb.device, dtype=torch.float32),
                    cur_view_dir.to(wb.device, dtype=torch.float32),
                    fov=fov, add=True, interpolate=True, interpolation_mode='bilinear',
                    panorama_cnt=wb_cnt, return_cnt=True,
                    temperature=weighted_average_temperature,
                )
                pbar.update()

        wb_cnt[wb_cnt == 0] = 1
        wb = wb / wb_cnt

        image_out = self.image_processor.postprocess(wb[:, :, 0], output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_out,)
        return FluxPipelineOutput(images=image_out)
