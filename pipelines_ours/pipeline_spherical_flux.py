import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux import FluxPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from .spherical_functions import SphericalFunctions

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        # todo
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class SphericalFluxPipeline(FluxPipeline):
    @staticmethod
    def _pack_latents_for_spherical(latents, batch_size, num_channels_latents_for_spherical, height, width):
        latents = latents.permute(0, 2, 1)  # (batch_size, height, width, num_channels_latents_for_spherical)
        latents = latents.reshape(batch_size, height * width, num_channels_latents_for_spherical)
        return latents

    @staticmethod
    def _unpack_latents_for_spherical(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        latents = latents.permute(0, 2, 1)
        latents = latents.reshape(batch_size, channels, height, width)
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt_txt_path: str = None,  # (modified) SphereDiff
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_txt_path: str = "",
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        ### Spherical options ###
        n_spherical_points: int = 26500,
        weighted_average_temperature: float = 0.1,
        erp_height: int = 2048,
        erp_width: int = 4096,
        use_anisotropic_fov: bool = False,
        use_adaptive_temperature: bool = False,
        fibonacci_randomize: bool = False,
        view_batch_size: int = 1,
        vae_batch_size: int = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        device = self._execution_device

        # load prompts
        with open(prompt_txt_path, 'r') as f:
            lines = f.readlines()
        prompt_raw = [line.strip() for line in lines]
        assert len(prompt_raw) == 5, 'prompt_txt_path should contain 5 lines'
        prompt, thetas, phis, prompt_fovs = [], [], [], []
        phis_raw = [-90, -10, 0, 10, 90]
        for i in range(len(phis_raw)):
            for theta in [0, 90, 180, 270]:
                prompt.append(prompt_raw[i])
                thetas.append(math.radians(theta))
                phis.append(math.radians(phis_raw[i]))
                prompt_fovs.append((80, 80))
        thetas = torch.tensor(thetas, device=device, dtype=self.dtype)
        phis = torch.tensor(phis, device=device, dtype=self.dtype)
        prompt_dir = SphericalFunctions.spherical_to_cartesian(thetas, phis)

        if negative_prompt_txt_path != '' and negative_prompt_txt_path is not None:
            with open(negative_prompt_txt_path, 'r') as f:
                negative_prompt = f.read().strip('\n')
        else:
            negative_prompt = ''

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        batch_size = 1

        num_prompt = len(prompt)

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        num_channels_latents = num_channels_latents * 4
        spherical_points = SphericalFunctions.fibonacci_sphere(N=n_spherical_points, randomize=fibonacci_randomize).to(device, dtype=self.dtype)  # (N, 3)
        num_points_on_sphere = spherical_points.shape[0]
        shape = (batch_size, num_channels_latents, 1, num_points_on_sphere)
        spherical_points = spherical_points.repeat(batch_size, 1, 1, 1)

        view_dir = SphericalFunctions.horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator()
        view_dir = view_dir.to(device, dtype=self.dtype)  # (N, 3)
        num_inference_steps_view_dir = len(view_dir)
        multi_prompts_indices_main, fovs_main = SphericalFunctions.get_prompt_indices(view_dir, prompt_dir, prompt_fovs)

        print(f'num_points_on_sphere = {num_points_on_sphere}, num_inference_steps_view_dir = {num_inference_steps_view_dir}')

        # --- P2: Anisotropic FOV per view direction based on latitude ---
        phi_angles = torch.asin(view_dir[:, 1].clamp(-1, 1))  # (V,) in radians
        if use_anisotropic_fov:
            view_fovs = []
            for _j in range(len(view_dir)):
                phi_abs = phi_angles[_j].abs().item()
                fov_h = 80.0
                fov_v = float(max(80.0 * math.cos(phi_abs), 20.0))
                view_fovs.append((fov_h, fov_v))
        else:
            view_fovs = list(fovs_main)

        # --- P1: Adaptive blending temperature per view direction ---
        if use_adaptive_temperature:
            phi_normalized = phi_angles.abs() / (math.pi / 2)  # 0 at equator, 1 at poles
            temperatures_per_view = (0.05 + 0.15 * phi_normalized).tolist()
        else:
            temperatures_per_view = [weighted_average_temperature] * len(view_dir)

        # --- P1: Precompute dynamic latent sampling (timestep-invariant) ---
        print('Precomputing dynamic latent sampling indices...')
        precomputed_sampling = []
        for _j in range(len(view_dir)):
            _cur_view_dir = view_dir[_j].unsqueeze(0)  # (1, 3)
            _fov_j = view_fovs[_j]
            _temp_j = temperatures_per_view[_j]
            _idx_j, _w_j = SphericalFunctions.dynamic_laetent_sampling(
                spherical_points, _cur_view_dir, num_points_on_sphere, _fov_j,
                temperature=_temp_j, center_first=False,
            )
            precomputed_sampling.append((_idx_j, _w_j))
        print('Precomputation done.')

        latents = randn_tensor(shape, generator, device, dtype=self.dtype)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # --- P3a: Precompute view mini-batches grouped by equal patch size ---
        from collections import defaultdict as _defaultdict
        _patch_groups = _defaultdict(list)
        for _j in range(len(view_dir)):
            _n = precomputed_sampling[_j][0].shape[-1]
            _patch_groups[_n].append(_j)
        _view_batches = []
        for _n, _js in sorted(_patch_groups.items()):
            for _s in range(0, len(_js), view_batch_size):
                _view_batches.append(_js[_s:_s + view_batch_size])
        print(f'P3a: {len(view_dir)} views → {len(_view_batches)} batches (view_batch_size={view_batch_size})')
        # VAE decode uses its own (typically smaller) batch size to avoid OOM
        _vae_batches = []
        for _n, _js in sorted(_patch_groups.items()):
            for _s in range(0, len(_js), vae_batch_size):
                _vae_batches.append(_js[_s:_s + vae_batch_size])

        # 6. Denoising loop
        n_total = len(view_dir) * len(timesteps)

        def selected_j_inside(j_inside):  # use it for debugging
            # return j_inside == 2
            # return j_inside in (0, 1, 14, 15, 29, 43, 54, 65, 73, 81, 85)
            return True

        progress_bar = self.progress_bar(total=n_total)
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t

            latents_next = torch.zeros_like(latents)
            latents_next_cnt = torch.zeros_like(latents)

            _multi_prompts_indices = multi_prompts_indices_main

            if image_embeds is not None:
                self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

            for _batch in _view_batches:
                K_b = len(_batch)
                j0 = _batch[0]
                indices_0, _ = precomputed_sampling[j0]
                N_b = indices_0.shape[-1]
                h_b = round(N_b ** 0.5)

                # Collect per-view latent patches and prompt embeddings
                _all_lat = []
                _all_pe = []
                _all_ppe = []
                _all_idx = []
                _all_w = []
                for _jj in _batch:
                    _idx_jj, _w_jj = precomputed_sampling[_jj]
                    _lat_jj = latents[..., _idx_jj].squeeze(2)  # (B, C, N_b)
                    _lat_jj = self._pack_latents_for_spherical(
                        _lat_jj, batch_size, num_channels_latents, h_b, h_b)  # (B, N_b, C)
                    _all_lat.append(_lat_jj)
                    _all_pe.append(prompt_embeds[_multi_prompts_indices[_jj]])   # (seq_len, dim)
                    _all_ppe.append(pooled_prompt_embeds[_multi_prompts_indices[_jj]])  # (pooled_dim,)
                    _all_idx.append(_idx_jj)
                    _all_w.append(_w_jj)

                # Build batched inputs (B==1 assumed; generalises trivially)
                latent_model_input = torch.cat(_all_lat, dim=0).to(self.dtype)  # (K_b, N_b, C)
                _pe_batch = torch.stack(_all_pe, dim=0)                          # (K_b, seq_len, dim)
                _ppe_batch = torch.stack(_all_ppe, dim=0)                        # (K_b, pooled_dim)
                _img_ids_batch = self._prepare_latent_image_ids(
                    1, h_b, h_b, device, latents.dtype).squeeze(0)              # (N_b, 3) — 2D, transformer broadcasts
                _timestep_batch = t.expand(K_b).to(latents.dtype)                # (K_b,)
                _guidance_batch = guidance.expand(K_b) if guidance is not None else None

                ### Batched transformer forward (P3a) ###
                self.scheduler._step_index = None  # ! important
                noise_pred_batch = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=_timestep_batch / 1000,
                    guidance=_guidance_batch,
                    pooled_projections=_ppe_batch,
                    encoder_hidden_states=_pe_batch,
                    txt_ids=text_ids,        # (seq_len, 3) — positional-only, same for all
                    img_ids=_img_ids_batch,  # (K_b, N_b, 3)
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]  # (K_b, N_b, C)

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    # neg prompts are a single shared embedding; expand to K_b
                    _neg_pe_batch = negative_prompt_embeds.expand(K_b, -1, -1)
                    _neg_ppe_batch = negative_pooled_prompt_embeds.expand(K_b, -1)
                    self.scheduler._step_index = None
                    neg_noise_pred_batch = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=_timestep_batch / 1000,
                        guidance=_guidance_batch,
                        pooled_projections=_neg_ppe_batch,
                        encoder_hidden_states=_neg_pe_batch,
                        txt_ids=negative_text_ids,
                        img_ids=_img_ids_batch,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred_batch = neg_noise_pred_batch + true_cfg_scale * (noise_pred_batch - neg_noise_pred_batch)

                # Batched scheduler step (FlowMatch Euler is linear — safe to batch)
                latents_dtype = latents.dtype
                self.scheduler._step_index = None  # ! important
                latent_model_input_stepped = self.scheduler.step(
                    noise_pred_batch, t, latent_model_input, return_dict=False)[0]  # (K_b, N_b, C)

                # Scatter denoised patches back to spherical latents
                for _k, _jj in enumerate(_batch):
                    _idx_jj, _w_jj = _all_idx[_k], _all_w[_k]
                    _lat_out = latent_model_input_stepped[_k:_k + 1]  # (1, N_b, C)
                    _lat_out = self._unpack_latents_for_spherical(_lat_out, h_b, h_b, 1)  # (1, C, h_b, h_b)
                    _lat_out = rearrange(_lat_out, 'b c h w -> b c 1 (h w)')
                    for idx_b in range(batch_size):
                        latents_next[idx_b, ..., _idx_jj] += _lat_out[idx_b] * _w_jj
                        latents_next_cnt[idx_b, ..., _idx_jj] += _w_jj

                progress_bar.update(K_b)
                progress_bar.set_description_str(f'i: {i}, batch_j0: {j0}')
                progress_bar.set_postfix_str(f'K_b={K_b}, N_b={N_b}')

            latents_next_cnt[latents_next_cnt == 0] = 1
            latents = latents_next / latents_next_cnt

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

        progress_bar.close()

        self._current_timestep = None

        wb = torch.zeros((batch_size, 3, 1, erp_height, erp_width), device=device, dtype=torch.float)
        wb_cnt = torch.zeros_like(wb)

        with self.progress_bar(total=len(view_dir)) as progress_bar:
            for _batch in _vae_batches:
                K_b = len(_batch)
                j0 = _batch[0]
                indices_0, _ = precomputed_sampling[j0]
                N_b = indices_0.shape[-1]
                h_b = round(N_b ** 0.5)

                # Build batched VAE input (P3a: same patch size within group)
                _vae_inputs = []
                for _jj in _batch:
                    _idx_jj, _ = precomputed_sampling[_jj]
                    _lat_jj = latents[..., _idx_jj].squeeze(2)  # (1, C, N_b)
                    _lat_jj = self._unpack_latents(
                        _lat_jj.permute(0, 2, 1), h_b * 2, h_b * 2, 1)  # (1, C, H_b, W_b)
                    _vae_inputs.append(_lat_jj.to(self.vae.dtype))

                _vae_input_batch = torch.cat(_vae_inputs, dim=0)  # (K_b, C, H_b, W_b)
                images_batch = self.vae.decode(
                    _vae_input_batch / self.vae.config.scaling_factor, return_dict=False)[0]  # (K_b, C, H_b, W_b)

                # Paste each decoded view into the ERP canvas
                for _k, _jj in enumerate(_batch):
                    _image_k = images_batch[_k:_k + 1].unsqueeze(2)  # (1, C, 1, H_b, W_b)
                    _cur_view_dir = view_dir[_jj].repeat(batch_size, 1)
                    _fov_vae = view_fovs[_jj]  # P0 bugfix + P2 anisotropic FOV
                    wb, wb_cnt = SphericalFunctions.paste_perspective_to_erp_rectangle(
                        wb, _image_k.to(wb.device, wb.dtype),
                        _cur_view_dir.to(wb.device, wb.dtype), fov=_fov_vae,
                        add=True, interpolate=True, interpolation_mode='bilinear',
                        panorama_cnt=wb_cnt, return_cnt=True,
                        temperature=temperatures_per_view[_jj],  # P1 adaptive
                    )

                progress_bar.update(K_b)

        wb_cnt[wb_cnt == 0] = 1
        wb /= wb_cnt

        image = self.image_processor.postprocess(wb[:, :, 0, :, :], output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, )

        return FluxPipelineOutput(images=image)
