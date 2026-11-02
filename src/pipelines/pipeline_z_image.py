# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for GlyphAnchor

import ast
import inspect
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, PreTrainedModel

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, ZImageLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from ..models.transformer_z_image import ZImageTransformer2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput


from .layout_utils import (
    _bbox_iou_xywh_norm,
    _bbox_norm_xywh_to_xyxy_px,
    _boxes_intersect,
    _sort_matches_reading_order,
    _xyxy_px_to_bbox_norm,
    bbox_norm_xywh_to_pixel_wh,
    dedup_layout_matches,
    load_glyph_layout_items,
    merge_layout_matches_to_blocks,
    parse_layout_bbox,
)
from .glyph_utils import (
    anchor_glyph_patches,
    GLYPH_BG,
    GLYPH_FG,
    GLYPH_FONT_PATH,
    GLYPH_FONT_SIZE,
    GLYPH_IMAGE_HEIGHT,
    GLYPH_PAD_X,
    _break_at_word_boundary,
    _fit_layout_font,
    _get_text_bbox,
    _layout_get_line_height,
    _layout_get_render_lines,
    _measure_layout_lines,
    _measure_text,
    _measure_text_width,
    _rewrap_lines_by_width,
    _simulate_tight_bbox,
    _wrap_line_by_width,
    _wrap_text_by_width,
    ceil_to,
    compose_glyph_group_image,
    ensure_side,
    get_font,
    render_full_glyph_canvas as _render_full_glyph_canvas,
    render_glyph_image,
    render_one_glyph_patch,
    round_to,
)
from .scheduler_utils import (
    calculate_shift,
    retrieve_timesteps,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ─── VAE 适配辅助（同时支持 2D XQ VAE 和 3D Wan VAE）───────────────────────────
class _DecodeResult:
    __slots__ = ("sample",)
    def __init__(self, sample):
        self.sample = sample


def _vae_encode(vae, x):
    if getattr(vae, "_is_3d", False):
        return vae.encode(x.unsqueeze(2))
    return vae.encode(x)


def _vae_decode(vae, z, return_dict=True):
    if getattr(vae, "_is_3d", False):
        out = vae.decode(z.unsqueeze(2)).sample.squeeze(2)
        return _DecodeResult(out) if return_dict else (out,)
    return vae.decode(z, return_dict=return_dict)


def _vae_shift(vae):
    v = getattr(vae, "_shift_factor", None)
    return v if v is not None else vae.config.shift_factor


def _vae_scale(vae):
    v = getattr(vae, "_scaling_factor", None)
    return v if v is not None else vae.config.scaling_factor

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import ZImagePipeline

        >>> pipe = ZImagePipeline.from_pretrained("Z-a-o/Z-Image-Turbo", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> # Optionally, set the attention backend to flash-attn 2 or 3, default is SDPA in PyTorch.
        >>> # (1) Use flash attention 2
        >>> # pipe.transformer.set_attention_backend("flash")
        >>> # (2) Use flash attention 3
        >>> # pipe.transformer.set_attention_backend("_flash_3")

        >>> prompt = "一幅为名为“造相「Z-IMAGE-TURBO」”的项目设计的创意海报。画面巧妙地将文字概念视觉化：一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。"
        >>> image = pipe(
        ...     prompt,
        ...     height=1024,
        ...     width=1024,
        ...     num_inference_steps=9,
        ...     guidance_scale=0.0,
        ...     generator=torch.Generator("cuda").manual_seed(42),
        ... ).images[0]
        >>> image.save("zimage.png")
        ```
"""


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps


CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1536 * 1536


def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def adjust_matches_by_real_glyph_overlap_2d(matches, glyph_images, canvas_size, gap_px=2, row_thresh=0.6, col_thresh=0.6, max_iter=8):
    W, H = canvas_size
    out = []
    items = []
    for m, g in zip(matches, glyph_images):
        mm = dict(m)
        mm["bbox_norm_xywh"] = list(m["bbox_norm_xywh"])
        out.append(mm)
        x, y, w, h = mm["bbox_norm_xywh"]
        bw = w * W
        bh = h * H
        if not isinstance(g, Image.Image):
            g = Image.fromarray(g)
        gw, gh = g.size
        cx = x * W + bw / 2
        cy = y * H + bh / 2
        items.append({"w_norm": w, "h_norm": h, "bw": bw, "bh": bh, "gw": gw, "gh": gh, "base_cx": cx, "base_cy": cy, "shift_x": 0.0, "shift_y": 0.0})

    def rect(it):
        cx = it["base_cx"] + it["shift_x"]
        cy = it["base_cy"] + it["shift_y"]
        return {"left": cx - it["gw"] / 2, "top": cy - it["gh"] / 2, "right": cx + it["gw"] / 2, "bottom": cy + it["gh"] / 2, "cx": cx, "cy": cy}

    def overlap_1d(a1, a2, b1, b2):
        return min(a2, b2) - max(a1, b1)

    def move_pair_x(i, j, need):
        ri = rect(items[i])
        rj = rect(items[j])
        li, ri_idx = (i, j) if ri["cx"] <= rj["cx"] else (j, i)
        rl = rect(items[li])
        rr = rect(items[ri_idx])
        cap_l = max(0.0, rl["left"])
        cap_r = max(0.0, W - rr["right"])
        take_l = min(need / 2, cap_l)
        take_r = min(need / 2, cap_r)
        remain = need - take_l - take_r
        extra_r = min(remain, cap_r - take_r)
        take_r += extra_r
        remain -= extra_r
        extra_l = min(remain, cap_l - take_l)
        take_l += extra_l
        items[li]["shift_x"] -= take_l
        items[ri_idx]["shift_x"] += take_r
        return take_l + take_r

    def move_pair_y(i, j, need):
        ri = rect(items[i])
        rj = rect(items[j])
        di = j if ri["cy"] <= rj["cy"] else i
        items[di]["shift_y"] += need
        return need

    order = sorted(range(len(items)), key=lambda i: (items[i]["base_cy"], items[i]["base_cx"]))
    for _ in range(max_iter):
        changed = False
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                i = order[a]
                j = order[b]
                ri = rect(items[i])
                rj = rect(items[j])
                ox = overlap_1d(ri["left"], ri["right"], rj["left"], rj["right"])
                oy = overlap_1d(ri["top"], ri["bottom"], rj["top"], rj["bottom"])
                if ox <= 0 or oy <= 0:
                    continue
                same_row = abs(items[i]["base_cy"] - items[j]["base_cy"]) <= row_thresh * max(items[i]["bh"], items[j]["bh"])
                same_col = abs(items[i]["base_cx"] - items[j]["base_cx"]) <= col_thresh * max(items[i]["bw"], items[j]["bw"])
                need_x = ox + gap_px
                need_y = oy + gap_px
                first_axis = "x" if (same_row and not same_col) or (not (same_col and not same_row) and need_x <= need_y) else "y"
                moved = move_pair_x(i, j, need_x) if first_axis == "x" else move_pair_y(i, j, need_y)
                if moved > 1e-6:
                    changed = True
                ri = rect(items[i])
                rj = rect(items[j])
                ox = overlap_1d(ri["left"], ri["right"], rj["left"], rj["right"])
                oy = overlap_1d(ri["top"], ri["bottom"], rj["top"], rj["bottom"])
                if ox > 0 and oy > 0:
                    moved = move_pair_y(i, j, oy + gap_px) if first_axis == "x" else move_pair_x(i, j, ox + gap_px)
                    if moved > 1e-6:
                        changed = True
        if not changed:
            break

    # If any box's bottom overflows the canvas, uniformly scale the layout
    # vertically so the bottom-most box just fits within the canvas.
    if items:
        max_bottom_px = max(it["base_cy"] + it["shift_y"] + it["bh"] / 2 for it in items)
        if max_bottom_px > H:
            scale = H / max_bottom_px
            for it in items:
                it["base_cy"] *= scale
                it["shift_y"] *= scale
                it["bh"] *= scale
                it["h_norm"] *= scale

    for k, it in enumerate(items):
        w = it["w_norm"]
        h = it["h_norm"]
        new_x = (it["base_cx"] + it["shift_x"] - it["bw"] / 2) / W
        new_y = (it["base_cy"] + it["shift_y"] - it["bh"] / 2) / H
        out[k]["bbox_norm_xywh"] = [min(max(new_x, 0.0), max(0.0, 1.0 - w)), min(max(new_y, 0.0), max(0.0, 1.0 - h)), w, h]
    return out


def render_full_glyph_canvas(layout_items, width: int, height: int,
                             glyph_patch_mode: str = "match_box",
                             debug: bool = False) -> Image.Image:
    return _render_full_glyph_canvas(
        layout_items, width, height, glyph_patch_mode, debug,
        adjust_matches=adjust_matches_by_real_glyph_overlap_2d,
    )


class ZImagePipeline(DiffusionPipeline, ZImageLoraLoaderMixin, FromSingleFileMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        transformer: ZImageTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )
        if hasattr(self, "vae") and self.vae is not None:
            _cfg = self.vae.config
            self.vae_scale_factor = getattr(_cfg, "scale_factor_spatial", None) \
                or 2 ** (len(_cfg.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 128

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            assert len(prompt) == len(negative_prompt)
            negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = []
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    def build_layout_matches_and_glyphs(
        self,
        glyph_layout_items,
        canvas_size,
        glyph_patch_mode: str = "fixed_line_height",
        debug: bool = False,
    ):
        if not glyph_layout_items:
            return [], []

        raw_matches = []
        for item in glyph_layout_items:
            tight = _simulate_tight_bbox(item, canvas_size)
            bbox_norm_xywh = tight["bbox_norm_xywh"]
            if not bbox_norm_xywh:
                continue
            raw_matches.append(
                {
                    **({"align": item["align"]} if "align" in item else {}),
                    **({"layout_frame": list(item["bbox"])} if "align" in item or "valign" in item else {}),
                    **({"valign": item["valign"]} if "valign" in item else {}),
                    "text": tight["text"],
                    "bbox": tight["bbox"],
                    "bbox_norm_xywh": bbox_norm_xywh,
                }
            )

        raw_matches = dedup_layout_matches(raw_matches)
        raw_layout_glyph_images = [render_one_glyph_patch(m, canvas_size, glyph_patch_mode) for m in raw_matches]
        raw_matches = anchor_glyph_patches(raw_matches, raw_layout_glyph_images, canvas_size)
        raw_matches = adjust_matches_by_real_glyph_overlap_2d(raw_matches, raw_layout_glyph_images, canvas_size, gap_px=1)
        matches = merge_layout_matches_to_blocks(
            raw_matches,
            canvas_size,
            glyph_images=raw_layout_glyph_images,
            expand_x_ratio=0.05,
            expand_y_ratio=0.20,
            max_height_ratio=1.8,
        )

        vae_glyph_images = []
        for m in matches:
            children = m.get("children")
            if children and len(children) > 1:
                glyph = compose_glyph_group_image(children, raw_layout_glyph_images, m["bbox_norm_xywh"], canvas_size)
            elif children and len(children) == 1:
                glyph = raw_layout_glyph_images[children[0]["idx"]]
            else:
                glyph = render_one_glyph_patch(m, canvas_size, glyph_patch_mode)
            glyph = ensure_side(glyph, min_side=16, multiple=16)
            vae_glyph_images.append(glyph)

        if debug:
            debug_canvas = Image.new("RGB", canvas_size, "white")
            draw = ImageDraw.Draw(debug_canvas)
            img_w, img_h = canvas_size
            for match, glyph_img in zip(matches, vae_glyph_images):
                bbox = match.get("bbox_norm_xywh")
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                x1 = int(x * img_w)
                y1 = int(y * img_h)
                bw = int(w * img_w)
                bh = int(h * img_h)
                x2 = x1 + bw
                y2 = y1 + bh
                glyph_img = glyph_img.convert("RGBA")
                gw, gh = glyph_img.size
                cx = x1 + bw // 2
                cy = y1 + bh // 2
                paste_x = cx - gw // 2
                paste_y = cy - gh // 2
                debug_canvas.paste(glyph_img, (paste_x, paste_y), glyph_img)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            debug_canvas.save("glyph_debug.png")
            print("[debug] saved glyph debug image to: glyph_debug.png")

        return matches, vae_glyph_images

    def _prepare_condition_latents(self, condition_images: List[Image.Image], target_area: int, resize: bool = True):
        condition_feats = []
        vae_sizes = []
        for image in condition_images:
            image_width, image_height = image.size
            if resize:
                # 仅当输入图面积超过 target_area 时才按目标面积缩放，否则保留原尺寸
                if image_width * image_height > target_area:
                    calculated_width, calculated_height = calculate_dimensions(target_area, image_width / image_height)
                else:
                    # 保留原尺寸，但需对齐到 32 的倍数以满足 VAE 输入约束
                    calculated_width = max(32, round(image_width / 32) * 32)
                    calculated_height = max(32, round(image_height / 32) * 32)
                if (calculated_width, calculated_height) != (image_width, image_height):
                    image = self.image_processor.resize(image, calculated_height, calculated_width)
            else:
                # glyph image 已按 bbox 尺寸渲染，直接使用原始尺寸，不做 resize
                calculated_width, calculated_height = image_width, image_height
            image_tensor = self.image_processor.preprocess(image)
            image_latent_enc = _vae_encode(self.vae, image_tensor.to(self.vae.device).to(self.vae.dtype))
            image_latent = image_latent_enc.latent_dist.mode()
            if getattr(self.vae, "_is_3d", False):
                image_latent = image_latent.squeeze(2)
            image_latent = (image_latent - _vae_shift(self.vae)) * _vae_scale(self.vae)
            condition_feats.append(image_latent.permute(1, 0, 2, 3))
            vae_sizes.append((calculated_width, calculated_height))
        return condition_feats, vae_sizes

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        glyph_layout_path: Optional[Union[str, Path]] = None,
        glyph_layout_items: Optional[List[Dict[str, Any]]] = None,
        glyph_patch_mode: str = "fixed_line_height",
        canvas_area: int = VAE_IMAGE_SIZE,
        debug: bool = False,
        images: Optional[List[Image.Image]] = None,
        same_frame_cond: bool = False,
        full_glyph_ref: bool = False,
        sample_shift: Optional[float] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            cfg_normalization (`bool`, *optional*, defaults to False):
                Whether to apply configuration normalization.
            cfg_truncation (`float`, *optional*, defaults to 1.0):
                The truncation value for configuration.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Must be 1; only one output image per call is supported.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.ZImagePipelineOutput`] instead of a plain
                tuple.
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
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.z_image.ZImagePipelineOutput`] or `tuple`: [`~pipelines.z_image.ZImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """
        if num_images_per_prompt != 1:
            raise ValueError("Only one output image per call is supported; num_images_per_prompt must be 1.")
        if isinstance(prompt, list) and len(prompt) != 1:
            raise ValueError("Only one prompt per call is supported.")
        if prompt_embeds is not None and len(prompt_embeds) != 1:
            raise ValueError("Only one prompt embedding batch item per call is supported.")
        if glyph_layout_items is None and glyph_layout_path is not None:
            glyph_layout_items = load_glyph_layout_items(glyph_layout_path)

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        device = self._execution_device

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        # If prompt_embeds is provided and prompt is None, skip encoding
        if prompt_embeds is not None and prompt is None:
            if self.do_classifier_free_guidance and negative_prompt_embeds is None:
                raise ValueError(
                    "When `prompt_embeds` is provided without `prompt`, "
                    "`negative_prompt_embeds` must also be provided for classifier-free guidance."
                )
        else:
            (
                prompt_embeds,
                negative_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        # full_glyph_ref: 渲染 glyph 全图作为额外参考图，不走 GlyphAnchor 注入
        if full_glyph_ref and glyph_layout_items:
            glyph_canvas = render_full_glyph_canvas(glyph_layout_items, width, height,
                                                    glyph_patch_mode=glyph_patch_mode, debug=debug)
            if images is None:
                images = [glyph_canvas]
            else:
                images = list(images) + [glyph_canvas]
            glyph_layout_items = None

        # 1. source images
        ref_entries = []
        if images:
            ref_feats, _ = self._prepare_condition_latents(images, height * width, resize=True)
            ref_entries = ref_feats  # list of (C, 1, H, W) tensors

        # 2. glyph
        glyph_entries = []
        if glyph_layout_items:
            layout_matches, vae_glyph_images = self.build_layout_matches_and_glyphs(
                glyph_layout_items=glyph_layout_items,
                canvas_size=(width, height),
                glyph_patch_mode=glyph_patch_mode,
                debug=debug,
            )
            glyph_condition_feats, _ = self._prepare_condition_latents(vae_glyph_images, canvas_area, resize=False)
            glyph_entries = [{"feat": feat, "layout_match": match, "kind": "glyph"}
                             for feat, match in zip(glyph_condition_feats, layout_matches)]

        # 3. pos/neg condition_feats：source 两 pass 都给，glyph 只给正 pass
        if ref_entries or glyph_entries:
            pos_condition_feats = [ref_entries + glyph_entries] * batch_size
            neg_condition_feats = [ref_entries] * batch_size
        else:
            pos_condition_feats = None
            neg_condition_feats = None

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        # Repeat prompt_embeds for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)]

        actual_batch_size = batch_size * num_images_per_prompt
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        # 5. Prepare timesteps
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        scheduler_kwargs = {"mu": mu}
        shift = self.scheduler.config.shift if sample_shift is None else sample_shift
        self.scheduler.set_shift(float(shift))
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0])
                timestep = (1000 - timestep) / 1000
                # Normalized time for time-aware config (0 at start, 1 at end)
                t_norm = timestep[0].item()

                # Handle cfg truncation
                current_guidance_scale = self.guidance_scale
                if (
                    self.do_classifier_free_guidance
                    and self._cfg_truncation is not None
                    and float(self._cfg_truncation) <= 1
                ):
                    if t_norm > self._cfg_truncation:
                        current_guidance_scale = 0.0

                # Run CFG only if configured AND scale is non-zero
                apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0

                if apply_cfg:
                    latents_typed = latents.to(self.transformer.dtype)
                    latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                    prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                    timestep_model_input = timestep.repeat(2)
                    # positive pass: source + glyph；negative pass: source only
                    condition_feats_model_input = (
                        pos_condition_feats + neg_condition_feats
                        if pos_condition_feats is not None
                        else None
                    )
                else:
                    latent_model_input = latents.to(self.transformer.dtype)
                    prompt_embeds_model_input = prompt_embeds
                    timestep_model_input = timestep
                    condition_feats_model_input = pos_condition_feats

                latent_model_input = latent_model_input.unsqueeze(2)
                latent_model_input_list = list(latent_model_input.unbind(dim=0))

                model_out_list = self.transformer(
                    latent_model_input_list,
                    timestep_model_input,
                    prompt_embeds_model_input,
                    condition_feats=condition_feats_model_input,
                    return_dict=False,
                    same_frame_cond=same_frame_cond,
                )[0]

                if apply_cfg:
                    # Perform CFG
                    pos_out = model_out_list[:actual_batch_size]
                    neg_out = model_out_list[actual_batch_size:]

                    noise_pred = []
                    for j in range(actual_batch_size):
                        pos = pos_out[j].float()
                        neg = neg_out[j].float()

                        pred = pos + current_guidance_scale * (pos - neg)

                        # Renormalization
                        if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                            ori_pos_norm = torch.linalg.vector_norm(pos)
                            new_pos_norm = torch.linalg.vector_norm(pred)
                            max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                            if new_pos_norm > max_new_norm:
                                pred = pred * (max_new_norm / new_pos_norm)

                        noise_pred.append(pred)

                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

                noise_pred = noise_pred.squeeze(2)
                if getattr(self, "x_pred", False):
                    # transformer output is x0_hat; convert to velocity for diffusers scheduler
                    # v = (z_sigma - x0_hat) / sigma  (matches training target `noise - x0`)
                    sigma = self.scheduler.sigmas[i].to(noise_pred.device, torch.float32)
                    sigma_denom = torch.clamp(sigma, min=float(getattr(self, "x_pred_sigma_clip", 0.05)))
                    noise_pred = ((latents.float() - noise_pred.float()) / sigma_denom).to(noise_pred.dtype)
                else:
                    noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
                assert latents.dtype == torch.float32

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / _vae_scale(self.vae)) + _vae_shift(self.vae)

            image = _vae_decode(self.vae, latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ZImagePipelineOutput(images=image)