"""Model loading and image generation shared by CLI and Web UI."""
import json
from pathlib import Path

BASE_MODELS = {'zimage': 'Tongyi-MAI/Z-Image',
               'qwen': 'Qwen/Qwen-Image-Edit-2511',
               'firered': 'Qwen/Qwen-Image-Edit-2511'}


def resolve_model_type(model, subfolder=None, model_type=None):
    if model_type:
        return model_type
    name = (subfolder or model).lower()
    if 'z-image' in name or 'zimage' in name:
        return 'zimage'
    if 'firered' in name:
        return 'firered'
    if 'qwen' in name:
        return 'qwen'
    config_path = Path(model) / (subfolder or '') / 'config.json'
    if config_path.is_file():
        kind = json.loads(config_path.read_text()).get('_class_name', '')
        if kind == 'ZImageTransformer2DModel':
            return 'zimage'
        if kind == 'QwenImageTransformer2DModel':
            return 'qwen'
    raise ValueError('Cannot determine the model architecture; pass --model-type zimage, qwen, or firered.')


def load_pipeline(args, model_type):
    import torch
    if model_type == 'zimage':
        from .models.transformer_z_image import ZImageTransformer2DModel
        from .pipelines.pipeline_z_image import ZImagePipeline

        transformer = ZImageTransformer2DModel.from_pretrained(
            args.model, subfolder=args.subfolder, torch_dtype=torch.bfloat16,
        )
        pipe = ZImagePipeline.from_pretrained(
            args.base_model, transformer=transformer, torch_dtype=torch.bfloat16,
        ).to('cuda')
    else:
        from PIL import Image
        from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKLQwenImage
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
        from .models.transformer_qwenimage import QwenImageTransformer2DModel
        from .pipelines.pipeline_qwen_image_edit import QwenImageEditPlusPipeline

        transformer = QwenImageTransformer2DModel.from_pretrained(
            args.model, subfolder=args.subfolder, torch_dtype=torch.bfloat16,
        )
        pipe = QwenImageEditPlusPipeline(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model, subfolder='scheduler'),
            vae=AutoencoderKLQwenImage.from_pretrained(args.base_model, subfolder='vae', torch_dtype=torch.bfloat16),
            text_encoder=Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.base_model, subfolder='text_encoder', torch_dtype=torch.bfloat16),
            tokenizer=Qwen2Tokenizer.from_pretrained(args.base_model, subfolder='tokenizer'),
            processor=Qwen2VLProcessor.from_pretrained(args.base_model, subfolder='processor'),
        ).to('cuda')
    pipe.transformer.eval()
    return pipe


def generate_image(pipe, args, model_type, prompt, layout, images, width, height):
    import torch
    if model_type != 'zimage':
        from PIL import Image
        references = []
        for path in images:
            with Image.open(path) as source:
                rgba = source.convert('RGBA')
                reference = Image.new('RGB', rgba.size, 'white')
                reference.paste(rgba, mask=rgba.getchannel('A'))
                references.append(reference)
    with torch.inference_mode():
        if model_type == 'zimage':
            image = pipe(
                prompt=prompt, negative_prompt='', glyph_layout_items=layout,
                width=width, height=height, num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale, cfg_normalization=False,
                max_sequence_length=2048, glyph_patch_mode='fixed_line_height',
                canvas_area=1536 * 1536, images=None, same_frame_cond=False,
                generator=torch.Generator('cuda').manual_seed(args.seed),
            ).images[0]
        else:
            image = pipe(
                image=references, prompt=prompt, negative_prompt='', glyph_layout_items=layout,
                width=width, height=height, num_inference_steps=args.steps,
                true_cfg_scale=args.guidance_scale, glyph_patch_mode='fixed_line_height',
                canvas_area=1536 * 1536, no_glyph_layout=False,
                generator=torch.Generator('cuda').manual_seed(args.seed),
            ).images[0]
    return image
