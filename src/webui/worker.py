"""One persistent model per GPU worker; no CUDA imports in the parent process."""
import gc
import os
from pathlib import Path
from types import SimpleNamespace

_PIPE = None
_KEY = None


def initialize_worker(device):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


def generate_on_worker(model, subfolder, base_model, data, seed, steps, guidance, output, *, model_type):
    global _PIPE, _KEY
    from ..generation import BASE_MODELS, load_pipeline, generate_image
    kind = model_type
    base_model = base_model or BASE_MODELS[kind]
    key = (model, subfolder, base_model, kind)
    args = SimpleNamespace(model=model, subfolder=subfolder, base_model=base_model,
                           seed=int(seed), steps=int(steps), guidance_scale=guidance)
    if key != _KEY:
        _PIPE = None
        _KEY = None
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        _PIPE = load_pipeline(args, kind)
        _KEY = key
    image = generate_image(_PIPE, args, kind, data['prompt'], data['layout'],
                           [Path(p) for p in data['images']] if kind != 'zimage' else [],
                           data['width'], data['height'])
    image.save(output)
    return str(output)
