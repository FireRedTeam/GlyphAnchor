"""Load a prompt/layout example or request a layout from a VLM service."""
import asyncio
import json
import math
from pathlib import Path


def validate_layout(layout):
    if not isinstance(layout, list) or not layout:
        raise ValueError('The layout is empty or unfinished. Fill in layout before using --layout.')
    result = []
    for i, item in enumerate(layout, 1):
        if not isinstance(item, dict) or not isinstance(item.get('text'), str) or not item['text'].strip():
            raise ValueError(f'Layout item {i} must contain nonempty text')
        if item.get('align', 'left') not in ('left', 'center', 'right'):
            raise ValueError(f'Layout item {i}: align must be left, center, or right')
        if item.get('valign', 'center') not in ('top', 'center', 'bottom'):
            raise ValueError(f'Layout item {i}: valign must be top, center, or bottom')
        box = item.get('bbox')
        if isinstance(box, str):
            box = json.loads(box)
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f'Layout item {i}: bbox must be [left, top, width, height]')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            raise ValueError(f'Layout item {i}: bbox values must be finite numbers')
        x, y, w, h = box
        if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x+w <= 1.000001 and y+h <= 1.000001):
            raise ValueError(f'Layout item {i}: bbox must lie within the normalized canvas')
        result.append({**item, 'bbox': box})
    return result


def parse_layout_response(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data['layout'] if 'layout' in data else [data]
    return validate_layout(data)


def request_layout(prompt, images, width, height, config):
    from .vlm_backend import call_vlm_for_layout
    content, _ = asyncio.run(call_vlm_for_layout(
        prompt, images, (width, height), backend=config['backend'],
        base_url=config['url'], model=config['model'], api_key=config['api_key'],
        options=config['options']))
    return parse_layout_response(content)


def load_inputs(args, *, editing):
    data = {}
    if args.layout:
        if getattr(args, 'rewrite', None):
            raise ValueError('--rewrite requires --prompt, not --layout')
        if any(getattr(args, name, None) is not None for name in ('vlm_url', 'vlm_model', 'vlm_backend', 'vlm_config', 'vlm_options', 'vlm_api_key')):
            raise ValueError('--layout already contains a layout; omit VLM options')
        data = json.loads(args.layout.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('--layout must contain an object with prompt and layout fields')
        prompt = data.get('prompt')
        layout = validate_layout(data.get('layout'))
    else:
        prompt = args.prompt
        if prompt.lower().endswith('.txt'):
            prompt_path = Path(prompt)
            try:
                is_prompt_file = prompt_path.is_file()
            except OSError:
                is_prompt_file = False
            if is_prompt_file:
                prompt = prompt_path.read_text(encoding='utf-8')
        from .vlm_client import config_from_args, build_request
        config = config_from_args(args)
        build_request('', '', [], backend=config['backend'], base_url=config['url'],
                      api_key=config['api_key'], model=config['model'], options=config['options'])
        layout = None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('A nonempty prompt is required')
    if args.image:
        images = args.image
    else:
        paths = data.get('images', [])
        if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
            raise ValueError('images must be a list of file paths')
        images = [args.layout.parent / p for p in paths]
    if not editing and images:
        raise ValueError('Z-Image text-to-image does not accept reference images')
    for path in images:
        if not path.is_file():
            raise ValueError(f'Reference image not found: {path}')
    width = args.width if args.width is not None else data.get('width', 1536)
    height = args.height if args.height is not None else data.get('height', 1536)
    if any(type(v) is not int or v <= 0 or v % 32 for v in [width, height]):
        raise ValueError('width and height must be positive multiples of 32')
    if layout is None:
        if getattr(args, 'rewrite', None):
            from .vlm_backend import call_vlm_for_prompt_rewrite
            prompt, _ = asyncio.run(call_vlm_for_prompt_rewrite(
                prompt, images, backend=config['backend'], mode=args.rewrite,
                base_url=config['url'], model=config['model'], api_key=config['api_key'],
                options=config['options']))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError('VLM returned an empty rewritten prompt')
            prompt = prompt.strip()
            print(f'\nRewritten prompt ({args.rewrite}):\n{prompt}\n', flush=True)
        layout = request_layout(prompt, images, width, height, config)
    layout = [{'align': 'center', 'valign': 'center', **item} for item in layout]
    return prompt.strip(), layout, images, width, height
