"""Configurable OpenAI-compatible and Gemini HTTP protocols."""
import asyncio
import base64
import json
import math
import os
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests
from PIL import Image


def image_to_base64(image_path: str):
    """将图片转换为 base64；若面积 > 1024×1024，则缩放长边到 1024。
    返回 (mime_type, base64_str)。
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            area = width * height

            if area > 1024 * 1024:
                if width >= height:
                    new_width = 1024
                    new_height = int(height * (1024 / width))
                else:
                    new_height = 1024
                    new_width = int(width * (1024 / height))
                img = img.resize((new_width, new_height), Image.LANCZOS)

            ext = Path(image_path).suffix.lower()
            if ext in [".jpg", ".jpeg"]:
                mime_type, fmt = "image/jpeg", "JPEG"
            elif ext == ".png":
                mime_type, fmt = "image/png", "PNG"
            elif ext == ".gif":
                mime_type, fmt = "image/gif", "GIF"
            elif ext == ".webp":
                mime_type, fmt = "image/webp", "WEBP"
            else:
                mime_type, fmt = "image/jpeg", "JPEG"

            buf = BytesIO()
            if fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            img.save(buf, format=fmt, quality=100)
            base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            return mime_type, base64_str

    except Exception as e:
        raise RuntimeError(f"转换图片为base64时出错 {image_path}: {e}")



def parse_options(options):
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            raise ValueError('VLM options must be a JSON object') from None
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError('VLM options must be a JSON object')
    allowed = {'url_mode', 'model_location', 'auth_header', 'auth_scheme', 'timeout',
               'temperature', 'top_p', 'max_tokens', 'max_completion_tokens', 'extra_body'}
    if set(options) - allowed:
        raise ValueError('Unknown VLM option; supported options: ' + ', '.join(sorted(allowed)))
    return dict(options)


def add_vlm_arguments(parser):
    group = parser.add_argument_group('VLM service')
    group.add_argument('--vlm-backend', choices=['openai', 'google'],
                       help='API protocol (VLM_BACKEND; default: openai)')
    group.add_argument('--vlm-url', help='API base URL (VLM_URL); full endpoint via options.url_mode=full')
    group.add_argument('--vlm-model', help='Model served by the VLM API (VLM_MODEL)')
    group.add_argument('--vlm-api-key', help='API key (falls back to VLM_API_KEY)')
    group.add_argument('--vlm-config', type=Path, help='Optional JSON config; read only when explicitly specified')
    group.add_argument('--vlm-options', help='Advanced JSON options overriding matching config option keys')


def config_from_args(args):
    return load_config(getattr(args, 'vlm_config', None),
                       backend=getattr(args, 'vlm_backend', None),
                       url=getattr(args, 'vlm_url', None),
                       model=getattr(args, 'vlm_model', None),
                       api_key=getattr(args, 'vlm_api_key', None),
                       options=getattr(args, 'vlm_options', None))


def load_config(path=None, *, backend=None, url=None, model=None, api_key=None, options=None):
    data = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(data, dict):
        raise ValueError('VLM config must be a JSON object')
    merged_options = parse_options(data.get('options'))
    merged_options.update(parse_options(options))
    protocol = backend if backend is not None else data.get('backend', os.getenv('VLM_BACKEND', 'openai'))
    if protocol not in ('openai', 'google'):
        raise ValueError('VLM backend must be openai or google')
    return {
        'backend': protocol,
        'url': url if url is not None else data.get('url', os.getenv('VLM_URL', '')),
        'model': model if model is not None else data.get('model', os.getenv('VLM_MODEL', '')),
        'api_key': api_key if api_key is not None else data.get('api_key') or os.getenv('VLM_API_KEY', 'EMPTY'),
        'options': merged_options,
    }


def build_request(system_prompt, user_text, image_paths, *, backend='openai',
                  base_url, api_key, model=None, options=None):
    if backend not in ('openai', 'google'):
        raise ValueError('VLM backend must be openai or google')
    options = parse_options(options)
    if not isinstance(base_url, str) or urlsplit(base_url).scheme not in ('http', 'https') or not urlsplit(base_url).netloc:
        raise ValueError('VLM URL must be an HTTP(S) address')
    url_mode = options.get('url_mode', 'base')
    location = options.get('model_location', 'url' if backend == 'google' else 'body')
    if url_mode not in ('base', 'full') or location not in ('url', 'body'):
        raise ValueError('url_mode must be base/full; model_location must be url/body')
    if backend == 'openai' and location != 'body':
        raise ValueError('OpenAI-compatible requests require model_location=body')
    if (location == 'body' or url_mode == 'base') and (not isinstance(model, str) or not model.strip()):
        raise ValueError('A VLM model name is required')
    if url_mode == 'base' and (urlsplit(base_url).query or urlsplit(base_url).fragment):
        raise ValueError('Use url_mode=full for an endpoint with query parameters')
    timeout = options.get('timeout', 900.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('VLM timeout must be a positive finite number')
    if 'max_tokens' in options and 'max_completion_tokens' in options:
        raise ValueError('Use only one of max_tokens and max_completion_tokens')
    for name in ('max_tokens', 'max_completion_tokens'):
        if name in options and (type(options[name]) is not int or options[name] <= 0):
            raise ValueError(f'{name} must be a positive integer')
    for name in ('temperature', 'top_p'):
        if name in options and (isinstance(options[name], bool) or not isinstance(options[name], (int, float)) or not math.isfinite(options[name])):
            raise ValueError(f'{name} must be a finite number')
    extra = options.get('extra_body', {})
    if not isinstance(extra, dict) or set(extra) & {'model', 'messages', 'contents', 'systemInstruction', 'stream'}:
        raise ValueError('extra_body must be an object and cannot override model, messages, contents, systemInstruction or stream')
    header = options.get('auth_header', 'x-goog-api-key' if backend == 'google' else 'Authorization')
    scheme = options.get('auth_scheme', 'Bearer' if header.lower() == 'authorization' else '') if isinstance(header, str) else None
    if not isinstance(header, str) or not header or any(c.isspace() or c == ':' for c in header) or header.lower() == 'content-type':
        raise ValueError('Invalid VLM auth_header')
    if not isinstance(scheme, str) or '\r' in scheme or '\n' in scheme:
        raise ValueError('Invalid VLM auth_scheme')
    headers = {'Content-Type': 'application/json', header: f'{scheme} {api_key}'.strip()}
    url = base_url
    if backend == 'openai':
        if url_mode == 'base':
            url = base_url.rstrip('/') + '/chat/completions'
        content = [{'type': 'text', 'text': user_text}]
        for path in image_paths or []:
            mime, data = image_to_base64(path)
            content.append({'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{data}'}})
        body = {'model': model, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': content}]}
        for name in ('temperature', 'top_p', 'max_tokens', 'max_completion_tokens'):
            if name in options:
                body[name] = options[name]
        body.update(extra)
    else:
        if url_mode == 'base':
            root = base_url.rstrip('/')
            url = (root + '/models/' + quote(model.removeprefix('models/'), safe='') + ':generateContent'
                   if location == 'url' else root + ':generateContent')
        parts = []
        for path in image_paths or []:
            mime, data = image_to_base64(path)
            parts.append({'inlineData': {'mimeType': mime, 'data': data}})
        parts.append({'text': system_prompt + '\n\n' + user_text})
        body = {'contents': [{'role': 'user', 'parts': parts}]}
        if location == 'body':
            body['model'] = model
        generation = {}
        for name, field in [('temperature', 'temperature'), ('top_p', 'topP'), ('max_tokens', 'maxOutputTokens'), ('max_completion_tokens', 'maxOutputTokens')]:
            if name in options:
                generation[field] = options[name]
        body.update(extra)
        extra_generation = body.get('generationConfig', {})
        if not isinstance(extra_generation, dict):
            raise ValueError('generationConfig must be an object')
        generation.update(extra_generation)
        if generation:
            body['generationConfig'] = generation
    return url, headers, body, timeout


async def generate_text(system_prompt, user_text, image_paths, **config):
    url, headers, body, timeout = build_request(system_prompt, user_text, image_paths, **config)
    try:
        response = await asyncio.to_thread(requests.post, url, headers=headers, json=body,
                                           timeout=timeout, allow_redirects=False)
    except requests.RequestException:
        raise RuntimeError('VLM connection failed or timed out; check the endpoint and network') from None
    if not 200 <= response.status_code < 300:
        raise RuntimeError(f'VLM request failed (HTTP {response.status_code}); check endpoint, authentication and model')
    try:
        data = response.json()
        if config.get('backend', 'openai') == 'google':
            parts = data['candidates'][0]['content']['parts']
            text = '\n'.join(p['text'] for p in parts if p.get('text') and not p.get('thought'))
        else:
            text = data['choices'][0]['message']['content']
            if isinstance(text, list):
                text = '\n'.join(p['text'] for p in text if p.get('type') == 'text' and p.get('text'))
        if not isinstance(text, str) or not text.strip():
            raise ValueError('empty text')
    except (ValueError, KeyError, IndexError, TypeError):
        raise RuntimeError('VLM returned no usable final text for the selected protocol') from None
    return text.strip()
