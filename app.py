#!/usr/bin/env python3
"""Launch the GlyphAnchor Gradio demo."""
import argparse
import os

from src.vlm_client import add_vlm_arguments, config_from_args
from src.webui import interface
from src.webui.interface import build_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--root-path', default='', help='Proxy prefix; empty string for direct access')
    parser.add_argument('--gpu', default=interface.GPU_DEVICE, help='GPU device for the model worker')
    add_vlm_arguments(parser)
    args = parser.parse_args()
    interface.GPU_DEVICE = args.gpu.strip()
    if not interface.GPU_DEVICE or ',' in interface.GPU_DEVICE:
        parser.error('--gpu requires one GPU device')
    try:
        config = config_from_args(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if config.get('api_key'):
        os.environ['VLM_API_KEY'] = config['api_key']
    build_app(config).queue(max_size=4).launch(
        server_name=args.host, server_port=args.port, share=args.share, show_error=True,
        root_path=args.root_path)


if __name__ == '__main__':
    main()
