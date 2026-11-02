#!/usr/bin/env python3
"""Generate or edit an image with GlyphAnchor."""
import argparse
from pathlib import Path

from src.inputs import load_inputs
from src.vlm_client import add_vlm_arguments
from src.generation import BASE_MODELS, resolve_model_type, load_pipeline, generate_image


def add_input_arguments(parser):
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--layout', type=Path, help='JSON file containing prompt and layout')
    inputs.add_argument('--prompt', help='Prompt text or UTF-8 .txt file path to send to the VLM for layout planning')
    parser.add_argument('--rewrite', choices=['guess', 'enrich'],
                        help='Rewrite --prompt before layout planning: guess fills in renderable text; enrich adds visual details while preserving text')
    parser.add_argument('--image', type=Path, action='append', help='Reference image; repeat for multiple images')
    add_vlm_arguments(parser)



def main(default_model_type=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Local checkpoint directory, local model root, or Hub repository root; subfolder is selected by --model-type')
    parser.add_argument('--model-type', choices=list(BASE_MODELS), default=default_model_type,
                        help='Usually inferred from the checkpoint name or local configuration')
    parser.add_argument('--base-model', help='Local base-model directory or Hub ID; defaults to the matching Hub model')
    add_input_arguments(parser)
    parser.add_argument('--output', type=Path, default=Path('outputs/example_1.png'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=4.0)
    args = parser.parse_args()
    try:
        if args.model is None or args.model == 'innerway/GlyphAnchor':
            model_type = args.model_type or 'zimage'
            args.model = 'innerway/GlyphAnchor'
        else:
            args.model = str(Path(args.model).expanduser())
            model_type = resolve_model_type(args.model, model_type=args.model_type)
        args.subfolder = {
            'zimage': 'GlyphAnchor-Z-Image',
            'qwen': 'GlyphAnchor-Qwen-Image-Edit-2511',
            'firered': 'GlyphAnchor-FireRed-Image-Edit-1.1',
        }[model_type]
        if (Path(args.model) / 'config.json').is_file():
            args.subfolder = None
        args.base_model = args.base_model or BASE_MODELS[model_type]
        prompt, layout, images, width, height = load_inputs(args, editing=model_type != 'zimage')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    pipe = load_pipeline(args, model_type)
    image = generate_image(pipe, args, model_type, prompt, layout, images, width, height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
