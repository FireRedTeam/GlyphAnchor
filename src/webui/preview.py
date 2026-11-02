"""Render glyph layout previews without loading model weights."""
import json

from ..inputs import parse_layout_response
from .layout_editor import editor_layout


def render_layout_preview(model, layout, width, height):
    """Render actual model glyph patches on their final layout boxes, without loading weights."""
    from PIL import Image, ImageDraw
    if not layout or json.loads(layout) in ([], None):
        return None
    items = editor_layout(parse_layout_response(layout))
    if model == 'GlyphAnchor-Z-Image':
        from ..pipelines.pipeline_z_image import ZImagePipeline as Pipeline
    else:
        from ..pipelines.pipeline_qwen_image_edit import QwenImageEditPlusPipeline as Pipeline
    size = (int(width), int(height))
    matches, glyphs = Pipeline.build_layout_matches_and_glyphs(
        None, items, size, glyph_patch_mode='fixed_line_height', debug=False)
    canvas = Image.new('RGB', size, (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    step = max(16, min(size) // 32)
    for x in range(0, size[0], step):
        draw.line([(x, 0), (x, size[1])], fill=(220, 220, 220))
    for y in range(0, size[1], step):
        draw.line([(0, y), (size[0], y)], fill=(220, 220, 220))
    for match, glyph in zip(matches, glyphs):
        x, y, w, h = match['bbox_norm_xywh']
        left, top = int(x * size[0]), int(y * size[1])
        bw, bh = int(w * size[0]), int(h * size[1])
        glyph = glyph.convert('RGBA')
        canvas.paste(glyph, (left + bw // 2 - glyph.width // 2,
                             top + bh // 2 - glyph.height // 2), glyph)
        draw.rectangle([left, top, left + bw, top + bh], outline=(255, 0, 0), width=2)
    return canvas.resize((max(1, int(size[0] * 520 / size[1])), 520), Image.Resampling.LANCZOS)
