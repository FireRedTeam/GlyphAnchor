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

"""Shared glyph measurement and rendering."""
import math
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
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
    parse_layout_bbox
)

GLYPH_FONT_PATH = str(Path(__file__).resolve().parents[2] / "assets" / "SourceHanSansCN-Regular.otf")


GLYPH_FONT_SIZE = 24


GLYPH_IMAGE_HEIGHT = 32


GLYPH_PAD_X = 8


GLYPH_BG = (255, 255, 255)


GLYPH_FG = (0, 0, 0)


@lru_cache(maxsize=512)
def get_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=int(size))


@lru_cache(maxsize=1)
def _fallback_chars():
    path = Path(GLYPH_FONT_PATH).with_name('MiSans-fallback-chars.json')
    return frozenset(json.loads(path.read_text()))


def _font_runs(font, text):
    """Only replace unsupported primary glyphs; align runs on one baseline."""
    if str(font.path) != GLYPH_FONT_PATH or not any(c in _fallback_chars() for c in text):
        return [(text, font, 0, 0)]
    fallback = get_font(str(Path(GLYPH_FONT_PATH).with_name('MiSans-Regular.otf')), font.size)
    runs = []
    x = 0.0
    for char in text:
        selected = fallback if char in _fallback_chars() else font
        if runs and runs[-1][1] is selected:
            runs[-1][0] += char
        else:
            runs.append([char, selected])
    positioned = []
    for value, selected in runs:
        positioned.append((value, selected, x, font.getmetrics()[0] - selected.getmetrics()[0]))
        x += selected.getlength(value)
    return positioned


def _text_bbox(draw, xy, text, font):
    boxes = [draw.textbbox((xy[0] + x, xy[1] + y), value, font=selected)
             for value, selected, x, y in _font_runs(font, text)]
    return (math.floor(min(b[0] for b in boxes)), math.floor(min(b[1] for b in boxes)),
            math.ceil(max(b[2] for b in boxes)), math.ceil(max(b[3] for b in boxes)))


def _draw_text(draw, xy, text, font, fill):
    for value, selected, x, y in _font_runs(font, text):
        draw.text((xy[0] + x, xy[1] + y), value, font=selected, fill=fill)


def ceil_to(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def round_to(x: int, base: int) -> int:
    return ((x + base // 2) // base) * base


def _get_text_bbox(font, text: str):
    try:
        if not any(c in _fallback_chars() for c in text):
            return font.getbbox(text)
        return _text_bbox(ImageDraw.Draw(Image.new("L", (1, 1))), (0, 0), text, font)
    except Exception:
        w = max(1, int(font.getlength(text)))
        return 0, 0, w, font.size


def _measure_text(font, text: str):
    x0, y0, x1, y1 = _get_text_bbox(font, text)
    return max(1, x1 - x0), max(1, y1 - y0), x0, y0


def _measure_text_width(font, text: str) -> int:
    w, _, _, _ = _measure_text(font, text)
    return w


def _break_at_word_boundary(cur: str, ch: str):
    """断行辅助：若 cur 末尾和 ch 都是 ASCII 字母（即断在英文单词中间），
    则把 cur 末尾的完整单词退回，返回 (本行内容, 下一行起始内容)。
    若无法按单词边界拆分（整行本身就是一个单词），则直接硬断，返回 (cur, ch)。"""
    if ch.isascii() and ch.isalpha() and cur and cur[-1].isascii() and cur[-1].isalpha():
        # 找到当前行末尾单词的起始位置
        i = len(cur) - 1
        while i > 0 and cur[i - 1].isascii() and cur[i - 1].isalpha():
            i -= 1
        if i > 0:
            # 有非单词字符作为分隔，把末尾单词移到下一行
            return cur[:i], cur[i:] + ch
    # 无法按单词边界拆分，硬断
    return cur, ch


def _wrap_text_by_width(s: str, font, pad_x: int, max_width: int, overflow_tolerance: float = 0.1):
    """按宽度换行，允许最多 overflow_tolerance 比例的轻微超出后再换行。
    若断行点落在英文单词中间，则将整个单词移到下一行。"""
    lines = []
    cur = ""
    tol = max_width * overflow_tolerance
    for ch in s:
        if not cur or _measure_text_width(font, cur + ch) + pad_x * 2 <= max_width + tol:
            cur += ch
        else:
            line, cur = _break_at_word_boundary(cur, ch)
            lines.append(line)
    if cur:
        lines.append(cur)
    return lines or [" "]


def render_glyph_image(
    s: str,
    *,
    mode: str = "fixed_line_height",
    target_ratio: Optional[float] = None,
    target_size: Optional[Tuple[int, int]] = None,
    font_path: str = GLYPH_FONT_PATH,
    font_size: int = GLYPH_FONT_SIZE,
    h: int = GLYPH_IMAGE_HEIGHT,
    pad_x: int = GLYPH_PAD_X,
    box_pad_x: int = 0,
    box_pad_y: int = 0,
    bg=GLYPH_BG,
    fg=GLYPH_FG,
) -> Image.Image:
    s = (s or "").strip() or " "

    if mode == "match_box":
        assert target_size is not None, "target_size is required for match_box"
        W = max(1, int(round(target_size[0])))
        H = max(1, int(round(target_size[1])))

        lines = s.splitlines() if "\n" in s else [s]
        avail_w = max(1, W - box_pad_x * 2)
        avail_h = max(1, H - box_pad_y * 2)
        line_spacing = 0.15

        dummy = ImageDraw.Draw(Image.new("RGB", (8, 8), "white"))
        lo, hi = 1, max(1, avail_h * 4)
        best_font = get_font(font_path, 1)
        best_line_h = 1
        best_total_h = 1

        while lo <= hi:
            mid = (lo + hi) // 2
            font = get_font(font_path, mid)
            line_h = _layout_get_line_height(dummy, font)
            total_h = int(len(lines) * line_h + max(0, len(lines) - 1) * line_h * line_spacing)

            max_line_w = 0
            for line in lines:
                x0, y0, x1, y1 = _text_bbox(dummy, (0, 0), line, font=font)
                max_line_w = max(max_line_w, x1 - x0)

            if max_line_w <= avail_w and total_h <= avail_h:
                best_font = font
                best_line_h = line_h
                best_total_h = total_h
                lo = mid + 1
            else:
                hi = mid - 1

        img = Image.new("RGB", (W, H), bg)
        draw = ImageDraw.Draw(img)

        # 用字形的实际像素范围（glyph top ~ glyph bottom）做居中，
        # 而不是以 text origin 做居中，避免顶部/底部被裁切。
        # actual_extent = 从第一行字形顶部到最后一行字形底部的像素高度
        _, ref_y0, _, _ = _text_bbox(dummy, (0, 0), "Ag中文", font=best_font)
        step = int(best_line_h * (1 + line_spacing))
        actual_extent = (len(lines) - 1) * step + best_line_h
        center_pad = max(0, (avail_h - actual_extent) // 2)
        # cur_y 是 text origin，字形顶部在 cur_y + ref_y0，
        # 令字形顶部 = box_pad_y + center_pad，则：
        cur_y = box_pad_y + center_pad - ref_y0
        x = box_pad_x

        for line in lines:
            x0, y0, x1, y1 = _text_bbox(draw, (0, 0), line, font=best_font)
            _draw_text(draw, (x - x0, cur_y), line, font=best_font, fill=fg)
            cur_y += int(best_line_h * (1 + line_spacing))

        return img

    if mode == "fixed_line_height":
        font = get_font(font_path, font_size)

        if "\n" in s:
            lines = s.splitlines()
            max_line_w = 1
            for line in lines:
                max_line_w = max(max_line_w, _measure_text_width(font, line))
            W = ceil_to(max_line_w + pad_x * 2, 32)
        else:
            assert target_ratio is not None, "target_ratio is required for fixed_line_height when no explicit line breaks"
            single_line_w = ceil_to(_measure_text_width(font, s) + pad_x * 2, 32)
            min_char_w = max(_measure_text_width(font, ch) for ch in s)
            min_w = ceil_to(min_char_w + pad_x * 2, 32)
            target_ratio = max(float(target_ratio), 1e-6)

            # ratio(w) = w / (h * num_lines(w)) 关于 w 单调不减，
            # 用二分找到 ratio 首次 >= target_ratio 的位置，
            # 再在该点前后各取一个候选，共至多 3 个候选。
            n_steps = (single_line_w - min_w) // 32  # 共 n_steps+1 个候选

            def _eval(idx):
                _w = min_w + idx * 32
                _wrapped = _wrap_text_by_width(s, font, pad_x, _w)
                _total_h = h * len(_wrapped)
                _ratio = _w / max(_total_h, 1)
                return _ratio, _wrapped, _w

            # 二分：找最小 idx 使 ratio >= target_ratio
            lo_idx, hi_idx = 0, n_steps
            while lo_idx < hi_idx:
                mid_idx = (lo_idx + hi_idx) // 2
                if _eval(mid_idx)[0] < target_ratio:
                    lo_idx = mid_idx + 1
                else:
                    hi_idx = mid_idx

            # 检查跨越点前后各一个候选
            best = None
            for delta in (-1, 0, 1):
                idx = lo_idx + delta
                if not (0 <= idx <= n_steps):
                    continue
                ratio_now, wrapped, w = _eval(idx)
                total_h = h * len(wrapped)
                cand = (
                    abs(math.log(ratio_now / target_ratio)),
                    abs(ratio_now - target_ratio),
                    w * total_h,
                    w,
                    wrapped,
                )
                if best is None or cand < best:
                    best = cand

            _, _, _, W, lines = best

        H = h * len(lines)
        img = Image.new("RGB", (W, H), bg)
        draw = ImageDraw.Draw(img)

        for i, line in enumerate(lines):
            _, line_h, x0, y0 = _measure_text(font, line)
            x = pad_x - x0
            y = i * h + (h - line_h) // 2 - y0
            _draw_text(draw, (x, y), line, font=font, fill=fg)

        return img

    raise ValueError(f"Unknown mode: {mode}")


def _layout_get_render_lines(text: str):
    if not text:
        return [""]
    if "\n" in text:
        return [p if p else "" for p in text.splitlines()]
    return [text]


def _layout_get_line_height(draw, font):
    x0, y0, x1, y1 = _text_bbox(draw, (0, 0), "Ag中文", font=font)
    return y1 - y0


def _wrap_line_by_width(draw, text, font, max_width, overflow_tolerance=0.1):
    """按宽度换行，允许最多 overflow_tolerance 比例的轻微超出后再换行。
    若断行点落在英文单词中间，则将整个单词移到下一行。"""
    if not text:
        return [""]

    tol = max_width * overflow_tolerance
    wrapped = []
    cur = ""

    for ch in text:
        candidate = cur + ch
        x0, y0, x1, y1 = _text_bbox(draw, (0, 0), candidate, font=font)
        if not cur or (x1 - x0) <= max_width + tol:
            cur = candidate
        else:
            line, cur = _break_at_word_boundary(cur, ch)
            wrapped.append(line)

    if cur:
        wrapped.append(cur)

    return wrapped or [""]


def _rewrap_lines_by_width(draw, lines, font, box_w):
    wrapped = []
    for line in lines:
        wrapped.extend(_wrap_line_by_width(draw, line, font, box_w))
    return wrapped or [""]


def _fit_layout_font(draw, text, font_path, box_w, box_h, line_spacing=0.15, min_font_size=24, max_font_size=200):
    lines = _layout_get_render_lines(text)

    while True:
        best = None
        lo, hi = min_font_size, max_font_size

        while lo <= hi:
            mid = (lo + hi) // 2
            font = ImageFont.truetype(font_path, mid)
            wrapped_lines = _rewrap_lines_by_width(draw, lines, font, box_w)
            max_line_w, total_h, line_h, _ = _measure_layout_lines(draw, wrapped_lines, font, line_spacing)

            if max_line_w <= box_w and total_h <= box_h:
                best = (font, list(wrapped_lines), line_h, total_h)
                lo = mid + 1
            else:
                hi = mid - 1

        if best is not None:
            return best

        font = ImageFont.truetype(font_path, min_font_size)
        wrapped_lines = _rewrap_lines_by_width(draw, lines, font, box_w)
        max_line_w, total_h, line_h, _ = _measure_layout_lines(draw, wrapped_lines, font, line_spacing)
        return font, wrapped_lines, line_h, total_h


def _measure_layout_lines(draw, lines, font, line_spacing=0.15):
    line_h = _layout_get_line_height(draw, font)
    total_h = int(len(lines) * line_h + max(0, len(lines) - 1) * line_h * line_spacing)
    max_line_w = 0
    longest_idx = 0
    for i, line in enumerate(lines):
        x0, y0, x1, y1 = _text_bbox(draw, (0, 0), line, font=font)
        w = x1 - x0
        if w > max_line_w:
            max_line_w = w
            longest_idx = i
    return max_line_w, total_h, line_h, longest_idx


def _simulate_tight_bbox(item, image_size, font_path=GLYPH_FONT_PATH, padding_ratio=0.04, line_spacing=0.15):
    image_w, image_h = image_size
    text = str(item.get("text", ""))
    x, y, w, h = item["bbox"]
    px = int(x * image_w)
    py = int(y * image_h)
    pw = int(w * image_w)
    ph = int(h * image_h)
    pad_x = max(2, int(pw * padding_ratio))
    pad_y = max(2, int(ph * padding_ratio))
    inner_x = px + pad_x
    inner_y = py + pad_y
    inner_w = max(1, pw - 2 * pad_x)
    inner_h = max(1, ph - 2 * pad_y)
    dummy = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(dummy)
    font, lines, line_h, total_h = _fit_layout_font(draw, text, font_path, inner_w, inner_h, line_spacing)
    text = "\n".join(lines)
    cur_y = inner_y + max(0, (inner_h - total_h) // 2)
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")
    for line in lines:
        x0, y0, x1, y1 = _text_bbox(draw, (inner_x, cur_y), line, font=font)
        min_x = min(min_x, x0)
        min_y = min(min_y, y0)
        max_x = max(max_x, x1)
        max_y = max(max_y, y1)
        cur_y += int(line_h * (1 + line_spacing))
    if not math.isfinite(min_x):
        min_x, min_y, max_x, max_y = px, py, px + pw, py + ph
    min_x = min(max(min_x, 0.0), image_w)
    min_y = min(max(min_y, 0.0), image_h)
    max_x = min(max(max_x, 0.0), image_w)
    max_y = min(max(max_y, 0.0), image_h)
    return {
        "text": text,
        "bbox": [float(min_x), float(min_y), float(max_x), float(max_y)],
        "bbox_norm_xywh": _xyxy_px_to_bbox_norm(min_x, min_y, max_x, max_y, image_size),
    }


def anchor_glyph_patches(matches, glyphs, canvas_size):
    """Place each already-rendered patch within its user-drawn frame."""
    W, H = canvas_size
    result = []
    for match, glyph in zip(matches, glyphs):
        frame = match.get('layout_frame')
        if frame is None:
            result.append(match)
            continue
        x, y, w, h = frame
        bx, by, bw, bh = match['bbox_norm_xywh']
        gw, gh = glyph.size
        # Unspecified axes keep the original placement; explicit options anchor
        # the whole patch, including padding, without changing its pixels.
        left = (bx + bw / 2) * W - gw / 2
        top = (by + bh / 2) * H - gh / 2
        if 'align' in match:
            factor = {'left': 0, 'center': 0.5, 'right': 1}[match['align']]
            left = x * W + factor * (w * W - gw)
        if 'valign' in match:
            factor = {'top': 0, 'center': 0.5, 'bottom': 1}[match['valign']]
            top = y * H + factor * (h * H - gh)
        result.append({**match, 'bbox_norm_xywh': [left / W, top / H, gw / W, gh / H]})
    return result


def render_one_glyph_patch(match, image_size, glyph_patch_mode: str) -> Image.Image:
    _, _, w, h = match["bbox_norm_xywh"]
    text = (match.get("text") or "").strip() or " "
    if glyph_patch_mode == "match_box":
        target_w, target_h = bbox_norm_xywh_to_pixel_wh(match["bbox_norm_xywh"], image_size)
        return render_glyph_image(text, mode="match_box", target_size=(target_w, target_h))
    if glyph_patch_mode == "fixed_line_height":
        image_width, image_height = image_size
        ratio = (w * image_width) / max(h * image_height, 1e-6)
        return render_glyph_image(text, mode="fixed_line_height", target_ratio=ratio)
    raise ValueError(f"Unknown glyph_patch_mode: {glyph_patch_mode}")


def compose_glyph_group_image(children, child_glyph_images, merged_bbox_norm_xywh, image_size, bg=GLYPH_BG) -> Image.Image:
    mx1, my1, mx2, my2 = _bbox_norm_xywh_to_xyxy_px(merged_bbox_norm_xywh, image_size)
    W = max(1, int(round(mx2 - mx1)))
    H = max(1, int(round(my2 - my1)))
    canvas = Image.new("RGB", (W, H), bg)
    for child in _sort_matches_reading_order(children):
        child_idx = child["idx"]
        glyph = child_glyph_images[child_idx]
        if not isinstance(glyph, Image.Image):
            glyph = Image.fromarray(glyph)
        cx1, cy1, cx2, cy2 = _bbox_norm_xywh_to_xyxy_px(child["bbox_norm_xywh"], image_size)
        rel_x1 = int(round(cx1 - mx1))
        rel_y1 = int(round(cy1 - my1))
        rel_x2 = int(round(cx2 - mx1))
        rel_y2 = int(round(cy2 - my1))
        slot_w = max(1, rel_x2 - rel_x1)
        slot_h = max(1, rel_y2 - rel_y1)
        gw, gh = glyph.size
        paste_x = rel_x1 + (slot_w - gw) // 2
        paste_y = rel_y1 + (slot_h - gh) // 2
        canvas.paste(glyph, (paste_x, paste_y))
    return canvas


def ensure_side(image: Image.Image, min_side: int = 16, multiple: int = 16, resample=Image.BICUBIC) -> Image.Image:
    w, h = image.size
    if w <= 0 or h <= 0:
        return image
    scale = max(min_side / w, min_side / h, 1.0)
    scaled_w = max(1, int(math.ceil(w * scale)))
    scaled_h = max(1, int(math.ceil(h * scale)))
    new_w = round_to(scaled_w, multiple)
    new_h = round_to(scaled_h, multiple)
    if new_w == w and new_h == h:
        return image
    return image.resize((new_w, new_h), resample=resample)


def render_full_glyph_canvas(layout_items, width: int, height: int,
                             glyph_patch_mode: str = "match_box",
                             debug: bool = False, *, adjust_matches) -> Image.Image:
    """将所有 layout items 渲染到一张白底全图作为额外参考图输入 pipeline（不走 GlyphAnchor 注入）。
    处理流程与 use_glyph 完全一致，唯一差异是跳过 merge_layout_matches_to_blocks。
    """
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas_size = (width, height)

    if not layout_items:
        return canvas

    raw_matches = []
    for item in layout_items:
        tight = _simulate_tight_bbox(item, canvas_size)
        bbox_norm_xywh = tight["bbox_norm_xywh"]
        if not bbox_norm_xywh:
            continue
        raw_matches.append({
            **({"align": item["align"]} if "align" in item else {}),
            **({"layout_frame": list(item["bbox"])} if "align" in item or "valign" in item else {}),
            **({"valign": item["valign"]} if "valign" in item else {}),
            "text": tight["text"],
            "bbox": tight["bbox"],
            "bbox_norm_xywh": bbox_norm_xywh,
        })

    raw_matches = dedup_layout_matches(raw_matches)
    raw_layout_glyph_images = [render_one_glyph_patch(m, canvas_size, glyph_patch_mode) for m in raw_matches]
    raw_matches = anchor_glyph_patches(raw_matches, raw_layout_glyph_images, canvas_size)
    raw_matches = adjust_matches(raw_matches, raw_layout_glyph_images, canvas_size, gap_px=1)

    img_w, img_h = canvas_size
    for match, glyph in zip(raw_matches, raw_layout_glyph_images):
        bbox = match.get("bbox_norm_xywh")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        px = int(x * img_w)
        py = int(y * img_h)
        pw = max(1, int(w * img_w))
        ph = max(1, int(h * img_h))
        glyph = ensure_side(glyph, min_side=16, multiple=16)
        gw, gh = glyph.size
        paste_x = px + (pw - gw) // 2
        paste_y = py + (ph - gh) // 2
        canvas.paste(glyph, (paste_x, paste_y))

    if debug:
        canvas.save("full_glyph_ref_debug.png")
        print("[debug] saved full glyph ref image to: full_glyph_ref_debug.png")
    return canvas
