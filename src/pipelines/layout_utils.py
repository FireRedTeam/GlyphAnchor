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

"""Shared layout parsing and geometry; model-specific overlap policies stay in the pipelines."""
import ast
import json
from PIL import Image
from typing import Optional, Tuple

def parse_layout_bbox(bbox):
    if isinstance(bbox, str):
        bbox = ast.literal_eval(bbox)
    return [float(x) for x in bbox]


def load_glyph_layout_items(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({"idx": obj.get("idx"), "text": str(obj.get("text", "")), "bbox": parse_layout_bbox(obj["bbox"])})
    return items


def _xyxy_px_to_bbox_norm(x1, y1, x2, y2, image_size):
    image_w, image_h = image_size
    x1 = min(max(x1, 0.0), image_w)
    y1 = min(max(y1, 0.0), image_h)
    x2 = min(max(x2, 0.0), image_w)
    y2 = min(max(y2, 0.0), image_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 / image_w, y1 / image_h, (x2 - x1) / image_w, (y2 - y1) / image_h]


def bbox_norm_xywh_to_pixel_wh(bbox_norm_xywh, image_size):
    _, _, w, h = bbox_norm_xywh
    image_w, image_h = image_size
    return max(1, int(round(w * image_w))), max(1, int(round(h * image_h)))


def _bbox_norm_xywh_to_xyxy_px(bbox_norm_xywh, image_size):
    x, y, w, h = bbox_norm_xywh
    image_w, image_h = image_size
    x1 = x * image_w
    y1 = y * image_h
    x2 = (x + w) * image_w
    y2 = (y + h) * image_h
    return x1, y1, x2, y2


def _boxes_intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)


def _sort_matches_reading_order(matches):
    return sorted(matches, key=lambda m: (m["bbox_norm_xywh"][1] + 0.5 * m["bbox_norm_xywh"][3], m["bbox_norm_xywh"][0]))


def _bbox_iou_xywh_norm(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    return inter / max(union, 1e-12)


def dedup_layout_matches(matches, iou_thresh=0.95, coord_tol=0.003):
    deduped = []
    for m in matches:
        text = (m.get("text") or "").strip()
        bbox = m.get("bbox_norm_xywh")
        if not text or not bbox or len(bbox) != 4:
            deduped.append(m)
            continue
        is_dup = False
        for kept in deduped:
            kept_text = (kept.get("text") or "").strip()
            kept_bbox = kept.get("bbox_norm_xywh")
            if kept_text != text or not kept_bbox or len(kept_bbox) != 4:
                continue
            same_box = _bbox_iou_xywh_norm(bbox, kept_bbox) >= iou_thresh or max(abs(bbox[i] - kept_bbox[i]) for i in range(4)) <= coord_tol
            if same_box:
                is_dup = True
                break
        if not is_dup:
            deduped.append(m)
    return deduped


def merge_layout_matches_to_blocks(matches, image_size, glyph_images=None, expand_x_ratio=0.12, expand_y_ratio=0.30, max_height_ratio=1.8, min_x_overlap_ratio=0.12, max_center_dx_ratio=0.45):
    if len(matches) <= 1:
        return matches
    image_w, image_h = image_size
    items = []
    for idx, m in enumerate(matches):
        bbox = m.get("bbox_norm_xywh")
        text = (m.get("text") or "").strip()
        if not bbox or len(bbox) != 4 or not text:
            continue
        bx1, by1, bx2, by2 = _bbox_norm_xywh_to_xyxy_px(bbox, image_size)
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        cx = 0.5 * (bx1 + bx2)
        cy = 0.5 * (by1 + by2)
        if glyph_images is not None:
            g = glyph_images[idx]
            if not isinstance(g, Image.Image):
                g = Image.fromarray(g)
            w = float(max(1, g.size[0]))
            h = float(max(1, g.size[1]))
        else:
            w = bw
            h = bh
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        ex = max(2.0, expand_x_ratio * w)
        ey = max(2.0, expand_y_ratio * h)
        expanded = (max(0.0, x1 - ex), max(0.0, y1 - ey), min(float(image_w), x2 + ex), min(float(image_h), y2 + ey))
        items.append({"idx": idx, "text": text, "bbox_norm_xywh": bbox, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": w, "h": h, "cx": cx, "cy": cy, "expanded": expanded})
    n = len(items)
    if n <= 1:
        return matches
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]
            height_ratio = max(a["h"], b["h"]) / max(1.0, min(a["h"], b["h"]))
            if height_ratio > max_height_ratio:
                continue
            if not _boxes_intersect(a["expanded"], b["expanded"]):
                continue
            x_overlap = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
            x_overlap_ratio = x_overlap / max(1.0, min(a["w"], b["w"]))
            center_dx = abs(a["cx"] - b["cx"])
            max_center_dx = max_center_dx_ratio * max(a["w"], b["w"])
            if x_overlap_ratio < min_x_overlap_ratio and center_dx > max_center_dx:
                continue
            union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(items[i])

    merged = []
    for comp in groups.values():
        comp = _sort_matches_reading_order(comp)
        x1 = min(m["x1"] for m in comp)
        y1 = min(m["y1"] for m in comp)
        x2 = max(m["x2"] for m in comp)
        y2 = max(m["y2"] for m in comp)
        merged_bbox = _xyxy_px_to_bbox_norm(x1, y1, x2, y2, image_size)
        if merged_bbox is None:
            continue
        merged.append({"text": "\n".join(m["text"] for m in comp if m["text"]), "bbox_norm_xywh": merged_bbox, "children": comp})
    return _sort_matches_reading_order(merged)
