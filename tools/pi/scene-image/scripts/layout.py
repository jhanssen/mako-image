#!/usr/bin/env python3
"""Turns a layout draft (boxes in pixels from the top-left) into a MOSS-Image layout prompt.

    layout.py DRAFT        prints WIDTHxHEIGHT on the first line, the prompt (one line of JSON) on the second, and
                           with control maps, {"color_map": PATH, "depth_map": PATH} on a third
DRAFT is a file, "-" for standard input, or the JSON itself (starting with "{").

The draft: {"width": W, "height": H, "summary": {"caption": "...", "meta": [...], "canvas": "single"},
            "items": [{"label": "character" | "object" | "text", "caption": "...", "bbox": [x0, y0, x1, y1],
                       "text": "..." (text items; \\n for a line break), "tags": [...], "palette": [...]}, ...]}
canvas: single, multi_panel or collage.
With "color_map": FILE (a flat colour per subject, for a model with a control adapter; "depth_map": FILE beside it),
an item may give "color": "#rrggbb" in place of its bbox: the box around that colour in the map. Paths are relative
to the draft's file; width and height default to the map's. Optional: "bbox_coord_space": "abs", "bbox_format": "xyxy" (what boxes are).
The prompt's boxes are [y0, x0, y1, x1] from the image's centre, edges rounded outward.
"""
import json, math, sys

ITEM_KEYS = {"label", "caption", "bbox", "text", "tags", "palette", "color"}
SUMMARY_KEYS = {"caption", "meta", "canvas", "palette"}
ROOT_KEYS = {"width", "height", "summary", "items", "bbox_coord_space", "bbox_format", "color_map", "depth_map"}


def fail(msg):
    sys.exit(f"layout: {msg}")


def strings(v, where):
    if not (isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v)):
        fail(f"{where} must be a list of nonempty strings")


def edge(v, up):
    v = round(v, 9)  # float noise must not move an edge a pixel
    return math.ceil(v) if up else math.floor(v)


def color_boxes(path, colors):
    """The box [x0, y0, x1, y1] around each colour's pixels (within 24 a channel) in the map, and its size."""
    try:
        from PIL import Image
    except ImportError:
        fail("boxes from a colour map need Pillow (python3 -m pip install pillow)")
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:
        fail(f"color_map: {e}")
    w, h = im.size
    px = im.load()
    want = {c: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in colors}
    box = {}
    for y in range(h):
        for x in range(w):
            p = px[x, y]
            for c, rgb in want.items():
                if max(abs(p[0] - rgb[0]), abs(p[1] - rgb[1]), abs(p[2] - rgb[2])) <= 24:
                    b = box.get(c)
                    box[c] = [x, y, x + 1, y + 1] if b is None else [min(b[0], x), min(b[1], y), max(b[2], x + 1),
                                                                      max(b[3], y + 1)]
    missing = [c for c in colors if c not in box]
    if missing:
        fail(f"color_map {path}: no pixels of {', '.join(missing)}")
    return box, (w, h)


def main():
    if len(sys.argv) != 2:
        fail("usage: layout.py DRAFT (a file, - for standard input, or the JSON)")
    src = sys.argv[1]
    try:
        if src == "-":
            d = json.load(sys.stdin)
        elif src.lstrip().startswith("{"):
            d = json.loads(src)
        else:
            d = json.load(open(src))
    except Exception as e:
        fail(f"not a JSON draft: {e}")
    if not isinstance(d, dict):
        fail("the draft must be a JSON object")
    if set(d) - ROOT_KEYS:
        fail(f"unknown fields {sorted(set(d) - ROOT_KEYS)}")
    import os
    here = os.path.dirname(os.path.abspath(src)) if src != "-" and not src.lstrip().startswith("{") else os.getcwd()
    maps = {}
    for k in ("color_map", "depth_map"):
        if k in d:
            if not isinstance(d[k], str) or not os.path.isfile(os.path.join(here, os.path.expanduser(d[k]))):
                fail(f"{k}: no file {d[k]!r}")
            maps[k] = os.path.abspath(os.path.join(here, os.path.expanduser(d[k])))
    items0 = d.get("items") if isinstance(d.get("items"), list) else []
    colors = [it["color"] for it in items0 if isinstance(it, dict) and "color" in it]
    if colors:
        if "color_map" not in maps:
            fail("items give colours but the draft has no color_map")
        for c in colors:
            if not (isinstance(c, str) and len(c) == 7 and c[0] == "#" and all(ch in "0123456789abcdefABCDEF"
                                                                                for ch in c[1:])):
                fail(f"color {c!r}: not #rrggbb")
        boxes, msize = color_boxes(maps["color_map"], colors)
        d.setdefault("width", msize[0])
        d.setdefault("height", msize[1])
        if (d["width"], d["height"]) != msize:
            fail(f"the color map is {msize[0]}x{msize[1]}, the draft {d['width']}x{d['height']}: make them the same")
        for it in items0:
            if isinstance(it, dict) and "color" in it:
                if "bbox" in it:
                    fail("an item gives either a color or a bbox")
                it["bbox"] = boxes[it["color"]]
    if d.get("bbox_coord_space", "abs") != "abs" or d.get("bbox_format", "xyxy") != "xyxy":
        fail('boxes are [x0, y0, x1, y1] in pixels from the top-left: bbox_coord_space "abs", bbox_format "xyxy"')
    w, h = d.get("width"), d.get("height")
    if not (isinstance(w, int) and isinstance(h, int) and w >= 256 and h >= 256 and w % 32 == 0 and h % 32 == 0
            and w * h <= 2048 * 2048):
        fail("width and height must be multiples of 32 from 256, at most 2048x2048 pixels (as the server takes)")
    s = d.get("summary")
    if not isinstance(s, dict) or not isinstance(s.get("caption"), str) or not s["caption"].strip():
        fail("summary.caption must describe the whole image")
    if set(s) - SUMMARY_KEYS:
        fail(f"summary: unknown fields {sorted(set(s) - SUMMARY_KEYS)}")
    if s.get("canvas", "single") not in ("single", "multi_panel", "collage"):
        fail("summary.canvas must be single, multi_panel or collage")
    for k in ("meta", "palette"):
        if k in s:
            strings(s[k], f"summary.{k}")
    items = d.get("items")
    if not isinstance(items, list) or not items:
        fail("items must be a nonempty list")
    out = []
    for i, it in enumerate(items):
        if not isinstance(it, dict) or set(it) - ITEM_KEYS:
            fail(f"item {i}: unknown fields {sorted(set(it) - ITEM_KEYS) if isinstance(it, dict) else it}")
        if it.get("label") not in ("character", "object", "text"):
            fail(f"item {i}: label must be character, object or text")
        if not isinstance(it.get("caption"), str) or not it["caption"].strip():
            fail(f"item {i}: needs a caption")
        if it["label"] == "text" and "text" not in it:
            fail(f"item {i}: a text item needs its text")
        if "text" in it and not (isinstance(it["text"], str) and it["text"].strip()):
            fail(f"item {i}: text must be the nonempty words to render")
        for k in ("tags", "palette"):
            if k in it:
                strings(it[k], f"item {i}: {k}")
        b = it.get("bbox")
        if not (isinstance(b, list) and len(b) == 4
                and all(type(v) in (int, float) and math.isfinite(v) for v in b)):
            fail(f"item {i}: bbox must be [x0, y0, x1, y1]")
        x0, y0, x1, y1 = b
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            fail(f"item {i}: bbox {b} is not inside the {w}x{h} image")
        o = {k: it[k] for k in ("label", "caption", "text", "tags", "palette") if k in it}  # not "color"
        o["bbox"] = [edge(y0, False) - h // 2, edge(x0, False) - w // 2, edge(y1, True) - h // 2,
                     edge(x1, True) - w // 2]
        if not (o["bbox"][0] < o["bbox"][2] and o["bbox"][1] < o["bbox"][3]):
            fail(f"item {i}: bbox {b} is less than a pixel")
        out.append(o)
    layout = {"summary": s, "items": out, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    print(f"{w}x{h}")
    print(json.dumps(layout, ensure_ascii=False, separators=(",", ":")))
    if maps:
        print(json.dumps(maps))


main()
