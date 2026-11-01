#!/usr/bin/env python3
"""Turns a 3D scene spec (people and objects on a floor, a camera) into a layout draft for layout.py / render.sh.

    scene.py SPEC [--seed N] [--n N]     prints one draft per line (JSON)
SPEC is a file, "-" for standard input, or the JSON itself.

Spec:
  {"summary": "the setting, light, style",
   "people": [{"id": "A", "posture": "standing", "at": [x, z], "facing": DEG, "height": 1.75, "base": 0,
               "caption": "..."}, ...],
   "objects": [{"id": "bed", "box": [w, d, h], "at": [x, z], "facing": DEG, "base": 0, "caption": "..."}, ...],
   "camera": {"yaw": DEG | [lo, hi], "elevation": DEG | [lo, hi], "distance": M | [lo, hi],
              "framing": "full" | "medium" | "close" | [choices], "size": "WxH" | "auto",
              "preset": NAME | [names]}}   (a preset's ranges; fields given beside it win)
Metres on the floor: x to the right and z toward the camera as seen from yaw 0. facing F: turned toward a camera at
yaw F (0: toward the yaw-0 camera, 90: toward its right). Lying and on all fours: facing is where the head points.

Each draft adds what depends on its camera: where a body is in the frame, how it is turned to the camera, which
body is in front of another, where the frame cuts it. Captions go into every draft as written, so they say nothing
about the camera or the frame ("facing the camera", "on the left"); they place things relative to the people and
objects ("her back to the window", "his hand on the woman's shoulder"). What the posture leaves open is the
caption's: lying on the back, front or side; arms and legs.
A body is one box: a person's covers the whole body in its posture, and a pose the posture does not describe (legs
hanging over a ledge, a body bent at the waist) needs its own "box" [w, d, h] and "base". A layout places subjects
in the frame, not in depth: "A between B and C" shows only from a camera that looks across the gap, so that it runs
left to right in the image.
Objects behind the camera are left out of a draft; a person there is an error.
"""
import json, math, random, sys

# width, depth (along facing), height for a 1.75 m person; kind: how to describe orientation.
POSTURES = {
    "standing": (0.5, 0.3, 1.75, "upright"),
    "sitting": (0.5, 0.6, 1.3, "upright"),
    "sitting-floor": (0.5, 0.8, 0.9, "upright"),
    "kneeling": (0.5, 0.5, 1.25, "upright"),
    "kneeling-sitting": (0.5, 0.6, 0.95, "upright"),
    "crouching": (0.55, 0.6, 1.0, "upright"),
    "all-fours": (0.5, 1.2, 0.75, "prone"),
    "lying": (0.5, 1.8, 0.3, "prone"),
}
POSE_TEXT = {"standing": "standing", "sitting": "seated", "sitting-floor": "sitting on the floor",
             "kneeling": "kneeling upright", "kneeling-sitting": "kneeling, sitting back on the heels",
             "crouching": "crouching", "all-fours": "on all fours", "lying": "lying down"}
# Named cameras: yaw (mirrored at random when "mirror"), elevation, distance ranges.
PRESETS = {
    "front": {"yaw": [-15, 15], "elevation": [0, 15]},
    "three-quarter": {"yaw": [30, 55], "elevation": [5, 25], "mirror": True},
    "side": {"yaw": [75, 100], "elevation": [0, 20], "mirror": True},
    "behind": {"yaw": [150, 210], "elevation": [5, 30]},
    "high": {"yaw": [-45, 45], "elevation": [35, 60]},
    "overhead": {"yaw": [-30, 30], "elevation": [70, 85]},
    "low": {"yaw": [-40, 40], "elevation": [-20, -8]},
}
# Metres one body's footprint must reach nearer the camera (along its view, on the floor) than another's to count as
# in front of it.
DEPTH_APART = 0.1
SIZES = [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216), (1344, 768), (768, 1344)]


def fail(msg):
    sys.exit(f"scene: {msg}")


def sub(a, b): return [a[i] - b[i] for i in range(3)]
def dot(a, b): return sum(a[i] * b[i] for i in range(3))
def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def norm(a):
    n = math.sqrt(dot(a, a))
    return [v / n for v in a]


def heading(deg):
    r = math.radians(deg)
    return [math.sin(r), 0.0, math.cos(r)]


def pick(v, rng):
    if isinstance(v, list):
        if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            return rng.uniform(v[0], v[1])
        return rng.choice(v)
    return v


class Body:
    def __init__(self, d, person):
        self.id, self.caption, self.person = d.get("id", "?"), d.get("caption", ""), person
        if person:
            p = d.get("posture", "standing")
            if p not in POSTURES:
                fail(f"{self.id}: posture must be one of {', '.join(POSTURES)}")
            w, dp, h, self.kind = POSTURES[p]
            self.posture = p
            s = d.get("height", 1.75) / 1.75
            self.dims = d.get("box", [w * s, dp * s, h * s])
        else:
            self.kind = "object"
            self.dims = d.get("box")
            if not self.dims:
                fail(f"{self.id}: an object needs box [w, d, h]")
        x, z = d.get("at", [0, 0])
        self.base = d.get("base", 0.0)
        self.facing = d.get("facing", 0.0)
        self.centre = [x, self.base + self.dims[2] / 2, z]

    def corners(self):
        w, dp, h = self.dims
        f = heading(self.facing)
        r = [f[2], 0.0, -f[0]]  # the body's own right
        x, _, z = self.centre
        out = []
        for a in (-0.5, 0.5):
            for b in (-0.5, 0.5):
                for y in (self.base, self.base + h):
                    out.append([x + a * w * r[0] + b * dp * f[0], y, z + a * w * r[2] + b * dp * f[2]])
        return out

    def end(self, sign):  # head (+1) or feet (-1) of a prone body, at mid height
        f = heading(self.facing)
        return [self.centre[0] + sign * f[0] * self.dims[1] / 2, self.centre[1],
                self.centre[2] + sign * f[2] * self.dims[1] / 2]


class Camera:
    def __init__(self, target, yaw, elev, dist):
        self.yaw, self.elev, self.dist = yaw, elev, dist
        e = math.radians(max(-80.0, min(85.0, elev)))
        h = heading(yaw)
        self.pos = [target[0] + dist * math.cos(e) * h[0], target[1] + dist * math.sin(e),
                    target[2] + dist * math.cos(e) * h[2]]
        self.fwd = norm(sub(target, self.pos))
        self.right = norm(cross(self.fwd, [0, 1, 0]))
        self.up = cross(self.right, self.fwd)

    def project(self, p):  # image-plane coordinates at focal length 1, y down; None behind the camera
        q = sub(p, self.pos)
        zc = dot(q, self.fwd)
        if zc < 0.05:
            return None
        return dot(q, self.right) / zc, -dot(q, self.up) / zc, zc


def side(cx, w):
    t = cx / w
    return "on the left" if t < 0.36 else "on the right" if t > 0.64 else "in the centre"


def orientation(b, cam, project):
    if b.kind == "upright":
        to_cam = norm(sub(cam.pos, b.centre))
        to_cam[1] = 0
        if dot(to_cam, to_cam) < 1e-6:
            return "seen from directly above"
        to_cam = norm(to_cam)
        f = heading(b.facing)
        a = math.degrees(math.acos(max(-1.0, min(1.0, dot(f, to_cam)))))
        if abs(a) < 30:
            return "facing the camera"
        turn = "the viewer's right" if dot(f, cam.right) > 0 else "the viewer's left"
        if abs(a) < 65:
            return f"three-quarter view, turned toward {turn}"
        if abs(a) < 115:
            return f"in profile, facing {turn}"
        if abs(a) < 150:
            return f"turned away from the camera toward {turn}, seen mostly from behind, face barely visible"
        return "back to the camera, head facing away, face not visible"
    hd, ft = project(b.end(1)), project(b.end(-1))
    dx, dy = hd[0] - ft[0], hd[1] - ft[1]
    if abs(dx) >= abs(dy):
        return "head to the viewer's " + ("right" if dx > 0 else "left")
    if cam.elev >= 50:
        return "head toward the " + ("top" if dy < 0 else "bottom") + " of the frame"
    return "feet toward the camera, head pointing away" if dy < 0 else "head toward the camera, feet pointing away"


# The lowest visible fraction of a body's height: where the frame's bottom edge cuts it, and what that leaves out.
CUTS = [(0.72, "neck", "everything below the shoulders"), (0.6, "chest", "the belly, hips and legs"),
        (0.5, "waist", "the hips and legs"), (0.4, "hips", "the legs"),
        (0.25, "thighs", "the knees, lower legs and feet"), (0.08, "knees", "the lower legs and feet")]
SEATED = [(0.75, "neck", "everything below the shoulders"), (0.55, "chest", "the belly, lap and legs"),
          (0.35, "waist", "the lap and legs"), (0.1, "lap", "the lower legs and feet")]


def cutoff(b, project, H):
    """(part, hidden) where the frame's bottom edge cuts an upright body; None if it is fully in."""
    h = b.dims[2]
    for k in range(0, 41):
        t = k / 40
        if project([b.centre[0], b.base + t * h, b.centre[2]])[1] <= H:
            break
    else:
        return None
    if t <= 0.02:
        return None
    for lim, part, hidden in (CUTS if b.posture == "standing" else SEATED):
        if t >= lim:
            return part, hidden
    return None


def viewpoint(cam, framing):
    e = cam.elev
    v = ("low angle from near the floor, looking up at them, the ceiling visible above" if e < -8 else "eye level" if e < 15 else "high angle, looking down" if e < 40
         else "steep high angle from above" if e < 70 else "overhead view from directly above")
    shot = {"full": "full-body shot", "medium": "medium shot", "close": "close-up"}[framing]
    return f"{shot}, {v}"


def render(spec, rng):
    bodies = [Body(d, True) for d in spec.get("people", [])] + [Body(d, False) for d in spec.get("objects", [])]
    people = [b for b in bodies if b.person]
    if not people:
        fail("people must be a nonempty list")
    c = dict(spec.get("camera", {}))
    preset = pick(c.pop("preset", None), rng)
    if preset is not None:
        if preset not in PRESETS:
            fail(f"camera preset must be one of {', '.join(PRESETS)}")
        c = {**PRESETS[preset], **c}
    yaw, elev = pick(c.get("yaw", 0), rng), pick(c.get("elevation", 10), rng)
    if c.get("mirror") and rng.random() < 0.5:
        yaw = -yaw
    dist, framing = pick(c.get("distance", 3.5), rng), pick(c.get("framing", "full"), rng)
    if framing not in ("full", "medium", "close"):
        fail("framing must be full, medium or close")
    # Aim at the people's middle, higher for tighter framing (faces and torsos).
    target = [sum(b.centre[i] for b in people) / len(people) for i in range(3)]
    top = max(b.base + b.dims[2] for b in people)
    target[1] = {"full": target[1], "medium": target[1] + 0.3 * (top - target[1]),
                 "close": target[1] + 0.6 * (top - target[1])}[framing]
    cam = Camera(target, yaw, elev, dist)

    raw, kept = [], []
    for b in bodies:
        pts = [cam.project(p) for p in b.corners()]
        if any(p is None for p in pts):
            if b.person:
                fail(f"{b.id} is behind or at the camera; increase distance")
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        raw.append((min(xs), min(ys), max(xs), max(ys), sum(p[2] for p in pts) / len(pts)))
        kept.append(b)
    bodies = kept
    pr = [r for r, b in zip(raw, bodies) if b.person]
    ux0, uy0, ux1, uy1 = min(r[0] for r in pr), min(r[1] for r in pr), max(r[2] for r in pr), max(r[3] for r in pr)

    size = c.get("size", "auto")
    if size == "auto":
        asp = (ux1 - ux0) / (uy1 - uy0) * {"full": 1.0, "medium": 1.45, "close": 2.3}[framing]
        if len(people) == 1 and framing != "full":
            asp = min(asp, 0.8)  # a portrait
        W, H = min(SIZES, key=lambda s: abs(math.log(s[0] / s[1] / asp)))
    else:
        W, H = (int(v) for v in size.lower().split("x"))
    fill = {"full": 0.88, "medium": 1.45, "close": 2.3}[framing]  # the people's height over the frame's
    k = min(W * 0.92 / (ux1 - ux0), H * fill / (uy1 - uy0))
    ox = W / 2 - k * (ux0 + ux1) / 2
    oy = H / 2 - k * (uy0 + uy1) / 2 if framing == "full" else H * 0.05 - k * uy0
    project = lambda p: (lambda q: (ox + k * q[0], oy + k * q[1]))(cam.project(p))

    view = norm([cam.fwd[0], 0.0, cam.fwd[2]]) if abs(cam.fwd[1]) < 0.999 else heading(cam.yaw + 180)
    boxes = []
    for r, b in zip(raw, bodies):
        x0, y0, x1, y1 = ox + k * r[0], oy + k * r[1], ox + k * r[2], oy + k * r[3]
        area = (x1 - x0) * (y1 - y0)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        vis = max(0, cx1 - cx0) * max(0, cy1 - cy0) / area
        near = min(dot([p[0], 0.0, p[2]], view) for p in b.corners())
        boxes.append(((cx0, cy0, cx1, cy1), vis, r[4], near))

    items, cuts = [], []
    for i, b in enumerate(bodies):
        (x0, y0, x1, y1), vis, depth, near = boxes[i]
        if vis < 0.05:
            continue
        notes = [side((x0 + x1) / 2, W)]
        if b.person:
            notes.append(POSE_TEXT[b.posture])
            notes.append(orientation(b, cam, project))
            for j, o in enumerate(bodies):
                if j == i or not o.person:
                    continue
                (a0, b0, a1, b1), _, _, near2 = boxes[j]
                ov = max(0, min(x1, a1) - max(x0, a0)) * max(0, min(y1, b1) - max(y0, b0))
                # Bodies whose fronts are about level overlap side by side (touching), not one hiding the other.
                if ov > 0.15 * (x1 - x0) * (y1 - y0) and abs(near - near2) >= DEPTH_APART:
                    notes.append(f"in front of {o.id}" if near < near2 else f"partly behind {o.id}")
            cut = cutoff(b, project, H) if b.kind == "upright" else None
            if cut:
                cuts.append(f"{b.id}'s {cut[0]}")
                notes.append(f"cut off at the {cut[0]} by the bottom edge of the frame, {cut[1]} not in the image")
            elif vis < 0.9:
                notes.append("partly out of frame")
        cap = b.caption.rstrip(". ")
        items.append({"label": "character" if b.person else "object", "caption": f"{b.id}: {cap}; " + ", ".join(notes),
                      "bbox": [round(x0), round(y0), round(x1), round(y1)], "_depth": depth})
    items.sort(key=lambda it: -it["_depth"])  # far to near
    for it in items:
        del it["_depth"]
    names = [it["caption"].split(":")[0] for it in items if it["label"] == "character"]
    where = "; ".join(f"{b.id} {side((boxes[i][0][0] + boxes[i][0][2]) / 2, W)}"
                      for i, b in enumerate(bodies) if b.person and boxes[i][1] >= 0.05)
    caption = f"{spec.get('summary', '').rstrip('. ')}. {viewpoint(cam, framing)}. {where}."
    if cuts:
        caption += f" The bottom edge of the frame cuts across {' and '.join(cuts)}."
    return {"width": W, "height": H, "summary": {"caption": caption.lstrip('. ')}, "items": items,
            "camera": {"preset": preset, "yaw": round(yaw, 1), "elevation": round(elev, 1), "distance": round(dist, 2),
                       "framing": framing, "people": len(names)}}


def main():
    args = sys.argv[1:]
    seed, n, src = None, 1, None
    while args:
        a = args.pop(0)
        if a == "--seed":
            seed = int(args.pop(0))
        elif a == "--n":
            n = int(args.pop(0))
        elif src is None:
            src = a
        else:
            fail(f"unexpected argument {a}")
    if src is None:
        fail("usage: scene.py SPEC [--seed N] [--n N]")
    try:
        spec = json.load(sys.stdin) if src == "-" else json.loads(src) if src.lstrip().startswith("{") \
            else json.load(open(src))
    except Exception as e:
        fail(f"not a JSON spec: {e}")
    rng = random.Random(seed)
    for _ in range(n):
        d = render(spec, rng)
        print(json.dumps(d, ensure_ascii=False))


main()
