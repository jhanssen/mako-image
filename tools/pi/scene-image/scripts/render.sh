#!/bin/sh
# Renders an image with mako_image_serve's /v1/images/generations and saves it as a PNG.
#   render.sh [--model NAME] [--lora NAME[:SCALE][:merge|:unmerged]]... [--no-loras] [--steps N] [--negative TEXT]
#             [--guidance F] [--sampler NAME] [--skip-blocks LIST] [--seed N] [--image FILE [--strength F]]
#             [--color-map FILE] [--depth-map FILE] [--color-end F] [--depth-end F] [--upscale NAME[:SCALE]]
#             WIDTHxHEIGHT|auto "prompt"
#   render.sh [options] --layout DRAFT     (MOSS-Image: a layout draft, see layout.py, as a file, - for standard input or
#                                           the JSON itself; the size comes from the draft)
# --model renders with that image model (default: the server's); --lora (repeatable) with these LoRAs in place of the
# model's defaults (:merge into the weights or :unmerged beside them; default the model's way); --no-loras with none;
# --steps N denoising steps (0: the model's own count; default: the model's setting); --negative what the image should
# not show (a second transformer pass a step); --guidance its scale (default: the model's); --sampler euler, euler_a
# (SDXL models only) or er_sde (default: the model's); --skip-blocks the blocks the negative prompt's pass leaves out,
# e.g. 22,24 ("" for none; MOSS-Image only; default: the model's); --seed N the noise's seed (default: a new one each
# render; the PNG records it); --image FILE starts from that image (MOSS-Image), scaled to cover the size and
# centred, --strength the share of the steps run on it (0 to 1, default 0.75: more changes more); size "auto" takes
# the image's (at most 1024x1024 pixels' area). --color-map FILE (a flat colour per subject) and --depth-map FILE
# (near white) steer the layout on a model with a control adapter, each for the first --color-end / --depth-end share
# of the steps (0 to 1, default 1); a layout draft's color_map / depth_map are passed the same way. --upscale NAME
# runs the server's ESRGAN upscaler NAME on the image (scripts/upscalers.sh lists them), to SCALE times the size
# (default: the upscaler's own, 4 for most; the server resamples down from its output for less).
# MAKO_IMAGE_URL (mako_image_serve; default http://127.0.0.1:8082/v1), MAKO_IMAGE_API_KEY (when the server wants
# one), MAKO_IMAGE_DIR (default ~/Pictures/mako), MAKO_IMAGE_OPEN=0 (not opened in the image viewer). Prints the saved
# file's path.
set -eu
usage() {
    echo "usage: render.sh [--model NAME] [--lora NAME[:SCALE][:merge|:unmerged]]... [--no-loras] [--steps N]" >&2
    echo "                 [--negative TEXT] [--guidance F] [--sampler NAME] [--skip-blocks LIST] [--seed N]" >&2
    echo "                 [--image FILE [--strength F]] [--color-map FILE] [--depth-map FILE]" >&2
    echo "                 [--color-end F] [--depth-end F] [--upscale NAME[:SCALE]] WIDTHxHEIGHT|auto \"prompt\"" >&2
    echo "       render.sh [options] --layout DRAFT.json|-|'{...}'" >&2
    exit 2
}
loras=
steps=
negative=
guidance=
sampler=
skip=
skip_set=
seed=
image=
strength=
color_map=
depth_map=
color_end=
depth_end=
model=
layout=
upscale=
while [ $# -gt 0 ]; do
    case $1 in
    --lora)
        [ $# -ge 2 ] || usage
        loras="$loras$2
"
        shift 2
        ;;
    --model)
        [ $# -ge 2 ] || usage
        model=$2
        shift 2
        ;;
    --layout)
        [ $# -ge 2 ] || usage
        layout=$2
        shift 2
        ;;
    --steps)
        [ $# -ge 2 ] || usage
        steps=$2
        shift 2
        ;;
    --negative)
        [ $# -ge 2 ] || usage
        negative=$2
        shift 2
        ;;
    --guidance)
        [ $# -ge 2 ] || usage
        guidance=$2
        shift 2
        ;;
    --sampler)
        [ $# -ge 2 ] || usage
        sampler=$2
        shift 2
        ;;
    --skip-blocks)
        [ $# -ge 2 ] || usage
        skip=$2
        skip_set=1
        shift 2
        ;;
    --seed)
        [ $# -ge 2 ] || usage
        seed=$2
        shift 2
        ;;
    --image)
        [ $# -ge 2 ] || usage
        image=$2
        [ -r "$image" ] || { echo "render.sh: cannot read $image" >&2; exit 2; }
        shift 2
        ;;
    --color-map | --depth-map)
        [ $# -ge 2 ] || usage
        [ -r "$2" ] || { echo "render.sh: cannot read $2" >&2; exit 2; }
        if [ "$1" = --color-map ]; then color_map=$2; else depth_map=$2; fi
        shift 2
        ;;
    --color-end | --depth-end)
        [ $# -ge 2 ] || usage
        if [ "$1" = --color-end ]; then color_end=$2; else depth_end=$2; fi
        shift 2
        ;;
    --strength)
        [ $# -ge 2 ] || usage
        strength=$2
        shift 2
        ;;
    --upscale)
        [ $# -ge 2 ] || usage
        upscale=$2
        shift 2
        ;;
    --no-loras)
        loras="
"
        shift
        ;;
    -*) usage ;;
    *) break ;;
    esac
done
if [ -n "$layout" ]; then
    [ $# -eq 0 ] || usage
    compiled=$(python3 "$(dirname "$0")/layout.py" "$layout")
    size=$(printf '%s\n' "$compiled" | sed -n 1p)
    prompt=$(printf '%s\n' "$compiled" | sed -n 2p)
    maps=$(printf '%s\n' "$compiled" | sed -n 3p)
    if [ -n "$maps" ]; then
        [ -n "$color_map" ] || color_map=$(MAPS=$maps python3 -c 'import json, os; print(json.loads(os.environ["MAPS"]).get("color_map", ""))')
        [ -n "$depth_map" ] || depth_map=$(MAPS=$maps python3 -c 'import json, os; print(json.loads(os.environ["MAPS"]).get("depth_map", ""))')
    fi
else
    [ $# -eq 2 ] || usage
    size=$1
    prompt=$2
fi
url=${MAKO_IMAGE_URL:-http://127.0.0.1:8082/v1}
dir=${MAKO_IMAGE_DIR:-$HOME/Pictures/mako}
mkdir -p "$dir"
base="$dir/$(date +%Y%m%d-%H%M%S)"
tmp=$(mktemp)
req=$(mktemp)
trap 'rm -f "$tmp" "$req"' EXIT
REQ=$req SIZE=$size PROMPT=$prompt LORAS=$loras STEPS=$steps MODEL=$model NEGATIVE=$negative GUIDANCE=$guidance \
    SAMPLER=$sampler SKIP=$skip SKIP_SET=$skip_set SEED=$seed IMAGE=$image STRENGTH=$strength \
    COLOR_MAP=$color_map DEPTH_MAP=$depth_map COLOR_END=$color_end DEPTH_END=$depth_end UPSCALE=$upscale python3 -c '
import json, os, sys
b = {"prompt": os.environ["PROMPT"], "size": os.environ["SIZE"], "n": 1, "response_format": "b64_json"}
if os.environ["IMAGE"]:
    import base64
    b["image"] = base64.b64encode(open(os.environ["IMAGE"], "rb").read()).decode()
control = {}
for key in ("color", "depth"):
    path = os.environ[key.upper() + "_MAP"]
    if path:
        import base64
        control[key] = base64.b64encode(open(path, "rb").read()).decode()
    end = os.environ[key.upper() + "_END"]
    if end:
        try:
            control[key + "_end"] = float(end)
        except ValueError:
            sys.exit(f"--{key}-end takes a number")
if control:
    b["control"] = control
if os.environ["STRENGTH"]:
    if not os.environ["IMAGE"]:
        sys.exit("--strength goes with --image")
    try:
        b["strength"] = float(os.environ["STRENGTH"])
    except ValueError:
        sys.exit("--strength takes a number")
if os.environ["MODEL"]:
    b["model"] = os.environ["MODEL"]
if os.environ["UPSCALE"]:
    name, _, scale = os.environ["UPSCALE"].rpartition(":")
    try:
        b["upscale"] = {"model": name, "scale": float(scale)} if name else {"model": os.environ["UPSCALE"]}
    except ValueError:
        sys.exit("--upscale takes NAME or NAME:SCALE")
if os.environ["STEPS"]:
    try:
        b["steps"] = int(os.environ["STEPS"])
    except ValueError:
        sys.exit("--steps takes a whole number")
if os.environ["NEGATIVE"]:
    b["negative_prompt"] = os.environ["NEGATIVE"]
if os.environ["GUIDANCE"]:
    try:
        b["guidance_scale"] = float(os.environ["GUIDANCE"])
    except ValueError:
        sys.exit("--guidance takes a number")
if os.environ["SAMPLER"]:
    b["sampler"] = os.environ["SAMPLER"]
if os.environ["SKIP_SET"]:
    try:
        b["skip_blocks"] = [int(x) for x in os.environ["SKIP"].split(",") if x.strip()]
    except ValueError:
        sys.exit("--skip-blocks takes block numbers, e.g. 22,24")
if os.environ["SEED"]:
    if not os.environ["SEED"].isdigit():
        sys.exit("--seed takes a whole number")
    b["seed"] = os.environ["SEED"]
if os.environ["LORAS"]:
    b["loras"] = []
    for spec in filter(None, os.environ["LORAS"].split("\n")):
        lora = {}
        for suffix, merge in ((":merge", True), (":unmerged", False)):
            if spec.endswith(suffix):
                spec, lora["merge"] = spec[: -len(suffix)], merge
                break
        name, _, scale = spec.rpartition(":")
        try:
            lora.update({"name": name, "scale": float(scale)} if name else {"name": spec})
        except ValueError:
            lora["name"] = spec
        b["loras"].append(lora)
json.dump(b, open(os.environ["REQ"], "w"))' || exit 2
status=$(curl -sS -o "$tmp" -w '%{http_code}' --max-time 1800 "$url/images/generations" \
    -H 'Content-Type: application/json' ${MAKO_IMAGE_API_KEY:+-H "Authorization: Bearer $MAKO_IMAGE_API_KEY"} \
    --data-binary @"$req")
out=$(BASE=$base STATUS=$status python3 - "$tmp" <<'PY'
import base64, json, os, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(f"mako_image_serve answered HTTP {os.environ['STATUS']} with no JSON")
if os.environ["STATUS"] != "200" or "data" not in d:
    sys.exit(f"mako_image_serve: HTTP {os.environ['STATUS']}: {d.get('error', {}).get('message', d)}")
png = base64.b64decode(d["data"][0]["b64_json"])
# Renders finishing in the same second each get a file of their own.
for i in range(1, 1000):
    path = os.environ["BASE"] + (f"-{i}" if i > 1 else "") + ".png"
    try:
        with open(path, "xb") as f:
            f.write(png)
        break
    except FileExistsError:
        continue
else:
    sys.exit(f"no free name for {os.environ['BASE']}.png")
print(path)
PY
)
if [ "${MAKO_IMAGE_OPEN:-1}" != 0 ]; then
    open "$out"
fi
echo "$out"
