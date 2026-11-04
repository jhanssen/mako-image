#!/bin/sh
# Lists the ESRGAN upscalers mako_image_serve can run (GET /v1/images/upscalers): a name per line with what it
# multiplies a side by and its blocks. render.sh --upscale NAME[:SCALE] uses one.
# MAKO_IMAGE_URL (default http://127.0.0.1:8082/v1), MAKO_IMAGE_API_KEY (when the server wants one).
set -eu
url=${MAKO_IMAGE_URL:-http://127.0.0.1:8082/v1}
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
status=$(curl -sS -o "$tmp" -w '%{http_code}' --max-time 30 "$url/images/upscalers" \
    ${MAKO_IMAGE_API_KEY:+-H "Authorization: Bearer $MAKO_IMAGE_API_KEY"})
STATUS=$status python3 - "$tmp" <<'PY'
import json, os, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(f"mako_image_serve answered HTTP {os.environ['STATUS']} with no JSON")
if os.environ["STATUS"] != "200" or "data" not in d:
    sys.exit(f"mako_image_serve: HTTP {os.environ['STATUS']}: {d.get('error', {}).get('message', d)}")
if not d["data"]:
    print("no upscalers (mako_image_serve --upscalers DIR)")
for u in d["data"]:
    print(f"{u['name']}  {u['factor']}x, {u['blocks']} blocks")
PY
