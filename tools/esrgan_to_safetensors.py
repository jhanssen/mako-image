#!/usr/bin/env python3
"""Converts an ESRGAN upscaler's PyTorch checkpoint (.pth) to safetensors for mako_upscale and mako_image_serve.

usage: esrgan_to_safetensors.py MODEL.pth [OUT.safetensors]

Reads torch's zip and legacy pickle formats without torch (the tensors' bytes are copied as they are; F64 becomes F32)
and names the layers as Real-ESRGAN does (conv_first, body.N.rdbM.convK, conv_body, conv_upN, conv_hr, conv_last),
renaming the original ESRGAN's model.N.sub.M layers. A state dict under "params_ema" or "params" is unwrapped. The
output defaults to the input's name with .safetensors.
"""
import array
import collections
import json
import os
import pickle
import re
import struct
import sys
import zipfile

DTYPES = {"FloatStorage": ("F32", 4), "HalfStorage": ("F16", 2), "BFloat16Storage": ("BF16", 2),
          "DoubleStorage": ("F64", 8)}


def fail(msg):
    print(f"esrgan_to_safetensors: {msg}", file=sys.stderr)
    sys.exit(1)


class Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def rebuild_tensor(storage, offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    return Tensor(storage, offset, size, stride)


class Unpickler(pickle.Unpickler):
    """Loads a state dict's pickle: tensors become Tensor records over (dtype name, storage key, numel)."""

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, requires_grad, hooks: data
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name in DTYPES:
            return name
        raise pickle.UnpicklingError(f"{module}.{name} is not part of a state dict")

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or pid[0] != "storage" or pid[1] not in DTYPES:
            raise pickle.UnpicklingError(f"unknown storage {pid!r}")
        if len(pid) > 5 and pid[5] is not None:
            raise pickle.UnpicklingError("storage views are not supported")
        return (pid[1], pid[2], pid[4])  # dtype name, key, numel


def load_zip(path):
    """Torch's zip format: data.pkl and the storages as data/KEY, under one directory."""
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n == "data.pkl" or n.endswith("/data.pkl")]
    if len(pkl) != 1:
        fail(f"{path}: no data.pkl in the archive")
    prefix = pkl[0][: -len("data.pkl")]
    with z.open(pkl[0]) as f:
        obj = Unpickler(f).load()
    return obj, lambda key: z.read(f"{prefix}data/{key}")


def load_legacy(path):
    """The legacy (pre-1.6) format: pickles, then each storage's int64 element count and bytes in key order."""
    with open(path, "rb") as f:
        if pickle.load(f) != 0x1950A86A20F9469CFC6C:
            fail(f"{path}: not a torch checkpoint")
        pickle.load(f)  # protocol version
        pickle.load(f)  # sys_info
        obj = Unpickler(f).load()
        keys = pickle.load(f)
        sizes = {}  # element size of each storage key, from the tensors that use it
        def collect(o):
            if isinstance(o, Tensor):
                sizes[o.storage[1]] = DTYPES[o.storage[0]][1]
            elif isinstance(o, dict):
                for v in o.values():
                    collect(v)
        collect(obj)
        data = {}
        for key in keys:
            (numel,) = struct.unpack("<q", f.read(8))
            if key not in sizes:
                fail(f"storage {key} is not used by any tensor")
            data[key] = f.read(numel * sizes[key])
            if len(data[key]) != numel * sizes[key]:
                fail(f"{path}: truncated")
    return obj, lambda key: data[key]


def state_dict(obj):
    for key in ("params_ema", "params"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict) or not all(isinstance(v, Tensor) for v in obj.values()):
        fail("not a state dict of tensors")
    return obj


def renamed(keys):
    """Real-ESRGAN names for each tensor name (the original ESRGAN's model.N layers renamed)."""
    if not any(k.startswith("model.") for k in keys):
        return {k: k for k in keys}
    seq = sorted({int(m[1]) for k in keys if (m := re.fullmatch(r"model\.(\d+)\.(weight|bias)", k))})
    subs = sorted({int(m[1]) for k in keys if (m := re.match(r"model\.1\.sub\.(\d+)\.", k))})
    if not subs or seq[:1] != [0] or len(seq) < 3:
        fail("not the original ESRGAN layout (model.0, model.1.sub.N, model.N)")
    trunk = max(subs)
    out = {}
    for k in keys:
        if m := re.fullmatch(r"model\.0\.(weight|bias)", k):
            out[k] = f"conv_first.{m[1]}"
        elif m := re.fullmatch(r"model\.1\.sub\.(\d+)\.RDB(\d)\.conv(\d)\.0\.(weight|bias)", k):
            out[k] = f"body.{m[1]}.rdb{m[2]}.conv{m[3]}.{m[4]}"
        elif m := re.fullmatch(rf"model\.1\.sub\.{trunk}\.(weight|bias)", k):
            out[k] = f"conv_body.{m[1]}"
        elif m := re.fullmatch(r"model\.(\d+)\.(weight|bias)", k):
            i = int(m[1])
            name = "conv_last" if i == seq[-1] else "conv_hr" if i == seq[-2] else f"conv_up{seq.index(i)}"
            out[k] = f"{name}.{m[2]}"
        else:
            fail(f"unknown tensor {k}")
    return out


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) == 3 else os.path.splitext(src)[0] + ".safetensors"
    if zipfile.is_zipfile(src):
        obj, read = load_zip(src)
    else:
        obj, read = load_legacy(src)
    sd = state_dict(obj)
    names = renamed(list(sd.keys()))
    tensors = []
    for k, t in sd.items():
        dtype, es = DTYPES[t.storage[0]]
        numel = 1
        for s in t.size:
            numel *= s
        expect, acc = [], 1
        for s in reversed(t.size):
            expect.insert(0, acc)
            acc *= s
        if numel and tuple(expect) != t.stride:
            fail(f"{k}: not contiguous")
        data = read(t.storage[1])[t.offset * es:(t.offset + numel) * es]
        if len(data) != numel * es:
            fail(f"{k}: storage too short")
        if dtype == "F64":
            data = array.array("f", array.array("d", data)).tobytes()
            dtype = "F32"
        tensors.append((names[k], dtype, t.size, data))
    blocks = len({n.split(".")[1] for n, *_ in tensors if n.startswith("body.")})
    ups = len({n for n, *_ in tensors if re.fullmatch(r"conv_up\d+\.weight", n)})
    header = {"__metadata__": {"architecture": "ESRGAN", "blocks": str(blocks), "scale": str(2 ** ups),
                               "source": os.path.basename(src)}}
    offset = 0
    for name, dtype, size, data in tensors:
        header[name] = {"dtype": dtype, "shape": list(size), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _, _, _, data in tensors:
            f.write(data)
    print(f"{dst}: {len(tensors)} tensors, {blocks} blocks, {2 ** ups}x")


if __name__ == "__main__":
    main()
