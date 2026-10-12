"""Bridge between ComfyUI tensors and the headless ``lightcraft-cli`` binary.

A node writes its input image to a PNG in a temporary folder, runs
``lightcraft-cli run --import in.png --script steps.jsonl`` (one engine command per line, the
last one an ``app.export`` to a PNG) and reads the PNG back. The binary is found through the
``LIGHTCRAFT_CLI`` environment variable, a ``bin/`` folder next to this file, or the ``PATH``.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BINARY = "lightcraft-cli.exe" if sys.platform.startswith("win") else "lightcraft-cli"
TIMEOUT = float(os.environ.get("LIGHTCRAFT_CLI_TIMEOUT", "600"))
#: Side of the square grid a ComfyUI mask is stored on for LightCraft (its cap is 1024).
SEG_SIDE = 512


class LightCraftError(RuntimeError):
    """A command failed or the CLI could not be found; the message is meant for the ComfyUI user."""


def find_cli() -> str:
    candidates = []
    env = os.environ.get("LIGHTCRAFT_CLI")
    if env:
        candidates.append(Path(env))
    candidates.append(HERE / "bin" / BINARY)
    on_path = shutil.which("lightcraft-cli")
    if on_path:
        candidates.append(Path(on_path))
    for c in candidates:
        if c.is_file():
            return str(c)
    raise LightCraftError(
        "lightcraft-cli not found. Build it with `cargo build --release -p lightcraft-cli` and either "
        "set the LIGHTCRAFT_CLI environment variable to the binary, copy it to "
        f"{HERE / 'bin' / BINARY}, or put it on the PATH."
    )


# ---------------------------------------------------------------- tensors <-> PNG


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def from_numpy(a: np.ndarray):
    try:
        import torch
    except ImportError:  # pragma: no cover - only outside ComfyUI
        return a
    return torch.from_numpy(np.ascontiguousarray(a))


def write_png(path: Path, image: np.ndarray) -> None:
    """One ``[H, W, 3]`` float image (0..1) as an 8-bit RGB PNG."""
    if image.ndim != 3 or image.shape[2] < 3:
        raise LightCraftError(f"expected an image of shape [H, W, 3], got {list(image.shape)}")
    rgb = (np.clip(image[:, :, :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(path, format="PNG")


def read_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0)


# ---------------------------------------------------------------- ComfyUI MASK -> LightCraft segmentation


def seg_from_mask(mask: np.ndarray, side: int = SEG_SIDE) -> dict:
    """A ``[H, W]`` coverage mask (0..1) as LightCraft's stored segmentation (``SegMask``).

    LightCraft keeps a model segmentation as a ``side x side`` grid of logits over the whole
    image stretched to a square, quantized to int8 in steps of 1/8, raw-deflated and base64
    encoded; coverage is the sigmoid of the bilinearly sampled logit. Passing one with
    ``mask.add {kind: "object", seg}`` needs no model, so any ComfyUI mask becomes a local
    adjustment mask."""
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if m.ndim != 2:
        raise LightCraftError(f"a mask must be [H, W], got {list(m.shape)}")
    side = int(max(8, min(1024, side)))
    grid = np.asarray(Image.fromarray((m * 255.0 + 0.5).astype(np.uint8), "L").resize((side, side), Image.BILINEAR), dtype=np.float32) / 255.0
    eps = 1e-6
    logits = np.log((grid + eps) / (1.0 - grid + eps))
    q = np.clip(np.round(logits * 8.0), -127, 127).astype(np.int8)
    co = zlib.compressobj(8, zlib.DEFLATED, -15)
    z = co.compress(q.astype(np.uint8).tobytes()) + co.flush()
    return {"side": side, "data": base64.b64encode(z).decode("ascii")}


def mask_from_seg(seg: dict) -> np.ndarray:
    """The inverse of `seg_from_mask` (coverage grid); used by the tests."""
    side = int(seg["side"])
    z = base64.b64decode(seg["data"])
    q = np.frombuffer(zlib.decompress(z, -15), dtype=np.int8).astype(np.float32) / 8.0
    return 1.0 / (1.0 + np.exp(-q.reshape(side, side)))


# ---------------------------------------------------------------- running commands


def run_steps(in_png: Path, steps: list[tuple[str, dict]], workdir: Path) -> list[dict]:
    """Import ``in_png`` into a headless session, run ``steps`` (command id, params) in order.

    Returns each command's result. Raises `LightCraftError` naming the failing command."""
    script = workdir / "steps.jsonl"
    script.write_text("".join(json.dumps({"command": c, "params": p or {}}) + "\n" for c, p in steps), encoding="utf-8")
    args = [find_cli(), "run", "--import", str(in_png), "--script", str(script)]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=TIMEOUT, check=False, cwd=str(workdir))
    except subprocess.TimeoutExpired as e:
        raise LightCraftError(f"lightcraft-cli timed out after {TIMEOUT:.0f}s") from e
    except OSError as e:
        raise LightCraftError(f"could not start lightcraft-cli ({args[0]}): {e}") from e
    results = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("ok") is False:
            raise LightCraftError(f"LightCraft {r.get('command')}: {r.get('error')}")
        results.append(r)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise LightCraftError("LightCraft: " + (msg[-1] if msg else f"exit code {proc.returncode}"))
    return results


def develop(images, steps: list[tuple[str, dict]], export: dict | None = None):
    """Run ``steps`` on every image of a ComfyUI batch ``[B, H, W, 3]`` and export each as PNG.

    ``export`` adds ``app.export`` options (``longEdge``, ``quality``…). Returns the batch tensor
    and the per-image command results."""
    imgs = to_numpy(images)
    if imgs.ndim == 3:
        imgs = imgs[None]
    out, results = [], []
    with tempfile.TemporaryDirectory(prefix="lightcraft-comfy-") as tmp:
        tmp = Path(tmp)
        for i in range(imgs.shape[0]):
            src, dst = tmp / f"in-{i}.png", tmp / f"out-{i}.png"
            write_png(src, imgs[i])
            exp = {"path": str(dst), "format": "png", "conflict": "overwrite"}
            exp.update(export or {})
            results.append(run_steps(src, list(steps) + [("app.export", exp)], tmp))
            if not dst.is_file():
                raise LightCraftError("LightCraft exported no file")
            out.append(read_png(dst))
    sizes = {o.shape[:2] for o in out}
    if len(sizes) > 1:
        raise LightCraftError(f"the images of the batch came out at different sizes {sorted(sizes)}")
    return from_numpy(np.stack(out)), results


def query(steps: list[tuple[str, dict]]) -> list[dict]:
    """Run commands that need no photo (``presets.list``, ``develop.controls``…)."""
    with tempfile.TemporaryDirectory(prefix="lightcraft-comfy-") as tmp:
        tmp = Path(tmp)
        script = tmp / "steps.jsonl"
        script.write_text("".join(json.dumps({"command": c, "params": p or {}}) + "\n" for c, p in steps), encoding="utf-8")
        try:
            proc = subprocess.run([find_cli(), "run", "--script", str(script)], capture_output=True, text=True, timeout=TIMEOUT, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise LightCraftError(f"lightcraft-cli: {e}") from e
        results = []
        for line in proc.stdout.splitlines():
            if line.strip().startswith("{"):
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return results


def is_default(value: float, default: float) -> bool:
    return math.isclose(float(value), float(default), abs_tol=1e-9)
