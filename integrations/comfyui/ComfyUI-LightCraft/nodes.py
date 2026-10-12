"""ComfyUI nodes that develop an image with LightCraft (Lightroom-class tone and colour).

``image`` is ComfyUI's ``IMAGE`` batch ``[B, H, W, 3]`` in 0..1; every node returns an ``IMAGE``
of the same size. Each node is one headless LightCraft session: import the frame, apply the
edit, export a PNG. Chain nodes to stack edits, or use *LightCraft Command* for anything in the
registry (``lightcraft-cli commands`` lists it, ``lightcraft-cli controls`` the sliders).
"""

from __future__ import annotations

import json

from .lightcraft_cli import LightCraftError, develop, is_default, query, seg_from_mask, to_numpy

CATEGORY = "LightCraft"

#: Global develop sliders: (node input, control id, default, min, max, step)
GLOBAL = [
    ("exposure", "light.exposure", 0.0, -5.0, 5.0, 0.01),
    ("contrast", "light.contrast", 0.0, -100.0, 100.0, 1.0),
    ("highlights", "light.highlights", 0.0, -100.0, 100.0, 1.0),
    ("shadows", "light.shadows", 0.0, -100.0, 100.0, 1.0),
    ("whites", "light.whites", 0.0, -100.0, 100.0, 1.0),
    ("blacks", "light.blacks", 0.0, -100.0, 100.0, 1.0),
    ("temperature_k", "wb.temp", 6500.0, 2000.0, 50000.0, 50.0),
    ("tint", "wb.tint", 0.0, -150.0, 150.0, 1.0),
    ("vibrance", "color.vibrance", 0.0, -100.0, 100.0, 1.0),
    ("saturation", "color.saturation", 0.0, -100.0, 100.0, 1.0),
    ("texture", "effects.texture", 0.0, -100.0, 100.0, 1.0),
    ("clarity", "effects.clarity", 0.0, -100.0, 100.0, 1.0),
    ("dehaze", "effects.dehaze", 0.0, -100.0, 100.0, 1.0),
    ("vignette", "vignette.amount", 0.0, -100.0, 100.0, 1.0),
    ("grain", "grain.amount", 0.0, 0.0, 100.0, 1.0),
    ("sharpen", "detail.sharpenAmount", 0.0, 0.0, 150.0, 1.0),
    ("noise_reduction", "detail.nrLuminance", 0.0, 0.0, 100.0, 1.0),
]

#: Local (masked) sliders: (node input, LocalAdjustments key, default, min, max, step)
LOCAL = [
    ("exposure", "exposure", 0.0, -4.0, 4.0, 0.01),
    ("contrast", "contrast", 0.0, -100.0, 100.0, 1.0),
    ("highlights", "highlights", 0.0, -100.0, 100.0, 1.0),
    ("shadows", "shadows", 0.0, -100.0, 100.0, 1.0),
    ("whites", "whites", 0.0, -100.0, 100.0, 1.0),
    ("blacks", "blacks", 0.0, -100.0, 100.0, 1.0),
    ("temperature", "temp", 0.0, -100.0, 100.0, 1.0),
    ("tint", "tint", 0.0, -100.0, 100.0, 1.0),
    ("texture", "texture", 0.0, -100.0, 100.0, 1.0),
    ("clarity", "clarity", 0.0, -100.0, 100.0, 1.0),
    ("dehaze", "dehaze", 0.0, -100.0, 100.0, 1.0),
    ("hue", "hue", 0.0, -100.0, 100.0, 1.0),
    ("saturation", "saturation", 0.0, -100.0, 100.0, 1.0),
    ("sharpness", "sharpness", 0.0, -100.0, 100.0, 1.0),
    ("noise", "noise", 0.0, -100.0, 100.0, 1.0),
]

MASK_KINDS = ["input_mask", "sky", "subject", "background", "luminance", "linear", "radial", "prompt"]


def _slider_inputs(table):
    return {name: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": st}) for name, _, d, lo, hi, st in table}


def _extra(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        v = json.loads(text)
    except json.JSONDecodeError as e:
        raise LightCraftError(f"extra controls are not valid JSON: {e}") from e
    if not isinstance(v, dict):
        raise LightCraftError("extra controls must be a JSON object {control: value}")
    return v


class LightCraftDevelop:
    """Global develop: light, white balance, presence, effects, detail."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {"image": ("IMAGE",), "auto_tone": ("BOOLEAN", {"default": False, "tooltip": "Lightroom-style Auto first; the sliders then add to it"})}
        req.update(_slider_inputs(GLOBAL))
        req["extra_controls"] = ("STRING", {"default": "", "multiline": True, "tooltip": 'Any other control: {"curve.lights": 20, "grading.shadows.hue": 220}'})
        return {"required": req}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Develop the image with LightCraft's global sliders (exposure, highlights, shadows, white balance, clarity, dehaze…)."

    def run(self, image, auto_tone, extra_controls, **sliders):
        steps = []
        if auto_tone:
            steps.append(("develop.auto", {}))
        values = {ctl: float(sliders[name]) for name, ctl, d, *_ in GLOBAL if not is_default(sliders[name], d)}
        values.update(_extra(extra_controls))
        if values:
            steps.append(("develop.set", {"values": values}))
        img, _ = develop(image, steps)
        return (img,)


class LightCraftMaskedDevelop:
    """Relight one region: a mask (yours, or LightCraft's sky/subject/background/luminance/
    linear/radial/prompt) with Lightroom's local adjustment sliders inside it."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "image": ("IMAGE",),
            "mask_kind": (MASK_KINDS, {"default": "input_mask", "tooltip": "input_mask uses the MASK input; the others are LightCraft's own masks"}),
            "invert": ("BOOLEAN", {"default": False}),
            "amount": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 200.0, "step": 1.0, "tooltip": "Strength of the whole adjustment"}),
        }
        req.update(_slider_inputs(LOCAL))
        req.update(
            {
                "prompt": ("STRING", {"default": "the sky", "tooltip": 'mask_kind = prompt: a noun phrase ("the red car"); needs the SAM 3 model installed in LightCraft'}),
                "luminance_lo": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "luminance_hi": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "radial: edge softness; luminance: range feather x100"}),
                "x0": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 2.0, "step": 0.01, "tooltip": "linear: start; radial: centre (normalized 0..1)"}),
                "y0": ("FLOAT", {"default": 0.25, "min": -1.0, "max": 2.0, "step": 0.01}),
                "x1": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 2.0, "step": 0.01, "tooltip": "linear: end; radial: radius x and y (normalized)"}),
                "y1": ("FLOAT", {"default": 0.6, "min": -1.0, "max": 2.0, "step": 0.01}),
                "extra_values": ("STRING", {"default": "", "multiline": True, "tooltip": 'Other local values: {"color_hue": 30, "color_sat": 40, "defringe": 20}'}),
            }
        )
        return {"required": req, "optional": {"mask": ("MASK", {"tooltip": "1 = adjust here (for mask_kind = input_mask)"})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Local relighting: exposure, shadows, highlights, warmth, clarity… inside a mask (a ComfyUI MASK, or LightCraft's sky / subject / background / luminance / gradient / text-prompt masks)."

    def run(self, image, mask_kind, invert, amount, prompt, luminance_lo, luminance_hi, feather, x0, y0, x1, y1, extra_values, mask=None, **sliders):
        values = {key: float(sliders[name]) for name, key, d, *_ in LOCAL if not is_default(sliders[name], d)}
        values.update(_extra(extra_values))
        if not is_default(amount, 100.0):
            values["amount"] = float(amount)
        if not values:
            raise LightCraftError("set at least one local slider (all are at their defaults)")
        imgs = to_numpy(image)
        if imgs.ndim == 3:
            imgs = imgs[None]
        masks = None
        if mask_kind == "input_mask":
            if mask is None:
                raise LightCraftError("mask_kind = input_mask needs the MASK input")
            masks = to_numpy(mask)
            if masks.ndim == 2:
                masks = masks[None]
            if masks.shape[0] not in (1, imgs.shape[0]):
                raise LightCraftError(f"{masks.shape[0]} masks for {imgs.shape[0]} images")

        def add_step(i: int) -> dict:
            if mask_kind == "input_mask":
                return {"kind": "object", "seg": seg_from_mask(masks[min(i, masks.shape[0] - 1)]), "name": "ComfyUI mask"}
            if mask_kind == "luminance":
                f = max(0.0, float(feather)) / 100.0
                return {"kind": "luminanceRange", "lo": float(luminance_lo), "hi": float(luminance_hi), "loFeather": f, "hiFeather": f}
            if mask_kind == "linear":
                return {"kind": "linear", "start": [float(x0), float(y0)], "end": [float(x1), float(y1)]}
            if mask_kind == "radial":
                return {"kind": "radial", "center": [float(x0), float(y0)], "rx": abs(float(x1)), "ry": abs(float(y1)), "feather": float(feather)}
            if mask_kind == "prompt":
                if not prompt.strip():
                    raise LightCraftError("mask_kind = prompt needs a prompt")
                return {"kind": "prompt", "text": prompt.strip()}
            return {"kind": mask_kind}

        outs = []
        from .lightcraft_cli import from_numpy

        for i in range(imgs.shape[0]):
            steps = [("mask.add", add_step(i))]
            if invert:
                steps.append(("mask.invert", {}))
            steps.append(("mask.adjust", {"values": values}))
            img, _ = develop(imgs[i : i + 1], steps)
            outs.append(to_numpy(img)[0])
        import numpy as np

        return (from_numpy(np.stack(outs)),)


class LightCraftPreset:
    """Apply a develop preset by id (LightCraft Preset List prints the ids)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": ("STRING", {"default": "", "tooltip": "A preset id from the LightCraft Preset List node"}),
                "amount": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "auto_tone": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Apply one of LightCraft's develop presets (built-in looks or your own)."

    def run(self, image, preset, amount, auto_tone):
        if not preset.strip():
            raise LightCraftError("choose a preset id (see LightCraft Preset List)")
        steps = []
        if auto_tone:
            steps.append(("develop.auto", {}))
        p = {"id": preset.strip()}
        if not is_default(amount, 100.0):
            p["amount"] = float(amount)
        steps.append(("preset.apply", p))
        img, _ = develop(image, steps)
        return (img,)


class LightCraftPresetList:
    """The preset ids and the develop control ids, as text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"what": (["presets", "controls", "profiles"], {"default": "presets"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "List LightCraft's presets, develop controls or profiles (ids to use in the other nodes)."

    def run(self, what):
        cmd = {"presets": "presets.list", "controls": "develop.controls", "profiles": "profiles.list"}[what]
        rs = query([(cmd, {})])
        if not rs or rs[-1].get("ok") is False:
            raise LightCraftError(f"{cmd}: {rs[-1].get('error') if rs else 'no answer from lightcraft-cli'}")
        r = rs[-1].get("result")
        items = r if isinstance(r, list) else (r.get(what) or r.get("items") or r) if isinstance(r, dict) else r
        if isinstance(items, list):
            lines = []
            for it in items:
                if isinstance(it, dict):
                    ident = it.get("id") or it.get("name")
                    label = it.get("name") or it.get("label") or ""
                    extra = it.get("group") or it.get("section") or ""
                    rng = f" [{it['min']}..{it['max']}]" if "min" in it and "max" in it else ""
                    lines.append(f"{ident}\t{label}{rng}\t{extra}".rstrip())
                else:
                    lines.append(str(it))
            text = "\n".join(lines)
        else:
            text = json.dumps(items, indent=2)
        return {"ui": {"text": [text]}, "result": (text,)}


class LightCraftCommand:
    """Any chain of engine commands on the image, as JSON lines."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "steps": (
                    "STRING",
                    {
                        "default": '{"command": "develop.auto"}\n{"command": "develop.set", "params": {"values": {"light.exposure": 0.3}}}',
                        "multiline": True,
                        "tooltip": 'One command per line: {"command": id, "params": {...}} (or a JSON list of them)',
                    },
                ),
                "long_edge": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "Export size; 0 = the full size"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "results")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Run any LightCraft engine commands (develop, masks, crop, presets…) on the image and export it."

    def run(self, image, steps, long_edge):
        parsed = []
        text = steps.strip()
        if text.startswith("["):
            try:
                items = json.loads(text)
            except json.JSONDecodeError as e:
                raise LightCraftError(f"steps: {e}") from e
        else:
            items = []
            for n, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise LightCraftError(f"steps line {n}: {e}") from e
        for it in items:
            if isinstance(it, str):
                parsed.append((it, {}))
            elif isinstance(it, dict) and isinstance(it.get("command") or it.get("id"), str):
                parsed.append((it.get("command") or it.get("id"), it.get("params") or {}))
            else:
                raise LightCraftError(f"bad step {it!r}")
        export = {"longEdge": int(long_edge)} if long_edge else None
        img, results = develop(image, parsed, export)
        return (img, json.dumps(results[0] if len(results) == 1 else results, indent=2))


NODE_CLASS_MAPPINGS = {
    "LightCraftDevelop": LightCraftDevelop,
    "LightCraftMaskedDevelop": LightCraftMaskedDevelop,
    "LightCraftPreset": LightCraftPreset,
    "LightCraftPresetList": LightCraftPresetList,
    "LightCraftCommand": LightCraftCommand,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LightCraftDevelop": "LightCraft Develop",
    "LightCraftMaskedDevelop": "LightCraft Masked Develop (relight)",
    "LightCraftPreset": "LightCraft Preset",
    "LightCraftPresetList": "LightCraft Preset List",
    "LightCraftCommand": "LightCraft Command",
}
