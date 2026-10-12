# LightCraft for ComfyUI

`ComfyUI-LightCraft/` is a ComfyUI custom-node package that develops images with LightCraft
through the headless command-line tool `lightcraft-cli`: Lightroom-class tone and colour,
globally or inside a mask. Each node writes its input to a PNG, runs one headless session
(`lightcraft-cli run --import in.png --script steps.jsonl`, one engine command per line, ending
in an `app.export`) and reads the PNG back. ComfyUI is Python, so the nodes are Python; LightCraft
stays Rust.

## Nodes (category **LightCraft**)

| Node | Does |
|---|---|
| LightCraft Develop | Global sliders: exposure, contrast, highlights, shadows, whites, blacks, white balance (Kelvin, tint), vibrance, saturation, texture, clarity, dehaze, vignette, grain, sharpening, noise reduction; optional Auto Tone first; `extra_controls` JSON for any other control (`lightcraft-cli controls` lists the 86) |
| LightCraft Masked Develop (relight) | Local adjustments inside a mask: exposure, contrast, highlights, shadows, whites, blacks, temperature, tint, texture, clarity, dehaze, hue, saturation, sharpness, noise, amount. The mask is a ComfyUI `MASK` input (any segmentation node), or LightCraft's own: `sky`, `subject`, `background` (heuristic, no model), `luminance` range, `linear` / `radial` gradients, or a text `prompt` (needs the SAM 3 model, see below). `invert` flips it |
| LightCraft Preset | Apply a develop preset by id, with an amount |
| LightCraft Preset List | Prints the preset ids, the develop control ids with ranges, or the profiles |
| LightCraft Command | Any chain of engine commands as JSON lines (develop, masks, crop, presets…), with an optional export size |

"Relighting" here means what a raw developer does: brightening or darkening a region, lifting
shadows, warming or cooling it, adding clarity or dehaze, inside a precise mask. It does not move
a light source or add new shadows (that is a generative task). Combine with a ComfyUI
segmentation or depth node for the mask and this node for the look.

**How a ComfyUI mask reaches LightCraft.** LightCraft stores an AI mask as a square grid of
logits (int8, deflated, base64). `lightcraft_cli.seg_from_mask` builds that from a ComfyUI mask,
and `mask.add {kind: "object", seg}` accepts it with no model installed. The grid is 512 x 512
over the whole image, bilinearly sampled, so edges stay soft at any size.

Images travel as 8-bit sRGB PNG; the output is the same size as the input (crop and geometry
controls would change it). Batches are processed one image at a time.

## Install

1. Build the CLI (Rust 1.95+): `cargo build --release -p lightcraft-cli` → `target/release/lightcraft-cli[.exe]`.
   Text-prompt masks need the engine's `sam` feature and the SAM 3 weights (a download the app
   offers after showing the licence, or `segment.model.download`): `cargo build --release -p
   lightcraft-cli --features lightcraft-engine/sam`. Everything else works without it.
2. Copy `ComfyUI-LightCraft/` into `ComfyUI/custom_nodes/`.
3. Tell the nodes where the binary is: the `LIGHTCRAFT_CLI` environment variable, or copy it to
   `ComfyUI/custom_nodes/ComfyUI-LightCraft/bin/`, or put it on the `PATH`.
4. Restart ComfyUI. The nodes need only `numpy` and `Pillow`, which ComfyUI already has.

### Windows (PowerShell)

```powershell
cd C:\src\ale_lightcraft
cargo build --release -p lightcraft-cli

$comfy = "C:\ComfyUI"
Copy-Item -Recurse -Force integrations\comfyui\ComfyUI-LightCraft "$comfy\custom_nodes\ComfyUI-LightCraft"
New-Item -ItemType Directory -Force "$comfy\custom_nodes\ComfyUI-LightCraft\bin" | Out-Null
Copy-Item target\release\lightcraft-cli.exe "$comfy\custom_nodes\ComfyUI-LightCraft\bin\"
```

## Tests

```sh
cargo build --release -p lightcraft-cli
python3 -m unittest discover -s integrations/comfyui/ComfyUI-LightCraft/tests
```

numpy only (no torch, no ComfyUI): the tests develop a synthetic scene and check that exposure
brightens, that a ComfyUI mask brightens only its region (and the inverse with `invert`), that
luminance, linear and radial masks touch the right parts, that presets list and apply, and that
bad input is an error rather than a crash.
