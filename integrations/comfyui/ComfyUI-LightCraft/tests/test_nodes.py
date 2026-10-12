"""End-to-end tests of the nodes against a built lightcraft-cli (numpy only, no torch).

    python3 -m unittest discover -s integrations/comfyui/ComfyUI-LightCraft/tests
Set LIGHTCRAFT_CLI when the binary is not at target/release/lightcraft-cli.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parents[1]
REPO = PKG_DIR.parents[2]
os.environ.setdefault("LIGHTCRAFT_CLI", str(REPO / "target" / "release" / ("lightcraft-cli.exe" if sys.platform.startswith("win") else "lightcraft-cli")))

nodes = cli = None


def _load():
    spec = importlib.util.spec_from_file_location("lightcraft_nodes", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lightcraft_nodes"] = mod
    spec.loader.exec_module(mod)
    return sys.modules["lightcraft_nodes.nodes"], sys.modules["lightcraft_nodes.lightcraft_cli"]


try:
    nodes, cli = _load()
    HAVE_CLI = Path(cli.find_cli()).is_file()
except Exception:  # noqa: BLE001
    HAVE_CLI = False


def scene(w=160, h=120):
    """Dark ground, bright sky, a mid-grey disc: regions with distinct luminance."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.empty((h, w, 3), np.float32)
    img[:] = (0.15, 0.12, 0.10)
    img[: h // 2] = (0.55, 0.65, 0.85)
    disc = (xx - w / 2) ** 2 + (yy - 3 * h / 4) ** 2 <= (h / 6) ** 2
    img[disc] = (0.45, 0.45, 0.45)
    rng = np.random.default_rng(3)
    img += rng.normal(0, 0.004, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)[None], disc


def lum(a):
    return float(np.mean(a[..., :3]))


class SegTests(unittest.TestCase):
    def test_seg_round_trip(self):
        if cli is None:
            self.skipTest("package failed to import")
        m = np.zeros((60, 80), np.float32)
        m[10:40, 20:70] = 1.0
        seg = cli.seg_from_mask(m, side=64)
        self.assertEqual(seg["side"], 64)
        back = cli.mask_from_seg(seg)
        self.assertEqual(back.shape, (64, 64))
        self.assertGreater(back[32, 32], 0.99)   # inside
        self.assertLess(back[2, 2], 0.01)        # outside
        self.assertLess(abs(back.mean() - m.mean()), 0.03)


@unittest.skipUnless(HAVE_CLI, "lightcraft-cli not built (set LIGHTCRAFT_CLI)")
class NodeTests(unittest.TestCase):
    def test_develop_exposure_brightens(self):
        img, _ = scene()
        (out,) = nodes.LightCraftDevelop().run(img, False, "", **{n: d for n, _, d, *_ in nodes.GLOBAL} | {"exposure": 1.0})
        self.assertEqual(out.shape, img.shape)
        self.assertGreater(lum(out), lum(img) + 0.1)
        (same,) = nodes.LightCraftDevelop().run(img, False, "", **{n: d for n, _, d, *_ in nodes.GLOBAL})
        self.assertLess(abs(lum(same) - lum(img)), 0.03)

    def test_develop_extra_controls_and_auto(self):
        img, _ = scene()
        (out,) = nodes.LightCraftDevelop().run(img, True, '{"color.saturation": -100}', **{n: d for n, _, d, *_ in nodes.GLOBAL})
        self.assertEqual(out.shape, img.shape)
        o = np.asarray(out)[0]
        self.assertLess(float(np.std(o, axis=2).mean()), 0.01, "desaturated")
        with self.assertRaises(cli.LightCraftError):
            nodes.LightCraftDevelop().run(img, False, "not json", **{n: d for n, _, d, *_ in nodes.GLOBAL})

    def _masked(self, img, kind, mask=None, **kw):
        base = {n: d for n, _, d, *_ in nodes.LOCAL}
        base.update(kw.pop("sliders", {}))
        args = dict(mask_kind=kind, invert=False, amount=100.0, prompt="the sky", luminance_lo=0.6, luminance_hi=1.0, feather=50.0, x0=0.5, y0=0.25, x1=0.5, y1=0.6, extra_values="")
        args.update(kw)
        (out,) = nodes.LightCraftMaskedDevelop().run(img, mask=mask, **args, **base)
        return np.asarray(out)[0]

    def test_masked_develop_with_input_mask(self):
        img, disc = scene()
        m = disc.astype(np.float32)[None]
        out = self._masked(img, "input_mask", mask=m, sliders={"exposure": 2.0})
        src = img[0]
        inside = lum(out[disc]) - lum(src[disc])
        outside = lum(out[~disc]) - lum(src[~disc])
        self.assertGreater(inside, 0.2, f"inside brightened by {inside:.3f}")
        self.assertLess(abs(outside), 0.03, f"outside changed by {outside:.3f}")
        # inverted: the rest brightens, the disc stays
        out = self._masked(img, "input_mask", mask=m, invert=True, sliders={"exposure": 2.0})
        self.assertLess(abs(lum(out[disc]) - lum(src[disc])), 0.03)
        self.assertGreater(lum(out[~disc]) - lum(src[~disc]), 0.1)

    def test_masked_develop_luminance_and_gradients(self):
        img, _ = scene()
        src = img[0]
        h = src.shape[0]
        out = self._masked(img, "luminance", luminance_lo=0.5, luminance_hi=1.0, feather=5.0, sliders={"exposure": -2.0})
        self.assertLess(lum(out[: h // 2]) - lum(src[: h // 2]), -0.15, "bright sky darkened")
        self.assertLess(abs(lum(out[h // 2 + 10 :, :10]) - lum(src[h // 2 + 10 :, :10])), 0.03, "dark ground untouched")
        out = self._masked(img, "linear", x0=0.5, y0=0.1, x1=0.5, y1=0.4, sliders={"exposure": -2.0})
        self.assertLess(lum(out[:10]) - lum(src[:10]), -0.15)
        out = self._masked(img, "radial", x0=0.5, y0=0.75, x1=0.2, y1=0.2, feather=20.0, sliders={"exposure": 2.0})
        self.assertGreater(lum(out[3 * h // 4, 80]) - lum(src[3 * h // 4, 80]), 0.1)
        self.assertLess(abs(lum(out[5, 5]) - lum(src[5, 5])), 0.03)
        out = self._masked(img, "sky", sliders={"saturation": -100})
        self.assertEqual(out.shape, src.shape)

    def test_masked_develop_errors(self):
        img, _ = scene()
        with self.assertRaises(cli.LightCraftError):
            self._masked(img, "input_mask", mask=None, sliders={"exposure": 1.0})
        with self.assertRaises(cli.LightCraftError):
            self._masked(img, "sky")  # no slider set

    def test_preset_list_and_apply(self):
        text, = nodes.LightCraftPresetList().run("presets")["result"]
        self.assertTrue(text.strip(), "presets listed")
        first = text.strip().splitlines()[0].split("\t")[0]
        img, _ = scene()
        (out,) = nodes.LightCraftPreset().run(img, first, 100.0, False)
        self.assertEqual(out.shape, img.shape)
        with self.assertRaises(cli.LightCraftError):
            nodes.LightCraftPreset().run(img, "no-such-preset", 100.0, False)
        controls, = nodes.LightCraftPresetList().run("controls")["result"]
        self.assertIn("light.exposure", controls)

    def test_command_node_and_batch(self):
        img, _ = scene()
        batch = np.concatenate([img, img[:, ::-1]], axis=0)
        out, results = nodes.LightCraftCommand().run(batch, '{"command": "develop.set", "params": {"values": {"light.exposure": 1}}}', 80)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(max(out.shape[1], out.shape[2]), 80)
        self.assertIn("develop.set", results)
        with self.assertRaises(cli.LightCraftError):
            nodes.LightCraftCommand().run(img, '{"command": "no.such.command"}', 0)


if __name__ == "__main__":
    unittest.main()
