"""
api/watermark.py
----------------
Stamps a corrected image with a colour calibration certification badge.

The badge is engraved at the bottom of the image and shows:
  - Title and ΔE₀₀ values
  - ISO 13655 / D65 Illuminant reference
  - A colour swatch strip: measured / corrected / ideal Macbeth patches

Uses Pillow (PIL) only — no external font files required.
"""

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def stamp_certification(
    img_array:      np.ndarray,
    de_before:      float,
    de_after:       float,
    measured_colors: np.ndarray,
    corrected_colors: np.ndarray,
    macbeth_ref:    np.ndarray,
    swatch_height:  int = 18,
    banner_height:  int = 56,
) -> np.ndarray:
    """
    Add a certification badge to the bottom of the corrected image.

    Parameters
    ----------
    img_array        : (H, W, 3) float32 [0,1]  RGB corrected image
    de_before        : mean ΔE₀₀ before correction
    de_after         : mean ΔE₀₀ after correction
    measured_colors  : (24, 3) float32  original camera measurements
    corrected_colors : (24, 3) float32  after matrix correction
    macbeth_ref      : (24, 3) float32  ideal Macbeth target
    swatch_height    : height in px of each patch swatch row
    banner_height    : height in px of the text banner

    Returns
    -------
    stamped : (H + banner_height + 3*swatch_height, W, 3) uint8 [0,255]
    """
    H, W, _ = img_array.shape
    img_uint8 = (np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8)

    # ── Build patch swatch strip ──────────────────────────────────────────────
    # Each of 24 patches shown in 3 rows: measured | corrected | ideal
    sw = W // 24   # width of each swatch cell
    strip_h = swatch_height * 3
    strip = np.zeros((strip_h, W, 3), dtype=np.uint8)

    for i in range(24):
        x0 = i * sw
        x1 = x0 + sw
        strip[0          :swatch_height,   x0:x1] = (measured_colors[i]  * 255).astype(np.uint8)
        strip[swatch_height:2*swatch_height, x0:x1] = (corrected_colors[i] * 255).astype(np.uint8)
        strip[2*swatch_height:3*swatch_height, x0:x1] = (macbeth_ref[i] * 255).astype(np.uint8)

    # ── Build text banner ─────────────────────────────────────────────────────
    banner = Image.new("RGB", (W, banner_height), color=(20, 20, 20))
    draw   = ImageDraw.Draw(banner)

    # Use default PIL font (no external files needed)
    try:
        font_big   = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except (IOError, OSError):
        font_big   = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Row 1: title
    draw.text((8, 4), "Colour Calibration Certified", fill=(255, 220, 50),
              font=font_big)

    # Row 2: metrics
    improvement = de_before - de_after
    de_label = f"DE2000 Before: {de_before:.2f}   After: {de_after:.2f}   Improvement: {improvement:+.2f}"
    draw.text((8, 22), de_label, fill=(200, 255, 200), font=font_small)

    # Row 3: standard reference
    draw.text((8, 38), "ISO 13655  |  Illuminant D65  |  CIE 1931 2 deg Observer",
              fill=(180, 180, 180), font=font_small)

    # Swatch labels on right side of banner
    label_x = W - 120
    draw.text((label_x, 4),  "Measured",  fill=(100, 180, 255), font=font_small)
    draw.text((label_x, 18), "Corrected", fill=(100, 255, 130), font=font_small)
    draw.text((label_x, 32), "Ideal",     fill=(255, 180, 80),  font=font_small)

    banner_np = np.array(banner, dtype=np.uint8)

    # ── Stack image + banner + swatches ──────────────────────────────────────
    stamped = np.vstack([img_uint8, banner_np, strip])
    return stamped
