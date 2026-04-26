"""
api/color_utils.py
------------------
Standalone colour space utilities for the webapp.
NumPy only — no PyTorch dependency.

Provides:
  - sRGB → Linear RGB → XYZ → CIELAB conversion
  - CIE76 ΔE and CIEDE2000 (ΔE₀₀) colour difference
  - Macbeth ColorChecker reference values
  - _color.txt parsing (NUS dataset sensor measurements)
  - sRGB pixel parsing (from in-browser patch sampling)
"""

import numpy as np


# ── Macbeth ColorChecker ideal sRGB values (24 patches, standard D50) ────────
# Row order: top row (dark skin → white), bottom row (black → neutral 3.5)
MACBETH_SRGB = np.array([
    [115,  82,  68], [194, 150, 130], [ 98, 122, 157], [ 87, 108,  67],
    [133, 128, 177], [103, 189, 170], [214, 126,  44], [ 80,  91, 166],
    [193,  90,  99], [ 94,  60, 108], [157, 188,  64], [224, 163,  46],
    [ 56,  61, 150], [ 70, 148,  73], [175,  54,  60], [231, 199,  31],
    [187,  86, 149], [  8, 133, 161], [243, 243, 242], [200, 200, 200],
    [160, 160, 160], [122, 122, 121], [ 85,  85,  85], [ 52,  52,  52],
], dtype=np.uint8)


# Reorder index: _color.txt stores patches bottom-row-first (neutrals first),
# but MACBETH_SRGB above is top-row-first.
_COLOR_TXT_TO_MACBETH = [18, 19, 20, 21, 22, 23,
                          12, 13, 14, 15, 16, 17,
                           6,  7,  8,  9, 10, 11,
                           0,  1,  2,  3,  4,  5]


def parse_color_txt(color_txt_content: str,
                    dark_level: float = 0.0,
                    sat_level:  float = 65535.0) -> np.ndarray:
    """
    Parse a NUS _color.txt file and return normalised sRGB patch colours.

    Each line of _color.txt contains:
        R G B   (raw sensor counts, one line per patch, 24 lines total)

    Normalisation pipeline:
        1. Subtract dark level (sensor black point)
        2. Divide by sensor range (sat - dark)
        3. Global max normalisation (preserves illuminant colour cast)
        4. Gamma correction (^1/2.2) to convert to sRGB

    Parameters
    ----------
    color_txt_content : str   — full text content of the _color.txt file
    dark_level        : float — sensor darkness level (from _gt.mat, default 0)
    sat_level         : float — sensor saturation level (default 65535)

    Returns
    -------
    patches : np.ndarray  (24, 3)  float32  sRGB in [0, 1], Macbeth order
    """
    lines = [l.strip() for l in color_txt_content.strip().splitlines()
             if l.strip() and not l.startswith('#')]
    raw = np.array([list(map(float, l.split())) for l in lines], dtype=np.float32)
    if raw.shape[0] < 24:
        raise ValueError(f"Expected 24 patches in _color.txt, got {raw.shape[0]}")
    raw = raw[:24]   # use first 24 rows

    dark_sub     = np.maximum(raw - dark_level, 0.0)
    sensor_range = max(sat_level - dark_level, 1.0)
    linear       = dark_sub / sensor_range

    # Global max: divide by the single largest value across all 24 patches × 3 channels.
    # Preserves inter-channel ratios (illuminant color cast).
    # Per-channel normalization was wrong — it erased the warm/cool cast signal.
    global_max = np.maximum(linear.max(), 1e-8)
    linear     = linear / global_max

    srgb = np.power(np.clip(linear, 1e-8, 1.0), 1.0 / 2.2).astype(np.float32)
    srgb = srgb[_COLOR_TXT_TO_MACBETH]
    return srgb


def parse_srgb_txt(content: str) -> np.ndarray:
    """
    Parse 24 lines of 'R G B' (0–255 sRGB) from in-browser pixel sampling.

    Used when the user clicks 4 corners of the colour card in a JPEG photo.
    Browser JPEG pixels are already gamma-encoded sRGB — no dark-level
    subtraction, no sat-range division, no gamma correction needed.

    Parameters
    ----------
    content : str  — 24 lines, each containing 'R G B' (0–255 integers)

    Returns
    -------
    patches : np.ndarray  (24, 3)  float32  [0, 1], Macbeth standard order
              (top-left to bottom-right, row by row — same as MACBETH_SRGB)
    """
    lines = [l.strip() for l in content.splitlines()
             if l.strip() and not l.startswith('#')]
    if len(lines) < 24:
        raise ValueError(f"Expected 24 patch lines, got {len(lines)}")
    vals = []
    for line in lines[:24]:
        parts = line.split()
        vals.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.clip(np.array(vals, dtype=np.float32) / 255.0, 0.0, 1.0)


# ── Colour space conversion ───────────────────────────────────────────────────

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] → CIELAB.  Input: (..., 3). Output: (..., 3)."""
    rgb = np.clip(rgb, 0.0, 1.0)
    # sRGB → Linear
    linear = np.where(rgb <= 0.04045, rgb / 12.92,
                      ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)
    # Linear → XYZ D65
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz  = linear @ M.T
    # XYZ → LAB
    D65  = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    xyz_n = xyz / D65
    d  = 6.0 / 29.0
    f  = np.where(xyz_n > d**3, np.cbrt(xyz_n),
                  xyz_n / (3.0 * d**2) + 4.0 / 29.0)
    L  = 116.0 * f[..., 1] - 16.0
    a  = 500.0 * (f[..., 0] - f[..., 1])
    b  = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 ΔE — Euclidean distance in LAB. (..., 3) → (...)."""
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1)).astype(np.float32)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    CIEDE2000 (ΔE₀₀) — most perceptually accurate colour difference.
    (..., 3) → (...)
    """
    eps = 1e-8
    L1 = lab1[..., 0]; a1 = lab1[..., 1]; b1 = lab1[..., 2]
    L2 = lab2[..., 0]; a2 = lab2[..., 1]; b2 = lab2[..., 2]

    C1 = np.sqrt(a1**2 + b1**2 + eps)
    C2 = np.sqrt(a2**2 + b2**2 + eps)
    Cb7 = ((C1 + C2) / 2.0) ** 7
    G   = 0.5 * (1.0 - np.sqrt(Cb7 / (Cb7 + 25.0**7)))

    a1p = a1 * (1.0 + G);  a2p = a2 * (1.0 + G)
    C1p = np.sqrt(a1p**2 + b1**2 + eps)
    C2p = np.sqrt(a2p**2 + b2**2 + eps)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p
    hd  = h2p - h1p
    ach = (C1p * C2p) < eps
    dhp = np.where(ach, 0.0,
          np.where(np.abs(hd) <= 180.0, hd,
          np.where(hd > 180.0, hd - 360.0, hd + 360.0)))
    dHp = 2.0 * np.sqrt(C1p * C2p + eps) * np.sin(np.radians(dhp / 2.0))

    Lb  = (L1 + L2) / 2.0
    Cb  = (C1p + C2p) / 2.0
    hs  = h1p + h2p
    hb  = np.where(ach, hs,
          np.where(np.abs(h1p - h2p) <= 180.0, hs / 2.0,
          np.where(hs < 360.0, (hs + 360.0) / 2.0, (hs - 360.0) / 2.0)))

    T   = (1.0 - 0.17*np.cos(np.radians(hb-30.0))
               + 0.24*np.cos(np.radians(2.0*hb))
               + 0.32*np.cos(np.radians(3.0*hb+6.0))
               - 0.20*np.cos(np.radians(4.0*hb-63.0)))
    L50 = (Lb - 50.0)**2
    SL  = 1.0 + 0.015 * L50 / np.sqrt(20.0 + L50 + eps)
    SC  = 1.0 + 0.045 * Cb
    SH  = 1.0 + 0.015 * Cb * T

    Cb7 = Cb**7
    RC  = 2.0 * np.sqrt(Cb7 / (Cb7 + 25.0**7 + eps))
    dt  = 30.0 * np.exp(-((hb - 275.0) / 25.0)**2)
    RT  = -np.sin(np.radians(2.0 * dt)) * RC

    out = np.sqrt(np.maximum(
        (dLp/SL)**2 + (dCp/SC)**2 + (dHp/SH)**2 + RT*(dCp/SC)*(dHp/SH),
        0.0) + eps)
    return out.astype(np.float32)
