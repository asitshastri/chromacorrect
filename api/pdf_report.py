"""
api/pdf_report.py
-----------------
Generates a one-page professional PDF calibration report using fpdf2.

Report layout:
  - Header: title + timestamp + model version
  - Section 1: Original | Corrected image thumbnails side-by-side
  - Section 2: 24-patch table (patch name, measured L/a/b, ideal L/a/b, ΔE₀₀, pass/fail)
  - Section 3: Summary metrics

Requirements:
    pip install fpdf2 Pillow
"""

import io
import base64
from datetime import datetime

import numpy as np
from PIL import Image


# Standard Macbeth patch names in display order (top-left to bottom-right)
PATCH_NAMES = [
    "Dark Skin", "Light Skin", "Blue Sky", "Foliage",
    "Blue Flower", "Bluish Green", "Orange", "Purplish Blue",
    "Moderate Red", "Purple", "Yellow Green", "Orange Yellow",
    "Blue", "Green", "Red", "Yellow",
    "Magenta", "Cyan", "White 9.5", "Neutral 8",
    "Neutral 6.5", "Neutral 5", "Neutral 3.5", "Black 2",
]


def generate_pdf(
    original_img:    np.ndarray,
    corrected_img:   np.ndarray,
    measured_lab:    np.ndarray,
    corrected_lab:   np.ndarray,
    ideal_lab:       np.ndarray,
    de_per_patch:    np.ndarray,
    de_before_mean:  float,
    de_after_mean:   float,
    psnr_before:     float,
    psnr_after:      float,
    model_version:   str = "V2",
) -> bytes:
    """
    Generate a PDF calibration report and return it as bytes.

    Parameters
    ----------
    original_img    : (H, W, 3) float32 [0,1]  original image
    corrected_img   : (H, W, 3) float32 [0,1]  corrected image
    measured_lab    : (24, 3)   L*a*b* of camera measurements
    corrected_lab   : (24, 3)   L*a*b* after correction
    ideal_lab       : (24, 3)   L*a*b* ideal Macbeth targets
    de_per_patch    : (24,)     ΔE₀₀ per patch after correction
    de_before_mean  : float     mean ΔE₀₀ before
    de_after_mean   : float     mean ΔE₀₀ after
    psnr_before     : float     PSNR before (dB)
    psnr_after      : float     PSNR after (dB)
    model_version   : str       label for the model

    Returns
    -------
    pdf_bytes : bytes
    """
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(20, 20, 50)
    pdf.rect(0, 0, 210, 22, "F")
    pdf.set_text_color(255, 220, 50)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_xy(8, 5)
    pdf.cell(0, 10, "Colour Calibration Report", ln=False)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 180, 220)
    pdf.set_xy(130, 7)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 5, f"ISO 13655  |  D65  |  {ts}", ln=True)
    pdf.set_xy(130, 13)
    pdf.cell(0, 5, f"Model: ChromaCorrect {model_version}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # ── Section 1: Image thumbnails ───────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Image Comparison", ln=True)
    pdf.set_font("Helvetica", "", 8)

    thumb_w = 85   # mm
    thumb_h = 56   # mm

    def img_to_pdf_bytes(arr):
        """Convert float32 (H,W,3) array to JPEG bytes for embedding."""
        uint8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        pil   = Image.fromarray(uint8)
        buf   = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return buf

    x0 = 8
    y0 = pdf.get_y()

    orig_buf = img_to_pdf_bytes(original_img)
    corr_buf = img_to_pdf_bytes(corrected_img)

    pdf.image(orig_buf, x=x0,           y=y0, w=thumb_w, h=thumb_h)
    pdf.image(corr_buf, x=x0+thumb_w+4, y=y0, w=thumb_w, h=thumb_h)

    pdf.set_xy(x0, y0 + thumb_h + 1)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(thumb_w, 5, "Original",  align="C")
    pdf.cell(4, 5, "")
    pdf.cell(thumb_w, 5, "Corrected", align="C")
    pdf.ln(8)
    pdf.set_text_color(0, 0, 0)

    # ── Section 2: Summary metrics ────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Summary Metrics", ln=True)

    def metric_row(label, before, after, unit=""):
        pdf.set_font("Helvetica", "", 9)
        improvement = after - before
        arrow = "v" if improvement < 0 else "^"
        colour = (0, 140, 0) if improvement < 0 else (180, 0, 0)
        pdf.cell(60, 5, label)
        pdf.cell(30, 5, f"{before:.2f}{unit}")
        pdf.cell(30, 5, f"{after:.2f}{unit}")
        pdf.set_text_color(*colour)
        pdf.cell(40, 5, f"{arrow} {abs(improvement):.2f}{unit}")
        pdf.set_text_color(0, 0, 0)
        pdf.ln()

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(60, 5, "Metric")
    pdf.cell(30, 5, "Before")
    pdf.cell(30, 5, "After")
    pdf.cell(40, 5, "Change")
    pdf.ln()
    pdf.set_draw_color(180, 180, 180)
    pdf.line(8, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(1)

    metric_row("Mean dE2000",   de_before_mean, de_after_mean)
    metric_row("PSNR (dB)",    psnr_before,    psnr_after)

    pct_ok_before = 0.0   # before correction, assume 0% patches pass
    pct_ok_after  = float((de_per_patch < 2.0).sum()) / len(de_per_patch) * 100.0
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(60, 5, "Patches with dE2000 < 2")
    pdf.cell(30, 5, f"{pct_ok_before:.0f}%")
    pdf.cell(30, 5, f"{pct_ok_after:.1f}%")
    pdf.ln(8)

    # ── Section 3: Per-patch table ────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Per-Patch Calibration Data", ln=True)

    col_w = [8, 30, 15, 15, 15, 15, 15, 15, 15, 18]
    headers = ["#", "Patch Name", "Meas L*", "Meas a*", "Meas b*",
               "Ideal L*", "Ideal a*", "Ideal b*", "dE2000", "Status"]

    pdf.set_fill_color(40, 40, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 7)
    for i, (h, w) in enumerate(zip(headers, col_w)):
        pdf.cell(w, 5, h, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    for i in range(24):
        de = de_per_patch[i]
        passed = de < 2.0
        bg = (235, 255, 235) if passed else (255, 240, 240)
        pdf.set_fill_color(*bg)
        pdf.set_font("Helvetica", "", 7)
        row = [
            str(i + 1),
            PATCH_NAMES[i],
            f"{measured_lab[i, 0]:.1f}",
            f"{measured_lab[i, 1]:.1f}",
            f"{measured_lab[i, 2]:.1f}",
            f"{ideal_lab[i, 0]:.1f}",
            f"{ideal_lab[i, 1]:.1f}",
            f"{ideal_lab[i, 2]:.1f}",
            f"{de:.2f}",
            "PASS" if passed else "FAIL",
        ]
        for j, (val, w) in enumerate(zip(row, col_w)):
            if j == 8:
                colour = (0, 120, 0) if passed else (180, 0, 0)
                pdf.set_text_color(*colour)
            elif j == 9:
                colour = (0, 120, 0) if passed else (180, 0, 0)
                pdf.set_text_color(*colour)
            else:
                pdf.set_text_color(0, 0, 0)
            pdf.cell(w, 4.5, val, fill=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.ln()

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4,
             "dE2000 < 1: imperceptible  |  1-2: noticeable to experts  |  "
             "> 2: noticeable  |  Standard: ISO 13655, CIE Publication 15",
             ln=True)

    return bytes(pdf.output())
