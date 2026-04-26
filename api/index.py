"""
api/index.py
------------
FastAPI backend for the ChromaCorrect colour calibration webapp.

Endpoints:
    POST /api/correct       — calibrate an image using its _color.txt
    GET  /api/download/{id}/{file}  — download corrected image or PDF
    GET  /                  — serve the frontend (Vercel handles this)

How inference works:
    1. Parse the uploaded _color.txt to get 24 measured patch RGB values
    2. Build solid-colour patch tiles (64×64 per patch)
    3a. Analytical path: solve least-squares CCM in linear-light space
    3b. Neural path: run ONNX model → correction matrix (+ optional bias)
    3c. Auto-select whichever path gives lower dE2000 on calibration patches
    4. Apply correction to every pixel of the full image
    5. Watermark the image with a dE2000 certification badge
    6. Generate a PDF calibration report
    7. Return base64-encoded corrected image + download URLs
"""

import base64
import io
import math
import os
import tempfile
import time
import uuid

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from color_utils import (
    MACBETH_SRGB,
    delta_e_2000,
    parse_color_txt,
    parse_srgb_txt,
    rgb_to_lab,
)
from pdf_report import generate_pdf
from watermark import stamp_certification

# ── ONNX model session (loaded once at startup) ───────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")
_ort_session = None
_model_has_bias = False   # True for V2 (matrix + bias), False for V1 (matrix only)
_model_is_poly  = False   # True when matrix is (9,3) polynomial, False for (3,3) affine


def _poly_expand(colors: np.ndarray) -> np.ndarray:
    """Expand (..., 3) RGB to (..., 9) polynomial features: [R,G,B,RG,RB,GB,R²,G²,B²]."""
    R, G, B = colors[..., 0:1], colors[..., 1:2], colors[..., 2:3]
    return np.concatenate([R, G, B, R*G, R*B, G*B, R*R, G*G, B*B], axis=-1)


def _srgb_to_linear(s: np.ndarray) -> np.ndarray:
    """sRGB gamma → linear light."""
    return np.where(s <= 0.04045, s / 12.92,
                    ((s + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    """Linear light → sRGB gamma."""
    return np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * lin ** (1.0 / 2.4) - 0.055).astype(np.float32)


def _compute_analytical_ccm(measured_colors: np.ndarray) -> np.ndarray:
    """
    Compute the least-squares optimal 3×3 CCM in linear-light space.

    Solves: linear(measured) @ M  ≈  linear(MACBETH_SRGB)
    This is the mathematically best affine correction for these 24 patches.
    Returns M (3,3) — apply as:  linear(img) @ M  then re-encode.
    """
    ideal = MACBETH_SRGB.astype(np.float32) / 255.0
    meas_lin  = _srgb_to_linear(np.clip(measured_colors, 0.0, 1.0))
    ideal_lin = _srgb_to_linear(ideal)
    M, _, _, _ = np.linalg.lstsq(meas_lin, ideal_lin, rcond=None)
    return M.astype(np.float32)   # (3, 3)


def _apply_linear_ccm(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a linear-space 3×3 CCM to an sRGB float32 image (H,W,3)→(H,W,3)."""
    H, W = img.shape[:2]
    lin  = _srgb_to_linear(img.reshape(-1, 3))
    corr = np.clip(lin @ M, 0.0, 1.0)
    return _linear_to_srgb(corr).reshape(H, W, 3)


def _get_session():
    global _ort_session, _model_has_bias, _model_is_poly
    if _ort_session is None:
        import onnxruntime as ort
        _ort_session = ort.InferenceSession(
            _MODEL_PATH, providers=["CPUExecutionProvider"]
        )
        output_names = [o.name for o in _ort_session.get_outputs()]
        _model_has_bias = "bias" in output_names
        # Detect poly: run a dummy pass and check matrix output shape
        dummy_p = np.zeros((1, 24, 3, 64, 64), dtype=np.float32)
        dummy_i = np.zeros((1, 3, 256, 256),   dtype=np.float32)
        dummy_out = _ort_session.run(None, {"patches": dummy_p, "image": dummy_i})
        _model_is_poly = dummy_out[0].shape[1] == 9   # (1,9,3) → poly, (1,3,3) → affine
    return _ort_session


# ── Temporary file store: session_id → {"corrected": path, "report": path} ───
_sessions: dict = {}


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ChromaCorrect", version="1.0.0")

# Serve the static frontend
_static_dir = os.path.join(os.path.dirname(__file__), "..", "public")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the frontend HTML file."""
    index_path = os.path.join(_static_dir, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _colors_to_patches(colors: np.ndarray, size: int = 64) -> np.ndarray:
    """
    Convert (24, 3) float32 colour values to (24, 3, H, W) ONNX-ready array.
    Each patch is a solid-colour tile of shape (3, size, size).
    """
    patches = np.zeros((24, 3, size, size), dtype=np.float32)
    for i in range(24):
        for c in range(3):
            patches[i, c, :, :] = colors[i, c]
    return patches   # (24, 3, 64, 64)


def _load_image_bytes(data: bytes, target_size: int = 256):
    """
    Load image bytes → float32 (H, W, 3) full-res + float32 (1, 3, S, S) model input.
    Returns (full_rgb, model_tensor_np).
    """
    from PIL import Image
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    full_rgb = np.array(pil, dtype=np.float32) / 255.0   # (H, W, 3)

    # Resize for model input
    pil_small = pil.resize((target_size, target_size), Image.BILINEAR)
    small_rgb = np.array(pil_small, dtype=np.float32) / 255.0   # (S, S, 3)
    model_input = small_rgb.transpose(2, 0, 1)[np.newaxis, ...]   # (1, 3, S, S)
    return full_rgb, model_input


def _apply_correction(img: np.ndarray, matrix: np.ndarray,
                      bias: np.ndarray = None) -> np.ndarray:
    """Apply affine or polynomial colour correction to a full image (H,W,3)→(H,W,3)."""
    H, W = img.shape[:2]
    flat = img.reshape(-1, 3)
    if _model_is_poly:
        # matrix is (9,3): expand to 9 features then multiply
        corrected = _poly_expand(flat) @ matrix   # (H*W, 9) @ (9, 3) = (H*W, 3)
    else:
        corrected = flat @ matrix.T               # (H*W, 3) @ (3, 3) = (H*W, 3)
    if bias is not None:
        corrected = corrected + bias
    return np.clip(corrected.reshape(H, W, 3), 0.0, 1.0).astype(np.float32)


def _compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = ((pred - target) ** 2).mean()
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


# ── Main calibration endpoint ─────────────────────────────────────────────────

@app.post("/api/correct")
async def correct_image(
    image:          UploadFile = File(...,  description="Scene photo (JPEG/PNG)"),
    color_txt:      UploadFile = File(...,  description="24 patch measurements (_color.txt or browser-sampled R G B lines)"),
    dark_level:     float      = Form(0.0,   description="Sensor darkness level (ignored when is_srgb_input=true)"),
    sat_level:      float      = Form(65535.0, description="Sensor saturation level (ignored when is_srgb_input=true)"),
    is_srgb_input:  bool       = Form(False, description="True when color_txt contains sRGB 0-255 values from browser pixel sampling"),
):
    """
    Calibrate a photo using its Macbeth ColorChecker measurements.

    Upload the scene image and the corresponding _color.txt file.
    Returns the corrected image (base64), ΔE₀₀ metrics, and download URLs.
    """
    t0 = time.time()

    # ── Parse colour measurements ─────────────────────────────────────────────
    color_content = (await color_txt.read()).decode("utf-8", errors="replace")
    try:
        if is_srgb_input:
            # Values came from in-browser pixel sampling — already sRGB 0-255,
            # Macbeth row order (top-left → bottom-right). No raw-sensor processing.
            measured_colors = parse_srgb_txt(color_content)
        else:
            # Classic NUS _color.txt: 16-bit raw sensor values, bottom-row-first.
            measured_colors = parse_color_txt(color_content, dark_level, sat_level)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse colour measurements: {e}")
    # (24, 3) float32 [0,1]

    # ── Load image ────────────────────────────────────────────────────────────
    img_bytes = await image.read()
    try:
        full_rgb, img_input = _load_image_bytes(img_bytes, target_size=256)
    except Exception as e:
        raise HTTPException(400, f"Failed to load image: {e}")

    # ── Build patch tiles for ONNX ────────────────────────────────────────────
    patches_np = _colors_to_patches(measured_colors, size=64)  # (24, 3, 64, 64)
    patches_input = patches_np[np.newaxis, ...]                # (1, 24, 3, 64, 64)

    # ── Analytical CCM (always computed — optimal least-squares solution) ────────
    analytical_M = _compute_analytical_ccm(measured_colors)   # (3, 3)
    meas_lin     = _srgb_to_linear(measured_colors)           # (24, 3)
    corr_lin_patches = np.clip(meas_lin @ analytical_M, 0.0, 1.0)
    analytical_colors = _linear_to_srgb(corr_lin_patches)    # (24, 3) sRGB

    # ── Run ONNX model ────────────────────────────────────────────────────────
    sess = _get_session()
    try:
        outputs = sess.run(None, {"patches": patches_input, "image": img_input})
    except Exception as e:
        raise HTTPException(500, f"Model inference failed: {e}")

    matrix = outputs[0][0]   # (3, 3) or (9, 3)
    bias   = outputs[1][0] if _model_has_bias and len(outputs) > 1 else None

    if _model_is_poly:
        model_colors = _poly_expand(measured_colors) @ matrix
    else:
        model_colors = measured_colors @ matrix.T
    if bias is not None:
        model_colors = model_colors + bias
    model_colors = np.clip(model_colors, 0.0, 1.0).astype(np.float32)

    # ── Pick whichever method gives lower dE2000 on the 24 patches ────────────
    macbeth_f32 = MACBETH_SRGB.astype(np.float32) / 255.0
    ideal_lab   = rgb_to_lab(macbeth_f32)

    de_analytical = float(delta_e_2000(rgb_to_lab(analytical_colors), ideal_lab).mean())
    de_model      = float(delta_e_2000(rgb_to_lab(model_colors),      ideal_lab).mean())

    if de_analytical <= de_model:
        corrected_colors = analytical_colors
        corrected_full   = _apply_linear_ccm(full_rgb, analytical_M)
        method_used      = "analytical-ccm"
    else:
        corrected_colors = model_colors
        corrected_full   = _apply_correction(full_rgb, matrix, bias)
        method_used      = "neural-network"

    # ── LAB conversions ───────────────────────────────────────────────────────
    measured_lab  = rgb_to_lab(measured_colors)
    corrected_lab = rgb_to_lab(corrected_colors)

    de_before_patch = delta_e_2000(measured_lab,  ideal_lab)   # (24,)
    de_after_patch  = delta_e_2000(corrected_lab, ideal_lab)   # (24,)
    de_before_mean  = float(de_before_patch.mean())
    de_after_mean   = float(de_after_patch.mean())

    # PSNR (using measured vs ideal as proxy for full-image quality)
    psnr_before = _compute_psnr(measured_colors,  macbeth_f32)
    psnr_after  = _compute_psnr(corrected_colors, macbeth_f32)

    # ── Watermark corrected image ─────────────────────────────────────────────
    from watermark import stamp_certification
    stamped = stamp_certification(
        corrected_full, de_before_mean, de_after_mean,
        measured_colors, corrected_colors, macbeth_f32
    )   # (H+banner+strip, W, 3) uint8

    # ── Encode corrected image as base64 JPEG ─────────────────────────────────
    from PIL import Image as PILImage
    stamped_pil = PILImage.fromarray(stamped)
    img_buf = io.BytesIO()
    stamped_pil.save(img_buf, format="JPEG", quality=90)
    corrected_b64 = base64.b64encode(img_buf.getvalue()).decode("ascii")

    # ── Generate PDF report ───────────────────────────────────────────────────
    pdf_bytes = generate_pdf(
        original_img   = full_rgb,
        corrected_img  = corrected_full,
        measured_lab   = measured_lab,
        corrected_lab  = corrected_lab,
        ideal_lab      = ideal_lab,
        de_per_patch   = de_after_patch,
        de_before_mean = de_before_mean,
        de_after_mean  = de_after_mean,
        psnr_before    = psnr_before,
        psnr_after     = psnr_after,
        model_version  = "V2" if _model_has_bias else "V1",
    )

    # ── Save files to temp dir for download ──────────────────────────────────
    session_id = str(uuid.uuid4())[:8]
    tmp_dir    = tempfile.mkdtemp(prefix="chromacorrect_")

    corr_path = os.path.join(tmp_dir, "corrected.jpg")
    pdf_path  = os.path.join(tmp_dir, "report.pdf")

    stamped_pil.save(corr_path, format="JPEG", quality=90)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    _sessions[session_id] = {"corrected": corr_path, "report": pdf_path}

    # ── Build per-patch response ──────────────────────────────────────────────
    patch_data = []
    for i in range(24):
        patch_data.append({
            "patch":   i + 1,
            "name":    f"Patch {i + 1}",
            "meas_L":  round(float(measured_lab[i, 0]),  2),
            "meas_a":  round(float(measured_lab[i, 1]),  2),
            "meas_b":  round(float(measured_lab[i, 2]),  2),
            "ideal_L": round(float(ideal_lab[i, 0]),     2),
            "ideal_a": round(float(ideal_lab[i, 1]),     2),
            "ideal_b": round(float(ideal_lab[i, 2]),     2),
            "de":      round(float(de_after_patch[i]),   2),
        })

    elapsed = time.time() - t0
    return JSONResponse({
        "session_id":     session_id,
        "corrected_b64":  corrected_b64,
        "image_url":      f"/api/download/{session_id}/corrected.jpg",
        "pdf_url":        f"/api/download/{session_id}/report.pdf",
        "delta_e_before": round(de_before_mean, 3),
        "delta_e_after":  round(de_after_mean,  3),
        "psnr_before":    round(psnr_before, 2),
        "psnr_after":     round(psnr_after,  2),
        "elapsed_s":      round(elapsed, 2),
        "method":         method_used,
        "patches":        patch_data,
    })


@app.get("/api/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """Download the corrected image or PDF report for a session."""
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found or expired")

    sess = _sessions[session_id]
    if filename == "corrected.jpg":
        path      = sess["corrected"]
        media     = "image/jpeg"
    elif filename == "report.pdf":
        path      = sess["report"]
        media     = "application/pdf"
    else:
        raise HTTPException(400, "Unknown filename. Use 'corrected.jpg' or 'report.pdf'")

    if not os.path.exists(path):
        raise HTTPException(404, "File not found (may have been cleaned up)")

    return FileResponse(path, media_type=media,
                        filename=filename)


# ── Local development entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
