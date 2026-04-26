<div align="center">

# ChromaCorrect

**Professional colour calibration for photographers — powered by deep learning**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.18-ff9900?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Deploy to Vercel](https://img.shields.io/badge/Deploy_to-Vercel-black?logo=vercel)](https://vercel.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Upload a scene photo shot with a **Macbeth ColorChecker** reference card.  
ChromaCorrect measures the colour error, predicts a correction transform, and delivers a fully calibrated image — in seconds, right in your browser.

[**Live Demo**](https://your-app.vercel.app) &nbsp;·&nbsp; [**API Reference**](#-api-reference) &nbsp;·&nbsp; [**How It Works**](#-how-it-works) &nbsp;·&nbsp; [**Deploy**](#-deployment)

</div>

---

## What Is Colour Calibration?

Every camera sensor captures colour differently depending on lighting conditions, lens coatings, and sensor characteristics. The **Macbeth ColorChecker** is an industry-standard reference card with 24 precisely defined colour patches. ChromaCorrect compares what your camera measured against what those patches *should* look like, then computes a correction transform applied to every pixel.

> **ΔE₀₀ (Delta E 2000)** is the standard perceptual colour-difference metric. A mean ΔE₀₀ below **2.0** meets ISO 13655 colour accuracy. ChromaCorrect reports this metric before and after correction for all 24 patches.

---

## ✨ Features

| | Feature | Detail |
|-|---------|--------|
| 🎯 | **CIEDE2000 correction** | Most perceptually accurate colour-difference metric (ΔE₀₀) |
| 📷 | **In-scene card detection** | Click 4 corners of the card in your photo — no separate measurement file needed |
| 📊 | **Per-patch analysis** | Reports L\*, a\*, b\* and ΔE₀₀ for all 24 patches individually |
| 🖼️ | **Certified output** | Corrected image watermarked with ISO 13655 / D65 certification badge |
| 📄 | **PDF calibration report** | One-page professional report with before/after thumbnails and full patch table |
| 🧮 | **Dual correction engine** | Analytical least-squares CCM + trained neural CCM — auto-picks whichever gives lower ΔE₀₀ |
| ⚡ | **Fast inference** | ONNX Runtime on CPU — no GPU required, under 3 seconds per image |
| 🌐 | **Zero installation** | Fully serverless — deploy to Vercel free tier with one click |
| 🃏 | **Any compatible card** | Works with the original Macbeth card or any generic 24-patch clone |

---

## 🔬 How It Works

```
  Scene photo                   Colour measurements
  (JPEG / PNG)                  (24-patch RGB values)
       │                                │
       ▼                                ▼
 ┌──────────┐               ┌───────────────────────┐
 │  Resize  │               │  Parse + normalise    │
 │ 256×256  │               │  sRGB or raw sensor   │
 └────┬─────┘               └──────────┬────────────┘
      │                                │
      └───────────────┬────────────────┘
                      ▼
           ┌───────────────────┐
           │    ONNX Model     │
           │  (Attention CNN)  │
           │                   │
           │  24 patch tokens  │
           │  self-attention   │  ← patches learn from each other
           │  cross-attention  │  ← patches attend to scene image
           │  importance agg   │  ← neutral patches weighted higher
           │                   │
           │  3×3 matrix +     │
           │  3-vector bias    │
           └─────────┬─────────┘
                     │
                     ▼
      corrected_pixel = clamp(pixel @ M + bias, 0, 1)
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Corrected JPEG           PDF Report
   + ΔE₀₀ badge            + patch table
   + colour strip           + PSNR stats
```

The model is trained on the [NUS 8-Camera dataset](https://cvil.uvic.ca/projects/public/illuminant/illuminant.html) using a **Cross-Reference Attention** architecture — each of the 24 colour patches attends to the overall scene context before the correction matrix is predicted.

---

## 🚀 Quick Start

### Option A — Click Card Corners *(Recommended)*

> Include the ColorChecker in your scene photo. No separate measurement file needed.

1. Open the app and drag-and-drop your photo
2. The canvas appears — click the **4 corners** of the colour card in order:  
   `① Top-Left → ② Top-Right → ③ Bottom-Right → ④ Bottom-Left`
3. A 4×6 grid overlay confirms all 24 patches. Click **Confirm Card**
4. Click **Calibrate Image**
5. Download the corrected JPEG and PDF report

### Option B — Upload `_color.txt` *(Advanced — raw sensor values)*

> For users with camera software that exports raw patch measurements.

1. Upload your scene photo **and** the `_color.txt` file
2. Expand **Advanced Settings** to set Dark Level and Saturation Level if known
3. Click **Calibrate Image**

### Option C — Try the Demo

Click **Try Demo** below the Calibrate button to run the pipeline with synthetic data — no card required.

---

## 🃏 Compatible Colour Cards

The reference sRGB values for all 24 standard Macbeth patches are hardcoded in the app. You do **not** need printed reference values on the card.

| Card | Compatible |
|------|------------|
| X-Rite Macbeth ColorChecker Classic | ✅ |
| X-Rite ColorChecker Passport | ✅ |
| Generic 24-patch Macbeth clone (Amazon / AliExpress, ~₹800) | ✅ |
| Cards with fewer or more than 24 patches | ❌ |

---

## 🛠️ Local Development

### Prerequisites

- Python **3.10** or later
- The trained `model.onnx` file placed inside `api/` *(~8 MB — see [Obtaining the Model](#obtaining-the-model))*

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/chromacorrect.git
cd chromacorrect

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r api/requirements.txt

# 4. Start the development server
uvicorn api.index:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Obtaining the Model

`model.onnx` is not included in the repository due to its size. Options:

- **Train your own** — use the companion training repository (linked below)
- **Download** — a pre-trained checkpoint will be linked here after the initial release

---

## 📁 Project Structure

```
chromacorrect/
│
├── api/
│   ├── index.py          ← FastAPI app — all endpoints and inference logic
│   ├── color_utils.py    ← sRGB ↔ CIELAB conversion + CIEDE2000 (NumPy only)
│   ├── watermark.py      ← Stamps corrected image with ΔE₀₀ certification badge
│   ├── pdf_report.py     ← Generates downloadable PDF calibration report (fpdf2)
│   ├── model.onnx        ← Pre-trained ONNX model (~3.2 MB) — add before deploying
│   └── requirements.txt  ← Python dependencies (pinned versions)
│
├── public/
│   └── index.html        ← Complete single-file frontend (vanilla JS, no framework)
│
├── vercel.json           ← Serverless routing and function configuration
├── .gitignore
└── README.md
```

---

## ☁️ Deployment

ChromaCorrect is designed for **one-click deployment** to Vercel's free tier.

```bash
# 1. Push this repository to GitHub (with model.onnx included in api/)
git push origin main

# 2. Go to vercel.com → Add New Project → import your repository
# 3. Vercel auto-detects vercel.json — click Deploy
# 4. Your app is live at: https://your-repo-name.vercel.app
```

<details>
<summary><strong>Vercel resource limits (free tier)</strong></summary>

| Resource | Free Tier Limit | ChromaCorrect Usage |
|----------|----------------|---------------------|
| Memory | 1024 MB | ~200 MB (model + image in flight) |
| Max execution time | 60 s | < 5 s typical |
| Bundle size | 250 MB | ~120 MB (onnxruntime + model) |

</details>

<details>
<summary><strong>Self-hosting with Docker</strong></summary>

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t chromacorrect .
docker run -p 8000:8000 chromacorrect
# Open http://localhost:8000
```

</details>

---

## 📡 API Reference

### `POST /api/correct`

Calibrate an image using Macbeth ColorChecker measurements.

**Form fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image` | file | ✅ | — | Scene photo (JPEG or PNG) |
| `color_txt` | file | ✅ | — | 24-line file, one `R G B` value per line |
| `dark_level` | float | ❌ | `0.0` | Sensor black level (raw workflow only) |
| `sat_level` | float | ❌ | `65535.0` | Sensor saturation level (raw workflow only) |
| `is_srgb_input` | bool | ❌ | `false` | `true` when `color_txt` contains sRGB 0–255 values from browser corner-click sampling |

**Response:**

```json
{
  "session_id":     "a1b2c3d4",
  "corrected_b64":  "<base64-encoded JPEG>",
  "image_url":      "/api/download/a1b2c3d4/corrected.jpg",
  "pdf_url":        "/api/download/a1b2c3d4/report.pdf",
  "delta_e_before": 14.83,
  "delta_e_after":  1.92,
  "psnr_before":    18.40,
  "psnr_after":     29.10,
  "elapsed_s":      2.34,
  "patches": [
    {
      "patch": 1,
      "name": "Patch 1",
      "meas_L": 37.20, "meas_a": 11.40, "meas_b": 9.80,
      "ideal_L": 38.30, "ideal_a": 12.10, "ideal_b": 10.60,
      "de": 1.41
    }
  ]
}
```

### `GET /api/download/{session_id}/{filename}`

Download the corrected image or PDF report for a session.

| Parameter | Accepted Values |
|-----------|-----------------|
| `filename` | `corrected.jpg` or `report.pdf` |

> **Note:** Sessions are in-memory and expire when the serverless function restarts. Download files immediately after calibration.

**Example with `curl`:**

```bash
# Calibrate
curl -X POST https://your-app.vercel.app/api/correct \
  -F "image=@scene.jpg" \
  -F "color_txt=@measured.txt" \
  -F "is_srgb_input=true" \
  -o response.json

# Extract session ID and download corrected image
SESSION=$(python -c "import json; print(json.load(open('response.json'))['session_id'])")
curl "https://your-app.vercel.app/api/download/$SESSION/corrected.jpg" -o corrected.jpg
curl "https://your-app.vercel.app/api/download/$SESSION/report.pdf"    -o report.pdf
```

---

## 🧰 Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Backend | FastAPI + Uvicorn | 0.95 / 0.29 |
| ML inference | ONNX Runtime (CPU) | 1.18 |
| Image processing | Pillow | 10.3 |
| Colour mathematics | NumPy | 1.26 |
| PDF generation | fpdf2 | 2.7.9 |
| Frontend | Vanilla JS + CSS Grid | — |
| Deployment | Vercel (Python serverless) | — |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made for photographers who care about accurate colour.

</div>
