---
title: Sanskrit Ocr Crnn
emoji: 🏆
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
---

# Sanskrit OCR Using CRNN

Sanskrit OCR is an end-to-end optical character recognition project for printed Devanagari Sanskrit. It combines a PyTorch CRNN + CTC model, a Tesseract baseline, a Flask inference API, and a lightweight browser frontend for interactive OCR runs.

## Description

The repository is designed as a practical OCR workflow rather than only a research model. It includes synthetic data generation, preprocessing, model training, inference, benchmarking against Tesseract, CER/WER evaluation, and a drag-and-drop web UI connected to the existing Flask backend.

## Features

- Deep learning OCR with a CRNN architecture trained using CTC loss
- Tesseract OCR baseline for comparison and fallback benchmarking
- Flask API endpoint at `POST /ocr` for application integration
- Vanilla JavaScript frontend with preview, drag-and-drop, dark mode, and copy-to-clipboard
- Synthetic Sanskrit dataset generation with augmentation for printed Devanagari text
- CER and WER evaluation utilities for side-by-side model comparison

## Demo

- Add screenshots or a short screen recording under `docs/` or in the GitHub README media section
- Recommended screenshots:
  - frontend upload state
  - OCR result state
  - comparison CLI output showing ground truth vs Tesseract vs CRNN

## Tech Stack

- Python 3.11+
- PyTorch
- OpenCV
- NumPy
- Flask
- Tesseract OCR
- HTML, CSS, Vanilla JavaScript

## Project Structure

```text
OCR_Sanskrit/
  frontend/
    index.html
    style.css
    script.js
  sanskrit_ocr/
    data/
    models/
    utils/
    app.py
    dataset.py
    infer.py
    train.py
  dataset/
    labels.txt
    images/              # generated locally, ignored in Git
  checkpoints/           # generated locally, ignored in Git
  outputs/               # generated locally, ignored in Git
  app.py
  compare.py
  config.py
  data_gen.py
  infer.py
  metrics.py
  requirements.txt
  tesseract_baseline.py
  train.py
  README.md
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pytesseract
```

Optional but recommended for the baseline:

```bash
# Install the Tesseract executable separately, then point to it with --tesseract-cmd
```

## Usage

### 1. Generate Synthetic Data

```bash
py -3.11 data_gen.py --count 5000
```

Dataset label format:

```text
image_name.png	धर्मक्षेत्रे कुरुक्षेत्रे
image_name_2.png	योगः कर्मसु कौशलम्
```

### 2. Train the CRNN Model

```bash
py -3.11 train.py --dataset-dir dataset --epochs 30 --batch-size 8 --device cpu
```

Resume from an existing checkpoint:

```bash
py -3.11 train.py --dataset-dir dataset --epochs 30 --batch-size 8 --device cpu --resume checkpoints/sanskrit_crnn.pt
```

### 3. Run Inference

```bash
py -3.11 infer.py --checkpoint checkpoints/sanskrit_crnn.pt --image path/to/input.png --decoder beam --device cpu
```

### 4. Run the Flask API

```bash
py -3.11 app.py
```

Example request:

```bash
curl -X POST -F "image=@path/to/input.png" http://127.0.0.1:5000/ocr
```

### 5. Launch the Frontend

- Start the Flask API first
- Open `frontend/index.html` in a browser
- Upload an image and click `Run OCR`

### 6. Compare CRNN vs Tesseract

If Tesseract is installed outside `PATH`, pass the executable explicitly:

```bash
$env:TESSDATA_PREFIX=(Resolve-Path .\tessdata)
py -3.11 compare.py --image dataset/images/synthetic_00000.png --checkpoint checkpoints/sanskrit_crnn.pt --device cpu --decoder beam --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Results

Recent comparison snapshot on generated data:

- Tesseract average CER: `0.3354`
- Tesseract average WER: `0.8821`
- CRNN greedy average CER: `0.4430`
- CRNN greedy average WER: `0.6167`
- CRNN beam average CER: `0.4344`
- CRNN beam average WER: `0.6167`

Example sample:

```text
Ground truth: गीता ज्ञानम्
Tesseract:   गीता जुजानम
CRNN:        गीता ज्ञानम्
```

These numbers show that the CRNN can outperform Tesseract on some samples, especially at the word level, while Tesseract remains a strong character-level baseline.

## Future Improvements

- Add a proper train/validation/test split and report metrics on held-out data
- Fine-tune on scanned real-world Sanskrit documents instead of synthetic data only
- Improve beam-search confidence scoring and lexicon-aware post-processing
- Serve the frontend directly from Flask for a single-command local demo
- Add automated tests for metrics, decoding, and API contract checks

## Notes

- Generated images, model checkpoints, and training outputs are intentionally excluded from version control
- UTF-8 output is enabled in the CLI tools so Devanagari text prints correctly on Windows
- The backend API already exists; the frontend only consumes `POST /ocr`
