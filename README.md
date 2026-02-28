# PDF AI Parser

Build an AI-ready version of a PDF while preserving human-visible appearance as closely as possible.

## What this does

- Produces an OCR-enhanced PDF (`*.ai_ready.pdf`) that looks the same to a human in normal viewing.
- Keeps original graphics and page geometry.
- Adds a machine-readable text layer (searchable/selectable text).
- Exports structured layout metadata to `*.ai_manifest.json`.
- Exports extracted text summary to `*.ai.md`.

## Important note on "100% same look"

True pixel-perfect identity across all PDF viewers cannot be absolutely guaranteed for every source PDF.
This pipeline aims for practical equivalence by preserving page render and adding hidden text, not redesigning pages.

## Requirements

- Python 3.11+
- `ocrmypdf` CLI available in PATH
- OCRmyPDF system dependencies:
  - Tesseract OCR
  - Ghostscript
  - qpdf

Install Python deps:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m pdf_ai_parser input.pdf -o output --lang eng
```

Options:

- `--force-ocr` to OCR all pages even if text is already present.
- `--lang eng+ita` to process multiple OCR languages.
- `--ocr-dpi 200` to control fallback page OCR render quality.
- `--psm 6` to set Tesseract page segmentation mode.
- `--no-ocr-cleanup` to disable OCR text normalization.
- `--disable-region-ocr` to force full-page OCR fallback.
- `--no-export-images` to skip exporting page/cropped PNG visuals.
- `--image-dpi 144` to set exported image DPI.

## Output files

- `output/<name>.ai_ready.pdf`
- `output/<name>.ocr.txt`
- `output/<name>.ai_manifest.json`
- `output/<name>.ai.md`

## Streamlit app

Run a user-friendly UI:

```bash
streamlit run app.py
```

Then upload a PDF, choose OCR settings, and download generated artifacts.

## Why this is AI-first

- `ai_ready.pdf`: good for multimodal models with embedded searchable text.
- `ai_manifest.json`: explicit page structure for retrieval/chunking pipelines.
- `ai.md`: immediate ingestion into LLM workflows.
