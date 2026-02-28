import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-ai-parser",
        description=(
            "Make a PDF AI-ready while preserving human-visible appearance. "
            "Output includes an OCR-enhanced PDF and a structured JSON manifest."
        ),
    )
    parser.add_argument("input_pdf", type=Path, help="Path to input PDF")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="OCR language(s) for ocrmypdf, e.g. 'eng' or 'eng+ita'",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on all pages, even if text already exists",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=200,
        help="DPI for page-level OCR fallback rendering (default: 200)",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (default: 6)",
    )
    parser.add_argument(
        "--no-ocr-cleanup",
        action="store_true",
        help="Disable OCR text cleanup/normalization",
    )
    parser.add_argument(
        "--disable-region-ocr",
        action="store_true",
        help="Disable region-based OCR and use full-page OCR fallback",
    )
    parser.add_argument(
        "--no-export-images",
        action="store_true",
        help="Disable exporting page/cropped image files",
    )
    parser.add_argument(
        "--image-dpi",
        type=int,
        default=144,
        help="DPI for exported page/cropped images (default: 144)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = PipelineConfig(
        input_pdf=args.input_pdf,
        output_dir=args.output_dir,
        ocr_language=args.lang,
        force_ocr=args.force_ocr,
        ocr_dpi=args.ocr_dpi,
        tesseract_psm=args.psm,
        apply_ocr_cleanup=not args.no_ocr_cleanup,
        region_ocr=not args.disable_region_ocr,
        export_images=not args.no_export_images,
        export_image_dpi=args.image_dpi,
    )
    run_pipeline(config)
    print(f"Done. AI-ready artifacts written to: {config.output_dir.resolve()}")
    return 0
