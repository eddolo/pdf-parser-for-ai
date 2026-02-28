import io
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pdf_ai_parser.pipeline import PipelineConfig, run_pipeline


st.set_page_config(page_title="PDF AI Parser", page_icon="??", layout="wide")

st.title("PDF AI Parser")
st.caption(
    "Upload a PDF, preserve visual appearance, and generate AI-readable outputs "
    "(searchable PDF + structured manifest + extracted text)."
)

with st.sidebar:
    st.header("Settings")
    ocr_language = st.text_input("OCR language(s)", value="eng", help="Examples: eng, eng+ita")
    force_ocr = st.toggle("Force OCR on all pages", value=False)
    ocr_dpi = st.slider("OCR DPI (fallback)", min_value=120, max_value=400, value=200, step=10)
    tesseract_psm = st.selectbox(
        "Tesseract PSM",
        options=[3, 4, 6, 11, 12],
        index=2,
        help="6 is a strong default for blocks of text.",
    )
    apply_ocr_cleanup = st.toggle("Post-OCR cleanup", value=True)
    region_ocr = st.toggle("Region-based OCR fallback", value=True)
    export_images = st.toggle("Export images", value=True)
    export_image_dpi = st.slider("Export image DPI", min_value=96, max_value=300, value=144, step=12)

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded is not None:
    st.success(f"Loaded: {uploaded.name} ({uploaded.size:,} bytes)")

run_clicked = st.button("Process PDF", type="primary", disabled=uploaded is None)


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


if run_clicked and uploaded is not None:
    with tempfile.TemporaryDirectory(prefix="pdf_ai_parser_") as tmp:
        tmp_dir = Path(tmp)
        safe_name = Path(uploaded.name).name
        input_pdf = tmp_dir / safe_name
        output_dir = tmp_dir / "output"

        input_pdf.write_bytes(uploaded.getvalue())

        config = PipelineConfig(
            input_pdf=input_pdf,
            output_dir=output_dir,
            ocr_language=ocr_language.strip() or "eng",
            force_ocr=force_ocr,
            ocr_dpi=ocr_dpi,
            tesseract_psm=tesseract_psm,
            apply_ocr_cleanup=apply_ocr_cleanup,
            region_ocr=region_ocr,
            export_images=export_images,
            export_image_dpi=export_image_dpi,
        )

        try:
            with st.spinner("Running OCR + layout extraction..."):
                run_pipeline(config)
        except Exception as exc:
            st.error("Processing failed.")
            st.exception(exc)
        else:
            base = input_pdf.stem
            ai_ready_pdf = output_dir / f"{base}.ai_ready.pdf"
            ocr_txt = output_dir / f"{base}.ocr.txt"
            manifest = output_dir / f"{base}.ai_manifest.json"
            markdown = output_dir / f"{base}.ai.md"
            images_dir = output_dir / "images"

            st.subheader("Output files")
            for f in [ai_ready_pdf, ocr_txt, manifest, markdown]:
                st.write(f"- `{f.name}` ({f.stat().st_size:,} bytes)")
            if images_dir.exists():
                image_files = sorted(images_dir.rglob("*.png"))
                st.write(f"- `images/` ({len(image_files)} PNG files)")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download AI-ready PDF",
                    data=_read_bytes(ai_ready_pdf),
                    file_name=ai_ready_pdf.name,
                    mime="application/pdf",
                )
                st.download_button(
                    "Download Manifest JSON",
                    data=_read_bytes(manifest),
                    file_name=manifest.name,
                    mime="application/json",
                )

            with col2:
                st.download_button(
                    "Download OCR Text",
                    data=_read_bytes(ocr_txt),
                    file_name=ocr_txt.name,
                    mime="text/plain",
                )
                st.download_button(
                    "Download Markdown",
                    data=_read_bytes(markdown),
                    file_name=markdown.name,
                    mime="text/markdown",
                )

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in [ai_ready_pdf, ocr_txt, manifest, markdown]:
                    zf.writestr(f.name, f.read_bytes())
                if images_dir.exists():
                    for img in images_dir.rglob("*"):
                        if img.is_file():
                            rel = img.relative_to(output_dir)
                            zf.writestr(str(rel).replace("\\", "/"), img.read_bytes())
            zip_buffer.seek(0)

            st.download_button(
                "Download all outputs (.zip)",
                data=zip_buffer.getvalue(),
                file_name=f"{base}.ai_outputs.zip",
                mime="application/zip",
            )

            with st.expander("Preview: manifest JSON"):
                st.json(json.loads(manifest.read_text(encoding="utf-8", errors="replace")))

            with st.expander("Preview: OCR text"):
                st.text(ocr_txt.read_text(encoding="utf-8", errors="replace")[:10000])

st.markdown("---")
st.caption("Runtime requirements: OCRmyPDF + Tesseract + Ghostscript + qpdf.")
