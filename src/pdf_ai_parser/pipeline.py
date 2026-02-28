from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:  # pragma: no cover - optional runtime dependency
    pdfminer_extract_text = None


@dataclass(slots=True)
class PipelineConfig:
    input_pdf: Path
    output_dir: Path
    ocr_language: str = "eng"
    force_ocr: bool = False
    ocr_dpi: int = 200
    tesseract_psm: int = 6
    apply_ocr_cleanup: bool = True
    region_ocr: bool = True
    export_images: bool = True
    export_image_dpi: int = 144


PIPELINE_VERSION = "2026-02-28-region-ocr"


def run_pipeline(config: PipelineConfig) -> None:
    _validate_inputs(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    searchable_pdf = config.output_dir / f"{config.input_pdf.stem}.ai_ready.pdf"
    sidecar_txt = config.output_dir / f"{config.input_pdf.stem}.ocr.txt"
    manifest_json = config.output_dir / f"{config.input_pdf.stem}.ai_manifest.json"
    markdown_out = config.output_dir / f"{config.input_pdf.stem}.ai.md"

    input_layout = _extract_layout(config.input_pdf)
    input_text, input_debug = _extract_plain_text_with_debug(config.input_pdf)

    _run_ocrmypdf(
        input_pdf=config.input_pdf,
        output_pdf=searchable_pdf,
        sidecar_txt=sidecar_txt,
        language=config.ocr_language,
        force_ocr=config.force_ocr,
        tesseract_psm=config.tesseract_psm,
    )

    output_layout = _extract_layout(searchable_pdf)
    output_text, output_debug = _extract_plain_text_with_debug(searchable_pdf)
    sidecar_text = sidecar_txt.read_text(encoding="utf-8", errors="replace") if sidecar_txt.exists() else ""

    pre_input_chars = _clean_char_count(input_text)
    pre_output_chars = _clean_char_count(output_text)
    used_fallback_ocr = False
    fallback_reason = ""
    # If either branch is near-empty, force a full OCR pass for deterministic recovery.
    needs_forced_ocr = not config.force_ocr and (pre_input_chars < 20 or pre_output_chars < 20)
    if needs_forced_ocr:
        fallback_reason = f"low_text_signal(input={pre_input_chars},output={pre_output_chars})"
        _run_ocrmypdf(
            input_pdf=config.input_pdf,
            output_pdf=searchable_pdf,
            sidecar_txt=sidecar_txt,
            language=config.ocr_language,
            force_ocr=True,
            tesseract_psm=config.tesseract_psm,
        )
        output_layout = _extract_layout(searchable_pdf)
        output_text, output_debug = _extract_plain_text_with_debug(searchable_pdf)
        sidecar_text = sidecar_txt.read_text(encoding="utf-8", errors="replace") if sidecar_txt.exists() else ""
        used_fallback_ocr = True

    best_input_score = _text_quality(input_text, input_layout)
    best_output_score = _text_quality(output_text, output_layout)
    chosen_source = "ocr_output"
    if best_output_score >= best_input_score:
        layout = output_layout
        extracted_text = output_text
    else:
        layout = input_layout
        extracted_text = input_text
        chosen_source = "original_pdf"

    plain_text = _postprocess_text(_clean_extracted_text(extracted_text), config.apply_ocr_cleanup)
    if not plain_text:
        plain_text = _postprocess_text(_clean_extracted_text(sidecar_text), config.apply_ocr_cleanup)

    used_page_ocr_fallback = False
    page_ocr_chars = 0
    page_ocr_debug: dict[str, Any] = {"attempted": False}
    if _clean_char_count(plain_text) == 0:
        page_ocr_text, page_ocr_debug = _ocr_pages_with_tesseract(
            config.input_pdf,
            config.ocr_language,
            config.ocr_dpi,
            config.tesseract_psm,
            config.region_ocr,
        )
        page_ocr_chars = _clean_char_count(page_ocr_text)
        if page_ocr_chars > 0:
            plain_text = _postprocess_text(_clean_extracted_text(page_ocr_text), config.apply_ocr_cleanup)
            used_page_ocr_fallback = True

    layout = _merge_text_into_layout(layout, plain_text)
    exported_images = {"enabled": config.export_images, "dir": "", "files": []}
    if config.export_images:
        exported_images = _export_visual_assets(
            pdf_path=searchable_pdf,
            layout=layout,
            output_dir=config.output_dir,
            dpi=config.export_image_dpi,
        )

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "source_pdf": str(config.input_pdf.resolve()),
        "ai_ready_pdf": str(searchable_pdf.resolve()),
        "ocr_sidecar": str(sidecar_txt.resolve()),
        "ocr_language": config.ocr_language,
        "ocr_dpi": config.ocr_dpi,
        "tesseract_psm": config.tesseract_psm,
        "apply_ocr_cleanup": config.apply_ocr_cleanup,
        "region_ocr": config.region_ocr,
        "export_images": config.export_images,
        "export_image_dpi": config.export_image_dpi,
        "exported_images": exported_images,
        "used_fallback_ocr": used_fallback_ocr,
        "used_page_ocr_fallback": used_page_ocr_fallback,
        "page_ocr_chars": page_ocr_chars,
        "page_ocr_debug": page_ocr_debug,
        "fallback_reason": fallback_reason,
        "chosen_text_source": chosen_source,
        "extraction_debug": {
            "input_score": best_input_score,
            "output_score": best_output_score,
            "input_clean_chars": _clean_char_count(input_text),
            "output_clean_chars": _clean_char_count(output_text),
            "pre_fallback_input_clean_chars": pre_input_chars,
            "pre_fallback_output_clean_chars": pre_output_chars,
            "input_extractors": input_debug,
            "output_extractors": output_debug,
            "sidecar_clean_chars": _clean_char_count(sidecar_text),
        },
        "pages": layout,
        "plain_text": plain_text,
    }

    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    markdown_out.write_text(_build_markdown(layout, plain_text, exported_images), encoding="utf-8")


def _validate_inputs(config: PipelineConfig) -> None:
    if not config.input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {config.input_pdf}")
    if config.input_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Input file must be a .pdf: {config.input_pdf}")


def _run_ocrmypdf(
    input_pdf: Path,
    output_pdf: Path,
    sidecar_txt: Path,
    language: str,
    force_ocr: bool,
    tesseract_psm: int,
) -> None:
    if shutil.which("ocrmypdf") is not None:
        command = ["ocrmypdf"]
    else:
        command = ["python", "-m", "ocrmypdf"]

    command.extend(
        [
            "--jobs",
            "4",
            "--output-type",
            "pdfa",
            "--optimize",
            "0",
            "--tesseract-timeout",
            "0",
            "--sidecar",
            str(sidecar_txt),
            "--tesseract-pagesegmode",
            str(tesseract_psm),
            "-l",
            language,
        ]
    )

    if force_ocr:
        command.append("--force-ocr")
    else:
        command.append("--skip-text")

    command.extend([str(input_pdf), str(output_pdf)])

    proc = subprocess.run(command, capture_output=True, text=True, env=_build_ocr_env())
    if proc.returncode != 0:
        raise RuntimeError(
            "ocrmypdf failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )


def _build_ocr_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PATH", "")

    candidate_dirs = [str(p) for p in _candidate_binary_dirs() if p.exists()]
    if candidate_dirs:
        env["PATH"] = os.pathsep.join(candidate_dirs + [existing]) if existing else os.pathsep.join(candidate_dirs)

    tessdata_prefix = _resolve_tessdata_prefix()
    if tessdata_prefix:
        env["TESSDATA_PREFIX"] = tessdata_prefix
    return env


@lru_cache(maxsize=1)
def _candidate_binary_dirs() -> tuple[Path, ...]:
    dirs: list[Path] = []

    # User-level Python scripts often contain ocrmypdf.exe on Windows.
    py_tag = f"Python{sys.version_info.major}{sys.version_info.minor}"
    user_scripts = Path.home() / "AppData" / "Roaming" / "Python" / py_tag / "Scripts"
    dirs.append(user_scripts)

    # Prefer active Conda/Miniconda install when available.
    conda_base = _get_conda_base()
    if conda_base:
        dirs.extend(
            [
                conda_base / "Library" / "bin",
                conda_base / "Scripts",
                conda_base / "bin",
            ]
        )

    # Common non-Conda Windows installation locations.
    dirs.extend(
        [
            Path(r"C:\Program Files\Tesseract-OCR"),
            Path(r"C:\Program Files\gs\gs10.06.0\bin"),
            Path(r"C:\Program Files\qpdf\bin"),
        ]
    )

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    ordered: list[Path] = []
    for item in dirs:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


@lru_cache(maxsize=1)
def _get_conda_base() -> Path | None:
    conda_exe = shutil.which("conda")
    if not conda_exe:
        return None

    proc = subprocess.run(
        [conda_exe, "info", "--base"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None

    value = proc.stdout.strip()
    if not value:
        return None
    return Path(value)


def _extract_layout(pdf_path: Path) -> list[dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages: list[dict[str, Any]] = []

    for page_index, page in enumerate(doc, start=1):
        text_dict = page.get_text("dict")

        text_blocks = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            lines = []
            block_text_parts: list[str] = []
            block_font_sizes: list[float] = []
            for line in block.get("lines", []):
                spans = []
                line_text_parts: list[str] = []
                for span in line.get("spans", []):
                    text = (span.get("text") or "").strip()
                    if not text:
                        continue
                    size = span.get("size")
                    if isinstance(size, (int, float)):
                        block_font_sizes.append(float(size))
                    line_text_parts.append(text)
                    spans.append(
                        {
                            "text": text,
                            "bbox": span.get("bbox"),
                            "font": span.get("font"),
                            "size": size,
                        }
                    )
                if spans:
                    line_text = " ".join(line_text_parts).strip()
                    if line_text:
                        block_text_parts.append(line_text)
                    lines.append({"spans": spans, "bbox": line.get("bbox"), "text": line_text})

            if lines:
                block_text = "\n".join(block_text_parts).strip()
                text_blocks.append(
                    {
                        "bbox": block.get("bbox"),
                        "lines": lines,
                        "text": block_text,
                        "role": _guess_block_role(block_text, block_font_sizes),
                    }
                )

        images = []
        for image_meta in page.get_images(full=True):
            xref = image_meta[0]
            rects = [tuple(r) for r in page.get_image_rects(xref)]
            width = image_meta[2]
            height = image_meta[3]
            bpc = image_meta[4]
            colorspace = image_meta[5]
            images.append(
                {
                    "xref": xref,
                    "width": width,
                    "height": height,
                    "bits_per_component": bpc,
                    "colorspace": colorspace,
                    "rects": rects,
                }
            )

        pages.append(
            {
                "page_number": page_index,
                "width": page.rect.width,
                "height": page.rect.height,
                "text_blocks": text_blocks,
                "page_text": page.get_text("text"),
                "images": images,
            }
        )

    doc.close()
    return pages


def _build_markdown(layout: list[dict[str, Any]], plain_text: str, exported_images: dict[str, Any]) -> str:
    lines = ["# AI-ready PDF content", "", "## Full text", "", plain_text.strip(), ""]
    page_images: dict[int, str] = {}
    cropped_images: dict[int, list[str]] = {}
    for item in exported_images.get("files", []):
        page_number = int(item.get("page_number", 0))
        rel_path = str(item.get("rel_path", ""))
        kind = str(item.get("kind", ""))
        if not page_number or not rel_path:
            continue
        if kind == "page":
            page_images[page_number] = rel_path
        elif kind == "crop":
            cropped_images.setdefault(page_number, []).append(rel_path)

    for page in layout:
        lines.append(f"## Page {page['page_number']}")
        lines.append("")
        lines.append(f"- Size: {page['width']} x {page['height']}")
        lines.append(f"- Text blocks: {len(page['text_blocks'])}")
        lines.append(f"- Images: {len(page['images'])}")
        lines.append("")
        page_text = (page.get("page_text") or "").strip()
        page_number = int(page["page_number"])
        if page_number in page_images:
            lines.append("### Page image")
            lines.append("")
            lines.append(f"![Page {page_number}]({page_images[page_number]})")
            lines.append("")
        if page_number in cropped_images:
            lines.append("### Cropped visuals")
            lines.append("")
            for rel in cropped_images[page_number]:
                lines.append(f"![Crop]({rel})")
            lines.append("")
        if page_text:
            lines.append("### Page text")
            lines.append("")
            lines.append(page_text)
            lines.append("")
        if page["text_blocks"]:
            lines.append("### Text blocks")
            lines.append("")
            for idx, block in enumerate(page["text_blocks"], start=1):
                block_text = (block.get("text") or "").strip()
                if not block_text:
                    continue
                lines.append(f"#### Block {idx} ({block.get('role', 'paragraph')})")
                lines.append("")
                lines.append(block_text)
                lines.append("")

    return "\n".join(lines).strip() + "\n"


def _export_visual_assets(
    pdf_path: Path,
    layout: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    zoom = max(1.0, float(dpi) / 72.0)
    matrix = fitz.Matrix(zoom, zoom)
    files: list[dict[str, Any]] = []

    doc = fitz.open(pdf_path)
    try:
        for page_idx, page in enumerate(doc, start=1):
            page_name = f"page_{page_idx:03d}.png"
            page_path = images_dir / page_name
            page.get_pixmap(matrix=matrix, alpha=False).save(page_path)
            files.append(
                {
                    "kind": "page",
                    "page_number": page_idx,
                    "rel_path": f"images/{page_name}",
                }
            )

        for page_idx, page_data in enumerate(layout, start=1):
            page = doc[page_idx - 1]
            crop_counter = 0
            for image_item in page_data.get("images", []):
                xref = image_item.get("xref", "x")
                for rect in image_item.get("rects", []):
                    clip = fitz.Rect(rect)
                    if clip.width <= 1 or clip.height <= 1:
                        continue
                    crop_counter += 1
                    crop_name = f"page_{page_idx:03d}_img_{xref}_{crop_counter:02d}.png"
                    crop_path = images_dir / crop_name
                    page.get_pixmap(matrix=matrix, clip=clip, alpha=False).save(crop_path)
                    files.append(
                        {
                            "kind": "crop",
                            "page_number": page_idx,
                            "xref": xref,
                            "bbox": [float(clip.x0), float(clip.y0), float(clip.x1), float(clip.y1)],
                            "rel_path": f"images/{crop_name}",
                        }
                    )
    finally:
        doc.close()

    return {"enabled": True, "dir": str(images_dir.resolve()), "files": files}


def _extract_plain_text(pdf_path: Path) -> str:
    text, _ = _extract_plain_text_with_debug(pdf_path)
    return text


def _extract_plain_text_with_debug(pdf_path: Path) -> tuple[str, dict[str, int]]:
    fitz_text = _extract_plain_text_fitz(pdf_path)
    miner_text = _extract_plain_text_pdfminer(pdf_path)
    selected = _pick_richer_text(fitz_text, miner_text)
    return selected, {
        "fitz_raw_chars": len(fitz_text),
        "pdfminer_raw_chars": len(miner_text),
        "fitz_clean_chars": _clean_char_count(fitz_text),
        "pdfminer_clean_chars": _clean_char_count(miner_text),
        "selected_clean_chars": _clean_char_count(selected),
    }


def _extract_plain_text_fitz(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text").strip())
    doc.close()
    return "\f".join(pages).strip()


def _extract_plain_text_pdfminer(pdf_path: Path) -> str:
    if pdfminer_extract_text is None:
        return ""
    try:
        text = pdfminer_extract_text(str(pdf_path)) or ""
    except Exception:
        return ""
    # pdfminer uses form feed for page boundaries as well.
    return text.replace("\x0c", "\f").strip()


def _looks_text_empty(plain_text: str, layout: list[dict[str, Any]]) -> bool:
    if any(page.get("text_blocks") for page in layout):
        return False
    normalized = "".join(ch for ch in _clean_extracted_text(plain_text) if ch.isalnum())
    return len(normalized) < 20


def _guess_block_role(block_text: str, font_sizes: list[float]) -> str:
    text = block_text.strip()
    if not text:
        return "paragraph"
    if text.startswith(("-", "*", "\u2022")):
        return "list_item"
    max_size = max(font_sizes) if font_sizes else 0.0
    if max_size >= 18:
        return "heading"
    if max_size >= 13:
        return "subheading"
    return "paragraph"


def _clean_extracted_text(text: str) -> str:
    cleaned_pages: list[str] = []
    for page in text.split("\f"):
        line_items: list[str] = []
        for raw_line in page.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower in {"[skipped page]", "skipped page"}:
                continue
            if "ocr skipped on page" in lower:
                continue
            line_items.append(line)
        cleaned_pages.append("\n".join(line_items).strip())
    return "\f".join(cleaned_pages).strip("\f \n\t")


def _text_quality(plain_text: str, layout: list[dict[str, Any]]) -> int:
    text_len = _clean_char_count(plain_text)
    block_count = sum(len(page.get("text_blocks", [])) for page in layout)
    return text_len + (block_count * 25)


def _pick_richer_text(primary: str, secondary: str) -> str:
    primary_score = len("".join(ch for ch in _clean_extracted_text(primary) if ch.isalnum()))
    secondary_score = len("".join(ch for ch in _clean_extracted_text(secondary) if ch.isalnum()))
    return secondary if secondary_score > primary_score else primary


def _merge_text_into_layout(layout: list[dict[str, Any]], plain_text: str) -> list[dict[str, Any]]:
    page_texts = _split_text_by_pages(plain_text, len(layout))
    merged: list[dict[str, Any]] = []
    for idx, page in enumerate(layout):
        item = dict(page)
        existing = _clean_extracted_text(str(item.get("page_text", "")))
        candidate = page_texts[idx] if idx < len(page_texts) else ""
        page_text = existing or candidate
        item["page_text"] = page_text

        text_blocks = item.get("text_blocks") or []
        if not text_blocks and page_text.strip():
            item["text_blocks"] = [
                {
                    "bbox": [0.0, 0.0, float(item.get("width", 0.0)), float(item.get("height", 0.0))],
                    "lines": [
                        {
                            "spans": [],
                            "bbox": [0.0, 0.0, float(item.get("width", 0.0)), float(item.get("height", 0.0))],
                            "text": page_text,
                        }
                    ],
                    "text": page_text,
                    "role": "paragraph",
                    "synthetic": True,
                }
            ]
        merged.append(item)
    return merged


def _split_text_by_pages(text: str, expected_pages: int) -> list[str]:
    cleaned = _clean_extracted_text(text)
    if expected_pages <= 0:
        return []
    if not cleaned:
        return [""] * expected_pages

    parts = [p.strip() for p in cleaned.split("\f")]
    if len(parts) == expected_pages:
        return parts
    if len(parts) > expected_pages:
        return parts[:expected_pages]

    padded = parts + ([""] * (expected_pages - len(parts)))
    return padded


def _clean_char_count(text: str) -> int:
    clean = _clean_extracted_text(text)
    return len("".join(ch for ch in clean if ch.isalnum()))


def _postprocess_text(text: str, enabled: bool) -> str:
    if not enabled:
        return text

    value = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)

    # Normalize common OCR spacing artifacts for contact and URLs.
    value = re.sub(r"\s+@\s+", "@", value)
    value = re.sub(r"linkedin\.com\s*/\s*in\s*/?", "linkedin.com/in/", value, flags=re.IGNORECASE)
    value = re.sub(r"[ \t]{2,}", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _ocr_pages_with_tesseract(
    pdf_path: Path,
    language: str,
    dpi: int,
    tesseract_psm: int,
    region_ocr: bool,
) -> tuple[str, dict[str, Any]]:
    env = _build_ocr_env()
    tesseract_exe = _resolve_tesseract_executable(env)
    debug: dict[str, Any] = {
        "attempted": True,
        "tesseract_exe": tesseract_exe or "",
        "tessdata_prefix": env.get("TESSDATA_PREFIX", ""),
        "language": language or "eng",
        "dpi": dpi,
        "tesseract_psm": tesseract_psm,
        "region_ocr": region_ocr,
        "page_results": [],
    }
    if tesseract_exe is None:
        debug["error"] = "tesseract_not_found"
        return "", debug

    doc = fitz.open(pdf_path)
    pages: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="pdf_ai_page_ocr_") as tmp:
            tmp_dir = Path(tmp)
            for index, page in enumerate(doc, start=1):
                page_text, page_debug = _ocr_single_page(
                    page=page,
                    page_number=index,
                    tmp_dir=tmp_dir,
                    tesseract_exe=tesseract_exe,
                    language=(language or "eng"),
                    dpi=dpi,
                    tesseract_psm=tesseract_psm,
                    env=env,
                    use_regions=region_ocr,
                )
                debug["page_results"].append(page_debug)
                pages.append(page_text)
    finally:
        doc.close()
    return "\f".join(pages).strip(), debug


def _resolve_tesseract_executable(env: dict[str, str]) -> str | None:
    path_value = env.get("PATH", "")
    direct = shutil.which("tesseract", path=path_value) or shutil.which("tesseract.exe", path=path_value)
    if direct:
        return direct

    for directory in _candidate_binary_dirs():
        candidate = directory / "tesseract.exe"
        if candidate.exists():
            return str(candidate)
    return None


def _ocr_single_page(
    page: fitz.Page,
    page_number: int,
    tmp_dir: Path,
    tesseract_exe: str,
    language: str,
    dpi: int,
    tesseract_psm: int,
    env: dict[str, str],
    use_regions: bool,
) -> tuple[str, dict[str, Any]]:
    regions = _detect_page_regions(page) if use_regions else []
    if not regions:
        regions = [page.rect]

    texts: list[str] = []
    region_results: list[dict[str, Any]] = []
    zoom = max(1.0, float(dpi) / 72.0)
    matrix = fitz.Matrix(zoom, zoom)

    for ridx, rect in enumerate(regions, start=1):
        image_path = tmp_dir / f"page_{page_number:04d}_r{ridx:03d}.png"
        pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
        pix.save(image_path)

        command = [
            tesseract_exe,
            str(image_path),
            "stdout",
            "-l",
            language,
            "--psm",
            str(tesseract_psm),
            "--dpi",
            str(dpi),
        ]
        proc = subprocess.run(command, capture_output=True, text=True, env=env)
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode == 0 and out:
            texts.append(out)

        region_results.append(
            {
                "region_index": ridx,
                "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                "returncode": proc.returncode,
                "stdout_chars": len(out),
                "stderr_preview": err[:160],
            }
        )

    page_text = "\n\n".join(t for t in texts if t).strip()
    if use_regions and not page_text:
        # If all image regions fail or are empty, try full-page OCR once.
        full_text, full_result = _ocr_full_page(page, page_number, tmp_dir, tesseract_exe, language, dpi, tesseract_psm, env)
        page_text = full_text
        region_results.append(full_result)

    page_debug = {
        "page_number": page_number,
        "mode": "regions" if use_regions else "full_page",
        "region_count": len(regions),
        "returncode": 0 if page_text else 1,
        "stdout_chars": len(page_text),
        "regions": region_results,
    }
    return page_text, page_debug


def _ocr_full_page(
    page: fitz.Page,
    page_number: int,
    tmp_dir: Path,
    tesseract_exe: str,
    language: str,
    dpi: int,
    tesseract_psm: int,
    env: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    image_path = tmp_dir / f"page_{page_number:04d}_full.png"
    zoom = max(1.0, float(dpi) / 72.0)
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(image_path)

    command = [
        tesseract_exe,
        str(image_path),
        "stdout",
        "-l",
        language,
        "--psm",
        str(tesseract_psm),
        "--dpi",
        str(dpi),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    result = {
        "region_index": -1,
        "bbox": [float(page.rect.x0), float(page.rect.y0), float(page.rect.x1), float(page.rect.y1)],
        "returncode": proc.returncode,
        "stdout_chars": len(out),
        "stderr_preview": err[:160],
        "fallback_full_page": True,
    }
    if proc.returncode != 0:
        return "", result
    return out, result


def _detect_page_regions(page: fitz.Page) -> list[fitz.Rect]:
    rects: list[fitz.Rect] = []
    for image_meta in page.get_images(full=True):
        xref = image_meta[0]
        for rect in page.get_image_rects(xref):
            rects.append(rect)
    unique = _dedupe_rects(rects)
    unique.sort(key=lambda r: (round(r.y0, 1), round(r.x0, 1)))
    return unique


def _dedupe_rects(rects: list[fitz.Rect]) -> list[fitz.Rect]:
    out: list[fitz.Rect] = []
    seen: set[tuple[int, int, int, int]] = set()
    for rect in rects:
        key = (int(round(rect.x0)), int(round(rect.y0)), int(round(rect.x1)), int(round(rect.y1)))
        if key in seen:
            continue
        seen.add(key)
        out.append(rect)
    return out


@lru_cache(maxsize=1)
def _resolve_tessdata_prefix() -> str:
    # Respect user-provided value when valid.
    current = os.environ.get("TESSDATA_PREFIX", "").strip()
    if current and _tessdata_has_language(Path(current), "eng"):
        return current

    for directory in _candidate_tessdata_dirs():
        if _tessdata_has_language(directory, "eng"):
            return str(directory)
    return ""


def _tessdata_has_language(directory: Path, language: str) -> bool:
    if not directory.exists():
        return False
    return (directory / f"{language}.traineddata").exists()


@lru_cache(maxsize=1)
def _candidate_tessdata_dirs() -> tuple[Path, ...]:
    dirs: list[Path] = []
    conda_base = _get_conda_base()
    if conda_base:
        dirs.extend(
            [
                conda_base / "share" / "tessdata",
                conda_base / "Library" / "share" / "tessdata",
            ]
        )

    dirs.extend(
        [
            Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
        ]
    )

    # Also infer from any binary candidate path.
    for bin_dir in _candidate_binary_dirs():
        dirs.append(bin_dir / "tessdata")
        dirs.append(bin_dir.parent / "share" / "tessdata")

    seen: set[Path] = set()
    ordered: list[Path] = []
    for item in dirs:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)
