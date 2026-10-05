"""
Document and attachment loader with strict absolute path validation and Hybrid PDF parsing.
Supports digital text extraction, scanned page rasterization & OCR, and embedded image inspection.
"""

import asyncio
import base64
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image

from app.multimodal.context_formatter import AnnotatedContextObject
from app.multimodal.ocr_engine import OCREngine


class PathValidationError(Exception):
    """Raised when a user provides a relative or invalid path."""
    pass


class DocumentLoader:
    """Loads and formats documents, code files, images, and image-heavy PDFs."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
    DOC_EXTENSIONS = {
        ".pdf", ".docx", ".txt", ".md", ".json", ".csv", ".py", ".html",
        ".js", ".ts", ".c", ".cpp", ".rs", ".go", ".yaml", ".yml"
    }

    _ocr_engine = OCREngine()

    @staticmethod
    def is_absolute_path(path_str: str) -> bool:
        """
        Validates if the provided string is an absolute path.
        Rejects relative paths (e.g., .\\path, ../file, subdir/file).
        """
        path_str = path_str.strip().strip("'\"")
        p = Path(path_str)

        # On Windows: must have drive letter (e.g., C:\) or be UNC (\\server\share)
        if os.name == "nt":
            if path_str.startswith(".") or path_str.startswith("~") or not p.is_absolute() or not p.drive:
                return False
            return True
        else:
            return p.is_absolute() and not path_str.startswith(".")

    @classmethod
    def validate_and_resolve_path(cls, path_str: str) -> Path:
        """
        Validates that the path is strictly an absolute path and exists.
        Raises PathValidationError if relative or non-existent.
        """
        path_str = path_str.strip().strip("'\"")
        if not cls.is_absolute_path(path_str):
            raise PathValidationError(
                f"Relative paths ('{path_str}') are not allowed! "
                f"Please provide the FULL absolute path (e.g. 'C:\\Users\\...\\file.ext')."
            )

        p = Path(path_str)
        if not p.exists():
            raise PathValidationError(f"File not found at absolute path: '{path_str}'")

        if not p.is_file():
            raise PathValidationError(f"Path is a directory, not a file: '{path_str}'")

        return p

    @classmethod
    def is_image(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() in cls.IMAGE_EXTENSIONS

    @classmethod
    async def parse_pdf_hybrid(cls, file_path: Path) -> AnnotatedContextObject:
        """
        Hybrid PDF Parser:
        1. Extracts digital selectable text.
        2. For pages with sparse/scanned text (< 40 characters), rasterizes the page image via pypdfium2 and runs OCR.
        3. Extracts embedded images / diagrams from pages.
        """
        import pypdf
        import pypdfium2 as pdfium

        try:
            reader = pypdf.PdfReader(str(file_path))
            pdf_doc = pdfium.PdfDocument(str(file_path))
        except Exception as e:
            return AnnotatedContextObject(
                type="document_context",
                source=str(file_path),
                method="HYBRID_PDF_PARSER",
                confidence=0.0,
                content=f"[Error opening PDF: {str(e)}]"
            )

        pages_output: List[str] = []
        scanned_pages_count = 0
        embedded_images_count = 0
        total_pages = len(reader.pages)

        for page_idx in range(total_pages):
            pypdf_page = reader.pages[page_idx]
            raw_text = (pypdf_page.extract_text() or "").strip()

            page_header = f"--- Page {page_idx + 1} of {total_pages} ---"
            page_text_blocks: List[str] = []

            # Check if page has sufficient digital text
            alphanumeric_count = sum(1 for c in raw_text if c.isalnum())
            is_scanned_or_sparse = alphanumeric_count < 35

            if not is_scanned_or_sparse:
                page_text_blocks.append(raw_text)
            else:
                # Scanned page / sparse text: rasterize page and run OCR
                try:
                    page_img = pdf_doc[page_idx].render(scale=2.0).to_pil()
                    temp_img_path = file_path.parent / f"_temp_pdf_p{page_idx + 1}.png"
                    page_img.save(temp_img_path)

                    extracted_ocr, conf = await cls._ocr_engine.extract_text(temp_img_path)
                    try:
                        temp_img_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                    if extracted_ocr and not extracted_ocr.startswith("[OCR Engine:"):
                        page_text_blocks.append(f"[Scanned Page OCR Transcription]:\n{extracted_ocr}")
                        scanned_pages_count += 1
                    elif raw_text:
                        page_text_blocks.append(raw_text)
                    else:
                        page_text_blocks.append("[Visual Graphic / Scanned Page - No readable text extracted]")
                except Exception as ex:
                    if raw_text:
                        page_text_blocks.append(raw_text)
                    else:
                        page_text_blocks.append(f"[Error rasterizing scanned page: {ex}]")

            # Check for embedded images/figures on the page
            try:
                for img_idx, img_obj in enumerate(pypdf_page.images):
                    embedded_images_count += 1
                    img_bytes = img_obj.data
                    pil_embedded = Image.open(io.BytesIO(img_bytes))
                    # If embedded image has notable size, try OCR if text might be in diagram
                    if pil_embedded.width > 120 and pil_embedded.height > 60:
                        temp_emb = file_path.parent / f"_temp_emb_p{page_idx + 1}_{img_idx}.png"
                        pil_embedded.save(temp_emb)
                        emb_ocr, _ = await cls._ocr_engine.extract_text(temp_emb)
                        try:
                            temp_emb.unlink(missing_ok=True)
                        except Exception:
                            pass

                        if emb_ocr and len(emb_ocr.strip()) > 15 and not emb_ocr.startswith("["):
                            page_text_blocks.append(f"[Embedded Diagram / Figure Text ({img_obj.name})]:\n{emb_ocr}")
            except Exception:
                pass

            combined_page_text = "\n".join(page_text_blocks)
            pages_output.append(f"{page_header}\n{combined_page_text}")

        final_content = "\n\n".join(pages_output)

        return AnnotatedContextObject(
            type="document_context",
            source=str(file_path),
            method="HYBRID_PDF_PARSER",
            confidence=0.95,
            content=final_content,
            metadata={
                "page_count": total_pages,
                "scanned_pages_ocred": scanned_pages_count,
                "embedded_images_found": embedded_images_count,
                "file_type": "pdf"
            }
        )

    @classmethod
    async def load_document_async(cls, file_path: Path) -> AnnotatedContextObject:
        """Asynchronously loads documents, handling hybrid PDF OCR and standard files."""
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return await cls.parse_pdf_hybrid(file_path)

        # Word Docx handling
        if ext == ".docx":
            try:
                import docx
                doc = docx.Document(str(file_path))
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                content = "\n\n".join(paragraphs)
                return AnnotatedContextObject(
                    type="document_context",
                    source=str(file_path),
                    method="DOC_LOADER",
                    confidence=1.0,
                    content=content,
                    metadata={"paragraph_count": len(paragraphs), "file_type": "docx"}
                )
            except Exception as e:
                return AnnotatedContextObject(
                    type="document_context",
                    source=str(file_path),
                    method="DOC_LOADER",
                    confidence=0.0,
                    content=f"[Error parsing DOCX: {str(e)}]"
                )

        # Standard plain text / code files
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return AnnotatedContextObject(
                type="document_context",
                source=str(file_path),
                method="DOC_LOADER",
                confidence=1.0,
                content=content,
                metadata={"file_type": ext.lstrip(".")}
            )
        except Exception as e:
            return AnnotatedContextObject(
                type="document_context",
                source=str(file_path),
                method="DOC_LOADER",
                confidence=0.0,
                content=f"[Error reading file: {str(e)}]"
            )

    @classmethod
    def load_document(cls, file_path: Path) -> AnnotatedContextObject:
        """Synchronous wrapper for load_document_async."""
        return asyncio.run(cls.load_document_async(file_path))

    @staticmethod
    def encode_image_base64(file_path: Path) -> str:
        """Encodes an image to a base64 string for vision model consumption."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
