"""
OCR Engine implementation.
Provides lightweight and reliable OCR using Windows native Media OCR (winocr)
with fallback to pytesseract or pure image text extraction.
"""

import asyncio
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image


class OCREngine:
    """Extracts text from images using native OS APIs or OCR backends."""

    def __init__(self):
        self._winocr_available = False
        try:
            import winocr
            self._winocr_available = True
        except ImportError:
            pass

        self._tesseract_available = False
        try:
            import pytesseract
            self._tesseract_available = True
        except ImportError:
            pass

    async def extract_text(self, image_path: Path, lang: str = "en") -> Tuple[str, float]:
        """
        Extracts text from an image file.
        Returns (extracted_text, confidence_score).
        """
        path = Path(image_path)
        if not path.exists():
            return f"[Error: Image file '{path}' does not exist]", 0.0

        try:
            pil_img = Image.open(path)
        except Exception as e:
            return f"[Error loading image: {str(e)}]", 0.0

        # Method 1: Windows Native OCR (winocr)
        if self._winocr_available:
            try:
                import winocr
                # winocr uses asyncio
                result = await winocr.recognize_pil(pil_img, lang)
                if hasattr(result, "text"):
                    text = result.text.strip()
                    # Calculate simple heuristic confidence
                    confidence = 0.95 if text else 0.50
                    return text if text else "[No text detected in image]", confidence
            except Exception:
                pass

        # Method 2: pytesseract
        if self._tesseract_available:
            try:
                import pytesseract
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, pytesseract.image_to_string, pil_img)
                text = text.strip()
                if text:
                    return text, 0.88
            except Exception:
                pass

        return "[OCR Engine: Image contains visual data or unreadable text. Recommended to route to Vision model]", 0.40
