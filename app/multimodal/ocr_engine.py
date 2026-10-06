import warnings
import logging
import asyncio
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Optional, Tuple, Iterator
from PIL import Image

# manually suppressing logs from PaddleOCR and other libraries to avoid cluttering stdout
logging.basicConfig(level=logging.ERROR) 
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('paddle').setLevel(logging.ERROR)

class OCREngine:
    """Extracts text from images using a hierarchy of OCR backends."""

    def __init__(self):
        self._paddle_available = False
        try:
            from paddleocr import PaddleOCR
            import paddle.core.ops as ops # major source of logs/errors that pollute stdout
            self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
            self._paddle_available = True
        except ImportError:
            pass

            
        self._easy_available = False
        try:
            import easyocr
            self._easy_reader = easyocr.Reader(['en'], verbose=False)
        except ImportError:
            pass

        self._tesseract_available = False
        try:
            import pytesseract
            self._pytesseract = pytesseract
            self._tesseract_available = True
        except ImportError:
            pass

    async def extract_text(self, image_path: Path, lang: str = "en") -> Tuple[str, float]:
        """Extracts text from an image file using the defined hierarchy."""
        path = Path(image_path)
        if not path.exists():
            return f"[Error: Image file '{path}' does not exist]", 0.0

        try:
            pil_img = Image.open(path)
        except Exception as e:
            return f"[Error loading image: {str(e)}]", 0.0

        # Method 1: PaddleOCR (Primary)
        if self._paddle_available:
            try:
                import numpy as np
                import cv2
                open_cv_image = np.array(pil_img)
                if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
                    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
                elif len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 4:
                    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
                
                # Use silent context for the actual inference as well
                result = self._paddle_ocr.ocr(open_cv_image, cls=True)

                if result and len(result) > 0 and result[0]:
                    texts, max_conf = [], 0.0
                    for res in result[0]:
                        line_text, score = res[1][0], res[1][1]
                        if line_text:
                            texts.append(str(line_text))
                            max_conf = max(max_conf, float(score))
                    
                    combined_text = "\n".join(texts).strip()
                    if combined_text:
                        return combined_text, max_conf
            except Exception:
                pass

        # Method 2: EasyOCR (Secondary)
        if self._easy_available:
            try:
                result = self._easy_reader.readtext(str(path))
                if result:
                    texts, max_conf = [], 0.0
                    for res in result:
                        _, line_text, conf = res
                        if line_text:
                            texts.append(str(line_text))
                            max_conf = max(max_conf, float(conf))
                    combined_text = "\n".join(texts).strip()
                    if combined_text:
                        return combined_text, max_conf
            except Exception:
                pass

        # Method 3: Tesseract (Fallback)
        if self._tesseract_available:
            try:
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, self._pytesseract.image_to_string, pil_img)
                text = text.strip()
                if text:
                    return text, 0.75
            except Exception:
                pass

        return "[OCR Engine: Unreadable/No Text. Route to Vision Model]", 0.0