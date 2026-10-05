"""
Media Router for Multimodal Decision Logic (OCR vs Vision).
Calculates text density ratio and performs semantic prompt analysis to decide
between lightweight OCR or a full Multimodal Vision model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageFilter, ImageStat

from app.config.settings import TEXT_DENSITY_OCR_THRESHOLD
from app.multimodal.ocr_engine import OCREngine


@dataclass
class MediaRoutingResult:
    method: str  # "OCR" or "Vision"
    text_density: float
    confidence: float
    reason: str
    extracted_text: Optional[str] = None
    needs_vision_model: bool = False


class MediaRouter:
    """Decides whether an uploaded image should be preprocessed via OCR or sent directly to a Vision model."""

    OCR_PROMPT_KEYWORDS = {
        "read", "transcribe", "ocr", "extract text", "what does it say",
        "copy the text", "document", "receipt", "invoice", "article",
        "paper", "text in image", "code in screenshot", "words"
    }

    VISION_PROMPT_KEYWORDS = {
        "what is in", "describe", "identify", "color", "colors", "drawing",
        "diagram", "chart", "graph", "plot", "picture of", "scene",
        "where is", "how many", "look like", "visualize", "style", "facial",
        "detect", "explain the flow", "architecture diagram", "ui design"
    }

    def __init__(self, ocr_engine: Optional[OCREngine] = None):
        self.ocr_engine = ocr_engine or OCREngine()

    def estimate_text_density(self, image_path: Path) -> float:
        """
        Estimates the ratio of high-contrast text/edge regions to total image area.
        Returns a float between 0.0 (pure visual/gradient) and 1.0 (dense text document).
        """
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale
                gray = img.convert("L")
                w, h = gray.size
                if w == 0 or h == 0:
                    return 0.0

                # Edge detection filter
                edges = gray.filter(ImageFilter.FIND_EDGES)
                # Binarize edges
                threshold = 50
                bin_edges = edges.point(lambda p: 255 if p > threshold else 0)

                stat = ImageStat.Stat(bin_edges)
                edge_pixel_ratio = (stat.mean[0] / 255.0)

                # Scaling factor: documents typically have 0.05 - 0.25 raw edge density
                # which maps to high text density (0.4 - 0.9)
                density = min(1.0, edge_pixel_ratio * 3.5)
                return round(density, 3)
        except Exception:
            return 0.20

    async def decide_route(
        self,
        image_path: Path,
        user_prompt: str,
        has_vision_model: bool = True
    ) -> MediaRoutingResult:
        """
        Evaluates image and user prompt to choose between OCR or Vision pipeline.
        """
        # If no vision model is available in the current session, force OCR
        if not has_vision_model:
            extracted_text, conf = await self.ocr_engine.extract_text(image_path)
            return MediaRoutingResult(
                method="OCR",
                text_density=1.0,
                confidence=conf,
                reason="No vision model configured for session; automatically falling back to OCR pipeline.",
                extracted_text=extracted_text,
                needs_vision_model=False
            )

        prompt_clean = user_prompt.lower()
        density = self.estimate_text_density(image_path)

        # Semantic Analysis on prompt
        ocr_kw_count = sum(1 for kw in self.OCR_PROMPT_KEYWORDS if kw in prompt_clean)
        vision_kw_count = sum(1 for kw in self.VISION_PROMPT_KEYWORDS if kw in prompt_clean)

        # High Text Density (> 30%) with text-oriented prompt -> OCR Path
        if density >= TEXT_DENSITY_OCR_THRESHOLD and ocr_kw_count >= vision_kw_count:
            extracted_text, conf = await self.ocr_engine.extract_text(image_path)
            return MediaRoutingResult(
                method="OCR",
                text_density=density,
                confidence=conf,
                reason=f"High text density ({density:.1%}) & document inquiry -> Efficient OCR Path.",
                extracted_text=extracted_text,
                needs_vision_model=False
            )

        # Explicit vision question
        if vision_kw_count > ocr_kw_count:
            return MediaRoutingResult(
                method="Vision",
                text_density=density,
                confidence=0.92,
                reason=f"Visual / diagram inquiry detected -> Multimodal Vision Path.",
                needs_vision_model=True
            )

        # Low density (< 30%)
        if density < TEXT_DENSITY_OCR_THRESHOLD:
            return MediaRoutingResult(
                method="Vision",
                text_density=density,
                confidence=0.85,
                reason=f"Low text density ({density:.1%}) -> Multimodal Vision Path.",
                needs_vision_model=True
            )

        # Indecisive / low probability score -> Default to Vision model if available
        return MediaRoutingResult(
            method="Vision",
            text_density=density,
            confidence=0.60,
            reason="Ambiguous image modality; defaulting to Multimodal Vision model for maximum accuracy.",
            needs_vision_model=True
        )
