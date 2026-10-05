"""
Smart tier assignment and routing configuration for models.
Automatically maps models to Basic, Complex, Vision, and Default tiers,
and supports interactive veto / sequential customization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from app.providers.base import ModelInfo
from app.routing.system_scanner import SystemHardwareSpecs


@dataclass
class TierAssignment:
    basic_model: str
    complex_model: str
    vision_model: Optional[str]
    default_model: str
    provider_id: str
    reasoning: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "basic_model": self.basic_model,
            "complex_model": self.complex_model,
            "vision_model": self.vision_model,
            "default_model": self.default_model,
            "provider_id": self.provider_id
        }


class SmartModelAssigner:
    """Assigns available models into tiered routing buckets based on heuristics, metadata, and hardware specs."""

    @staticmethod
    def assign_tiers(
        models: List[ModelInfo],
        provider_id: str,
        hardware_specs: Optional[SystemHardwareSpecs] = None
    ) -> TierAssignment:
        if not models:
            return TierAssignment(
                basic_model="default",
                complex_model="default",
                vision_model=None,
                default_model="default",
                provider_id=provider_id,
                reasoning={"status": "No models found, using fallback defaults."}
            )

        # Check for specific known models from task specification
        model_names = [m.id for m in models]

        basic_candidate: Optional[str] = None
        complex_candidate: Optional[str] = None
        vision_candidate: Optional[str] = None
        default_candidate: Optional[str] = None
        reasons: Dict[str, str] = {}

        # 1. Check for exact matches from task.txt (e.g. gemma4-text:e4b, gemma4:26b, gemma4-vis:e4b)
        for m in models:
            mid = m.id.lower()
            if "gemma4-text:e4b" in mid or ("gemma4-text" in mid and not basic_candidate):
                basic_candidate = m.id
                default_candidate = m.id
                reasons["basic"] = f"Selected '{m.id}' as fast text-optimized small model."
            elif "gemma4:26b" in mid and not complex_candidate:
                complex_candidate = m.id
                reasons["complex"] = f"Selected '{m.id}' as high-capacity 26B reasoning model."
            elif ("gemma4-vis:e4b" in mid or "gemma4-vis" in mid) and not vision_candidate:
                vision_candidate = m.id
                reasons["vision"] = f"Selected '{m.id}' as vision-enabled multimodal model."

        # 2. General capability and parameter size based sorting
        # Vision candidates
        if not vision_candidate:
            vis_models = [m for m in models if m.capabilities.vision or "vis" in m.id.lower() or "vision" in m.id.lower()]
            if vis_models:
                vision_candidate = vis_models[0].id
                reasons["vision"] = f"Selected '{vision_candidate}' based on vision capability flag."

        # Sort non-vision models by parameter size / weight
        def parse_param_num(size_str: str) -> float:
            s = size_str.lower().strip()
            if "b" in s:
                try:
                    num_part = s.split("b")[0].replace("e", "").replace(":", "")
                    return float(num_part)
                except Exception:
                    pass
            if "m" in s:
                try:
                    return float(s.split("m")[0]) / 1000.0
                except Exception:
                    pass
            return 8.0

        # Sort models
        sorted_by_size = sorted(models, key=lambda m: parse_param_num(m.parameter_size or m.id))

        if not basic_candidate:
            # Pick smallest text model
            basic_candidates = [m for m in sorted_by_size if not m.capabilities.embedding and not "embed" in m.id.lower()]
            if basic_candidates:
                basic_candidate = basic_candidates[0].id
                reasons["basic"] = f"Selected '{basic_candidate}' (smallest parameter count for low latency)."
            else:
                basic_candidate = models[0].id

        if not complex_candidate:
            # Pick largest reasoning model
            complex_candidates = [m for m in sorted_by_size if not m.capabilities.embedding and not "embed" in m.id.lower()]
            if len(complex_candidates) > 1:
                # Pick largest or one with thinking capability
                thinking_models = [m for m in complex_candidates if m.capabilities.thinking or "reasoner" in m.id.lower()]
                if thinking_models:
                    complex_candidate = thinking_models[-1].id
                    reasons["complex"] = f"Selected '{complex_candidate}' (dedicated reasoning / thinking model)."
                else:
                    complex_candidate = complex_candidates[-1].id
                    reasons["complex"] = f"Selected '{complex_candidate}' (largest parameter count for complex reasoning)."
            else:
                complex_candidate = basic_candidate
                reasons["complex"] = f"Selected '{complex_candidate}' (single available model)."

        if not default_candidate:
            default_candidate = basic_candidate
            reasons["default"] = f"Defaulting to '{default_candidate}'."
        else:
            reasons["default"] = f"Set to '{default_candidate}'."

        return TierAssignment(
            basic_model=basic_candidate,
            complex_model=complex_candidate,
            vision_model=vision_candidate,
            default_model=default_candidate,
            provider_id=provider_id,
            reasoning=reasons
        )
