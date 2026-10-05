"""
Enhanced Tiered Complexity Router with Semantic Classification.
Implements Heuristic (Tier 1), Semantic Classifier (Tier 2), and Execution (Tier 3).
"""

from dataclasses import dataclass, field
import re
from typing import List, Optional, Tuple, Any
from app.routing.smart_assignment import TierAssignment
from app.providers.base import BaseProvider, Message, StreamChunk

@dataclass
class RoutingDecision:
    tier: str  # "basic", "complex", "vision", "override"
    selected_model: str
    confidence: float
    reason: str
    is_multimodal: bool = False
    semantic_check_performed: bool = True

class IntelligentRouter:
    """
    Evaluates prompt complexity using a combination of fast heuristics and 
    lightweight semantic classification to select the optimal model tier.
    """
    
    COMPLEXITY_KEYWORDS = {
        "prove", "derive", "calculate", "algorithm", "architecture",
        "implement", "optimize", "refactor", "debug", "explain the mathematical",
        "differential", "integral", "theorem", "step-by-step reasoning",
        "formal proof", "distributed system", "quantum", "concurrency",
        "critique", "evaluate trade-offs", "write a complete program",
        "design a system", "benchmark", "analysis", "compare and contrast",
        "data structure", "recursive", "complexity", "asymptotic", "o(n)"
    }
    
    SIMPLE_KEYWORDS = {
        "hello", "hi", "hey", "who are you", "what is", "define",
        "translate", "synonym", "antonym", "capital of", "short summary",
        "spelling", "grammar check", "format as json", "tell me a joke",
        "what color", "how many days", "convert", "greeting"
    }
    
    CODE_MARKERS = {
        "```", "def ", "class ", "import ", "function", "const ", "let ", "var ",
        "return ", "if __name__", "select ", "from ", "where ", "join ", "async ",
        "await ", "public static", "void ", "struct ", "impl "
    }

    def __init__(self, tier_assignment: TierAssignment, override_model: Optional[str] = None):
        self.tier_assignment = tier_assignment
        self.override_model = override_model

    def set_override_model(self, model_name: Optional[str]) -> None:
        self.override_model = model_name

    def clear_override(self) -> None:
        self.override_model = None

    async def route_prompt(
        self,
        prompt: str,
        provider: BaseProvider,
        classification_model: Optional[str] = None,
        has_image: bool = False,
        requires_vision_model: bool = False
    ) -> RoutingDecision:
        """
        Routes prompts to the optimal tier. Uses async semantic classification for ambiguous cases.
        """
        # 1. Manual User Override
        if self.override_model:
            return RoutingDecision(
                tier="override",
                selected_model=self.override_model,
                confidence=1.0,
                reason=f"Manual user override: {self.override_model}.",
                is_multimodal=has_image
            )

        # 2. Multimodal Path
        if has_image and requires_vision_model:
            target = self.tier_assignment.vision_model or self.tier_assignment.default_model
            return RoutingDecision(
                tier="vision",
                selected_model=target,
                confidence=0.95,
                reason="Multimodal vision requirement detected.",
                is_multimodal=True
            )

        prompt_clean = prompt.lower().strip()
        word_count = len(prompt_clean.split())

        # 3. Tier 1: Rapid Heuristics (Fast Path)
        # A. Simple Query Check
        simple_matches = [kw for kw in self.SIMPLE_KEYWORDS if kw in prompt_clean]
        if simple_matches and word_count < 25 and not any(ck in prompt_clean for ck in self.COMPLEXITY_KEYWORDS):
            return RoutingDecision(
                tier="basic",
                selected_model=self.tier_assignment.basic_model,
                confidence=0.90,
                reason=f"Heuristic: Simple query ({', '.join(simple_matches[:1])})."
            )

        # B. Code/Structure Check
        code_matches = [cm for cm in self.CODE_MARKERS if cm in prompt_clean]
        if code_matches or "```" in prompt:
            return RoutingDecision(
                tier="complex",
                selected_model=self.tier_assignment.complex_model,
                confidence=0.95,
                reason=f"Heuristic: Code/Structured syntax detected ({', '.join(code_matches[:1])})."
            )

        # C. Direct Complex Keyword Check
        complex_matches = [kw for kw in self.COMPLEXITY_KEYWORDS if kw in prompt_clean]
        if complex_matches and word_count > 20:
            return RoutingDecision(
                tier="complex",
                selected_model=self.tier_assignment.complex_model,
                confidence=0.92,
                reason=f"Heuristic: High complexity keywords ({', '.join(complex_matches[:1])})."
            )

        # 4. Tier 2: Semantic Classification (Intelligence Path)
        model_to_use = classification_model or self.tier_assignment.basic_model
        is_complex = await self._semantic_classify(prompt, provider, model_to_use)
        
        if is_complex:
            return RoutingDecision(
                tier="complex",
                selected_model=self.tier_assignment.complex_model,
                confidence=0.98, 
                reason="Semantic Classification: Model determined this is a complex reasoning task.",
                semantic_check_performed=True
            )
        else:
            return RoutingDecision(
                tier="basic",
                selected_model=self.tier_assignment.basic_model,
                confidence=0.98,
                reason="Semantic Classification: Model determined this is a simple task.",
                semantic_check_performed=True
            )

    async def _semantic_classify(self, prompt: str, provider: BaseProvider, model: str) -> bool:
        """Asks the LLM to classify if a prompt is 'complex' or 'simple'."""
        messages = [
            Message(role="system", content="You are a complexity classifier. Respond with exactly one word: 'complex' or 'simple'."),
            Message(role="user", content=f"Classify this user prompt: {prompt}")
        ]
        
        try:
            async for chunk in provider.generate_stream(messages, model=model):
                if chunk.event_type == "content_delta":
                    result = chunk.content.lower().strip()
                    return "complex" in result
        except Exception as e:
            print(f"[Router] Semantic classification failed ({e}), falling back to heuristics.")
            
        return len(prompt.split()) > 40 # Default fallback