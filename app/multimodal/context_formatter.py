"""
Annotated Context Object formatter.
Wraps extracted media and documents in structured metadata to prevent context drift.
"""

from dataclasses import dataclass, asdict
import json
from typing import Any, Dict, Optional


@dataclass
class AnnotatedContextObject:
    type: str = "media_context"  # "media_context", "document_context", "system_context"
    source: str = ""
    method: str = "OCR"  # "OCR", "Vision", "DOC_LOADER", "RAG"
    confidence: float = 1.0
    content: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.metadata is None:
            del d["metadata"]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_prompt_block(self) -> str:
        """Formats the context object as an annotated XML/JSON block for LLM consumption."""
        return (
            f"[ATTACHED CONTEXT: {self.source}]\n"
            f"Method: {self.method} | Confidence: {self.confidence:.2f}\n"
            f"--- BEGIN CONTENT ---\n"
            f"{self.content}\n"
            f"--- END CONTENT ---"
        )
