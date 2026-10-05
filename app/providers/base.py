"""
Base abstractions for LLM inference providers and models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple


@dataclass
class ModelCapabilities:
    completion: bool = True
    vision: bool = False
    tools: bool = False
    thinking: bool = False
    audio: bool = False
    embedding: bool = False


@dataclass
class ModelInfo:
    id: str
    name: str
    provider_id: str
    family: str = "generic"
    parameter_size: str = "Unknown"
    context_length: int = 8192
    quantization: str = "N/A"
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    is_local: bool = False
    description: str = ""

    @property
    def display_tag(self) -> str:
        tags = []
        if self.capabilities.vision:
            tags.append("Vision")
        if self.capabilities.thinking:
            tags.append("Thinking")
        if self.capabilities.tools:
            tags.append("Tools")
        if self.is_local:
            tags.append("Local/Free")
        else:
            tags.append(f"${self.cost_per_1m_input:.2f}/M in")
        return " | ".join(tags)


@dataclass
class StreamChunk:
    event_type: str  # "status", "thought_delta", "content_delta", "metadata", "error"
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str
    images: Optional[List[str]] = None  # Base64 or local paths for vision models
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """Abstract base class for all inference providers."""

    def __init__(self, provider_id: str, base_url: str, api_key: Optional[str] = None):
        self.provider_id = provider_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    @abstractmethod
    async def test_connection(self) -> Tuple[bool, str]:
        """
        Tests connectivity to the provider endpoint / API.
        Returns (success, message_or_error).
        """
        pass

    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """
        Lists available LLM models from the provider.
        Embedding-only models MUST be filtered out from standard chat models.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[StreamChunk]:
        """
        Streams generation output (thought tokens, content tokens, and final metadata).
        """
        pass

    @abstractmethod
    async def warmup_model(self, model: str, timeout_seconds: int = 30) -> Tuple[bool, str]:
        """
        Warms up or verifies that the model is loaded and ready for inference.
        """
        pass
