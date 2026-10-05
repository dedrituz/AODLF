"""
Cloud and OpenAI-compatible inference providers (OpenAI, Groq, DeepSeek, Anthropic, OpenRouter, Custom).
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import httpx

from app.config.settings import MODEL_PRICING, TIMEOUT_CLOUD_SECONDS
from app.providers.base import (
    BaseProvider,
    Message,
    ModelCapabilities,
    ModelInfo,
    StreamChunk,
)


class OpenAICompatibleProvider(BaseProvider):
    """Generic provider for any OpenAI-compatible endpoint (OpenAI, Groq, DeepSeek, OpenRouter, Custom)."""

    def __init__(
        self,
        provider_id: str,
        base_url: str,
        api_key: Optional[str] = None,
        default_models: Optional[List[ModelInfo]] = None
    ):
        super().__init__(provider_id=provider_id, base_url=base_url, api_key=api_key)
        self.default_models = default_models or []

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def test_connection(self) -> Tuple[bool, str]:
        """Tests connectivity by querying /models endpoint."""
        url = f"{self.base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url, headers=self._get_headers())
                if res.status_code == 200:
                    data = res.json()
                    models = data.get("data", [])
                    return True, f"Connected to {self.provider_id.title()} ({len(models)} models found)"
                elif res.status_code == 401:
                    return False, f"Authentication failed for {self.provider_id.title()}: Invalid or expired API key"
                else:
                    return False, f"{self.provider_id.title()} API error (HTTP {res.status_code}): {res.text[:120]}"
        except httpx.ConnectError:
            return False, f"Could not connect to {self.provider_id.title()} at {self.base_url}"
        except httpx.TimeoutException:
            return False, f"Connection to {self.provider_id.title()} timed out."
        except Exception as e:
            return False, f"Error testing {self.provider_id.title()}: {str(e)}"

    async def list_models(self) -> List[ModelInfo]:
        """Lists models from /models endpoint, augmented with pricing and capability metadata."""
        url = f"{self.base_url}/models"
        models: List[ModelInfo] = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url, headers=self._get_headers())
                if res.status_code == 200:
                    data = res.json()
                    for item in data.get("data", []):
                        mid = item.get("id", "")
                        # Filter out audio/embedding/tts only models
                        mid_lower = mid.lower()
                        if any(x in mid_lower for x in ["embed", "embedding", "whisper", "tts", "dall-e", "moderation"]):
                            continue

                        has_vision = any(x in mid_lower for x in ["vision", "4o", "claude", "vl", "visual"])
                        has_thinking = any(x in mid_lower for x in ["o1", "o3", "reasoner", "r1", "thinking"])
                        pricing = MODEL_PRICING.get(mid, {"input": 0.0, "output": 0.0})

                        models.append(
                            ModelInfo(
                                id=mid,
                                name=mid,
                                provider_id=self.provider_id,
                                family=self.provider_id,
                                parameter_size="Cloud API",
                                context_length=128000,
                                capabilities=ModelCapabilities(
                                    completion=True,
                                    vision=has_vision,
                                    tools=True,
                                    thinking=has_thinking
                                ),
                                cost_per_1m_input=pricing.get("input", 0.0),
                                cost_per_1m_output=pricing.get("output", 0.0),
                                is_local=False,
                                description=f"{self.provider_id.title()} Model"
                            )
                        )
        except Exception:
            pass

        # If live fetch returned nothing, fallback to predefined known models
        if not models and self.default_models:
            return self.default_models

        return models

    async def warmup_model(self, model: str, timeout_seconds: int = TIMEOUT_CLOUD_SECONDS) -> Tuple[bool, str]:
        """Tests cloud model readiness with a single token generation."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        }
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=float(timeout_seconds)) as client:
                res = await client.post(url, headers=self._get_headers(), json=payload)
                elapsed = time.time() - start_time
                if res.status_code == 200:
                    return True, f"Model '{model}' verified in {elapsed:.2f}s"
                return False, f"Failed to connect to model '{model}' (HTTP {res.status_code}): {res.text[:120]}"
        except httpx.TimeoutException:
            return False, f"Cloud provider timed out after {timeout_seconds}s connecting to '{model}'."
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    async def generate_stream(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[StreamChunk]:
        """Streams OpenAI-compatible chat completions."""
        url = f"{self.base_url}/chat/completions"

        formatted_messages = []
        for m in messages:
            if m.images:
                # Vision multimodal format
                content_parts: List[Dict[str, Any]] = [{"type": "text", "text": m.content}]
                for img_b64 in m.images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}" if not img_b64.startswith("data:") else img_b64}
                    })
                formatted_messages.append({"role": m.role, "content": content_parts})
            else:
                formatted_messages.append({"role": m.role, "content": m.content})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "stream": True,
            "temperature": temperature
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        yield StreamChunk(event_type="status", content=f"Requesting {self.provider_id.title()} ({model})...")

        start_time = time.time()
        token_count = 0

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, headers=self._get_headers(), json=payload) as response:
                    if response.status_code != 200:
                        err_bytes = await response.aread()
                        yield StreamChunk(
                            event_type="error",
                            content=f"{self.provider_id.title()} Error (HTTP {response.status_code}): {err_bytes.decode('utf-8', errors='ignore')}"
                        )
                        return

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})

                        # DeepSeek or OpenAI reasoning_content
                        reasoning = delta.get("reasoning_content") or delta.get("thinking")
                        if reasoning:
                            yield StreamChunk(event_type="thought_delta", content=reasoning)

                        text_delta = delta.get("content", "")
                        if text_delta:
                            token_count += 1
                            yield StreamChunk(event_type="content_delta", content=text_delta)

                    elapsed = time.time() - start_time
                    tokens_per_sec = (token_count / elapsed) if elapsed > 0 else 0

                    yield StreamChunk(
                        event_type="metadata",
                        content="",
                        metadata={
                            "model": model,
                            "provider": self.provider_id,
                            "elapsed_seconds": round(elapsed, 2),
                            "eval_tokens": token_count,
                            "tokens_per_second": round(tokens_per_sec, 1),
                            "done": True
                        }
                    )
        except Exception as e:
            yield StreamChunk(event_type="error", content=f"Cloud stream error: {str(e)}")


def create_cloud_provider(provider_id: str, base_url: str, api_key: Optional[str] = None) -> BaseProvider:
    """Factory creating appropriate provider instances with default known models."""
    default_models_map = {
        "openai": [
            ModelInfo(
                id="gpt-4o",
                name="gpt-4o",
                provider_id="openai",
                family="openai",
                parameter_size="Flagship",
                context_length=128000,
                capabilities=ModelCapabilities(completion=True, vision=True, tools=True),
                cost_per_1m_input=2.50,
                cost_per_1m_output=10.00,
                description="OpenAI flagship multimodal intelligence"
            ),
            ModelInfo(
                id="gpt-4o-mini",
                name="gpt-4o-mini",
                provider_id="openai",
                family="openai",
                parameter_size="Mini",
                context_length=128000,
                capabilities=ModelCapabilities(completion=True, vision=True, tools=True),
                cost_per_1m_input=0.15,
                cost_per_1m_output=0.60,
                description="Fast and cheap everyday model"
            ),
            ModelInfo(
                id="o3-mini",
                name="o3-mini",
                provider_id="openai",
                family="openai",
                parameter_size="Reasoning",
                context_length=200000,
                capabilities=ModelCapabilities(completion=True, thinking=True),
                cost_per_1m_input=1.10,
                cost_per_1m_output=4.40,
                description="Cost-effective reasoning model"
            )
        ],
        "groq": [
            ModelInfo(
                id="llama-3.3-70b-versatile",
                name="llama-3.3-70b-versatile",
                provider_id="groq",
                family="llama3",
                parameter_size="70B",
                context_length=128000,
                capabilities=ModelCapabilities(completion=True, tools=True),
                cost_per_1m_input=0.59,
                cost_per_1m_output=0.79,
                description="Fast high-intelligence reasoning on Groq LPU"
            ),
            ModelInfo(
                id="llama-3.1-8b-instant",
                name="llama-3.1-8b-instant",
                provider_id="groq",
                family="llama3",
                parameter_size="8B",
                context_length=128000,
                capabilities=ModelCapabilities(completion=True),
                cost_per_1m_input=0.05,
                cost_per_1m_output=0.08,
                description="Near-instant responses for simple prompts"
            )
        ],
        "deepseek": [
            ModelInfo(
                id="deepseek-chat",
                name="deepseek-chat",
                provider_id="deepseek",
                family="deepseek",
                parameter_size="671B MoE",
                context_length=64000,
                capabilities=ModelCapabilities(completion=True, tools=True),
                cost_per_1m_input=0.14,
                cost_per_1m_output=0.28,
                description="DeepSeek-V3 general reasoning"
            ),
            ModelInfo(
                id="deepseek-reasoner",
                name="deepseek-reasoner",
                provider_id="deepseek",
                family="deepseek",
                parameter_size="671B MoE",
                context_length=64000,
                capabilities=ModelCapabilities(completion=True, thinking=True),
                cost_per_1m_input=0.55,
                cost_per_1m_output=2.19,
                description="DeepSeek-R1 deep chain-of-thought model"
            )
        ]
    }

    defaults = default_models_map.get(provider_id, [])
    return OpenAICompatibleProvider(
        provider_id=provider_id,
        base_url=base_url,
        api_key=api_key,
        default_models=defaults
    )
