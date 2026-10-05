"""
Ollama Provider integration for local LLM inference.
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import httpx

from app.config.settings import TIMEOUT_LOCAL_SECONDS
from app.providers.base import (
    BaseProvider,
    Message,
    ModelCapabilities,
    ModelInfo,
    StreamChunk,
)


class OllamaProvider(BaseProvider):
    """Provider implementation for local Ollama instance."""

    def __init__(self, base_url: str = "http://localhost:11434", api_key: Optional[str] = None):
        super().__init__(provider_id="ollama", base_url=base_url, api_key=api_key)

    async def test_connection(self) -> Tuple[bool, str]:
        """Tests connectivity by querying the Ollama /api/version or /api/tags endpoint."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    model_count = len(data.get("models", []))
                    return True, f"Connected to Ollama ({model_count} models found at {self.base_url})"
                return False, f"Ollama returned HTTP status {res.status_code}"
        except httpx.ConnectError:
            return False, f"Could not connect to Ollama at {self.base_url}. Is Ollama running?"
        except httpx.TimeoutException:
            return False, f"Connection to Ollama at {self.base_url} timed out."
        except Exception as e:
            return False, f"Error connecting to Ollama: {str(e)}"

    async def list_models(self) -> List[ModelInfo]:
        """
        Lists available models from Ollama.
        Filters out embedding-only models (*embed*, *embedding*).
        """
        url = f"{self.base_url}/api/tags"
        models: List[ModelInfo] = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url)
                if res.status_code != 200:
                    return []
                data = res.json()
                for item in data.get("models", []):
                    name = item.get("name", "")
                    # Note in task.txt: Models with "embed" or "embedding" are text embedding models and not LLMs
                    is_embed = "embed" in name.lower() or "embedding" in name.lower()
                    if is_embed:
                        continue

                    details = item.get("details", {})
                    caps_list = item.get("capabilities", [])
                    has_vision = "vision" in caps_list or "vis" in name.lower() or "vl" in name.lower()
                    has_thinking = "thinking" in caps_list or "reason" in name.lower() or "r1" in name.lower()
                    has_tools = "tools" in caps_list

                    param_size = details.get("parameter_size", "Unknown")
                    if param_size == "Unknown" and ":" in name:
                        # e.g. gemma4:26b -> 26B
                        tag = name.split(":")[-1]
                        if any(tag.endswith(x) for x in ["b", "m", "e4b"]):
                            param_size = tag.upper()

                    context_len = details.get("context_length", 8192)
                    quant = details.get("quantization_level", "N/A")
                    family = details.get("family", "gemma")

                    capabilities = ModelCapabilities(
                        completion=True,
                        vision=has_vision,
                        tools=has_tools,
                        thinking=has_thinking,
                        audio="audio" in caps_list,
                        embedding=False
                    )

                    models.append(
                        ModelInfo(
                            id=name,
                            name=name,
                            provider_id="ollama",
                            family=family,
                            parameter_size=param_size,
                            context_length=context_len,
                            quantization=quant,
                            capabilities=capabilities,
                            cost_per_1m_input=0.0,
                            cost_per_1m_output=0.0,
                            is_local=True,
                            description=f"Local Ollama model ({param_size}, quant: {quant}, ctx: {context_len:,})"
                        )
                    )
        except Exception:
            return []
        return models

    async def warmup_model(self, model: str, timeout_seconds: int = TIMEOUT_LOCAL_SECONDS) -> Tuple[bool, str]:
        """
        Loads the model into memory and verifies it is ready.
        Times out after timeout_seconds (default 3 minutes for local backend).
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "options": {"num_predict": 1}
        }
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=float(timeout_seconds)) as client:
                res = await client.post(url, json=payload)
                elapsed = time.time() - start_time
                if res.status_code == 200:
                    return True, f"Model '{model}' loaded successfully in {elapsed:.1f}s"
                return False, f"Model load returned HTTP status {res.status_code}: {res.text}"
        except httpx.TimeoutException:
            return False, f"Model load timed out after {timeout_seconds} seconds ({timeout_seconds//60} min)."
        except Exception as e:
            return False, f"Failed to warm up model '{model}': {str(e)}"

    async def generate_stream(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[StreamChunk]:
        """
        Streams completions from Ollama with real-time thought_delta and content_delta events.
        """
        url = f"{self.base_url}/api/chat"

        formatted_messages = []
        for m in messages:
            msg_dict: Dict[str, Any] = {"role": m.role, "content": m.content}
            if m.images:
                msg_dict["images"] = m.images
            formatted_messages.append(msg_dict)

        options: Dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens

        payload = {
            "model": model,
            "messages": formatted_messages,
            "stream": True,
            "options": options
        }

        yield StreamChunk(event_type="status", content=f"Connecting to model '{model}'...")

        start_time = time.time()
        total_eval_tokens = 0
        in_thinking_tag = False

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        yield StreamChunk(
                            event_type="error",
                            content=f"Ollama API Error ({response.status_code}): {err_text.decode('utf-8', errors='ignore')}"
                        )
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except Exception:
                            continue

                        # Check for thinking field or content
                        msg = data.get("message", {})
                        thinking_content = msg.get("thinking")
                        text_content = msg.get("content", "")

                        if thinking_content:
                            yield StreamChunk(event_type="thought_delta", content=thinking_content)

                        if text_content:
                            # Also check for inline <thought> or <think> tags if model produces them in content
                            if "<think>" in text_content or "<thought>" in text_content:
                                in_thinking_tag = True
                                cleaned = text_content.replace("<think>", "").replace("<thought>", "")
                                if cleaned:
                                    yield StreamChunk(event_type="thought_delta", content=cleaned)
                            elif "</think>" in text_content or "</thought>" in text_content:
                                in_thinking_tag = False
                                parts = text_content.replace("</think>", "</thought>").split("</thought>")
                                if parts[0]:
                                    yield StreamChunk(event_type="thought_delta", content=parts[0])
                                if len(parts) > 1 and parts[1]:
                                    yield StreamChunk(event_type="content_delta", content=parts[1])
                            else:
                                if in_thinking_tag:
                                    yield StreamChunk(event_type="thought_delta", content=text_content)
                                else:
                                    yield StreamChunk(event_type="content_delta", content=text_content)

                        if data.get("done", False):
                            elapsed = time.time() - start_time
                            prompt_tokens = data.get("prompt_eval_count", 0)
                            eval_tokens = data.get("eval_count", 0)
                            eval_duration_ns = data.get("eval_duration", 0)
                            tokens_per_sec = (eval_tokens / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else (eval_tokens / elapsed if elapsed > 0 else 0)

                            yield StreamChunk(
                                event_type="metadata",
                                content="",
                                metadata={
                                    "model": model,
                                    "provider": "ollama",
                                    "elapsed_seconds": round(elapsed, 2),
                                    "prompt_tokens": prompt_tokens,
                                    "eval_tokens": eval_tokens,
                                    "tokens_per_second": round(tokens_per_sec, 1),
                                    "done": True
                                }
                            )
        except httpx.ConnectError:
            yield StreamChunk(event_type="error", content=f"Connection lost to Ollama at {self.base_url}")
        except Exception as e:
            yield StreamChunk(event_type="error", content=f"Streaming error: {str(e)}")
