"""
Settings and constants for the Automated Open-Domain Learning Framework (AODLF) Backend.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ProviderMeta:
    id: str
    name: str
    env_key: str
    default_base_url: str
    is_local: bool = False
    requires_key: bool = True
    description: str = ""


# Supported Providers Registry
SUPPORTED_PROVIDERS: Dict[str, ProviderMeta] = {
    "ollama": ProviderMeta(
        id="ollama",
        name="Ollama (Local Backend)",
        env_key="OLLAMA_BASE_URL",
        default_base_url="http://localhost:11434",
        is_local=True,
        requires_key=False,
        description="Local LLM runner with zero cost, high privacy, and GPU acceleration."
    ),
    "openai": ProviderMeta(
        id="openai",
        name="OpenAI",
        env_key="OPENAI_API_KEY",
        default_base_url="https://api.openai.com/v1",
        is_local=False,
        requires_key=True,
        description="GPT-4o, GPT-4o-mini, o1, o3-mini models."
    ),
    "groq": ProviderMeta(
        id="groq",
        name="Groq",
        env_key="GROQ_API_KEY",
        default_base_url="https://api.groq.com/openai/v1",
        is_local=False,
        requires_key=True,
        description="Ultra-fast LPU inference for Llama 3, Mixtral, and Gemma."
    ),
    "deepseek": ProviderMeta(
        id="deepseek",
        name="DeepSeek",
        env_key="DEEPSEEK_API_KEY",
        default_base_url="https://api.deepseek.com",
        is_local=False,
        requires_key=True,
        description="DeepSeek-V3 and DeepSeek-R1 reasoning models."
    ),
    "anthropic": ProviderMeta(
        id="anthropic",
        name="Anthropic Claude",
        env_key="ANTHROPIC_API_KEY",
        default_base_url="https://api.anthropic.com/v1",
        is_local=False,
        requires_key=True,
        description="Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus."
    ),
    "openrouter": ProviderMeta(
        id="openrouter",
        name="OpenRouter",
        env_key="OPENROUTER_API_KEY",
        default_base_url="https://openrouter.ai/api/v1",
        is_local=False,
        requires_key=True,
        description="Universal gateway to 200+ AI models."
    ),
    "custom": ProviderMeta(
        id="custom",
        name="Custom OpenAI-Compatible / llama.cpp",
        env_key="CUSTOM_ENDPOINT_URL",
        default_base_url="http://localhost:8080/v1",
        is_local=True,
        requires_key=False,
        description="Custom endpoint (llama.cpp server, vLLM, TGI, LocalAI)."
    )
}

# Known model pricing (input / output per 1M tokens in USD)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o1": {"input": 15.00, "output": 60.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}

# Timeouts in seconds
TIMEOUT_LOCAL_SECONDS = 180  # 3 minutes for local backend model loading
TIMEOUT_CLOUD_SECONDS = 30   # 30 seconds for cloud providers

# Context & routing defaults
DEFAULT_MAX_CONTEXT_TOKENS = 8192
CONTEXT_COMPACT_THRESHOLD_RATIO = 0.70  # Compact when context hits 70% of limit
TEXT_DENSITY_OCR_THRESHOLD = 0.30       # Text density ratio threshold (>30% -> OCR)
DEFAULT_SYSTEM_PROMPT = "do not use markdown in your response."

# Directory configurations
DEFAULT_APP_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_ENV_PATH = DEFAULT_APP_DIR / ".env"
DATA_DIR = DEFAULT_APP_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
