"""
Environment and API key manager.
Handles loading, parsing, validating, and updating the .env file,
including detecting incomplete/placeholder keys and checking provider configurations.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from app.config.settings import SUPPORTED_PROVIDERS, ProviderMeta, DEFAULT_ENV_PATH


class EnvStatus:
    def __init__(
        self,
        provider_id: str,
        configured: bool,
        raw_value: Optional[str] = None,
        is_empty_or_placeholder: bool = False,
        error_message: Optional[str] = None
    ):
        self.provider_id = provider_id
        self.configured = configured
        self.raw_value = raw_value
        self.is_empty_or_placeholder = is_empty_or_placeholder
        self.error_message = error_message

    @property
    def masked_value(self) -> str:
        if not self.raw_value:
            return "(not set)"
        if self.is_empty_or_placeholder:
            return f"(empty/incomplete: '{self.raw_value}')"
        if len(self.raw_value) <= 8:
            return "****"
        return f"{self.raw_value[:4]}...{self.raw_value[-4:]}"


class EnvManager:
    """Manages reading and writing environment variables and .env configuration."""

    def __init__(self, env_path: Optional[Path] = None):
        if env_path is None:
            # Look in gen/.env or current working directory .env
            cwd_env = Path.cwd() / ".env"
            if cwd_env.exists():
                self.env_path = cwd_env
            else:
                self.env_path = DEFAULT_ENV_PATH
        else:
            self.env_path = Path(env_path)
        
        self.load_env()

    def load_env(self) -> Dict[str, str]:
        """Loads environment variables from the .env file if it exists."""
        env_vars: Dict[str, str] = {}
        if self.env_path.exists():
            with open(self.env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        env_vars[key] = val
                        os.environ[key] = val
        return env_vars

    def is_placeholder(self, value: Optional[str]) -> bool:
        """Determines if a key value is an empty string, placeholder, or incomplete."""
        if value is None:
            return False
        val_clean = value.strip().lower()
        if val_clean == "":
            return True
        placeholders = [
            "your_api_key_here", "your-api-key", "your_api_key",
            "sk-...", "enter_key_here", "<empty string or incomplete key>",
            "none", "null", "undefined", "insert_key", "change_me"
        ]
        if val_clean in placeholders or val_clean.startswith("<") and val_clean.endswith(">"):
            return True
        # Incomplete keys check (e.g. less than 5 characters for an API key)
        if len(val_clean) < 5 and not val_clean.startswith("http"):
            return True
        return False

    def get_provider_status(self, provider_id: str) -> EnvStatus:
        """Checks the status of a specific provider's API key or base URL."""
        if provider_id not in SUPPORTED_PROVIDERS:
            return EnvStatus(provider_id=provider_id, configured=False, error_message="Unknown provider")

        meta = SUPPORTED_PROVIDERS[provider_id]
        raw_val = os.getenv(meta.env_key)

        if raw_val is None and self.env_path.exists():
            # Check directly in the file
            env_vars = self.load_env()
            raw_val = env_vars.get(meta.env_key)

        if meta.is_local and not meta.requires_key:
            # Local provider (e.g. Ollama): configured by default with base URL fallback
            base_url = raw_val or meta.default_base_url
            return EnvStatus(provider_id=provider_id, configured=True, raw_value=base_url)

        if raw_val is None:
            return EnvStatus(
                provider_id=provider_id,
                configured=False,
                error_message=f"No {meta.env_key} found in .env or environment"
            )

        if self.is_placeholder(raw_val):
            return EnvStatus(
                provider_id=provider_id,
                configured=False,
                raw_value=raw_val,
                is_empty_or_placeholder=True,
                error_message=f"Found {meta.env_key} but value is empty or a placeholder ('{raw_val}')"
            )

        return EnvStatus(provider_id=provider_id, configured=True, raw_value=raw_val)

    def scan_all_providers(self) -> Dict[str, EnvStatus]:
        """Scans all supported providers and returns their configuration statuses."""
        self.load_env()
        statuses: Dict[str, EnvStatus] = {}
        for pid in SUPPORTED_PROVIDERS:
            statuses[pid] = self.get_provider_status(pid)
        return statuses

    def get_key_for_provider(self, provider_id: str) -> Optional[str]:
        """Retrieves valid API key for a provider, or None if invalid/placeholder."""
        status = self.get_provider_status(provider_id)
        if status.configured and not status.is_empty_or_placeholder:
            return status.raw_value
        return None

    def get_endpoint_for_provider(self, provider_id: str) -> str:
        """Retrieves base URL endpoint for a provider."""
        meta = SUPPORTED_PROVIDERS.get(provider_id)
        if not meta:
            return ""
        if meta.is_local:
            status = self.get_provider_status(provider_id)
            if status.configured and status.raw_value:
                return status.raw_value
            return meta.default_base_url
        return meta.default_base_url

    def update_key(self, provider_id: str, key_or_url: str) -> Tuple[bool, str]:
        """
        Updates or adds an API key / URL to .env file and current os.environ.
        Supports provider tags like 'openai', 'groq', 'ollama', etc.
        """
        provider_id_clean = provider_id.lower().strip()
        meta: Optional[ProviderMeta] = None
        for pid, pmeta in SUPPORTED_PROVIDERS.items():
            if pid == provider_id_clean or pmeta.env_key.lower() == provider_id_clean:
                meta = pmeta
                break

        if not meta:
            # Check if user entered direct env variable name
            env_var_name = provider_id.strip().upper()
        else:
            env_var_name = meta.env_key

        key_or_url_clean = key_or_url.strip().strip("'\"")

        # Update os.environ
        os.environ[env_var_name] = key_or_url_clean

        # Update or create .env file
        lines: List[str] = []
        found = False
        if self.env_path.exists():
            with open(self.env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        new_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(f"{env_var_name}=") or stripped.startswith(f"{env_var_name} ="):
                new_lines.append(f'{env_var_name}="{key_or_url_clean}"\n')
                found = True
            else:
                new_lines.append(line)

        if not found:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines.append("\n")
            new_lines.append(f'{env_var_name}="{key_or_url_clean}"\n')

        # Ensure parent directory exists
        self.env_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        return True, f"Successfully saved {env_var_name} to {self.env_path.name}"
