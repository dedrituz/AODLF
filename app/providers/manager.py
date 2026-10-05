"""
Provider Manager orchestrating discovery, connectivity testing, model cataloging, and provider instantiation.
"""

from typing import Dict, List, Optional, Tuple
from app.config.env_manager import EnvManager, EnvStatus
from app.config.settings import SUPPORTED_PROVIDERS, ProviderMeta
from app.providers.base import BaseProvider, ModelInfo
from app.providers.cloud_providers import create_cloud_provider
from app.providers.ollama_provider import OllamaProvider


class ProviderConnectionReport:
    """Detailed initialization and connection status for a provider."""
    def __init__(
        self,
        provider_id: str,
        name: str,
        is_local: bool,
        configured: bool,
        connected: bool,
        status_message: str,
        empty_or_placeholder: bool = False,
        models_count: int = 0
    ):
        self.provider_id = provider_id
        self.name = name
        self.is_local = is_local
        self.configured = configured
        self.connected = connected
        self.status_message = status_message
        self.empty_or_placeholder = empty_or_placeholder
        self.models_count = models_count


class ProviderManager:
    """Manages provider instances, connection verification, and model catalogs."""

    def __init__(self, env_manager: Optional[EnvManager] = None):
        self.env_manager = env_manager or EnvManager()
        self.active_provider: Optional[BaseProvider] = None
        self.provider_cache: Dict[str, BaseProvider] = {}
        self.model_cache: Dict[str, List[ModelInfo]] = {}

    def get_provider(self, provider_id: str) -> Optional[BaseProvider]:
        """Gets or instantiates a provider instance."""
        if provider_id in self.provider_cache:
            return self.provider_cache[provider_id]

        if provider_id not in SUPPORTED_PROVIDERS:
            return None

        meta = SUPPORTED_PROVIDERS[provider_id]
        endpoint = self.env_manager.get_endpoint_for_provider(provider_id)
        api_key = self.env_manager.get_key_for_provider(provider_id)

        if provider_id == "ollama":
            provider = OllamaProvider(base_url=endpoint)
        else:
            provider = create_cloud_provider(
                provider_id=provider_id,
                base_url=endpoint,
                api_key=api_key
            )

        self.provider_cache[provider_id] = provider
        return provider

    async def scan_and_test_all(self) -> List[ProviderConnectionReport]:
        """
        Scans all supported providers, checks .env keys and tests connectivity.
        Produces detailed diagnostic reports according to task specs.
        """
        reports: List[ProviderConnectionReport] = []
        statuses = self.env_manager.scan_all_providers()

        for pid, meta in SUPPORTED_PROVIDERS.items():
            status = statuses.get(pid)
            if not status:
                continue

            if not status.configured and status.is_empty_or_placeholder:
                # Key is present but empty or placeholder
                reports.append(
                    ProviderConnectionReport(
                        provider_id=pid,
                        name=meta.name,
                        is_local=meta.is_local,
                        configured=False,
                        connected=False,
                        empty_or_placeholder=True,
                        status_message=f"Key entry found in .env but was left empty or placeholder: '{status.raw_value}'"
                    )
                )
                continue

            if not status.configured:
                # No key present
                reports.append(
                    ProviderConnectionReport(
                        provider_id=pid,
                        name=meta.name,
                        is_local=meta.is_local,
                        configured=False,
                        connected=False,
                        empty_or_placeholder=False,
                        status_message=f"No {meta.env_key} found"
                    )
                )
                continue

            # Provider is configured; test connection
            provider = self.get_provider(pid)
            if not provider:
                continue

            success, msg = await provider.test_connection()
            models_count = 0
            if success:
                models = await provider.list_models()
                models_count = len(models)
                self.model_cache[pid] = models

            reports.append(
                ProviderConnectionReport(
                    provider_id=pid,
                    name=meta.name,
                    is_local=meta.is_local,
                    configured=True,
                    connected=success,
                    status_message=msg,
                    models_count=models_count
                )
            )

        return reports

    async def test_single_provider(self, provider_id: str) -> Tuple[bool, str, List[ModelInfo]]:
        """Tests connection for a single provider and loads its model catalog."""
        provider = self.get_provider(provider_id)
        if not provider:
            return False, f"Unknown provider '{provider_id}'", []

        success, msg = await provider.test_connection()
        models: List[ModelInfo] = []
        if success:
            models = await provider.list_models()
            self.model_cache[provider_id] = models
        return success, msg, models

    def set_active_provider(self, provider_id: str) -> Optional[BaseProvider]:
        """Sets the active provider."""
        provider = self.get_provider(provider_id)
        if provider:
            self.active_provider = provider
        return provider
