"""
FastAPI REST & SSE Application Routes for the AODLF Backend.
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.api.sse_handler import SSEFormatter
from app.config.env_manager import EnvManager
from app.core.orchestrator import BackendOrchestrator
from app.providers.manager import ProviderManager
from app.routing.smart_assignment import SmartModelAssigner
from app.routing.system_scanner import SystemScanner

app = FastAPI(
    title="Automated Open-Domain Learning Framework (AODLF) Backend",
    description="Adaptive modular AI backend with tiered routing, multimodal density branching, and SSE streaming.",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons
env_manager = EnvManager()
provider_manager = ProviderManager(env_manager=env_manager)
orchestrator = BackendOrchestrator(provider_manager=provider_manager)


# --- Request / Response Models ---

class ChatRequest(BaseModel):
    prompt: str
    attached_file_path: Optional[str] = None
    provider_id: Optional[str] = "ollama"
    model_override: Optional[str] = None
    system_prompt: Optional[str] = None


class ApiKeyUpdateRequest(BaseModel):
    provider_id: str
    key_or_url: str


# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "framework": "AODLF",
        "version": "0.2.0"
    }


@app.get("/api/providers")
async def get_providers():
    reports = await provider_manager.scan_and_test_all()
    return [
        {
            "provider_id": r.provider_id,
            "name": r.name,
            "is_local": r.is_local,
            "configured": r.configured,
            "connected": r.connected,
            "status_message": r.status_message,
            "empty_or_placeholder": r.empty_or_placeholder,
            "models_count": r.models_count
        }
        for r in reports
    ]


@app.get("/api/models")
async def get_models(provider_id: str = Query("ollama")):
    provider = provider_manager.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")

    models = await provider.list_models()
    specs = SystemScanner.scan()
    tier_assignment = SmartModelAssigner.assign_tiers(models, provider_id, specs)

    return {
        "provider_id": provider_id,
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "parameter_size": m.parameter_size,
                "context_length": m.context_length,
                "quantization": m.quantization,
                "cost_input": m.cost_per_1m_input,
                "cost_output": m.cost_per_1m_output,
                "is_local": m.is_local,
                "capabilities": {
                    "vision": m.capabilities.vision,
                    "thinking": m.capabilities.thinking,
                    "tools": m.capabilities.tools
                }
            }
            for m in models
        ],
        "smart_routing": tier_assignment.to_dict(),
        "routing_reasoning": tier_assignment.reasoning
    }


@app.get("/api/specs")
async def get_hardware_specs():
    specs = SystemScanner.scan()
    return {
        "cpu_model": specs.cpu_model,
        "cpu_cores_physical": specs.cpu_cores_physical,
        "cpu_cores_logical": specs.cpu_cores_logical,
        "ram_total_gb": specs.ram_total_gb,
        "ram_available_gb": specs.ram_available_gb,
        "gpu_name": specs.gpu_name,
        "vram_total_gb": specs.vram_total_gb,
        "cuda_available": specs.cuda_available,
        "recommended_max_param_size": specs.recommended_max_param_size,
        "summary": specs.summary
    }


@app.post("/api/config/api-key")
async def update_api_key(req: ApiKeyUpdateRequest):
    success, msg = env_manager.update_key(req.provider_id, req.key_or_url)
    return {"success": success, "message": msg}


@app.post("/api/compact")
async def force_compaction():
    result = orchestrator.context_manager.force_compact()
    return {
        "compacted": result.compacted,
        "original_tokens": result.original_tokens,
        "new_tokens": result.new_tokens,
        "messages_condensed": result.messages_condensed
    }


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    # Ensure provider is configured
    provider_id = req.provider_id or "ollama"
    provider = provider_manager.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=400, detail=f"Provider '{provider_id}' is not available.")

    # Check if session needs configuration
    if not orchestrator.tier_assignment or orchestrator.session_state.active_provider_id != provider_id:
        models = await provider.list_models()
        specs = SystemScanner.scan()
        tier_assign = SmartModelAssigner.assign_tiers(models, provider_id, specs)
        orchestrator.configure_session(
            provider_id=provider_id,
            tier_assignment=tier_assign,
            override_model=req.model_override,
            system_prompt=req.system_prompt
        )
    elif req.model_override:
        orchestrator.router.set_override_model(req.model_override)

    async def event_generator():
        async for chunk in orchestrator.process_turn_stream(
            user_input=req.prompt,
            attached_file_path=req.attached_file_path
        ):
            yield SSEFormatter.format_chunk(chunk)

    return EventSourceResponse(event_generator())
