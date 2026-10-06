"""
Central Backend Orchestrator & Decision Engine.
Implements the end-to-end request lifecycle:
Media Routing -> OCR / Vision -> Intelligent Routing -> Context Compaction -> Streaming Inference -> Post-Response Tasks.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from app.core.background_tasks import BackgroundIntelligence
from app.core.context_manager import ContextManager
from app.multimodal.context_formatter import AnnotatedContextObject
from app.multimodal.doc_loader import DocumentLoader, PathValidationError
from app.multimodal.media_router import MediaRouter, MediaRoutingResult
from app.providers.base import BaseProvider, Message, ModelInfo, StreamChunk
from app.providers.manager import ProviderManager
from app.routing.router import IntelligentRouter, RoutingDecision
from app.routing.smart_assignment import SmartModelAssigner, TierAssignment


@dataclass
class SessionState:
    session_id: str = "default"
    title: str = "New Conversation"
    active_provider_id: str = "ollama"
    active_mode: str = "smart_routing"  # "smart_routing" or "manual_override"
    override_model: Optional[str] = None
    has_vision_model: bool = False
    turn_count: int = 0
    total_tokens_used: int = 0
    system_prompt: str = "do not use markdown in your response."


class BackendOrchestrator:
    """Central Decision Engine and Execution Orchestrator."""

    def __init__(
        self,
        provider_manager: Optional[ProviderManager] = None,
        context_manager: Optional[ContextManager] = None
    ):
        self.provider_manager = provider_manager or ProviderManager()
        self.context_manager = context_manager or ContextManager()
        self.media_router = MediaRouter()
        self.tier_assignment: Optional[TierAssignment] = None
        self.router: Optional[IntelligentRouter] = None
        self.session_state = SessionState()

    def configure_session(
        self,
        provider_id: str,
        tier_assignment: TierAssignment,
        override_model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> None:
        """Configures the current session with provider, routing assignments, and system prompt."""
        self.session_state.active_provider_id = provider_id
        self.tier_assignment = tier_assignment
        self.session_state.has_vision_model = bool(tier_assignment.vision_model)
        self.session_state.override_model = override_model
        if system_prompt:
            self.session_state.system_prompt = system_prompt
        self.context_manager.set_system_prompt(self.session_state.system_prompt)
        self.router = IntelligentRouter(tier_assignment=tier_assignment, override_model=override_model)

    def extract_file_paths(self, input_text: str) -> List[str]:
        """
        Extracts potential file path references from user input
        (e.g., "C:\\path\\to\\file.txt", ".\\relative\\path.png", or quoted paths).
        """
        paths: List[str] = []
        # Match paths in quotes
        quoted = re.findall(r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']', input_text)
        paths.extend(quoted)

        # Match unquoted Windows paths (C:\... or .\...)
        unquoted = re.findall(r'(?:[a-zA-Z]:\\[^\s]+|\.\\[^\s]+|\/[^\s]+)', input_text)
        for p in unquoted:
            if p not in paths:
                paths.append(p)

        return paths

    async def process_turn_stream(
        self,
        user_input: str,
        attached_file_path: Optional[str] = None
    ) -> AsyncIterator[StreamChunk]:
        """
        Executes the full pipeline for a single user turn, yielding SSE-compatible StreamChunks.
        """
        if not self.tier_assignment or not self.router:
            yield StreamChunk(
                event_type="error",
                content="System not initialized! Please select a provider and model routing first."
            )
            return

        provider = self.provider_manager.get_provider(self.session_state.active_provider_id)
        if not provider:
            yield StreamChunk(
                event_type="error",
                content=f"Active provider '{self.session_state.active_provider_id}' is unavailable."
            )
            return

        # 1. Detect & Validate File Attachments
        paths_to_check = []
        if attached_file_path:
            paths_to_check.append(attached_file_path)
        else:
            # Check if user mentioned paths directly in the prompt
            extracted = self.extract_file_paths(user_input)
            paths_to_check.extend(extracted)

        resolved_file: Optional[Path] = None
        for raw_p in paths_to_check:
            try:
                resolved_file = DocumentLoader.validate_and_resolve_path(raw_p)
                break
            except PathValidationError as pe:
                yield StreamChunk(
                    event_type="error",
                    content=f"Path Validation Error: {str(pe)}"
                )
                return

        # 2. Multimodal / Media Branching Logic
        annotated_contexts: List[AnnotatedContextObject] = []
        vision_image_base64: Optional[str] = None
        requires_vision_model = False

        if resolved_file:
            if DocumentLoader.is_image(resolved_file):
                yield StreamChunk(event_type="status", content="Analyzing image text density and modality...")
                media_res: MediaRoutingResult = await self.media_router.decide_route(
                    image_path=resolved_file,
                    user_prompt=user_input,
                    has_vision_model=self.session_state.has_vision_model
                )

                if media_res.method == "OCR":
                    yield StreamChunk(
                        event_type="status",
                        content=f"Performing OCR on '{resolved_file.name}' (Density: {media_res.text_density:.1%})..."
                    )
                    annotated_contexts.append(
                        AnnotatedContextObject(
                            type="media_context",
                            source=str(resolved_file),
                            method="OCR",
                            confidence=media_res.confidence,
                            content=media_res.extracted_text or "[Empty OCR Text]",
                            metadata={"text_density": media_res.text_density}
                        )
                    )
                else:
                    yield StreamChunk(
                        event_type="status",
                        content=f"Routing image '{resolved_file.name}' to Multimodal Vision Path..."
                    )
                    requires_vision_model = True
                    vision_image_base64 = DocumentLoader.encode_image_base64(resolved_file)
                    annotated_contexts.append(
                        AnnotatedContextObject(
                            type="media_context",
                            source=str(resolved_file),
                            method="Vision",
                            confidence=media_res.confidence,
                            content=f"[Image binary attached: {resolved_file.name}]",
                            metadata={"text_density": media_res.text_density}
                        )
                    )
            else:
                yield StreamChunk(event_type="status", content=f"Parsing document '{resolved_file.name}' (Hybrid text & OCR)...")
                doc_ctx = await DocumentLoader.load_document_async(resolved_file)
                annotated_contexts.append(doc_ctx)

        # 3. Intelligent Tiered Routing
        # has_image must reflect the full context window, not just the current turn.
        # A raw image binary from an earlier vision-path turn still lives in context and
        # will be forwarded to the model — text-only models must remain excluded until
        # the context is cleared. context_has_images() is False for OCR sessions because
        # OCR stores extracted text only, preserving lightweight model routing for those.
        current_turn_has_image = bool(resolved_file and DocumentLoader.is_image(resolved_file))
        image_in_context = self.context_manager.context_has_images()
        yield StreamChunk(event_type="status", content="Evaluating query complexity & routing...")
        routing_decision: RoutingDecision = await self.router.route_prompt(
            prompt=user_input,
            provider=provider,
            has_image=current_turn_has_image or image_in_context,
            requires_vision_model=requires_vision_model
        )

        selected_model = routing_decision.selected_model
        yield StreamChunk(
            event_type="status",
            content=f"Routed to [{routing_decision.tier.upper()}] model '{selected_model}' ({routing_decision.reason})"
        )

        # 4. Context Assembly & Compaction Check
        # Build user message content with annotated context blocks
        final_prompt = user_input
        if annotated_contexts:
            context_blocks = "\n\n".join([ctx.to_prompt_block() for ctx in annotated_contexts])
            final_prompt = f"{context_blocks}\n\nUser Query: {user_input}"

        images_payload = [vision_image_base64] if vision_image_base64 else None
        self.context_manager.add_user_message(content=final_prompt, images=images_payload)

        # Context Compaction
        compaction_res = await self.context_manager.check_and_compact(provider, selected_model)
        if compaction_res.compacted:
            yield StreamChunk(
                event_type="status",
                content=f"Context Compacted: condensed {compaction_res.messages_condensed} turns ({compaction_res.original_tokens} -> {compaction_res.new_tokens} tokens)"
            )

        messages_for_llm = self.context_manager.get_context_for_inference()

        # 5. Execute Inference Stream
        full_response_text = ""
        total_eval_tokens = 0
        last_metadata: Dict[str, Any] = {}

        async for chunk in provider.generate_stream(
            messages=messages_for_llm,
            model=selected_model
        ):
            if chunk.event_type == "content_delta":
                full_response_text += chunk.content
            elif chunk.event_type == "metadata":
                last_metadata = chunk.metadata
                total_eval_tokens = chunk.metadata.get("eval_tokens", 0)

            yield chunk

        # 6. Save Assistant Response & Trigger Background Tasks
        self.context_manager.add_assistant_message(
            content=full_response_text,
            metadata={"model": selected_model, "tier": routing_decision.tier}
        )
        self.session_state.turn_count += 1
        self.session_state.total_tokens_used += total_eval_tokens

        # Background Task 1: Auto-titling on first turn
        if self.session_state.turn_count == 1:
            title = await BackgroundIntelligence.generate_auto_title(
                provider=provider,
                model=self.tier_assignment.basic_model,
                user_prompt=user_input
            )
            self.session_state.title = title

        # Background Task 2: Predictive Follow-Up Questions
        yield StreamChunk(event_type="status", content="Generating follow-up suggestions...")
        suggestions = await BackgroundIntelligence.generate_predictive_questions(
            provider=provider,
            model=self.tier_assignment.basic_model,
            user_prompt=user_input,
            assistant_response=full_response_text
        )

        yield StreamChunk(
            event_type="metadata",
            content="",
            metadata={
                **last_metadata,
                "routing": {
                    "tier": routing_decision.tier,
                    "model": selected_model,
                    "reason": routing_decision.reason,
                    "confidence": routing_decision.confidence
                },
                "suggestions": suggestions,
                "session_title": self.session_state.title,
                "total_tokens_used": self.session_state.total_tokens_used
            }
        )
