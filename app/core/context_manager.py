"""
Context and Memory Manager with automatic Context Compaction.
Critical for small parameter models and smart routing to prevent token overflow.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from app.config.settings import (
    CONTEXT_COMPACT_THRESHOLD_RATIO,
    DEFAULT_MAX_CONTEXT_TOKENS,
)
from app.providers.base import Message, BaseProvider

@dataclass
class CompactionResult:
    compacted: bool
    original_tokens: int
    new_tokens: int
    messages_condensed: int
    summary: Optional[str] = None

class ContextManager:
    """Manages chat messages, token accounting, and intelligent context compaction."""
    def __init__(
        self,
        system_prompt: str = "do not use markdown in your response.",
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        compact_threshold_ratio: float = CONTEXT_COMPACT_THRESHOLD_RATIO
    ):
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.compact_threshold_ratio = compact_threshold_ratio
        self.messages: List[Message] = []
        self.compacted_summary: Optional[str] = None
        self.compaction_history: List[Dict[str, Any]] = []

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def add_user_message(self, content: str, images: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> Message:
        msg = Message(role="user", content=content, images=images, metadata=metadata or {})
        self.messages.append(msg)
        return msg

    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        msg = Message(role="assistant", content=content, metadata=metadata or {})
        self.messages.append(msg)
        return msg

    def clear_history(self) -> None:
        self.messages.clear()
        self.compacted_summary = None
        self.compaction_history.clear()

    def context_has_images(self) -> bool:
        """Returns True only if a raw image binary (base64) exists in the active context window.

        This is an exact structural check, not a heuristic:
        - Vision path: stores the encoded image in Message.images → returns True.
        - OCR path: stores only extracted text in Message.content, images=None → returns False.

        Used by the router to block text-only model selection when an image payload
        is present in context, while fully preserving OCR-based routing to lightweight models.
        """
        return any(bool(m.images) for m in self.messages)

    def estimate_tokens(self, text: str) -> int:
        """Heuristic token estimation: ~4 chars per token for English text and code."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def get_total_tokens(self) -> int:
        """Computes current token usage across system prompt, summary, and all stored messages."""
        tokens = self.estimate_tokens(self.system_prompt)
        if self.compacted_summary:
            tokens += self.estimate_tokens(self.compacted_summary)
        for m in self.messages:
            tokens += self.estimate_tokens(m.content)
        return tokens

    async def check_and_compact(self, provider: BaseProvider, model: str, target_context_limit: Optional[int] = None) -> CompactionResult:
        """Automatically compacts conversation history if token count exceeds threshold."""
        limit = target_context_limit or self.max_context_tokens
        threshold = int(limit * self.compact_threshold_ratio)
        current_tokens = self.get_total_tokens()

        if current_tokens <= threshold or len(self.messages) <= 4:
            return CompactionResult(
                compacted=False,
                original_tokens=current_tokens,
                new_tokens=current_tokens,
                messages_condensed=0
            )

        return await self.force_compact(provider, model)

    async def force_compact(self, provider: BaseProvider, model: str) -> CompactionResult:
        """Forces context compaction on current history using semantic summarization."""
        original_tokens = self.get_total_tokens()
        if len(self.messages) <= 2:
            return CompactionResult(
                compacted=False,
                original_tokens=original_tokens,
                new_tokens=original_tokens,
                messages_condensed=0
            )

        # Keep the most recent 4 messages intact
        keep_count = min(4, len(self.messages))
        to_condense = self.messages[:-keep_count]
        retained = self.messages[-keep_count:]

        # Build history text for summarization
        history_text = ""
        for msg in to_condense:
            role = "User" if msg.role == "user" else "Assistant"
            history_text += f"{role}: {msg.content}\n"

        summary_prompt = f"""Summarize the following conversation history into concise bullet points, preserving key information and intent:\n{history_text}"""
        messages = [
            Message(role="system", content="You are a helpful assistant specialized in summarizing conversations concisely."),
            Message(role="user", content=summary_prompt)
        ]

        new_summary = ""
        try:
            # Collect the stream into a string
            async for chunk in provider.generate_stream(messages, model=model):
                if chunk.event_type == "content_delta":
                    new_summary += chunk.content
            new_summary = new_summary.strip()
        except Exception as e:
            # Fallback to heuristic summary if LLM fails
            print(f"[ContextManager] Semantic summarization failed ({e}), using heuristic snippet.")
            snippet_points = []
            for msg in to_condense:
                role = "User" if msg.role == "user" else "Assistant"
                clean_content = msg.content.strip().replace("\n", " ")
                snippet = clean_content[:160] + ("..." if len(clean_content) > 160 else "")
                snippet_points.append(f"- {role}: {snippet}")
            new_summary = "[CONVERSATION HISTORY SUMMARY - HEURISTIC]\n" + "\n".join(snippet_points)

        self.compacted_summary = new_summary
        self.messages = retained
        
        # Recalculate tokens after compaction
        new_total_tokens = self.get_total_tokens()
        condensed_count = len(to_condense)

        self.compaction_history.append({
            "original_tokens": original_tokens,
            "new_tokens": new_total_tokens,
            "condensed_messages": condensed_count
        })

        return CompactionResult(
            compacted=True,
            original_tokens=original_tokens,
            new_tokens=new_total_tokens,
            messages_condensed=condensed_count,
            summary=new_summary
        )

    def get_context_for_inference(self) -> List[Message]:
        """Constructs the final list of Message objects to send to LLM."""
        full_messages: List[Message] = []
        system_text = self.system_prompt
        if self.compacted_summary:
            system_text += f"\n\n{self.compacted_summary}"
        full_messages.append(Message(role="system", content=system_text))
        full_messages.extend(self.messages)
        return full_messages
