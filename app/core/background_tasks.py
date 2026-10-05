"""
Asynchronous background tasks: Predictive Questioning and Auto-Titling.
"""

import asyncio
import re
from typing import List, Optional
from app.providers.base import BaseProvider, Message


class BackgroundIntelligence:
    """Handles non-blocking background tasks after inference completes."""

    @staticmethod
    async def generate_predictive_questions(
        provider: Optional[BaseProvider],
        model: str,
        user_prompt: str,
        assistant_response: str
    ) -> List[str]:
        """
        Generates 3 to 4 relevant follow-up questions based on the dialogue turn.
        """
        # Fast heuristic generation fallback
        fallback_suggestions = [
            "Can you explain this with a practical example?",
            "What are the edge cases or potential pitfalls?",
            "How does this compare to alternative approaches?",
            "Could you provide a step-by-step implementation?"
        ]

        if not provider:
            return fallback_suggestions

        prompt = (
            f"Based on the following query and answer, generate exactly 3-4 concise, high-value follow-up questions.\n"
            f"Query: {user_prompt[:200]}\n"
            f"Answer: {assistant_response[:300]}\n\n"
            f"Return ONLY a numbered list of 3-4 questions, one per line (e.g. '1. Question'). Do not add any preamble."
        )

        try:
            # Short inference request
            messages = [Message(role="user", content=prompt)]
            text_result = ""
            async for chunk in provider.generate_stream(messages, model, temperature=0.3, max_tokens=150):
                if chunk.event_type == "content_delta":
                    text_result += chunk.content

            questions: List[str] = []
            for line in text_result.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Match "1. ...", "- ...", etc.
                cleaned = re.sub(r"^(\d+[\.\)]|\-|\*)\s*", "", line).strip()
                if cleaned and len(cleaned) > 5 and "?" in cleaned:
                    questions.append(cleaned)

            if len(questions) >= 2:
                return questions[:4]
        except Exception:
            pass

        return fallback_suggestions

    @staticmethod
    async def generate_auto_title(
        provider: Optional[BaseProvider],
        model: str,
        user_prompt: str
    ) -> str:
        """
        Generates a concise 2-4 word title for a new chat session.
        """
        default_title = " ".join(user_prompt.split()[:4]).title() or "New Chat"

        if not provider:
            return default_title

        prompt = (
            f"Generate a concise 3-word title for a conversation starting with this user prompt:\n"
            f"'{user_prompt[:150]}'\n\n"
            f"Rules: Return ONLY the title (2 to 4 words). Do not include quotes, periods, or intro."
        )

        try:
            messages = [Message(role="user", content=prompt)]
            title_accum = ""
            async for chunk in provider.generate_stream(messages, model, temperature=0.2, max_tokens=15):
                if chunk.event_type == "content_delta":
                    title_accum += chunk.content

            clean_title = title_accum.strip().strip("'\"").replace("\n", " ").strip()
            # Enforce 2-5 words
            words = clean_title.split()
            if 1 <= len(words) <= 6:
                return " ".join(words).title()
        except Exception:
            pass

        return default_title
