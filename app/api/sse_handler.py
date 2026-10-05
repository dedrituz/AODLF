"""
Server-Sent Events (SSE) stream formatting protocol.
Formats events according to backend specification:
event: status | thought_delta | content_delta | metadata | suggestions | error
"""

import json
from typing import Any, Dict
from app.providers.base import StreamChunk


class SSEFormatter:
    """Formats StreamChunk objects into standard SSE data packets."""

    @staticmethod
    def format_chunk(chunk: StreamChunk) -> Dict[str, Any]:
        """
        Returns a dict formatted for sse_starlette.ServerSentEvent or raw SSE line.
        """
        payload = {
            "event": chunk.event_type,
            "data": json.dumps({
                "content": chunk.content,
                "metadata": chunk.metadata
            })
        }
        return payload

    @staticmethod
    def to_sse_string(chunk: StreamChunk) -> str:
        """Raw SSE wire string representation."""
        data_json = json.dumps({
            "content": chunk.content,
            "metadata": chunk.metadata
        })
        return f"event: {chunk.event_type}\ndata: {data_json}\n\n"
