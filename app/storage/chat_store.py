"""
Session and chat history storage manager.
Persists chat histories, titles, and metadata to JSON files in data/sessions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config.settings import SESSIONS_DIR
from app.providers.base import Message


class ChatStore:
    """Manages reading and writing chat session records to disk."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or SESSIONS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        safe_id = "".join([c if c.isalnum() or c in "-_" else "_" for c in session_id])
        return self.storage_dir / f"{safe_id}.json"

    def save_session(
        self,
        session_id: str,
        title: str,
        provider_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Saves a session to disk."""
        path = self._get_session_path(session_id)
        data = {
            "session_id": session_id,
            "title": title,
            "provider_id": provider_id,
            "metadata": metadata or {},
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "metadata": m.metadata
                }
                for m in messages
            ]
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Loads a session from disk."""
        path = self._get_session_path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists all saved sessions."""
        sessions = []
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data.get("session_id", file.stem),
                        "title": data.get("title", "Untitled"),
                        "provider_id": data.get("provider_id", "unknown"),
                        "message_count": len(data.get("messages", []))
                    })
            except Exception:
                continue
        return sessions
