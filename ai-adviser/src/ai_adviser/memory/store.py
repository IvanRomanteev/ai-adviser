"""
Conversation memory store backed by SQLite.

This module provides a simple persistent store for conversation history. Each
message is stored with an associated thread identifier and optional user
identifier. Messages are retrieved in chronological order and can be used
to build conversational context. A separate table stores summarised
representations of long conversations to limit context window usage.

The store defaults to an in‑memory SQLite database when no path is
provided. For production deployments set the `SQLITE_DB_PATH` in the
environment or configuration to persist data on disk. The store is thread
safe and uses a re‑entrant lock around write and read operations.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Dict, List, Optional


class MemoryStore:
    """Simple conversational memory store implemented with SQLite."""

    def __init__(self, db_path: str = ":memory:") -> None:
        # Expand environment variables in the provided path
        self.db_path = os.path.expanduser(db_path or ":memory:")
        # Use a re‑entrant lock to allow nested operations within a single
        # thread while still preventing concurrent writes from multiple threads.
        self._lock = threading.RLock()
        # Ensure the database schema is created
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Return a new connection to the SQLite database."""
        # disable `check_same_thread` to allow connections from multiple threads
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        """Initialise the required tables if they do not already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    user_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_summaries_thread ON summaries(thread_id)
                """
            )

    def append(self, thread_id: str, user_id: Optional[str], role: str, content: str) -> None:
        """Append a new message to the conversation history.

        Args:
            thread_id: The identifier for the conversation thread.
            user_id: Optional identifier for the user.
            role: The role of the message ('system', 'user' or 'assistant').
            content: The text content of the message.
        """
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO messages (thread_id, user_id, role, content) VALUES (?, ?, ?, ?)",
                    (thread_id, user_id, role, content),
                )
                conn.commit()

    def get_history(self, thread_id: str) -> List[Dict[str, str]]:
        """Return the chronological history of messages for a thread.

        Args:
            thread_id: The identifier for the conversation thread.

        Returns:
            A list of dictionaries with keys 'role' and 'content'. The list is
            sorted by insertion order.
        """
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY id ASC",
                    (thread_id,),
                )
                rows = cursor.fetchall()
                return [{"role": role, "content": content} for (role, content) in rows]

    def save_summary(self, thread_id: str, summary: str) -> None:
        """Persist a summary of a conversation thread.

        Args:
            thread_id: The identifier for the conversation thread.
            summary: A summarised representation of the conversation.
        """
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO summaries (thread_id, summary) VALUES (?, ?)",
                    (thread_id, summary),
                )
                conn.commit()

    def get_latest_summary(self, thread_id: str) -> Optional[str]:
        """Return the most recent summary for the conversation thread, if any."""
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "SELECT summary FROM summaries WHERE thread_id = ? ORDER BY id DESC LIMIT 1",
                    (thread_id,),
                )
                row = cursor.fetchone()
                return row[0] if row else None