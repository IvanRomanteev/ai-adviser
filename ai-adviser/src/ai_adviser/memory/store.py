"""
Conversation memory store backed by SQLite.

This module provides a simple persistent store for conversation history. Each
message is stored with an associated thread identifier and optional user
identifier. Messages are retrieved in chronological order and can be used
to build conversational context. A separate table stores summarised
representations of long conversations to limit context window usage.

Important implementation notes:

- The store keeps **one persistent SQLite connection** per MemoryStore
  instance. This is required for the default ``:memory:`` mode because each
  new connection would otherwise create a separate in-memory database.
- The private ``_init_db()`` method is also used by the test-suite as a
  *reset* hook. When pytest is running (detected via the
  ``PYTEST_CURRENT_TEST`` environment variable), ``_init_db()`` will clear the
  existing tables so tests are deterministic.

Retention / growth control:

- To prevent the database from growing without bound, the store prunes old
  rows per thread. Configure via env vars (or the same-named attributes on
  ``ai_adviser.config.settings`` if you prefer):

    * ``MEMORY_MAX_MESSAGES_PER_THREAD`` (default: 200)
    * ``MEMORY_MAX_SUMMARIES_PER_THREAD`` (default: 20)

  Set a value ``<= 0`` to disable pruning for that table.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Dict, List, Optional


def _parse_int(value: object, default: int) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        return int(str(value).strip())
    except Exception:
        return default


def _parse_bool(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class MemoryStore:
    """Simple conversational memory store implemented with SQLite."""

    def __init__(self, db_path: str = ":memory:") -> None:
        # Expand environment variables in the provided path
        self.db_path = os.path.expanduser(db_path or ":memory:")

        # Use a re-entrant lock to allow nested operations within a single
        # thread while still preventing concurrent writes from multiple threads.
        self._lock = threading.RLock()

        # Keep a single connection for the lifetime of the store.
        # NOTE: For ":memory:" this is essential because every new connection
        # would create a new empty database.
        uri = self.db_path.startswith("file:")
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, uri=uri)

        # Ensure the database schema is created
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Return the (persistent) connection to the SQLite database."""
        return self._conn

    def _is_pytest(self) -> bool:
        # pytest sets PYTEST_CURRENT_TEST during test execution.
        return bool(os.getenv("PYTEST_CURRENT_TEST"))

    def _get_limit(self, key: str, default: int) -> int:
        """Read an integer limit from settings or environment.

        We support both:
        - ai_adviser.config.settings.<key>
        - os.environ[<key>]
        """
        # Try config.settings first (optional, avoid hard dependency).
        try:
            from ai_adviser.config import settings  # type: ignore
            val = getattr(settings, key, None)
            if val is not None:
                return _parse_int(val, default)
        except Exception:
            pass
        # Fallback to env var.
        return _parse_int(os.getenv(key), default)

    def _init_db(self) -> None:
        """Initialise the required tables.

        In tests this method is also used as a reset hook; when pytest is
        running we clear the existing data to avoid cross-test pollution.
        """
        with self._lock:
            conn = self._connect()

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
                "CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id)"
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
                "CREATE INDEX IF NOT EXISTS idx_summaries_thread ON summaries(thread_id)"
            )

            # When running under pytest, treat _init_db() as a full reset.
            if self._is_pytest():
                self.clear_all()
            conn.commit()

    def clear_thread(self, thread_id: str) -> None:
        """Delete all memory (messages + summaries) for a single thread."""
        if not thread_id:
            return
        with self._lock:
            conn = self._connect()
            conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM summaries WHERE thread_id = ?", (thread_id,))
            conn.commit()

    def clear_all(self) -> None:
        """Delete all memory (messages + summaries) for all threads."""
        with self._lock:
            conn = self._connect()
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM summaries")
            # Reset AUTOINCREMENT counters for cleaner test runs (optional).
            conn.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('messages', 'summaries')"
            )
            conn.commit()

    def _prune_messages(self, conn: sqlite3.Connection, thread_id: str) -> None:
        """Keep only the most recent N messages for a thread (if enabled)."""
        limit = self._get_limit("MEMORY_MAX_MESSAGES_PER_THREAD", 200)
        if limit <= 0:
            return
        conn.execute(
            """
            DELETE FROM messages
            WHERE thread_id = ?
              AND id NOT IN (
                SELECT id FROM messages
                WHERE thread_id = ?
                ORDER BY id DESC
                LIMIT ?
              )
            """,
            (thread_id, thread_id, limit),
        )

    def _prune_summaries(self, conn: sqlite3.Connection, thread_id: str) -> None:
        """Keep only the most recent N summaries for a thread (if enabled)."""
        limit = self._get_limit("MEMORY_MAX_SUMMARIES_PER_THREAD", 20)
        if limit <= 0:
            return
        conn.execute(
            """
            DELETE FROM summaries
            WHERE thread_id = ?
              AND id NOT IN (
                SELECT id FROM summaries
                WHERE thread_id = ?
                ORDER BY id DESC
                LIMIT ?
              )
            """,
            (thread_id, thread_id, limit),
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
            conn = self._connect()
            conn.execute(
                "INSERT INTO messages (thread_id, user_id, role, content) VALUES (?, ?, ?, ?)",
                (thread_id, user_id, role, content),
            )
            # Prevent unbounded growth.
            self._prune_messages(conn, thread_id)
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
            conn = self._connect()
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
            conn = self._connect()
            conn.execute(
                "INSERT INTO summaries (thread_id, summary) VALUES (?, ?)",
                (thread_id, summary),
            )
            # Prevent unbounded growth.
            self._prune_summaries(conn, thread_id)
            conn.commit()

    def get_latest_summary(self, thread_id: str) -> Optional[str]:
        """Return the most recent summary for the conversation thread, if any."""
        with self._lock:
            conn = self._connect()
            cursor = conn.execute(
                "SELECT summary FROM summaries WHERE thread_id = ? ORDER BY id DESC LIMIT 1",
                (thread_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
