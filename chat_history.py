"""
chat_memory.py

Lightweight, persistent chat memory using SQLite. See earlier version notes:
bounded per-session storage, bounded retrieval window, no background server.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path("data/chat_memory.db")


class ChatMemory:
    def __init__(self, db_path: Path = DB_PATH, max_messages_per_session: int = 40):
        self.db_path = db_path
        self.max_messages_per_session = max_messages_per_session
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_time ON messages(session_id, created_at)"
            )

    def save_message(self, session_id: str, role: str, content: str):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            conn.execute(
                """
                DELETE FROM messages
                WHERE session_id = ? AND id NOT IN (
                    SELECT id FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (session_id, session_id, self.max_messages_per_session),
            )

    def get_recent_history(
        self, session_id: str, max_messages: int = 10, max_age_days: int | None = 7
    ) -> list[dict]:
        query = "SELECT role, content, created_at FROM messages WHERE session_id = ?"
        params: list = [session_id]

        if max_age_days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
            query += " AND created_at >= ?"
            params.append(cutoff)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max_messages)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        rows.reverse()
        return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]

    def clear_session(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))