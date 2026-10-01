import json
import sqlite3
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, data TEXT NOT NULL)")

    def get(self, key: str) -> Any | None:
        row = self._conn.execute("SELECT data FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Any) -> None:
        with self._conn:
            self._conn.execute("INSERT OR REPLACE INTO cache (key, data) VALUES (?, ?)", (key, json.dumps(value)))

    def close(self) -> None:
        self._conn.close()
