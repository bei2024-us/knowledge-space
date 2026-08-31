from pathlib import Path
import sqlite3


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "mindspace.sqlite3"


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def connect() -> sqlite3.Connection:
    ensure_storage()
    # timeout + busy_timeout：后台 OCR 线程写库、前台搜索读库可能并发，
    # 用 WAL 允许“读写并行”，用 busy_timeout 让偶发的锁等待而不是直接报错
    # （此前日志里出现过 "database is locked" —— 两个搜索同时 backfill 向量所致）。
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS spaces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                space_id INTEGER NOT NULL,
                folder_id INTEGER,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                ocr_status TEXT DEFAULT 'not_needed',
                ocr_message TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (space_id) REFERENCES spaces(id) ON DELETE CASCADE,
                FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                space_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (space_id) REFERENCES spaces(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                space_id INTEGER NOT NULL,
                document_id INTEGER NOT NULL,
                location_label TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (space_id) REFERENCES spaces(id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES('delete', old.id, old.text);
            END;

            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (chunk_id, model),
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );
            """
        )
        _ensure_column(conn, "documents", "folder_id", "INTEGER")
        _ensure_column(conn, "documents", "ocr_status", "TEXT DEFAULT 'not_needed'")
        _ensure_column(conn, "documents", "ocr_message", "TEXT DEFAULT ''")
        _ensure_default_folders(conn)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _ensure_default_folders(conn: sqlite3.Connection) -> None:
    spaces = conn.execute("SELECT id FROM spaces").fetchall()
    for space in spaces:
        folder = conn.execute(
            "SELECT id FROM folders WHERE space_id = ? ORDER BY id LIMIT 1",
            (space["id"],),
        ).fetchone()
        if not folder:
            cur = conn.execute(
                "INSERT INTO folders(space_id, name) VALUES(?, ?)",
                (space["id"], "默认文件夹"),
            )
            folder_id = cur.lastrowid
        else:
            folder_id = folder["id"]
        conn.execute(
            "UPDATE documents SET folder_id = ? WHERE space_id = ? AND folder_id IS NULL",
            (folder_id, space["id"]),
        )
