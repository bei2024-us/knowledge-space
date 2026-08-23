from base64 import b64decode
from collections import Counter
from pathlib import Path
import re
from shutil import copyfileobj, rmtree
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from .embeddings import backfill_space_embeddings, embedding_backend_name, load_space_embeddings, semantic_rank_chunks
from .db import UPLOAD_DIR, connect, ensure_storage, init_db
from .parsers import ingest_file


app = FastAPI(title="MindSpace API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYNONYM_GROUPS = [
    {"知识", "资料", "文档", "文件", "笔记", "材料", "内容"},
    {"搜索", "检索", "查找", "查询", "寻找"},
    {"学习", "复习", "预习", "自学", "备考"},
    {"题目", "习题", "例题", "练习", "试题", "考题"},
    {"公式", "定理", "法则", "性质", "结论"},
    {"定义", "概念", "含义", "解释"},
    {"总结", "归纳", "概要", "提炼", "梳理"},
    {"重点", "要点", "核心", "关键", "考点"},
    {"证明", "推导", "论证", "演算"},
    {"函数", "映射", "function"},
    {"极限", "lim", "limit"},
    {"导数", "微分", "derivative"},
    {"积分", "定积分", "不定积分", "integral"},
    {"矩阵", "matrix"},
    {"向量", "vector"},
    {"概率", "随机", "probability"},
    {"高数", "高等数学", "微积分", "calculus"},
]


class SpaceCreate(BaseModel):
    name: str
    description: str = ""


class FolderCreate(BaseModel):
    name: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 20


class Base64UploadRequest(BaseModel):
    filename: str
    content_base64: str
    folder_id: int | None = None


class DebugLogRequest(BaseModel):
    source: str = "mobile"
    event: str
    detail: dict | str | None = None


@app.on_event("startup")
def startup() -> None:
    ensure_storage()
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/upload-ui", response_class=HTMLResponse)
def upload_ui() -> str:
    ui_path = Path(__file__).resolve().parents[1] / "static" / "upload-ui.html"
    return ui_path.read_text(encoding="utf-8")


@app.get("/app-ui", response_class=HTMLResponse)
def app_ui() -> str:
    ui_path = Path(__file__).resolve().parents[1] / "static" / "app-ui.html"
    return ui_path.read_text(encoding="utf-8")


@app.post("/debug/logs")
def receive_debug_log(payload: DebugLogRequest) -> dict[str, str]:
    from json import dumps

    log_dir = Path(__file__).resolve().parents[1] / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "mobile-debug.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(dumps(payload.model_dump(), ensure_ascii=False) + "\n")
    return {"status": "ok"}


@app.get("/debug/logs")
def read_debug_logs() -> list[str]:
    log_path = Path(__file__).resolve().parents[1] / "data" / "mobile-debug.log"
    if not log_path.exists():
        return []
    return log_path.read_text(encoding="utf-8").splitlines()[-80:]


@app.get("/spaces")
def list_spaces() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.*,
                   COUNT(DISTINCT d.id) AS document_count,
                   COUNT(c.id) AS chunk_count
            FROM spaces s
            LEFT JOIN documents d ON d.space_id = s.id
            LEFT JOIN chunks c ON c.space_id = s.id
            GROUP BY s.id
            ORDER BY s.created_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


@app.post("/spaces")
def create_space(payload: SpaceCreate) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Space name is required")
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO spaces(name, description) VALUES(?, ?)",
            (name, payload.description.strip()),
        )
        space_id = cur.lastrowid
        conn.execute("INSERT INTO folders(space_id, name) VALUES(?, ?)", (space_id, "默认文件夹"))
        row = conn.execute("SELECT * FROM spaces WHERE id = ?", (space_id,)).fetchone()
        return dict(row)


@app.delete("/spaces/{space_id}")
def delete_space(space_id: int) -> dict[str, str]:
    with connect() as conn:
        ensure_space(conn, space_id)
        rows = conn.execute(
            "SELECT stored_path FROM documents WHERE space_id = ?",
            (space_id,),
        ).fetchall()
        conn.execute("DELETE FROM chunks WHERE space_id = ?", (space_id,))
        conn.execute("DELETE FROM documents WHERE space_id = ?", (space_id,))
        conn.execute("DELETE FROM folders WHERE space_id = ?", (space_id,))
        conn.execute("DELETE FROM spaces WHERE id = ?", (space_id,))

    for row in rows:
        delete_upload_path(row["stored_path"])
    delete_upload_path(UPLOAD_DIR / str(space_id), directory=True)
    return {"status": "deleted"}


@app.get("/spaces/{space_id}/folders")
def list_folders(space_id: int) -> list[dict]:
    with connect() as conn:
        ensure_space(conn, space_id)
        rows = conn.execute(
            """
            SELECT f.id,
                   f.space_id,
                   f.name,
                   f.created_at,
                   COUNT(d.id) AS document_count
            FROM folders f
            LEFT JOIN documents d ON d.folder_id = f.id
            WHERE f.space_id = ?
            GROUP BY f.id
            ORDER BY f.created_at ASC, f.id ASC
            """,
            (space_id,),
        ).fetchall()
        return [dict(row) for row in rows]


@app.post("/spaces/{space_id}/folders")
def create_folder(space_id: int, payload: FolderCreate) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Folder name is required")
    with connect() as conn:
        ensure_space(conn, space_id)
        cur = conn.execute("INSERT INTO folders(space_id, name) VALUES(?, ?)", (space_id, name))
        row = conn.execute("SELECT * FROM folders WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


@app.delete("/folders/{folder_id}")
def delete_folder(folder_id: int) -> dict:
    with connect() as conn:
        folder = conn.execute("SELECT id, space_id FROM folders WHERE id = ?", (folder_id,)).fetchone()
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        rows = conn.execute(
            "SELECT stored_path FROM documents WHERE folder_id = ?",
            (folder_id,),
        ).fetchall()
        conn.execute(
            """
            DELETE FROM chunks
            WHERE document_id IN (
                SELECT id FROM documents WHERE folder_id = ?
            )
            """,
            (folder_id,),
        )
        conn.execute("DELETE FROM documents WHERE folder_id = ?", (folder_id,))
        conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))

    for row in rows:
        delete_upload_path(row["stored_path"])
    delete_upload_path(UPLOAD_DIR / str(folder["space_id"]) / str(folder_id), directory=True)
    return {"status": "deleted", "space_id": folder["space_id"]}


@app.post("/spaces/{space_id}/files")
def upload_file(
    space_id: int,
    folder_id: int | None = Form(None),
    query_folder_id: int | None = Query(None, alias="folder_id"),
    file: UploadFile = File(...),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, TXT, and MD are supported")

    with connect() as conn:
        folder_id = resolve_folder_id(conn, space_id, folder_id if folder_id is not None else query_folder_id)

    stored_path = save_upload_file(space_id, folder_id, suffix, file)
    try:
        ingest = ingest_file(stored_path, suffix)
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc

    return insert_document(
        space_id=space_id,
        folder_id=folder_id,
        filename=file.filename or stored_path.name,
        suffix=suffix,
        stored_path=ingest.stored_path,
        chunks=ingest.chunks,
        ocr_status=ingest.ocr_status,
        ocr_message=ingest.ocr_message,
    )


@app.post("/spaces/{space_id}/files/base64")
def upload_file_base64(space_id: int, payload: Base64UploadRequest) -> dict:
    suffix = Path(payload.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, TXT, and MD are supported")

    with connect() as conn:
        folder_id = resolve_folder_id(conn, space_id, payload.folder_id)

    try:
        file_bytes = b64decode(payload.content_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 file content") from exc

    stored_dir = UPLOAD_DIR / str(space_id) / str(folder_id)
    stored_dir.mkdir(parents=True, exist_ok=True)
    stored_path = stored_dir / f"{uuid4().hex}{suffix}"
    stored_path.write_bytes(file_bytes)

    try:
        ingest = ingest_file(stored_path, suffix)
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc

    return insert_document(
        space_id=space_id,
        folder_id=folder_id,
        filename=payload.filename,
        suffix=suffix,
        stored_path=ingest.stored_path,
        chunks=ingest.chunks,
        ocr_status=ingest.ocr_status,
        ocr_message=ingest.ocr_message,
    )


@app.get("/spaces/{space_id}/documents")
def list_documents(space_id: int, folder_id: int | None = Query(None)) -> list[dict]:
    where = "WHERE d.space_id = ?"
    params: list[int] = [space_id]
    if folder_id is not None:
        where += " AND d.folder_id = ?"
        params.append(folder_id)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT d.id,
                   d.folder_id,
                   d.filename,
                   d.file_type,
                   d.ocr_status,
                   d.ocr_message,
                   d.created_at,
                   COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            {where}
            GROUP BY d.id
            ORDER BY d.created_at DESC
            """,
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]


@app.delete("/documents/{document_id}")
def delete_document(document_id: int) -> dict[str, str]:
    with connect() as conn:
        doc = conn.execute(
            "SELECT stored_path FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    delete_upload_path(doc["stored_path"])
    return {"status": "deleted"}


@app.get("/documents/{document_id}")
def get_document(document_id: int, focus_chunk_id: int | None = Query(None)) -> dict:
    with connect() as conn:
        doc = conn.execute(
            """
            SELECT id, space_id, folder_id, filename, file_type, stored_path, ocr_status, ocr_message, created_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        if focus_chunk_id:
            chunks = conn.execute(
                """
                SELECT id AS chunk_id, location_label, text
                FROM chunks
                WHERE document_id = ?
                  AND id BETWEEN ? AND ?
                ORDER BY id
                """,
                (document_id, max(1, focus_chunk_id - 20), focus_chunk_id + 20),
            ).fetchall()
        else:
            chunks = conn.execute(
                """
                SELECT id AS chunk_id, location_label, text
                FROM chunks
                WHERE document_id = ?
                ORDER BY id
                LIMIT 80
                """,
                (document_id,),
            ).fetchall()
    return {**dict(doc), "chunks": [dict(row) for row in chunks]}


@app.get("/documents/{document_id}/viewer", response_class=HTMLResponse)
def document_viewer(
    document_id: int,
    focus_page: int | None = Query(None),
    focus_chunk_id: int | None = Query(None),
) -> str:
    with connect() as conn:
        doc = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        if focus_chunk_id:
            chunks = conn.execute(
                """
                SELECT id AS chunk_id, location_label, text
                FROM chunks
                WHERE document_id = ?
                  AND id BETWEEN ? AND ?
                ORDER BY id
                """,
                (document_id, max(1, focus_chunk_id - 20), focus_chunk_id + 20),
            ).fetchall()
        else:
            chunks = conn.execute(
                """
                SELECT id AS chunk_id, location_label, text
                FROM chunks
                WHERE document_id = ?
                ORDER BY id
                LIMIT 120
                """,
                (document_id,),
            ).fetchall()

    title = html_escape(doc["filename"])
    if doc["file_type"] == "pdf":
        body = render_pdf_pages(document_id, doc["stored_path"], focus_page)
    else:
        body = "\n".join(
            f'<section id="chunk-{row["chunk_id"]}" class="chunk"><strong>{html_escape(row["location_label"])}</strong><p>{html_escape(row["text"])}</p></section>'
            for row in chunks
        )
        if not body:
            body = "<p>没有解析出文字，可能需要 OCR。</p>"

    return f"""
    <!doctype html>
    <html lang="zh-CN">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{title}</title>
        <style>
          body {{ margin: 0; background: #f6f8fb; color: #111827; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
          main {{ width: min(760px, 100%); margin: 0 auto; padding: 16px; }}
          h1 {{ font-size: 22px; line-height: 1.3; margin: 0 0 12px; overflow-wrap: anywhere; }}
          .meta {{ color: #64748b; font-size: 13px; margin-bottom: 14px; }}
          .page {{ display: block; width: 100%; margin: 0 0 14px; border-radius: 12px; background: #fff; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); }}
          .chunk {{ padding: 14px; margin-bottom: 12px; border: 1px solid #e5edf6; border-radius: 18px; background: #fff; }}
          .chunk:target {{ border-color: #1478ff; box-shadow: 0 0 0 3px rgba(20, 120, 255, 0.12); }}
          .chunk strong {{ color: #2563eb; font-size: 13px; }}
          .chunk p {{ line-height: 1.7; overflow-wrap: anywhere; }}
        </style>
      </head>
      <body><main><h1>{title}</h1><div class="meta">OCR 状态：{html_escape(doc["ocr_status"] or "")}</div>{body}</main></body>
    </html>
    """


@app.get("/documents/{document_id}/pages/{page_number}.png")
def document_page_image(document_id: int, page_number: int) -> Response:
    with connect() as conn:
        doc = conn.execute("SELECT file_type, stored_path FROM documents WHERE id = ?", (document_id,)).fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc["file_type"] != "pdf":
        raise HTTPException(status_code=400, detail="Only PDF documents can render pages")

    import fitz

    pdf = fitz.open(doc["stored_path"])
    if page_number < 1 or page_number > pdf.page_count:
        raise HTTPException(status_code=404, detail="Page not found")
    page = pdf.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
    return Response(content=pix.tobytes("png"), media_type="image/png")


@app.get("/documents/{document_id}/file")
def get_document_file(document_id: int) -> FileResponse:
    with connect() as conn:
        doc = conn.execute(
            "SELECT filename, stored_path FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    path = Path(doc["stored_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored file not found")
    return FileResponse(path, filename=doc["filename"])


@app.post("/spaces/{space_id}/search")
def search(space_id: int, payload: SearchRequest) -> dict:
    query = payload.query.strip()
    if not query:
        return {"query": query, "expanded_terms": [], "search_mode": "empty", "results": []}

    limit = max(1, min(payload.limit, 50))
    expanded_terms = expand_search_terms(query) or [query]
    required_terms = extract_required_terms(query)
    fts_query = " OR ".join(escape_fts_term(term) for term in expanded_terms if term)
    results: list[dict] = []
    seen_chunk_ids: set[int] = set()
    has_embedding_backend = embedding_backend_name() is not None
    with connect() as conn:
        ensure_space(conn, space_id)
        for item in find_same_page_matches(conn, space_id, required_terms, query, min(limit, 12)):
            seen_chunk_ids.add(item["chunk_id"])
            seen_chunk_ids.update(int(chunk_id) for chunk_id in item.get("source_chunk_ids", []))
            item["score"] = 3.0
            results.append(item)

        if has_embedding_backend:
            for _ in range(8):
                if backfill_space_embeddings(conn, space_id, batch_size=64) == 0:
                    break
            for item in semantic_rank_chunks(query, load_space_embeddings(conn, space_id), min(limit * 3, 36)):
                if item["chunk_id"] in seen_chunk_ids:
                    continue
                item["snippet"] = make_snippet(item["text"], query, expanded_terms, radius=110)
                item["semantic_match"] = True
                item["score"] = 2.0 + float(item.get("semantic_score", 0.0))
                results.append(item)
                seen_chunk_ids.add(item["chunk_id"])

        if fts_query:
            try:
                rows = conn.execute(
                    """
                    SELECT c.id AS chunk_id,
                           c.document_id,
                           c.location_label,
                           snippet(chunks_fts, 0, '[', ']', '...', 18) AS snippet,
                           c.text,
                           d.filename,
                           d.folder_id,
                           f.name AS folder_name
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    JOIN documents d ON d.id = c.document_id
                    LEFT JOIN folders f ON f.id = d.folder_id
                    WHERE c.space_id = ?
                      AND chunks_fts MATCH ?
                    ORDER BY bm25(chunks_fts)
                    LIMIT ?
                    """,
                    (space_id, fts_query, limit),
                ).fetchall()
                for row in rows:
                    item = dict(row)
                    if item["chunk_id"] in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(item["chunk_id"])
                    item["score"] = 1.5
                    results.append(item)
            except Exception:
                pass

        like_terms = expanded_terms
        where_parts = ["c.text LIKE ?" for _ in like_terms]
        params = [f"%{term}%" for term in like_terms]
        rows = conn.execute(
            f"""
            SELECT c.id AS chunk_id,
                   c.document_id,
                   c.location_label,
                   c.text,
                   d.filename,
                   d.folder_id,
                   f.name AS folder_name
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN folders f ON f.id = d.folder_id
            WHERE c.space_id = ?
              AND ({' OR '.join(where_parts)})
            ORDER BY c.id
            LIMIT ?
            """,
            (space_id, *params, limit),
        ).fetchall()
        for row in rows:
            item = dict(row)
            if item["chunk_id"] in seen_chunk_ids:
                continue
            item["snippet"] = make_snippet(item["text"], query, expanded_terms)
            item["score"] = 1.0
            results.append(item)
            seen_chunk_ids.add(item["chunk_id"])
            if len(results) >= limit:
                break

        for item in results:
            item["context_text"] = build_chunk_context(conn, item["document_id"], item["chunk_id"])

    results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    search_mode = "hybrid" if has_embedding_backend else ("same_page" if len(required_terms) >= 2 else "local")
    return {
        "query": query,
        "expanded_terms": expanded_terms,
        "required_terms": required_terms,
        "search_mode": search_mode,
        "results": results[:limit],
    }


@app.get("/spaces/{space_id}/word-cloud")
def word_cloud(space_id: int, limit: int = 36) -> dict:
    with connect() as conn:
        rows = conn.execute(
            "SELECT text FROM chunks WHERE space_id = ? ORDER BY id LIMIT 5000",
            (space_id,),
        ).fetchall()
    words = extract_keywords("\n".join(row["text"] for row in rows))
    return {"words": [{"text": word, "weight": count} for word, count in words[: max(1, min(limit, 80))]]}


@app.get("/documents/{document_id}/word-cloud")
def document_word_cloud(document_id: int, limit: int = 40) -> dict:
    with connect() as conn:
        doc = conn.execute("SELECT id FROM documents WHERE id = ?", (document_id,)).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        rows = conn.execute(
            "SELECT text FROM chunks WHERE document_id = ? ORDER BY id",
            (document_id,),
        ).fetchall()
    words = extract_keywords("\n".join(row["text"] for row in rows))
    return {"words": [{"text": word, "weight": count} for word, count in words[: max(1, min(limit, 100))]]}


def delete_upload_path(path: str | Path, directory: bool = False) -> None:
    target = Path(path)
    try:
        resolved_upload_dir = UPLOAD_DIR.resolve()
        resolved_target = target.resolve()
    except Exception:
        return

    if resolved_target != resolved_upload_dir and resolved_upload_dir not in resolved_target.parents:
        return
    if directory:
        rmtree(resolved_target, ignore_errors=True)
        return
    resolved_target.unlink(missing_ok=True)


def save_upload_file(space_id: int, folder_id: int, suffix: str, file: UploadFile) -> Path:
    stored_dir = UPLOAD_DIR / str(space_id) / str(folder_id)
    stored_dir.mkdir(parents=True, exist_ok=True)
    stored_path = stored_dir / f"{uuid4().hex}{suffix}"
    with stored_path.open("wb") as handle:
        copyfileobj(file.file, handle)
    return stored_path


def insert_document(
    space_id: int,
    folder_id: int,
    filename: str,
    suffix: str,
    stored_path: Path,
    chunks: list[tuple[str, str]],
    ocr_status: str,
    ocr_message: str,
) -> dict:
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO documents(space_id, folder_id, filename, file_type, stored_path, ocr_status, ocr_message)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (space_id, folder_id, filename, suffix.lstrip("."), str(stored_path), ocr_status, ocr_message),
        )
        document_id = cur.lastrowid
        conn.executemany(
            """
            INSERT INTO chunks(space_id, document_id, location_label, text)
            VALUES(?, ?, ?, ?)
            """,
            [(space_id, document_id, label, text) for label, text in chunks],
        )
    return {
        "document_id": document_id,
        "filename": filename,
        "folder_id": folder_id,
        "chunk_count": len(chunks),
        "ocr_status": ocr_status,
        "ocr_message": ocr_message,
    }


def render_pdf_pages(document_id: int, stored_path: str, focus_page: int | None = None) -> str:
    try:
        import fitz

        page_count = fitz.open(stored_path).page_count
    except Exception:
        page_count = 0
    if focus_page and page_count:
        start_page = max(1, focus_page - 3)
        end_page = min(page_count, focus_page + 5)
    else:
        start_page = 1
        end_page = min(page_count, 30)
    pages = "\n".join(
        f'<img id="page-{page_number}" class="page" src="/documents/{document_id}/pages/{page_number}.png" alt="第 {page_number} 页" />'
        for page_number in range(start_page, end_page + 1)
    )
    return pages or "<p>这个 PDF 暂时不能渲染为页面。</p>"


def ensure_space(conn, space_id: int) -> None:
    space = conn.execute("SELECT id FROM spaces WHERE id = ?", (space_id,)).fetchone()
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")


def resolve_folder_id(conn, space_id: int, folder_id: int | None) -> int:
    ensure_space(conn, space_id)
    if folder_id is not None:
        folder = conn.execute(
            "SELECT id FROM folders WHERE id = ? AND space_id = ?",
            (folder_id, space_id),
        ).fetchone()
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        return int(folder["id"])

    folder = conn.execute(
        "SELECT id FROM folders WHERE space_id = ? ORDER BY id LIMIT 1",
        (space_id,),
    ).fetchone()
    if folder:
        return int(folder["id"])
    cur = conn.execute("INSERT INTO folders(space_id, name) VALUES(?, ?)", (space_id, "默认文件夹"))
    return int(cur.lastrowid)


def html_escape(value: object) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def extract_required_terms(query: str) -> list[str]:
    terms: list[str] = []
    for match in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", query):
        add_unique_term(terms, match)
    if not terms:
        for part in query.split():
            add_unique_term(terms, part)
    return terms[:8]


def find_same_page_matches(
    conn,
    space_id: int,
    required_terms: list[str],
    query: str,
    limit: int,
) -> list[dict]:
    if len(required_terms) < 2:
        return []

    where_parts = ["c.text LIKE ?" for _ in required_terms]
    rows = conn.execute(
        f"""
        SELECT c.id AS chunk_id,
               c.document_id,
               c.location_label,
               c.text,
               d.filename,
               d.folder_id,
               f.name AS folder_name
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        LEFT JOIN folders f ON f.id = d.folder_id
        WHERE c.space_id = ?
          AND d.file_type = 'pdf'
          AND ({' OR '.join(where_parts)})
        ORDER BY d.id, c.id
        LIMIT 1200
        """,
        (space_id, *[f"%{term}%" for term in required_terms]),
    ).fetchall()

    required_norms = {normalize_term(term) for term in required_terms}
    pages: dict[tuple[int, int], dict] = {}
    for row in rows:
        page_number = parse_page_number(row["location_label"])
        if page_number is None:
            continue
        key = (row["document_id"], page_number)
        page = pages.setdefault(
            key,
            {
                "document_id": row["document_id"],
                "page_number": page_number,
                "matched_terms": set(),
                "first_chunk_id": row["chunk_id"],
            },
        )
        page["first_chunk_id"] = min(page["first_chunk_id"], row["chunk_id"])
        lower_text = row["text"].lower()
        for term in required_terms:
            if term.lower() in lower_text:
                page["matched_terms"].add(normalize_term(term))

    matched_pages = [
        page
        for page in pages.values()
        if required_norms.issubset(page["matched_terms"])
    ]
    matched_pages.sort(key=lambda item: (item["document_id"], item["page_number"], item["first_chunk_id"]))

    results: list[dict] = []
    for page in matched_pages[:limit]:
        page_rows = conn.execute(
            """
            SELECT c.id AS chunk_id,
                   c.document_id,
                   c.location_label,
                   c.text,
                   d.filename,
                   d.folder_id,
                   f.name AS folder_name
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN folders f ON f.id = d.folder_id
            WHERE c.document_id = ?
              AND (c.location_label = ? OR c.location_label LIKE ?)
            ORDER BY c.id
            """,
            (
                page["document_id"],
                f"Page {page['page_number']}",
                f"Page {page['page_number']},%",
            ),
        ).fetchall()
        if not page_rows:
            continue
        first = dict(page_rows[0])
        page_text = "\n".join(row["text"] for row in page_rows if row["text"].strip())
        first["source_chunk_ids"] = [int(row["chunk_id"]) for row in page_rows]
        first["location_label"] = f"Page {page['page_number']} · 同页命中"
        first["snippet"] = make_snippet(page_text, query, required_terms, radius=130)
        first["text"] = page_text[:2500]
        first["same_page_match"] = True
        results.append(first)
    return results


def parse_page_number(label: str) -> int | None:
    match = re.search(r"\bPage\s+(\d+)\b", label or "", flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def normalize_term(term: str) -> str:
    return term.strip().lower()


def expand_search_terms(query: str) -> list[str]:
    terms: list[str] = []
    candidates = [query, *query.split()]
    candidates.extend(re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", query))

    try:
        import jieba

        candidates.extend(word.strip() for word in jieba.cut(query))
    except Exception:
        pass

    for candidate in candidates:
        add_unique_term(terms, candidate)

    normalized = {term.lower() for term in terms}
    for group in SYNONYM_GROUPS:
        if any(term.lower() in normalized or term in query for term in group):
            for synonym in group:
                add_unique_term(terms, synonym)

    return terms[:24]


def add_unique_term(terms: list[str], term: str) -> None:
    clean = term.strip().strip('"').strip("'")
    if len(clean) < 2:
        return
    if clean not in terms:
        terms.append(clean)


def escape_fts_term(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def make_snippet(text: str, query: str, terms: list[str] | None = None, radius: int = 70) -> str:
    search_terms = [query, *(terms or [])]
    match_index = -1
    match_text = query
    lower_text = text.lower()
    for term in search_terms:
        index = lower_text.find(term.lower())
        if index >= 0 and (match_index < 0 or index < match_index):
            match_index = index
            match_text = term
    index = match_index
    if index < 0:
        return text[: radius * 2]
    start = max(0, index - radius)
    end = min(len(text), index + len(match_text) + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:index]}[{text[index:index + len(match_text)]}]{text[index + len(match_text):end]}{suffix}"


def build_chunk_context(conn, document_id: int, chunk_id: int) -> str:
    rows = conn.execute(
        """
        SELECT text
        FROM chunks
        WHERE document_id = ?
          AND id BETWEEN ? AND ?
        ORDER BY id
        """,
        (document_id, max(1, chunk_id - 2), chunk_id + 2),
    ).fetchall()
    text = "".join(row["text"] for row in rows)
    return text[:420]


def extract_keywords(text: str) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "可以",
        "进行",
        "一个",
        "我们",
        "需要",
        "通过",
        "其中",
        "以及",
        "由于",
        "因此",
        "这些",
        "这个",
        "使用",
        "没有",
        "如果",
        "相关",
        "资料",
        "文档",
        "文件",
    }
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text):
        lower = word.lower()
        if lower not in stop_words:
            counter[lower] += 1

    try:
        import jieba

        for word in jieba.cut(text):
            clean = word.strip()
            if len(clean) >= 2 and re.search(r"[\u4e00-\u9fff]", clean) and clean not in stop_words:
                counter[clean] += 2
    except Exception:
        for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            if len(sequence) <= 6:
                if sequence not in stop_words:
                    counter[sequence] += 2
                continue
            for size in (2, 3, 4):
                for index in range(0, len(sequence) - size + 1):
                    word = sequence[index : index + size]
                    if word not in stop_words:
                        counter[word] += 1
    return counter.most_common()
