from base64 import b64decode
from collections import Counter
import os
from pathlib import Path
import re
from shutil import copyfileobj, rmtree
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from .embeddings import (
    backfill_space_embeddings,
    embedding_backend_name,
    load_space_embeddings,
    semantic_rank_chunks,
    store_chunk_embeddings,
)
from .db import UPLOAD_DIR, connect, ensure_storage, init_db
from .parsers import MEDIA_SUFFIXES, ingest_file


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
    {"物理", "力学", "physics"},
    {"化学", "chemistry", "反应"},
    {"编程", "代码", "程序", "coding", "programming"},
    {"算法", "algorithm"},
    {"数据结构", "data structure"},
    {"经济", "经济学", "economics"},
    {"统计", "统计学", "statistics", "概率论"},
    {"实验", "实践", "experiment"},
    {"方法", "步骤", "流程", "做法", "过程"},
    {"原理", "机理", "机制", "principle"},
    {"例子", "案例", "示例", "样例", "example"},
    {"区别", "差异", "不同", "对比", "difference"},
    {"优点", "优势", "好处", "长处"},
    {"缺点", "不足", "短板", "劣势"},
]


# ---- 编程语言识别：按“语法特征”判断片段是不是某语言的代码 ----
# 动机：搜“R语言”时，用户真正想要的是“真正的 R 代码 / 真正在讲 R 的内容”。
#   · 有的片段本身就是 R 代码，却通篇没出现“R语言”三个字 —— 按词匹配会漏掉；
#   · 有的片段只是顺口提一句“R语言”，并没有代码或讲解 —— 按词匹配会误命中。
# 所以这里不看“有没有出现语言名”，而是看“有没有该语言独有的语法特征”。
LANGUAGE_ALIASES = {
    "R": ["r语言", "r 语言", "r代码", "r script", "rstudio", "rlang", "ggplot", "tidyverse", "dplyr"],
    "Python": ["python", "py代码", "python代码", "pandas", "numpy"],
    "SQL": ["sql", "sql语句", "sql查询", "数据库查询"],
}

# 每种语言的“强特征”（命中≥1条基本可确定是该语言）与“弱特征”（辅助加分）。
LANGUAGE_SIGNATURES = {
    "R": {
        "strong": [
            r"<<?-",                              # x <- 1 / x <<- 1，R 最标志性的赋值符
            r"%>%|\|>",                           # 管道
            r"\blibrary\s*\(",                    # library(ggplot2)
            r"\brequire\s*\(",
            r"\binstall\.packages\s*\(",
            r"\bset\.seed\s*\(",                  # set.seed(1)，统计/重抽样代码里极常见且几乎 R 专属
            r"\bcv\.glm\s*\(",                    # 重抽样章节高频
            r"\bggplot\s*\(",
            r"\bdata\.frame\s*\(",
            r"\bread\.(csv|table)\s*\(",
            r"\bas\.(numeric|factor|character|data\.frame)\s*\(",
        ],
        "weak": [
            r"\bc\s*\([^)]*\)",                   # c(1, 2, 3) 向量
            r"\bfunction\s*\([^)]*\)\s*\{",       # function(x){ ... }
            r"\b(TRUE|FALSE|NULL|NA)\b",
            r"\b(summary|head|str|print|plot|mean|sd|lm|glm|factor|vector|matrix)\s*\(",
            # ISLR/统计课件常见 R 函数（帮 OCR 出来的代码提分、排到前面）
            r"\b(sample|predict|boot|knn|lda|qda|rpart|tree|randomForest|prcomp|kmeans|rnorm|runif|attach|subset|apply|sapply|lapply|scale|coef|resid|fitted|poly)\s*\(",
            r"\bdata\s*=\s*\w+",                  # data=Auto
            r"\bsubset\s*=\s*\w+",                # subset=train
            r"\$[A-Za-z.]",                       # df$col
        ],
    },
    "Python": {
        "strong": [
            r"^\s*def\s+\w+\s*\(",
            r"^\s*import\s+\w+",
            r"^\s*from\s+\w+\s+import\b",
            r"\b(pd|np|plt)\.\w+\s*\(",
        ],
        "weak": [
            r"\bprint\s*\(",
            r"\bself\.",
            r"\b(True|False|None)\b",
            r"\brange\s*\(",
            r":\s*$",
        ],
    },
    "SQL": {
        "strong": [
            r"\bSELECT\b[\s\S]*?\bFROM\b",
            r"\bINSERT\s+INTO\b",
            r"\bUPDATE\b[\s\S]*?\bSET\b",
            r"\bCREATE\s+TABLE\b",
        ],
        "weak": [
            r"\bWHERE\b",
            r"\bGROUP\s+BY\b",
            r"\bJOIN\b",
            r"\bORDER\s+BY\b",
        ],
    },
}

_COMPILED_SIGNATURES = {
    lang: {
        "strong": [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in spec["strong"]],
        "weak": [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in spec["weak"]],
    }
    for lang, spec in LANGUAGE_SIGNATURES.items()
}


def detect_query_language(query: str) -> str | None:
    """判断这次搜索是不是在找某种编程语言（如“R语言/Python/SQL”）。"""
    q = query.lower()
    for lang, aliases in LANGUAGE_ALIASES.items():
        if any(alias in q for alias in aliases):
            return lang
    return None


def score_code_language(text: str, lang: str) -> float:
    """片段“像不像该语言的代码”的分数：强特征各 2 分、弱特征各 1 分。

    要求“至少命中 1 条强特征且总分≥2”才算数，否则返回 0 —— 这样“只是顺口提到
    语言名、却没有任何代码”的片段不会被误判为代码，也就不会被加权。
    """
    spec = _COMPILED_SIGNATURES.get(lang)
    if not spec or not text:
        return 0.0
    strong_hits = sum(1 for pat in spec["strong"] if pat.search(text))
    if strong_hits == 0:
        return 0.0
    weak_hits = sum(1 for pat in spec["weak"] if pat.search(text))
    score = strong_hits * 2.0 + weak_hits * 1.0
    return score if score >= 2.0 else 0.0


# 数学公式/符号字体在 ToUnicode CMap 损坏或缺失时，PyMuPDF 会把符号错映射成这些字符，
# 是最典型的“乱码信号”（正常中文课件几乎不会出现 Ø ß Þ 之类）。
_GARBLE_CHARS = "ØØßÞþðÐ¶§¤¦®©µ¬±÷×"


def looks_garbled(text: str) -> bool:
    """判断片段是不是“公式/符号字体解析出来的乱码”，用于从搜索结果里隐藏它。

    只在“几乎没有可读内容、充斥错映射符号”时判为乱码——正常中文/英文/代码/数字表格都不会被误伤。
    """
    if not text:
        return False
    s = text.strip()
    n = len(s)
    if n < 8:
        return False
    cjk = sum(1 for ch in s if "一" <= ch <= "鿿")
    letters = sum(1 for ch in s if ch.isalpha() and ord(ch) < 128)
    digits = sum(1 for ch in s if ch.isdigit())
    spaces = sum(1 for ch in s if ch.isspace())
    garble = sum(1 for ch in s if ch in _GARBLE_CHARS)
    non_space = n - spaces
    if non_space <= 0:
        return False
    meaningful_ratio = (cjk + letters + digits) / non_space
    # 信号一：错映射符号密集出现
    if garble >= 3 and garble / non_space > 0.06:
        return True
    # 信号二：可读内容（中文/单词/数字）占比过低，且几乎没有中文——典型公式符号汤
    if meaningful_ratio < 0.45 and cjk < 4 and non_space >= 12:
        return True
    return False


# ---------------------------------------------------------------------------
# 通用“本质识别”
#
# 用户最初的抱怨（“文件里只是提到 R 语言，并没有真正在讲述”）本质上不是代码问题，
# 而是两个更一般的问题：
#   (1) 这段内容“是什么东西”——是代码、公式、定义、定理、例题、步骤还是图表？
#   (2) 这段内容“是在讲这个主题，还是只顺口提了一句”？
# 原来只为 R/Python/SQL 写了 (1) 的特例（score_code_language）。下面把两者都做成通用能力：
# content type 用结构特征识别，aboutness 用“出现密度 + 词覆盖 + 是否出现在标题”判别。
# ---------------------------------------------------------------------------

CONTENT_TYPE_PATTERNS: dict[str, list[str]] = {
    # 公式：等号两侧有运算、数学符号、希腊字母、上下标
    "formula": [
        r"[=＝][^=\n]{0,40}[+\-*/^]",
        r"[=＝]\s*[-−]?\d",
        r"[∑∫∏√±≈≤≥≠∞∈∂]",
        r"[α-ωΑ-Ω]",
        r"\^\s*\{?\d",
        r"\b(MSE|RSS|RSE|SSE|Var|Cov|E)\s*[（(]",
    ],
    "definition": [r"定义\s*[:：\d]", r"定义为", r"称为", r"叫做", r"是指", r"记作", r"所谓"],
    "theorem": [r"定理", r"引理", r"推论", r"性质\s*[\d:：]", r"证明\s*[:：]", r"满足.{0,10}条件"],
    "example": [r"例\s*[\d一二三四五六七八九十]", r"例题", r"举例", r"案例", r"习题", r"练习\s*[\d:：]"],
    "steps": [r"第[一二三四五六七八九十\d]+步", r"步骤", r"流程", r"首先[^。]{0,40}(其次|然后|接着)"],
    "figure": [r"图\s*\d", r"如图", r"图示", r"横轴|纵轴|坐标轴"],
    "table": [r"表\s*\d", r"如表", r"下表"],
    "conclusion": [r"结论", r"综上", r"因此[^。]{0,12}(得出|说明|表明)", r"可以看出", r"说明了"],
}

_COMPILED_CONTENT_TYPES = {
    name: [re.compile(p, re.IGNORECASE) for p in pats] for name, pats in CONTENT_TYPE_PATTERNS.items()
}

# 查询里出现这些词 → 用户想要的是对应类型的内容。
# 刻意不放单字“图”“表”：它们会被“地图/表示/表明/代表”之类误触发。
QUERY_INTENT_PATTERNS: dict[str, list[str]] = {
    "formula": ["公式", "算式", "表达式", "计算方法", "怎么算", "如何计算", "推导", "求解"],
    "definition": ["定义", "什么是", "是什么", "含义", "概念", "叫什么", "指的是"],
    "theorem": ["定理", "引理", "推论", "证明", "性质"],
    "example": ["例题", "例子", "案例", "习题", "举例", "样例", "练习"],
    "steps": ["步骤", "流程", "怎么做", "如何做", "操作方法", "过程是"],
    "figure": ["图表", "示意图", "曲线", "折线图", "散点图", "流程图", "图像"],
    "table": ["表格", "对照表", "数据表"],
    "conclusion": ["结论", "总结", "要点", "小结"],
}


def detect_chunk_types(text: str, label: str = "") -> set[str]:
    """识别片段“是什么东西”：公式/定义/定理/例题/步骤/图表/表格/结论。

    一个片段可以同时属于多个类型（例如“例题里带公式”），这是刻意的。
    """
    if not text:
        return set()
    blob = f"{label}\n{text}"
    found = {name for name, pats in _COMPILED_CONTENT_TYPES.items() if any(p.search(blob) for p in pats)}
    if "代码(OCR)" in (label or ""):
        found.add("code")
    return found


def detect_query_intent(query: str) -> set[str]:
    """判断这次搜索想要“哪一类内容”。没有明显意图时返回空集合（不干预排序）。"""
    if not query:
        return set()
    q = query.lower()
    return {name for name, words in QUERY_INTENT_PATTERNS.items() if any(w in q for w in words)}


def first_line(text: str, limit: int = 60) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:limit]
    return ""


def aboutness_score(text: str, label: str, terms: list[str]) -> float:
    """判别“在讲这个主题”还是“只提到了一下”。

    三个信号：出现密度（每百字命中次数）、查询词覆盖率、是否出现在标题/首行。
    只提一句的长段落密度极低、覆盖率低、标题里也没有，自然拿不到分。
    """
    if not text or not terms:
        return 0.0
    low = text.lower()
    length = max(len(text), 1)
    hits = 0
    covered = 0
    for term in terms:
        count = low.count(term.lower())
        if count:
            covered += 1
        hits += count
    if hits == 0:
        return 0.0
    density = hits / (length / 100.0)
    # 极短片段（多为只有一行标题）密度天然虚高，按长度折算，避免“只有标题没内容”的
    # 片段压过真正展开讲这个主题的段落。它仍可凭 coverage + 标题命中被召回。
    if length < 40:
        density *= length / 40.0
    coverage = covered / len(terms)
    head = f"{first_line(text)} {label or ''}".lower()
    head_hit = any(term.lower() in head for term in terms)
    score = min(density, 6.0) + coverage * 3.0
    if head_hit:  # 标题/首行就点名了这个主题 —— 最强的“整段在讲它”信号
        score += 3.0
    if hits >= 3:
        score += 1.0
    return round(score, 3)


class SpaceCreate(BaseModel):
    name: str
    description: str = ""


class FolderCreate(BaseModel):
    name: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 20


class SummarizeRequest(BaseModel):
    query: str
    chunk_ids: list[int] = []
    # 是否把片段所在页的原图一起喂给 AI（默认开）。关掉可换取更快、更省的纯文本整理。
    include_images: bool = True


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
    background_tasks: BackgroundTasks,
    folder_id: int | None = Form(None),
    query_folder_id: int | None = Query(None, alias="folder_id"),
    file: UploadFile = File(...),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md", *MEDIA_SUFFIXES}:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF, DOCX, TXT, MD, audio and video files are supported"
                f" ({', '.join(sorted(MEDIA_SUFFIXES))})"
            ),
        )

    with connect() as conn:
        folder_id = resolve_folder_id(conn, space_id, folder_id if folder_id is not None else query_folder_id)

    stored_path = save_upload_file(space_id, folder_id, suffix, file)
    try:
        ingest = ingest_file(stored_path, suffix)
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc

    result = insert_document(
        space_id=space_id,
        folder_id=folder_id,
        filename=file.filename or stored_path.name,
        suffix=suffix,
        stored_path=ingest.stored_path,
        chunks=ingest.chunks,
        ocr_status=ingest.ocr_status,
        ocr_message=ingest.ocr_message,
    )
    # PDF：响应发出后在后台 OCR 代码截图页，把代码补进库（不阻塞上传）。
    if suffix == ".pdf":
        background_tasks.add_task(ocr_code_recover, result["document_id"], space_id, str(ingest.stored_path))
    return result


@app.post("/spaces/{space_id}/files/base64")
def upload_file_base64(space_id: int, payload: Base64UploadRequest, background_tasks: BackgroundTasks) -> dict:
    suffix = Path(payload.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md", *MEDIA_SUFFIXES}:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF, DOCX, TXT, MD, audio and video files are supported"
                f" ({', '.join(sorted(MEDIA_SUFFIXES))})"
            ),
        )

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

    result = insert_document(
        space_id=space_id,
        folder_id=folder_id,
        filename=payload.filename,
        suffix=suffix,
        stored_path=ingest.stored_path,
        chunks=ingest.chunks,
        ocr_status=ingest.ocr_status,
        ocr_message=ingest.ocr_message,
    )
    # PDF：响应发出后在后台 OCR 代码截图页，把代码补进库（不阻塞上传）。
    if suffix == ".pdf":
        background_tasks.add_task(ocr_code_recover, result["document_id"], space_id, str(ingest.stored_path))
    return result


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
    focus_tip = (
        f'<div class="focus-tip">已定位到第 {focus_page} 页（蓝框标出）</div>'
        if focus_page and doc["file_type"] == "pdf"
        else ""
    )
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
          .page.focus {{ outline: 3px solid #1478ff; outline-offset: 2px; }}
          .focus-tip {{ position: sticky; top: 0; z-index: 5; background: #1478ff; color: #fff; font-size: 13px;
                        padding: 8px 12px; border-radius: 10px; margin-bottom: 12px; }}
          .chunk {{ padding: 14px; margin-bottom: 12px; border: 1px solid #e5edf6; border-radius: 18px; background: #fff; }}
          .chunk:target {{ border-color: #1478ff; box-shadow: 0 0 0 3px rgba(20, 120, 255, 0.12); }}
          .chunk strong {{ color: #2563eb; font-size: 13px; }}
          .chunk p {{ line-height: 1.7; overflow-wrap: anywhere; }}
        </style>
      </head>
      <body><main><h1>{title}</h1><div class="meta">OCR 状态：{html_escape(doc["ocr_status"] or "")}</div>{focus_tip}{body}</main>
        <script>
          // 定位：等目标页图片加载出尺寸后再滚动，避免图片未加载导致滚到错误位置。
          (function () {{
            var focusPage = {focus_page or 0};
            if (!focusPage) return;
            var target = document.getElementById('page-' + focusPage);
            if (!target) return;
            function go() {{ target.scrollIntoView({{ block: 'start' }}); }}
            if (target.complete) {{ go(); }} else {{ target.addEventListener('load', go); }}
            window.addEventListener('load', go);
          }})();
        </script>
      </body>
    </html>
    """


def render_page_png(stored_path: str, page_number: int, zoom: float) -> bytes:
    """把 PDF 的第 page_number 页（1 起）渲染成 PNG 字节。页码非法时抛 IndexError。"""
    import fitz

    pdf = fitz.open(stored_path)
    try:
        if page_number < 1 or page_number > pdf.page_count:
            raise IndexError("page out of range")
        page = pdf.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")
    finally:
        pdf.close()


# 喂给 AI 的原图分辨率：1.5 倍（约 108 DPI）实测足够让视觉模型逐字读出代码截图里的
# 数字，再高只会增加体积和耗时。
LLM_IMAGE_ZOOM = float(os.getenv("LLM_IMAGE_ZOOM", "1.5"))


def collect_page_images(ordered_rows: list[dict]) -> list[dict]:
    """为待整理的片段挑选并渲染“所在页原图”，供 AI 直接看图整理。

    按“文本有多不可靠”排序挑选：OCR 恢复的代码页（文本层本来一个字都没有）最优先，
    其次是乱码的公式页，最后才是普通 PDF 页；同一页只送一次，数量上限由 llm 侧控制。
    任何一页渲染失败都跳过，不影响其余页和整个整理流程。
    """
    from .llm import LLM_MAX_IMAGES

    candidates: list[tuple[int, int, dict]] = []
    for idx, row in enumerate(ordered_rows):
        if (row.get("file_type") or "") != "pdf" or not row.get("stored_path"):
            continue
        page_number = parse_page_number(row.get("location_label", ""))
        if not page_number:
            continue
        label = row.get("location_label") or ""
        if "代码(OCR)" in label:
            need = 0  # 代码是截图，文本层没有 → 最需要看图
        elif looks_garbled(row.get("text", "")):
            need = 1  # 公式乱码 → 很需要看图
        else:
            need = 2
        candidates.append((need, idx, {**row, "page_number": page_number, "n": idx + 1}))

    candidates.sort(key=lambda item: (item[0], item[1]))
    images: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for _need, _idx, row in candidates:
        if len(images) >= max(LLM_MAX_IMAGES, 0):
            break
        key = (int(row["document_id"]), int(row["page_number"]))
        if key in seen:
            continue
        seen.add(key)
        try:
            png = render_page_png(str(row["stored_path"]), int(row["page_number"]), LLM_IMAGE_ZOOM)
        except Exception:
            continue
        images.append(
            {
                "n": row["n"],
                "filename": row.get("filename", ""),
                "page_number": int(row["page_number"]),
                "png": png,
            }
        )
    return images


@app.get("/documents/{document_id}/pages/{page_number}.png")
def document_page_image(document_id: int, page_number: int, zoom: float = Query(1.6, ge=0.5, le=4.0)) -> Response:
    """渲染 PDF 某一页为 PNG。zoom 可调：手机上看代码截图/公式需要更高清晰度。"""
    with connect() as conn:
        doc = conn.execute("SELECT file_type, stored_path FROM documents WHERE id = ?", (document_id,)).fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc["file_type"] != "pdf":
        raise HTTPException(status_code=400, detail="Only PDF documents can render pages")

    try:
        png = render_page_png(doc["stored_path"], page_number, zoom)
    except IndexError:
        raise HTTPException(status_code=404, detail="Page not found")
    return Response(content=png, media_type="image/png")


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


RRF_K = 60
RRF_WEIGHTS = {"same_page": 3.0, "code": 3.0, "intent": 2.5, "about": 2.0, "semantic": 1.5, "bm25": 1.5, "like": 0.5}
RRF_PRIORITY = {"same_page": 3, "code": 3, "intent": 2, "semantic": 2, "bm25": 1, "about": 1, "like": 0}


def fuse_results(sources: dict[str, list[dict]]) -> list[dict]:
    """Reciprocal Rank Fusion：每一路按 weight/(K+rank) 累加，多路一致命中会叠加得分。

    这取代了原先的"固定档位分 + dedup-drop"：现在同一个片段被语义、bm25、
    LIKE 同时命中会互相加分，而不是被后面的来源直接丢弃。保留高权重的
    same_page（同页精确共现）让"精确命中优先"的既有优点不丢。
    """
    fused: dict[int, dict] = {}
    for name, rows in sources.items():
        weight = RRF_WEIGHTS.get(name, 1.0)
        priority = RRF_PRIORITY.get(name, 0)
        for rank, row in enumerate(rows, start=1):
            try:
                cid = int(row["chunk_id"])
            except (KeyError, TypeError, ValueError):
                continue
            contrib = weight / (RRF_K + rank)
            entry = fused.get(cid)
            if entry is None:
                fused[cid] = {"score": contrib, "sources": {name}, "row": dict(row), "priority": priority}
            else:
                entry["score"] += contrib
                entry["sources"].add(name)
                if priority > entry["priority"]:  # 保留更优来源的正文/snippet（same_page 整页文本最有用）
                    merged = dict(entry["row"])
                    merged.update(row)
                    entry["row"] = merged
                    entry["priority"] = priority

    items: list[dict] = []
    for entry in fused.values():
        row = dict(entry["row"])
        row["score"] = round(entry["score"], 6)
        row["match_sources"] = sorted(entry["sources"])
        items.append(row)
    items.sort(key=lambda r: r["score"], reverse=True)
    return items


@app.post("/spaces/{space_id}/search")
def search(space_id: int, payload: SearchRequest) -> dict:
    query = payload.query.strip()
    if not query:
        return {"query": query, "expanded_terms": [], "search_mode": "empty", "results": []}

    limit = max(1, min(payload.limit, 50))
    expanded_terms = expand_search_terms(query) or [query]
    required_terms = extract_required_terms(query)
    fts_query = " OR ".join(escape_fts_term(term) for term in expanded_terms if term)
    has_embedding_backend = embedding_backend_name() is not None

    # 收集四路召回，各自产出有序列表；不再 dedup-drop、不再给常数分，交给 RRF 融合
    same_page_rows: list[dict] = []
    semantic_rows: list[dict] = []
    bm25_rows: list[dict] = []
    like_rows: list[dict] = []

    with connect() as conn:
        ensure_space(conn, space_id)

        for item in find_same_page_matches(conn, space_id, required_terms, query, min(limit, 12)):
            same_page_rows.append(item)

        if has_embedding_backend:
            for _ in range(8):
                if backfill_space_embeddings(conn, space_id, batch_size=64) == 0:
                    break
            # 让语义分支也吃同义词：原始 query 拼上前几个扩展词一起 embed，
            # 这样搜"微积分"也能语义召回只写"高数/calculus"的片段。
            extra = [t for t in expanded_terms if t.lower() != query.lower()][:6]
            sem_query = (query + " " + " ".join(extra)).strip() if extra else query
            for item in semantic_rank_chunks(sem_query, load_space_embeddings(conn, space_id), min(limit * 3, 36)):
                item["snippet"] = make_snippet(item["text"], query, expanded_terms, radius=110)
                item["semantic_match"] = True
                semantic_rows.append(item)

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
                bm25_rows = [dict(row) for row in rows]
            except Exception:
                bm25_rows = []

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
            item["snippet"] = make_snippet(item["text"], query, expanded_terms)
            like_rows.append(item)

        # 编程语言识别召回：搜“R语言/Python/SQL”等时，按“语法特征”把真正是该语言
        # 代码的片段找出来（即使片段里没出现语言名），并天然排除“只是顺口提到语言名、
        # 没有任何代码”的片段——后者拿不到这一路的高权重，自然会沉下去。
        code_rows: list[dict] = []
        query_lang = detect_query_language(query)
        if query_lang:
            scan_rows = conn.execute(
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
                WHERE c.space_id = ?
                ORDER BY c.id
                LIMIT 4000
                """,
                (space_id,),
            ).fetchall()
            scored: list[tuple[float, dict]] = []
            for row in scan_rows:
                code_score = score_code_language(row["text"], query_lang)
                # OCR 阶段已确认是代码的片段（label 带“代码(OCR)”）给一个兜底分，
                # 保证即使 OCR 噪声让语法特征没命中，这些代码也能被这一路召回；
                # 有语法特征命中的（score≥2）仍排在兜底分之前。
                if code_score <= 0 and "代码(OCR)" in (row["location_label"] or ""):
                    code_score = 1.5
                if code_score > 0:
                    scored.append((code_score, dict(row)))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            for code_score, item in scored[: min(limit * 3, 36)]:
                item["snippet"] = make_snippet(item["text"], query, expanded_terms, radius=110)
                item["code_match"] = True
                item["code_lang"] = query_lang
                item["code_score"] = round(code_score, 2)
                code_rows.append(item)

        # 通用“本质识别”两路：把只对代码生效的能力推广到公式/定义/定理/例题/步骤/图表。
        #  intent 路：查询里带“公式/定义/例题/步骤…”时，按结构特征找真正是那类内容的片段；
        #  about  路：不论查什么，都把“整段在讲这个主题”的片段提上来，压住“只顺口提一句”的。
        # 两路共用一次全表扫描，避免多一次 IO。
        intent_rows: list[dict] = []
        about_rows: list[dict] = []
        query_intents = detect_query_intent(query)
        scan_terms = [t for t in expanded_terms if len(t) >= 2][:12]
        if query_intents or scan_terms:
            scan_rows = conn.execute(
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
                WHERE c.space_id = ?
                ORDER BY c.id
                LIMIT 4000
                """,
                (space_id,),
            ).fetchall()
            intent_scored: list[tuple[float, dict]] = []
            about_scored: list[tuple[float, dict]] = []
            for row in scan_rows:
                text = row["text"] or ""
                label = row["location_label"] or ""
                about = aboutness_score(text, label, scan_terms) if scan_terms else 0.0
                # 先算 aboutness 再判类型：类型识别要跑几十条正则，只对“确实在讲这个主题”的
                # 片段跑，避免每次搜索都在全库上做上万次正则匹配。
                if query_intents and about > 0:
                    types = detect_chunk_types(text, label)
                    matched = types & query_intents
                    # 必须“既是那类内容、又确实在讲这个主题”才算命中，
                    # 否则搜“XX的公式”会把全库所有公式都捞上来。
                    if matched:
                        item = dict(row)
                        item["content_types"] = sorted(types)
                        intent_scored.append((len(matched) * 2.0 + about, item))
                if about >= 3.0:  # 阈值：低于此基本就是“只提到一次”的长段落
                    about_scored.append((about, dict(row)))
            intent_scored.sort(key=lambda pair: pair[0], reverse=True)
            about_scored.sort(key=lambda pair: pair[0], reverse=True)
            for _s, item in intent_scored[: min(limit * 2, 24)]:
                item["snippet"] = make_snippet(item["text"], query, expanded_terms, radius=110)
                item["intent_match"] = sorted(query_intents & set(item.get("content_types", [])))
                intent_rows.append(item)
            for about, item in about_scored[: min(limit * 2, 24)]:
                item["snippet"] = make_snippet(item["text"], query, expanded_terms)
                item["about_score"] = about
                about_rows.append(item)

        # RRF 融合：奖励多路一致命中，取代原先的固定档位分
        fused = fuse_results(
            {
                "same_page": same_page_rows,
                "code": code_rows,
                "intent": intent_rows,
                "about": about_rows,
                "semantic": semantic_rows,
                "bm25": bm25_rows,
                "like": like_rows,
            }
        )
        for item in fused:
            if not item.get("snippet"):
                item["snippet"] = make_snippet(item.get("text", ""), query, expanded_terms)

        # 页码先算出来：既给前端显示原页图片/一键定位，也决定乱码公式能不能“以图代文”保留。
        for item in fused:
            item["page_number"] = parse_page_number(item.get("location_label", ""))

        # 隐藏“公式/符号字体解析出来的乱码”片段（用户反馈搜索结果里有定位不到的乱码符号）。
        # 两个例外：
        #   1. 代码片段（含 OCR 恢复的代码）本来就以符号为主，不是乱码；
        #   2. 用户明确在找公式、而这段又能定位到 PDF 某一页时，改成“以图代文”——
        #      文本层还原不了公式，但那一页的原图里公式是完整正确的，前端会直接显示。
        def _keep(item: dict) -> bool:
            if item.get("code_match") or "代码(OCR)" in (item.get("location_label") or ""):
                return True
            if not looks_garbled(item.get("text", "")):
                return True
            if "formula" in query_intents and item.get("page_number"):
                item["garbled_formula"] = True
                item["snippet"] = f"该公式的文本层解析为乱码，原文第 {item['page_number']} 页原图里是完整的公式。"
                return True
            return False

        fused = [item for item in fused if _keep(item)]
        results = fused[:limit]

        for item in results:
            item["context_text"] = build_chunk_context(conn, item["document_id"], item["chunk_id"])

    search_mode = "hybrid" if has_embedding_backend else ("same_page" if len(required_terms) >= 2 else "local")
    return {
        "query": query,
        "expanded_terms": expanded_terms,
        "required_terms": required_terms,
        "search_mode": search_mode,
        # 这次识别出的“要什么类型的内容”（公式/定义/例题…），空表示无明显意图。
        "query_intents": sorted(query_intents),
        "results": results[:limit],
    }


@app.post("/spaces/{space_id}/summarize")
def summarize(space_id: int, payload: SummarizeRequest) -> dict:
    """用 DeepSeek 把检索到的片段整理成带引用的回答。未配置 key 时优雅降级（HTTP 200）。"""
    from .llm import llm_configured, summarize_chunks

    if not llm_configured():
        return {
            "usable": False,
            "answer": "",
            "citations": [],
            "message": "未配置 AI：在后端设置环境变量 DEEPSEEK_API_KEY 后即可使用。",
        }

    query = payload.query.strip()
    ids = payload.chunk_ids[:8]
    if not query or not ids:
        return {"usable": False, "answer": "", "citations": [], "message": "没有可整理的片段。"}

    placeholders = ",".join("?" * len(ids))
    with connect() as conn:
        ensure_space(conn, space_id)
        rows = conn.execute(
            f"""
            SELECT c.id AS chunk_id,
                   c.location_label,
                   c.text,
                   c.document_id,
                   d.filename,
                   d.file_type,
                   d.stored_path
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.space_id = ?
              AND c.id IN ({placeholders})
            """,
            (space_id, *ids),
        ).fetchall()

    order = {cid: i for i, cid in enumerate(ids)}  # 保持前端传入的顺序
    ordered = sorted((dict(r) for r in rows), key=lambda r: order.get(r["chunk_id"], 999))
    sources = [
        {
            "n": i + 1,
            "chunk_id": r["chunk_id"],
            "filename": r["filename"],
            "location_label": r["location_label"],
            "text": r["text"],
        }
        for i, r in enumerate(ordered)
    ]
    if not sources:
        return {"usable": False, "answer": "", "citations": [], "message": "没有可整理的片段。"}

    images = collect_page_images(ordered) if payload.include_images else []
    result = summarize_chunks(query, sources, images)
    citations = [
        {"n": s["n"], "chunk_id": s["chunk_id"], "filename": s["filename"], "location_label": s["location_label"]}
        for s in sources
    ]
    return {
        "usable": bool(result.get("usable")),
        "answer": result.get("answer", ""),
        "model": result.get("model"),
        "used_images": result.get("used_images", 0),
        "citations": citations,
        "message": result.get("error", ""),
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


def ocr_code_recover(document_id: int, space_id: int, stored_path: str) -> None:
    """后台任务：对 PDF 里的“代码截图页”做 OCR，把识别出的代码作为额外 chunk 补进库。

    设计：上传接口先秒回（内嵌文本层立即可搜），本函数在响应发出后于线程池里运行，
    ~几秒到几十秒后把代码补齐。任何异常都被吞掉，绝不影响上传/搜索主流程。
    """
    try:
        from .parsers import ocr_code_chunks
    except Exception:
        return

    try:
        code_chunks = ocr_code_chunks(Path(stored_path))
    except Exception as exc:
        try:
            with connect() as conn:
                conn.execute(
                    "UPDATE documents SET ocr_message = ? WHERE id = ?",
                    (f"代码 OCR 失败：{exc}"[:480], document_id),
                )
        except Exception:
            pass
        return

    if not code_chunks:
        return

    new_ids: list[int] = []
    new_texts: list[str] = []
    try:
        with connect() as conn:
            # 幂等：先清掉本文档旧的 OCR 代码 chunk（重复处理时不叠加）；
            # DELETE 会触发 FTS 删除触发器、并按外键级联删掉旧向量。
            conn.execute(
                "DELETE FROM chunks WHERE document_id = ? AND location_label LIKE '%代码(OCR)%'",
                (document_id,),
            )
            for label, text in code_chunks:
                cur = conn.execute(
                    "INSERT INTO chunks(space_id, document_id, location_label, text) VALUES(?, ?, ?, ?)",
                    (space_id, document_id, label, text),
                )
                new_ids.append(int(cur.lastrowid))
                new_texts.append(text)
            conn.execute(
                "UPDATE documents SET ocr_message = ? WHERE id = ?",
                (f"已通过 OCR 补充 {len(new_ids)} 段代码", document_id),
            )
    except Exception:
        return

    # 给新 chunk 生成向量（失败无所谓：搜索时 backfill_space_embeddings 会补上）。
    try:
        if new_ids and embedding_backend_name() is not None:
            with connect() as conn:
                store_chunk_embeddings(conn, new_ids, new_texts)
    except Exception:
        pass


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
        f'<img id="page-{page_number}" class="page{" focus" if page_number == focus_page else ""}"'
        f' src="/documents/{document_id}/pages/{page_number}.png?zoom=2.2" alt="第 {page_number} 页" />'
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
