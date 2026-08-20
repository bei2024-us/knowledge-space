from pathlib import Path
import os
import shutil
import subprocess
from typing import Iterable


OCR_TESSDATA_DIR = Path(r"D:\firstmodel\ocr-tools\tessdata")
TESSERACT_DIR = Path(r"C:\Program Files\Tesseract-OCR")
QPDF_DIR = Path(r"D:\firstmodel\ocr-tools\qpdf\qpdf-12.3.2-msvc64\bin")
GHOSTSCRIPT_DIRS = [
    Path(r"D:\firstmodel\ocr-tools\ghostscript\bin"),
    Path(r"D:\firstmodel\ocr-tools\ghostscript-user\bin"),
    Path(r"D:\firstmodel\ocr-tools\GhostscriptPortable\App\Ghostscript\bin"),
]


class IngestResult:
    def __init__(
        self,
        stored_path: Path,
        chunks: list[tuple[str, str]],
        ocr_status: str = "not_needed",
        ocr_message: str = "",
    ) -> None:
        self.stored_path = stored_path
        self.chunks = chunks
        self.ocr_status = ocr_status
        self.ocr_message = ocr_message


def ingest_file(path: Path, suffix: str) -> IngestResult:
    suffix = suffix.lower()
    chunks = parse_file(path, suffix)
    if suffix != ".pdf":
        return IngestResult(path, chunks)
    if has_enough_text(chunks):
        return IngestResult(path, chunks, "not_needed", "PDF already contains searchable text")

    ocr_path = path.with_name(f"{path.stem}.ocr.pdf")
    status, message = run_ocrmypdf(path, ocr_path)
    if status != "completed":
        return IngestResult(path, chunks, status, message)

    ocr_chunks = parse_pdf(ocr_path)
    if not has_enough_text(ocr_chunks):
        return IngestResult(ocr_path, ocr_chunks, "failed", "OCR completed but no searchable text was extracted")
    return IngestResult(ocr_path, ocr_chunks, "completed", message)


def parse_file(path: Path, suffix: str) -> list[tuple[str, str]]:
    suffix = suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix in {".txt", ".md"}:
        return parse_text(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def has_enough_text(chunks: list[tuple[str, str]]) -> bool:
    text_length = sum(len(text.strip()) for _, text in chunks)
    return text_length >= 80


def parse_pdf(path: Path) -> list[tuple[str, str]]:
    import fitz

    chunks: list[tuple[str, str]] = []
    doc = fitz.open(path)
    for page_index, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        for idx, part in enumerate(split_paragraphs(text), start=1):
            chunks.append((f"Page {page_index}, paragraph {idx}", part))
    return chunks


def run_ocrmypdf(source: Path, target: Path) -> tuple[str, str]:
    executable = shutil.which("ocrmypdf")
    if not executable:
        return "needs_ocr", "OCRmyPDF is not installed on this computer"

    env = os.environ.copy()
    if OCR_TESSDATA_DIR.exists():
        env["TESSDATA_PREFIX"] = str(OCR_TESSDATA_DIR)
    if TESSERACT_DIR.exists():
        env["PATH"] = f"{TESSERACT_DIR}{os.pathsep}{env.get('PATH', '')}"
    for tool_dir in [QPDF_DIR, *GHOSTSCRIPT_DIRS]:
        if tool_dir.exists():
            env["PATH"] = f"{tool_dir}{os.pathsep}{env.get('PATH', '')}"

    command = [
        executable,
        "--force-ocr",
        "--language",
        "chi_sim+eng",
        "--output-type",
        "pdf",
        "--optimize",
        "0",
        str(source),
        str(target),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180, env=env)
    except subprocess.TimeoutExpired:
        return "failed", "OCR timed out"
    except Exception as exc:
        return "failed", f"OCR failed to start: {exc}"

    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "OCRmyPDF failed").strip()
        return "failed", message[-500:]
    return "completed", "OCR completed with OCRmyPDF"


def parse_docx(path: Path) -> list[tuple[str, str]]:
    from docx import Document

    doc = Document(path)
    chunks: list[tuple[str, str]] = []
    for idx, paragraph in enumerate(doc.paragraphs, start=1):
        text = paragraph.text.strip()
        if text:
            chunks.append((f"Paragraph {idx}", text))
    return chunks


def parse_text(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(f"Paragraph {idx}", part) for idx, part in enumerate(split_paragraphs(text), start=1)]


def split_paragraphs(text: str) -> Iterable[str]:
    current = ""
    for raw in text.replace("\r\n", "\n").split("\n"):
        part = " ".join(raw.split())
        if len(part) < 2:
            if current:
                yield current
                current = ""
            continue

        if not current:
            current = part
            continue

        if should_start_new_chunk(current, part):
            yield current
            current = part
        else:
            current = join_text_lines(current, part)

    if current:
        yield current


def should_start_new_chunk(current: str, next_part: str) -> bool:
    if len(current) >= 260:
        return True
    if looks_like_heading(next_part) and len(current) >= 20:
        return True
    return current[-1] in "。！？!?；;"


def looks_like_heading(value: str) -> bool:
    if len(value) > 28:
        return False
    return bool(
        value.startswith(("第", "一、", "二、", "三、", "四、", "五、", "六、"))
        or value.endswith(("章", "节", "模型", "方法", "小结"))
    )


def join_text_lines(left: str, right: str) -> str:
    if not left:
        return right
    if left[-1].isascii() and right[:1].isascii():
        return f"{left} {right}"
    return f"{left}{right}"
