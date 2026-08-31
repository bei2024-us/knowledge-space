from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value) if value else None


OCR_TESSDATA_DIR = _optional_path("OCR_TESSDATA_DIR") or _optional_path("TESSDATA_PREFIX")
TESSERACT_DIR = _optional_path("TESSERACT_DIR")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
QPDF_DIR = _optional_path("QPDF_DIR")
GHOSTSCRIPT_DIRS = [
    Path(part)
    for part in os.getenv("GHOSTSCRIPT_DIRS", "").split(os.pathsep)
    if part.strip()
]

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm"}
VIDEO_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv",
    ".wmv", ".m4v", ".mpg", ".mpeg", ".ts", ".3gp",
}
# 同时支持音频和视频：音频直接转写，视频先提取音轨再转写
MEDIA_SUFFIXES = AUDIO_SUFFIXES | VIDEO_SUFFIXES

# 音频转录（ASR）的模型与参数可通过环境变量调整
ASR_MODEL_NAME = os.getenv("ASR_MODEL", "small")   # tiny / base / small / medium / large-v3 / distil-small
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")        # cpu 或 cuda
ASR_DTYPE = os.getenv("ASR_DTYPE", "int8")         # CPU 推荐 int8；GPU 推荐 float16 / int8_float16
ASR_LANG = os.getenv("ASR_LANG", "zh")             # "zh" 或 "auto" 自动检测
ASR_CHUNK_MAX_CHARS = int(os.getenv("ASR_CHUNK_MAX_CHARS", "260"))


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
    if suffix in MEDIA_SUFFIXES:
        media_kind = "video" if suffix in VIDEO_SUFFIXES else "audio"
        status, message, chunks = run_asr(path, media_kind=media_kind)
        return IngestResult(path, chunks, status, message)

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
    if suffix in MEDIA_SUFFIXES:
        # 音视频统一由 run_asr 在 ingest_file 内处理，这里兜底返回空
        media_kind = "video" if suffix in VIDEO_SUFFIXES else "audio"
        _, _, chunks = run_asr(path, media_kind=media_kind)
        return chunks
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
    if OCR_TESSDATA_DIR and OCR_TESSDATA_DIR.exists():
        env["TESSDATA_PREFIX"] = str(OCR_TESSDATA_DIR)
    if TESSERACT_DIR and TESSERACT_DIR.exists():
        env["PATH"] = f"{TESSERACT_DIR}{os.pathsep}{env.get('PATH', '')}"
    tool_dirs = [QPDF_DIR] if QPDF_DIR else []
    tool_dirs.extend(GHOSTSCRIPT_DIRS)
    for tool_dir in tool_dirs:
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


# ---------------------------------------------------------------------------
# 音频解析（ASR）—— 基于 faster-whisper（本地开源，无需额外系统 FFmpeg）
# 参考：https://github.com/SYSTRAN/faster-whisper
# ---------------------------------------------------------------------------


def format_ts(seconds: float) -> str:
    """把秒数格式化为 HH:MM:SS 或 MM:SS，便于 UI 展示与搜索定位。"""
    try:
        seconds_int = int(max(0, float(seconds)))
    except (TypeError, ValueError):
        return "00:00"
    h, remainder = divmod(seconds_int, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def merge_asr_segments(segments, max_chars: int = 260, media_kind: str = "audio") -> list[tuple[str, str]]:
    """
    将 faster-whisper 的 segment 列表合并为适合入库的 chunk。
    合并策略：
      - 优先遵守 Whisper 的自然分句边界；
      - 单个 segment 超过 max_chars 就单独一块并截断到 max_chars；
      - 否则在不超过 max_chars 时前后拼接，label 以该块的起止时间标识。
    media_kind: "audio" 或 "video"，仅影响 chunk label 的前缀。
    """
    kind_label = "Video" if media_kind == "video" else "Audio"
    chunks: list[tuple[str, str]] = []
    current_text = ""
    current_start: float | None = None
    current_end: float | None = None
    seg_idx = 0

    for seg in segments:
        seg_text = (getattr(seg, "text", "") or "").strip()
        if not seg_text:
            continue
        seg_start = float(getattr(seg, "start", 0.0) or 0.0)
        seg_end = float(getattr(seg, "end", seg_start) or seg_start)
        seg_idx += 1

        if current_start is None:
            current_start = seg_start
            current_end = seg_end
            current_text = seg_text
            continue

        # 预计合并后长度
        if current_text[-1:].isascii() and seg_text[:1].isascii():
            glue = " "
        else:
            glue = ""
        merged = current_text + glue + seg_text

        if len(merged) <= max_chars and not (len(current_text) >= 20 and looks_like_heading(seg_text)):
            current_text = merged
            current_end = seg_end
        else:
            label = f"{kind_label} {format_ts(current_start)} -> {format_ts(current_end)}, segment {len(chunks) + 1}"
            chunks.append((label, current_text))
            current_start = seg_start
            current_end = seg_end
            current_text = seg_text

    if current_text:
        label = f"{kind_label} {format_ts(current_start)} -> {format_ts(current_end)}, segment {len(chunks) + 1}"
        chunks.append((label, current_text))

    return chunks


def _extract_audio_to_wav(source: Path) -> tuple[Path | None, str]:
    """
    用 PyAV 把视频文件的音频轨提取为 16kHz 单声道 wav，返回 (临时 wav 路径, 错误信息)。
    成功时返回 (wav_path, "")；失败时返回 (None, error_message)。
    PyAV 是 faster-whisper 的自带依赖，无需额外安装。
    """
    try:
        import av  # type: ignore
        import tempfile
    except Exception as exc:
        return None, f"PyAV 未安装（{exc}），无法从视频提取音频"

    out_path = Path(tempfile.mktemp(suffix=".wav", prefix="asr_extract_"))
    try:
        inp = av.open(str(source))
        try:
            audio_streams = [s for s in inp.streams if s.type == "audio"]
            if not audio_streams:
                return None, "视频文件里没有音频轨，无法转写"
            out = av.open(str(out_path), mode='w')
            try:
                ostream = out.add_stream('pcm_s16le', rate=16000, layout='mono')
                # 新版 PyAV 用 AudioResampler 替代 AudioFrame.resample
                resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
                for frame in inp.decode(audio_streams[0]):
                    for resampled in resampler.resample(frame):
                        for packet in ostream.encode(resampled):
                            out.mux(packet)
                # flush resampler
                for resampled in resampler.resample(None):
                    for packet in ostream.encode(resampled):
                        out.mux(packet)
                for packet in ostream.encode():
                    out.mux(packet)
            finally:
                out.close()
        finally:
            inp.close()
    except Exception as exc:
        # 提取失败时清理半成品
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        return None, f"从视频提取音频失败：{exc}"
    return out_path, ""


def run_asr(source: Path, media_kind: str = "audio") -> tuple[str, str, list[tuple[str, str]]]:
    """
    调用本地 faster-whisper 对音/视频做 ASR 转写。
    - 音频：直接转写
    - 视频：先用 PyAV 提取音轨为 16kHz mono wav，再转写
    返回：(status, message, chunks)
      status:
        - "completed"  转写成功，chunks 非空
        - "failed"     转写过程报错，message 为错误摘要
        - "needs_asr"  faster-whisper 未安装
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # pragma: no cover - 依赖缺失时的兜底
        return (
            "needs_asr",
            "faster-whisper 未安装：请在后端虚拟环境里执行 pip install faster-whisper"
            f"（导入失败原因：{exc}）",
            [],
        )

    # 模型单例缓存：避免每次上传都重新加载模型
    model_cache_key = (ASR_MODEL_NAME, ASR_DEVICE, ASR_DTYPE)
    model: WhisperModel | None = getattr(_asr_model_store, "model", None)
    cache_key_now = getattr(_asr_model_store, "cache_key", None)
    if model is None or cache_key_now != model_cache_key:
        try:
            model = WhisperModel(ASR_MODEL_NAME, device=ASR_DEVICE, compute_type=ASR_DTYPE)
        except Exception as exc:
            return (
                "failed",
                f"加载 Whisper 模型失败（model={ASR_MODEL_NAME}，device={ASR_DEVICE}，"
                f"dtype={ASR_DTYPE}）：{exc}",
                [],
            )
        try:
            _asr_model_store.model = model
            _asr_model_store.cache_key = model_cache_key
        except Exception:
            pass

    # 视频先提取音频为 wav，提取失败则直接尝试用原视频路径（whisper 内部 PyAV 也可能解码）
    asr_source = source
    temp_wav: Path | None = None
    if media_kind == "video":
        wav_path, err = _extract_audio_to_wav(source)
        if wav_path is not None:
            asr_source = wav_path
            temp_wav = wav_path
        # 提取失败时继续用 source 尝试（whisper 内部 PyAV 也可能解码视频音轨）

    try:
        try:
            segments_iter, info = model.transcribe(
                str(asr_source),
                language=None if ASR_LANG == "auto" else ASR_LANG,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=800),
                condition_on_previous_text=False,
            )
            segments = list(segments_iter)
        except TimeoutError as exc:
            return "failed", f"ASR 转写超时：{exc}", []
        except Exception as exc:
            return "failed", f"ASR 转写异常：{exc}", []

        chunks = merge_asr_segments(segments, max_chars=ASR_CHUNK_MAX_CHARS, media_kind=media_kind)
        if not chunks:
            return "failed", "ASR 完成但没有提取到可搜索文字（可能是纯静音或内容过短）", []

        lang = getattr(info, "language", ASR_LANG) or ASR_LANG
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        kind_tag = "video" if media_kind == "video" else "audio"
        return (
            "completed",
            f"ASR completed with faster-whisper, model={ASR_MODEL_NAME}, "
            f"kind={kind_tag}, language={lang}, duration={format_ts(duration)}, segments={len(chunks)}",
            chunks,
        )
    finally:
        # 清理视频提取出的临时 wav
        if temp_wav is not None:
            try:
                if temp_wav.exists():
                    temp_wav.unlink()
            except Exception:
                pass


def parse_audio(path: Path, media_kind: str = "audio") -> list[tuple[str, str]]:
    """parse_file 入口会调用到的底层音/视频解析方法。"""
    _, _, chunks = run_asr(path, media_kind=media_kind)
    return chunks


class _AsrModelStore:
    """简单的进程内缓存，存放已加载的 Whisper 模型实例。"""

    model = None
    cache_key = None


_asr_model_store = _AsrModelStore()


# ---------------------------------------------------------------------------
# 代码截图 OCR 恢复 —— 很多 PDF 把“代码”当作截图/图片嵌入，PyMuPDF 的文本层
# 一个字都取不到（例如整本叫“R代码.pdf”却 0 段能被识别为 R 代码）。这里在后台
# 对“像代码截图”的页做 OCR，只保留“像代码的行”，作为额外 chunk 补进库里。
#
# 关键取舍：
#   · 只“补充代码”，不“替换正文”—— 中文正文的内嵌文本层往往是干净的，OCR 反而更差；
#   · 只 OCR“图片占比高”的页（代码截图页），跳过纯文字页，省时间；
#   · 顺序渲染（PyMuPDF 的 Document 非线程安全）、并行跑 tesseract（慢在这一步）。
# ---------------------------------------------------------------------------

OCR_RENDER_ZOOM = float(os.getenv("OCR_CODE_ZOOM", "3.0"))       # ~216 DPI，实测足够认出代码
OCR_IMAGE_AREA_RATIO = float(os.getenv("OCR_CODE_AREA_RATIO", "0.10"))  # 图片占页面≥10% 才算“代码截图页”
OCR_MAX_PAGES = int(os.getenv("OCR_CODE_MAX_PAGES", "80"))       # 单个文档最多 OCR 的页数上限
OCR_PER_PAGE_TIMEOUT = int(os.getenv("OCR_CODE_PAGE_TIMEOUT", "60"))
OCR_MAX_WORKERS = int(os.getenv("OCR_CODE_WORKERS", "0")) or min((os.cpu_count() or 4), 8)

# “像代码的行”要命中的语法记号：赋值 / 管道 / 比较 / 函数调用 / $取列。
# 注意：不能只凭“出现了括号或方括号”就算代码 —— OCR 噪声（如 "E35] RBS |) gta"）
# 里散落的括号会被误判成代码，反而把乱码塞进搜索结果。所以这里要求出现
# “标识符(” 的调用形式、“标识符=”的赋值，或明确的运算符。
_CODE_TOKEN_RE = re.compile(
    r"<-|<<-|%>%|\|>|->|!=|==|>=|<=|[A-Za-z_.][\w.]*\s*\(|\$[A-Za-z.]|[A-Za-z_.][\w.]*\s*="
)

# “强代码行”：含赋值或管道。用来确认一页 OCR 结果里确实有代码，而不只是控制台
# 输出表头（如 "Estinate Std.Error t value Pr(>|t|)"）或散落的符号噪声。
_CODE_STRONG_RE = re.compile(r"<-|<<-|%>%|\|>|[A-Za-z_.][\w.]*\s*=[^=]")


def _tesseract_env() -> dict:
    env = os.environ.copy()
    if OCR_TESSDATA_DIR and OCR_TESSDATA_DIR.exists():
        env["TESSDATA_PREFIX"] = str(OCR_TESSDATA_DIR)
    if TESSERACT_DIR and TESSERACT_DIR.exists():
        env["PATH"] = f"{TESSERACT_DIR}{os.pathsep}{env.get('PATH', '')}"
    return env


def _tesseract_exe() -> str | None:
    if TESSERACT_CMD:
        return TESSERACT_CMD
    if TESSERACT_DIR:
        candidate = TESSERACT_DIR / "tesseract.exe"
        if candidate.exists():
            return str(candidate)
    return shutil.which("tesseract")


def _page_image_ratio(page) -> float:
    """这一页里图片覆盖的面积占整页的比例（用来判断是不是代码截图页）。"""
    try:
        rect = page.rect
        page_area = float(rect.width) * float(rect.height)
        if page_area <= 0:
            return 0.0
        covered = 0.0
        for info in page.get_image_info():
            bbox = info.get("bbox")
            if not bbox:
                continue
            w = max(0.0, float(bbox[2]) - float(bbox[0]))
            h = max(0.0, float(bbox[3]) - float(bbox[1]))
            covered += w * h
        return covered / page_area
    except Exception:
        # 取不到图片信息时，退化为“有没有图片”
        try:
            return 1.0 if page.get_images() else 0.0
        except Exception:
            return 0.0


def looks_like_code_line(line: str) -> bool:
    """一行文本“像不像代码”：以 ASCII 为主 + 含代码语法记号。中文正文/标题会被排除。"""
    s = line.strip()
    if len(s) < 3:
        return False
    non_space = [ch for ch in s if not ch.isspace()]
    if not non_space:
        return False
    ascii_cnt = sum(1 for ch in non_space if ord(ch) < 128)
    if ascii_cnt / len(non_space) < 0.6:
        return False
    # 代码截图 OCR 出来的代码行基本是纯 ASCII；夹带多个汉字的多半是公式说明或
    # 中文标题（例如 "本总体 P ——~ Z = (21,22,...)"），不当作代码。
    cjk_cnt = sum(1 for ch in s if "一" <= ch <= "鿿")
    if cjk_cnt > 2:
        return False
    return bool(_CODE_TOKEN_RE.search(s))


def _keep_code_lines(ocr_text: str) -> str:
    """从一页 OCR 文本里只挑出“像代码的行”，拼成一段。达不到质量门槛时返回空串。"""
    code_lines = [ln.rstrip() for ln in ocr_text.splitlines() if looks_like_code_line(ln)]
    joined = "\n".join(code_lines)
    # 至少要有一定量，避免单个误判行也建 chunk
    if len(joined.replace("\n", "")) < 24 and len(code_lines) < 2:
        return ""
    # 必须至少有一行真正的赋值/管道，否则大概率只是 R 控制台输出表头或 OCR 噪声
    if not any(_CODE_STRONG_RE.search(ln) for ln in code_lines):
        return ""
    return joined


def _ocr_png(png_path: Path, exe: str, env: dict) -> str:
    """对单张 PNG 跑 tesseract（chi_sim+eng, psm 6），返回识别文本；失败返回空串。"""
    try:
        completed = subprocess.run(
            [exe, str(png_path), "stdout", "-l", "chi_sim+eng", "--psm", "6"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=OCR_PER_PAGE_TIMEOUT,
            env=env,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout or ""


def ocr_code_chunks(pdf_path: Path) -> list[tuple[str, str]]:
    """对 PDF 里“代码截图页”做 OCR，抽取代码行，返回 [(label, text), ...]。

    label 形如 "Page 13 代码(OCR)"，保留 "Page N" 前缀以便前端页码跳转 (parse_page_number)。
    任何环节失败都安全降级为空列表，绝不抛异常影响上传主流程。
    """
    exe = _tesseract_exe()
    if not exe:
        return []
    try:
        import fitz
    except Exception:
        return []

    env = _tesseract_env()
    tmp_dir = Path(tempfile.mkdtemp(prefix="code_ocr_"))
    rendered: list[tuple[int, Path]] = []   # (page_number, png_path)
    try:
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return []
        matrix = fitz.Matrix(OCR_RENDER_ZOOM, OCR_RENDER_ZOOM)
        try:
            # 顺序渲染候选页（fitz 的 Document 非线程安全）
            for page_index, page in enumerate(doc, start=1):
                if len(rendered) >= OCR_MAX_PAGES:
                    break
                if _page_image_ratio(page) < OCR_IMAGE_AREA_RATIO:
                    continue
                try:
                    pix = page.get_pixmap(matrix=matrix)
                    png_path = tmp_dir / f"p{page_index}.png"
                    pix.save(str(png_path))
                    rendered.append((page_index, png_path))
                except Exception:
                    continue
        finally:
            doc.close()

        if not rendered:
            return []

        # 并行 OCR（tesseract 是子进程，能真正并行）
        def worker(item: tuple[int, Path]) -> tuple[int, str]:
            page_number, png_path = item
            return page_number, _ocr_png(png_path, exe, env)

        chunks: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS) as pool:
            for page_number, ocr_text in pool.map(worker, rendered):
                code_text = _keep_code_lines(ocr_text)
                if code_text:
                    chunks.append((f"Page {page_number} 代码(OCR)", code_text))
        chunks.sort(key=lambda c: c[0])
        return chunks
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
