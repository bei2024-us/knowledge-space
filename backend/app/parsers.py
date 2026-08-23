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
