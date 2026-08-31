"""DeepSeek 客户端：把检索到的片段整理成带引用的回答。

设计原则：任何失败（无 key / 缺依赖 / 网络超时 / 非 200 / 解析失败）都返回
`{"usable": False, ...}`，绝不抛异常——这样搜索主流程永远不受 AI 影响。

配置沿用项目既有的 `os.getenv(NAME, default)` 风格。API key 优先读环境变量
DEEPSEEK_API_KEY，缺失时回退读取 backend/data/deepseek_key.txt（已 gitignore，
方便非技术用户直接把 key 粘进文件，无需改环境变量）。
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

try:
    import httpx
except Exception:  # pragma: no cover - httpx 装上前可选
    httpx = None


APP_DIR = Path(__file__).resolve().parents[1]
KEY_FILE = APP_DIR / "data" / "deepseek_key.txt"

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))
LLM_MAX_CONTEXT_CHARS = int(os.getenv("LLM_MAX_CONTEXT_CHARS", "6000"))
# v4 系列是推理模型，思考 token 也从 max_tokens 里扣。原来的 800 对“片段多、内容杂”
# 的整理会被思考占满、返回空正文，所以留足预算。
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2500"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
PER_SOURCE_CHARS = int(os.getenv("LLM_PER_SOURCE_CHARS", "800"))

# 视觉模型：把片段所在页的原图一起喂给 AI。必要性见 parsers.ocr_code_chunks 的注释——
# 课件里的代码是深色截图、关键参数还是青色高亮，Tesseract 读不出（灰度后几乎没对比度），
# 公式又因字体 CMap 损坏而乱码；只有原图里是完整正确的内容。实测视觉模型能逐字读出
# OCR 丢掉的 set.seed(1)、sample(392,196)。
DEEPSEEK_VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL", "deepseek-v4-flash-vision-exp").strip()
LLM_MAX_IMAGES = int(os.getenv("LLM_MAX_IMAGES", "3"))
# 视觉模型是推理模型：单张图就要烧掉约 1000-1400 个思考 token，正文得留足空间，
# 否则思考占满 max_tokens 会返回空正文。6000 实测足够容纳“思考 + 逐行抄代码的长回答”。
LLM_VISION_MAX_TOKENS = int(os.getenv("LLM_VISION_MAX_TOKENS", "6000"))
LLM_VISION_TIMEOUT = float(os.getenv("LLM_VISION_TIMEOUT", "180"))


SYSTEM_PROMPT = (
    "你是严谨的知识库助手。只依据用户提供的资料片段作答，不得编造或引入外部知识。"
    "用简体中文，输出纯文本，不要使用任何 Markdown 符号（不要出现 * # ` > - 等），"
    "需要分点时用“1. 2. 3.”或自然分段。"
    "以用户的问题为中心来组织回答；与问题明显无关的片段可以略去，"
    "但只要片段与问题相关，就应加以利用、不要轻易丢弃。"
    "引用某条片段时用【序号】标注来源。"
)

# 附了原图时追加的规则：文本片段可能缺字/乱码，原图才是真相。
VISION_RULES = (
    "本次还提供了部分片段所在页的原图。这些页的文字片段可能不完整或有乱码："
    "代码在原 PDF 里是截图（文本层没有代码，OCR 也会丢掉彩色高亮的数字），"
    "公式则因字体问题解析成乱码符号。因此凡是原图能看清的内容，一律以原图为准，"
    "不要照抄乱码，也不要说“资料未提及”。"
    "抄写代码或公式时逐字照抄（包括括号里的数字、下标、系数），"
    "代码用普通换行逐行写出，不要加 ``` 围栏，不要用任何 Markdown 符号。"
    "原图上的图表、坐标轴含义、结论数字也要一并利用。"
)


def _resolve_api_key() -> str:
    key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if key:
        return key
    try:
        if KEY_FILE.exists():
            return KEY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def llm_configured() -> bool:
    return bool(_resolve_api_key())


def _build_user_prompt(query: str, sources: list[dict], with_images: bool = False) -> str:
    lines = [f"问题：{query}", "", "检索到的资料片段（每条前为编号）："]
    budget = LLM_MAX_CONTEXT_CHARS
    for s in sources:
        text = (s.get("text") or "").strip().replace("\n", " ")
        if len(text) > PER_SOURCE_CHARS:
            text = text[:PER_SOURCE_CHARS] + "…"
        if budget - len(text) <= 0:
            break
        budget -= len(text)
        loc = s.get("location_label", "")
        fn = s.get("filename", "")
        lines.append(f"【{s['n']}】(文件:{fn} 位置:{loc}) {text}")
    lines += [
        "",
        "请只依据上面的片段、紧扣“问题”整理一段有条理的回答，要求：",
        "1. 先给一句总体结论；",
        "2. 若内容来自多个文件，按文件分别归纳（写明文件名），说明每个文件里与问题相关的部分；",
        "3. 每个要点后用【序号】标注它来自哪条片段；",
        "4. 紧扣问题组织内容，与问题明显无关的片段可以略去；但只要片段与问题相关，就应尽量整理进来，即使内容零散也要利用；",
        "5. 只有当这些片段几乎都与问题无关、确实无从作答时，才写“资料中未提及”；只要有相关内容，就正常整理，不要写这句；",
    ]
    if with_images:
        lines.append(
            "6. 附了原图的片段，请以原图内容为准，把图上的代码逐行照抄、公式按原样写出、"
            "图表里的关键数字和结论也一并说明；文本片段里的乱码一概忽略。"
        )
    lines.append("再次强调：输出纯文本，不要使用 * # ` - 等 Markdown 符号。")
    return "\n".join(lines)


def _build_content(query: str, sources: list[dict], images: list[dict]):
    """有原图时返回 OpenAI 多模态 content 数组，没有则返回纯字符串（走便宜的文本模型）。"""
    prompt = _build_user_prompt(query, sources, with_images=bool(images))
    if not images:
        return prompt
    parts: list[dict] = [{"type": "text", "text": prompt}]
    for img in images:
        png = img.get("png") or b""
        if not png:
            continue
        loc = f"{img.get('filename', '')} 第 {img.get('page_number')} 页"
        parts.append({"type": "text", "text": f"下面是【{img.get('n')}】所在页的原图（{loc}）："})
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()},
            }
        )
    return parts


def _chat(key: str, model: str, content, max_tokens: int, timeout: float) -> dict:
    """调一次 /chat/completions。任何异常都转成 usable=False，不往外抛。"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + (VISION_RULES if not isinstance(content, str) else "")},
            {"role": "user", "content": content},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        resp = httpx.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except Exception as exc:  # 网络 / 超时
        return {"usable": False, "answer": "", "error": f"请求 DeepSeek 失败：{exc}"}

    if resp.status_code != 200:
        return {"usable": False, "answer": "", "error": f"DeepSeek 返回 {resp.status_code}：{resp.text[:200]}"}

    try:
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        answer = ((choice.get("message") or {}).get("content") or "").strip()
        used_model = data.get("model", model)
        finish = choice.get("finish_reason", "")
        usage = data.get("usage") or {}
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    except Exception as exc:
        return {"usable": False, "answer": "", "error": f"解析 DeepSeek 响应失败：{exc}"}

    if not answer:
        # v4 系列是推理模型：思考 token 也算在 max_tokens 里，预算不够时会“思考占满、
        # 正文为空”，finish_reason 是 length。把这些信息带出来，否则只看到“返回空内容”
        # 根本无从判断是预算问题还是别的问题。
        return {
            "usable": False,
            "answer": "",
            "error": f"DeepSeek 返回空正文（finish={finish or '未知'}, 思考token={reasoning}, 预算={max_tokens}）",
            "finish_reason": finish,
            "reasoning_tokens": reasoning,
        }
    return {"usable": True, "answer": answer, "model": used_model}


def _chat_with_retry(key: str, model: str, content, max_tokens: int, timeout: float) -> dict:
    """调一次；若因“思考占满预算”导致正文为空（finish=length），加倍预算再试一次。"""
    result = _chat(key, model, content, max_tokens, timeout)
    if not result.get("usable") and result.get("finish_reason") == "length":
        result = _chat(key, model, content, max_tokens * 2, timeout)
    return result


def summarize_chunks(query: str, sources: list[dict], images: list[dict] | None = None) -> dict:
    """调用 DeepSeek 生成总结。失败一律返回 usable=False，不抛异常。

    images 里每项形如 {"n", "filename", "page_number", "png"}：片段所在页的原图。
    有图时走视觉模型；视觉调用失败会自动退回纯文本模型，保证“AI 整理”不会整块消失。
    """
    key = _resolve_api_key()
    if not key:
        return {"usable": False, "answer": "", "error": "未配置 DEEPSEEK_API_KEY"}
    if httpx is None:
        return {"usable": False, "answer": "", "error": "缺少依赖 httpx，请运行 pip install httpx"}
    if not sources:
        return {"usable": False, "answer": "", "error": "没有可整理的片段"}

    picked = [img for img in (images or []) if img.get("png")][:LLM_MAX_IMAGES]
    if picked:
        result = _chat_with_retry(
            key,
            DEEPSEEK_VISION_MODEL,
            _build_content(query, sources, picked),
            LLM_VISION_MAX_TOKENS,
            LLM_VISION_TIMEOUT,
        )
        if result.get("usable"):
            result["used_images"] = len(picked)
            return result
        vision_error = result.get("error", "")
    else:
        vision_error = ""

    result = _chat_with_retry(key, DEEPSEEK_MODEL, _build_content(query, sources, []), LLM_MAX_TOKENS, LLM_TIMEOUT)
    result["used_images"] = 0
    if vision_error and result.get("usable"):
        # 图片路径失败但文本兜底成功：仍算可用，把原因带回去便于排查。
        result["error"] = f"原图解析未成功（{vision_error}），本次仅根据文本整理"
    return result
