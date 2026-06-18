"""
API 客户端 — 统一的 LLM API 调用封装。

支持三种 Provider：
  - deepseek: 直连 DeepSeek API（urllib，零额外依赖）
  - litellm:  通过 liteLLM 统一接口调用 100+ 模型（OpenAI / Anthropic / Gemini / Groq / Qwen ...）
  - custom:   任意 OpenAI 兼容 API（Ollama / vLLM / 自定义端点）

内置重试机制（指数退避）。
"""

import json
from http.client import IncompleteRead
import socket
import ssl
import time
import urllib.error
import urllib.request
from rich.console import Console

console = Console()


class RetryableAPIError(RuntimeError):
    retryable = True


def build_chat_completion_payload(
    cfg: dict,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> bytes:
    """Build the JSON payload for /chat/completions."""
    payload = {
        "model": cfg.get("model", "deepseek-v4-pro"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": cfg.get("temperature", 0.3),
        "max_tokens": max_tokens,
        "stream": False,
    }

    api_base = cfg.get("api_base", "https://api.deepseek.com/v1")
    model = str(payload["model"])
    thinking = cfg.get("thinking")
    if thinking is None and "api.deepseek.com" in api_base and model.startswith("deepseek-v4"):
        thinking = "disabled"

    if thinking:
        payload["thinking"] = thinking if isinstance(thinking, dict) else {"type": str(thinking)}

    if cfg.get("reasoning_effort"):
        payload["reasoning_effort"] = cfg["reasoning_effort"]

    return json.dumps(payload).encode("utf-8")


def parse_chat_completion_response(body: bytes) -> tuple[str, dict]:
    """Parse a chat completion response into content and token usage."""
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API 返回了无效 JSON: {exc}") from exc

    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError("API 返回为空：没有 choices")

    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content") or ""
    usage = result.get("usage") or {}
    usage_summary = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }

    if content.strip():
        return content.strip(), usage_summary

    finish_reason = choice.get("finish_reason")
    completion_details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = completion_details.get("reasoning_tokens")
    reasoning_content = message.get("reasoning_content") or ""

    details = []
    if finish_reason:
        details.append(f"finish_reason={finish_reason}")
    if usage_summary["completion_tokens"]:
        details.append(f"completion_tokens={usage_summary['completion_tokens']}")
    if reasoning_tokens is not None:
        details.append(f"reasoning_tokens={reasoning_tokens}")
    if reasoning_content:
        details.append("reasoning_content 非空但最终 content 为空")

    suffix = f" ({', '.join(details)})" if details else ""
    advice = (
        "。模型可能把输出 token 用在 reasoning 上，未生成最终译文；"
        "请增大 max_tokens_per_chunk、减小 chunk_target_tokens，或换用非 reasoning 模型"
        if finish_reason == "length" or reasoning_tokens
        else ""
    )
    raise RuntimeError(f"LLM returned empty response{suffix}{advice}")


def format_incomplete_read_error(exc: IncompleteRead) -> str:
    partial = exc.partial or b""
    return (
        f"API 响应读取中断: IncompleteRead({len(partial)} bytes read)。"
        "服务端或网络代理提前关闭连接，可安全重试；如果频繁出现，"
        "请提高 request_timeout 或减小 chunk_target_tokens"
    )


def _iter_error_chain(exc: BaseException):
    seen = set()
    current = exc
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        yield current

        reason = current.reason if isinstance(current, urllib.error.URLError) else None
        if isinstance(reason, BaseException) and id(reason) not in seen:
            current = reason
            continue

        current = current.__cause__ or current.__context__


def _is_retryable_connection_error(exc: BaseException) -> bool:
    retryable_types = (
        IncompleteRead,
        ssl.SSLEOFError,
        ssl.SSLZeroReturnError,
        ConnectionResetError,
        BrokenPipeError,
        TimeoutError,
        socket.timeout,
    )
    retryable_fragments = (
        "unexpected_eof_while_reading",
        "eof occurred in violation of protocol",
        "connection reset",
        "connection aborted",
        "remote end closed connection",
        "timed out",
    )

    for current in _iter_error_chain(exc):
        if isinstance(current, retryable_types):
            return True

        message = str(current).lower()
        if any(fragment in message for fragment in retryable_fragments):
            return True

    return False


def _format_retryable_connection_error(exc: BaseException) -> str:
    for current in _iter_error_chain(exc):
        if isinstance(current, IncompleteRead):
            return format_incomplete_read_error(current)

    return (
        f"API 连接被提前关闭: {exc}。"
        "服务端或网络代理提前关闭连接，可安全重试；如果频繁出现，"
        "请提高 request_timeout、减小 chunk_target_tokens，或稍后重试"
    )


def call_openai_compatible_chat(
    cfg: dict,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
) -> tuple[str, dict]:
    """Call an OpenAI-compatible /chat/completions endpoint."""
    api_key = cfg.get("api_key") or cfg.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("未设置 API Key。请设置环境变量或配置 api_key。")

    api_base = cfg.get("api_base", "https://api.deepseek.com/v1").rstrip("/")
    timeout = cfg.get("request_timeout", 300)
    payload = build_chat_completion_payload(cfg, system_prompt, user_prompt, max_tokens)

    req = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return parse_chat_completion_response(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp else str(exc)
        raise RuntimeError(f"API 错误 ({exc.code}): {body[:500]}") from exc
    except IncompleteRead as exc:
        raise RetryableAPIError(format_incomplete_read_error(exc)) from exc
    except RuntimeError:
        raise
    except Exception as exc:
        if _is_retryable_connection_error(exc):
            raise RetryableAPIError(_format_retryable_connection_error(exc)) from exc
        raise RuntimeError(f"API 调用失败: {exc}") from exc


def call_api(
    api_key: str,
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
    retry_max_delay: float = 30.0,
    request_timeout: int | float = 300,
    provider: str = "deepseek",
) -> tuple[str, dict]:
    """调用 LLM API，带重试。

    Args:
        api_key: API 密钥
        api_base: API Base URL（如 https://api.deepseek.com/v1），liteLLM 模式下可留空
        model: 模型名。liteLLM 模式下使用 "provider/model" 格式
        system_prompt: 系统提示
        user_prompt: 用户提示
        max_tokens: 最大输出 token 数
        temperature: 温度参数
        max_retries: 最大重试次数
        retry_base_delay: 重试基础延迟（秒）
        retry_max_delay: 重试最大延迟（秒）
        request_timeout: 单次 HTTP 请求超时时间（秒）
        provider: 提供商类型 — "deepseek" | "litellm" | "custom"

    Returns:
        (response_text, usage_dict) — usage_dict 含 prompt_tokens, completion_tokens
    """
    if provider == "litellm":
        return _call_via_litellm(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
        )
    else:
        # deepseek / custom: 直连 OpenAI 兼容 API
        return _call_via_http(
            api_key=api_key,
            api_base=api_base,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            request_timeout=request_timeout,
        )


def _call_via_http(
    api_key: str,
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
    request_timeout: int | float,
) -> tuple[str, dict]:
    """直连 OpenAI 兼容 API（urllib 实现，无第三方依赖）。"""
    cfg = {
        "api_key": api_key,
        "api_base": api_base,
        "model": model,
        "temperature": temperature,
        "request_timeout": request_timeout,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            return call_openai_compatible_chat(cfg, system_prompt, user_prompt, max_tokens)
        except RetryableAPIError as e:
            last_error = e
        except RuntimeError as e:
            last_error = e
        except Exception as e:
            last_error = RuntimeError(f"API 调用失败: {e}")

        if attempt < max_retries - 1:
            delay = min(retry_base_delay * (2 ** attempt), retry_max_delay)
            console.print(f"  [yellow]⚠️ 第{attempt+1}次失败: {last_error}，{delay:.0f}s 后重试[/yellow]")
            time.sleep(delay)

    if last_error is None:
        raise RuntimeError("API 调用失败: unknown error")
    raise last_error


def _call_via_litellm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    api_key: str = "",
    api_base: str = "",
) -> tuple[str, dict]:
    """通过 liteLLM 调用任意 LLM 提供商。

    liteLLM 支持 100+ 模型，统一接口：
      - openai/gpt-4o
      - anthropic/claude-sonnet-4-20250514
      - gemini/gemini-2.5-pro
      - groq/llama-4-maverick-17b-128e
      - deepseek/deepseek-chat
      - ... 等
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "liteLLM 未安装。运行: uv sync --extra litellm 或 uv add litellm"
        )

    # liteLLM 自动从环境变量读取各 provider 的 API key
    # 如 OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY 等
    # 也可以显式传入 api_key
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    try:
        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content
        usage = response.usage
        return (content.strip() if content else ""), {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        }
    except Exception as e:
        raise RuntimeError(f"liteLLM 调用失败 [{model}]: {e}")


def get_available_litellm_models() -> list[str]:
    """返回 liteLLM 支持的常用翻译模型列表。"""
    return [
        # OpenAI
        "openai/gpt-5.5",
        "openai/gpt-5.5-mini",
        # Anthropic
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-fable-5",
        # Google
        "gemini/gemini-3.5-pro",
        "gemini/gemini-3.5-flash",
        # DeepSeek
        "deepseek/deepseek-chat",
        "deepseek/deepseek-reasoner",
        # Mimo
        "mimo/mimo-v2.5-pro",
        "mimo/mimo-v2.5-omni",
        # Mistral
        "mistral/mistral-large-latest",
        "mistral/mistral-small-latest",
    ]
