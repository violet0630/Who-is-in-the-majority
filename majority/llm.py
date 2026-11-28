"""OpenAI Chat Completions 兼容客户端；每次请求保留可审计记录。"""

import hashlib
import json
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from time import perf_counter
from threading import Lock
from typing import Callable, TextIO, TypeVar
from urllib.parse import urlsplit

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI
from openai.types.chat import ChatCompletion

from .storage import write_record

T = TypeVar("T")


def canonical_model(model: str) -> str:
    if model in {"deepseek-flash", "deepseek-v4-flash"}:
        return "deepseek-flash"
    return model


def gateway_model(base_url: str, model: str) -> str:
    if canonical_model(model) != "deepseek-flash":
        return model
    url = urlsplit(base_url)
    if url.hostname == "api.1nvictus.me":
        return "deepseek-flash"
    if url.hostname in {"127.0.0.1", "localhost"} and url.port == 3000:
        return "deepseek-v4-flash"
    return model


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://127.0.0.1:3000/v1"
    model: str = "deepseek-v4-flash"
    timeout: float = 60
    max_tokens: int = 4096
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if self.reasoning_effort not in (None, "none", "minimal", "low", "medium", "high"):
            raise ValueError("Invalid reasoning_effort")
        url = urlsplit(self.base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
            raise ValueError("base_url 需要不包含凭据的 HTTP(S) 地址")
        if self.base_url.rstrip("/").endswith("/chat/completions"):
            raise ValueError("SDK base_url 应以 /v1 结尾，不包含 /chat/completions")
        if self.timeout <= 0 or self.max_tokens < 1 or not self.model.strip():
            raise ValueError("模型、超时或最大输出长度无效")
        object.__setattr__(self, "model", gateway_model(self.base_url, self.model))


def load_settings(env_path: Path = Path(".env")) -> tuple[LLMConfig, str]:
    load_dotenv(env_path, override=False)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 或环境变量设置 DEEPSEEK_API_KEY")
    config = LLMConfig(
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "http://127.0.0.1:3000/v1"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        timeout=float(os.environ.get("DEEPSEEK_TIMEOUT", "60")),
        max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "4096")),
        reasoning_effort=os.environ.get("DEEPSEEK_REASONING_EFFORT") or None,
    )
    return config, api_key


def load_prompt(name: str) -> str:
    return files("majority").joinpath("prompts", f"{name}.md").read_text(encoding="utf-8")


def prompt_hash(name: str) -> str:
    return hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Completion:
    raw_json: str
    request_id: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    elapsed_seconds: float


class LLMError(RuntimeError):
    """调用或结构校验失败；日志保存详情，不替换成默认感知。"""


class LLMValidationError(LLMError):
    """模型输出需要纠正。"""


class JsonChat:
    def __init__(self, client: OpenAI, config: LLMConfig, audit: TextIO, reuse_log: Path | None = None) -> None:
        self.client, self.config, self.audit = client, config, audit
        self.lock = Lock()
        self.reuse = {}
        if reuse_log:
            for line in reuse_log.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)["data"]
                if item["status"] == "ok":
                    key = json.dumps([item["task"], item["payload"], item["context"], item["prompt_sha256"],
                                      canonical_model(item["model"]), item["temperature"], item["max_tokens"]], sort_keys=True)
                    self.reuse[key] = item["response"]

    def _record(self, record: dict) -> None:
        with self.lock:
            write_record(self.audit, "llm_call", record)
            self.audit.flush()

    def request(
        self, prompt: str, payload: dict, parse: Callable[[dict], T], *,
        temperature: float, context: dict | None = None,
    ) -> tuple[T, Completion]:
        system = load_prompt(prompt)
        control = ({"reasoning_effort": self.config.reasoning_effort,
                    "base_url": self.config.base_url}
                   if self.config.reasoning_effort is not None else {})
        record = {
            "task": prompt, "context": {**(context or {}), **control}, "model": self.config.model,
            "canonical_model": canonical_model(self.config.model),
            "prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "temperature": temperature, "max_tokens": self.config.max_tokens,
            "payload": payload,
        }
        if self.config.reasoning_effort is not None:
            record["reasoning_effort"] = self.config.reasoning_effort
        start = perf_counter()
        try:
            key = json.dumps([prompt, payload, record["context"], record["prompt_sha256"],
                              canonical_model(record["model"]), temperature, record["max_tokens"]], sort_keys=True)
            record["reused"] = key in self.reuse
            if record["reused"]:
                response = ChatCompletion.model_validate(self.reuse[key])
            else:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                    temperature=temperature,
                    max_tokens=self.config.max_tokens,
                    **({"reasoning_effort": self.config.reasoning_effort}
                       if self.config.reasoning_effort is not None
                       else {"response_format": {"type": "json_object"}}),
                )
        except (APIConnectionError, APIStatusError) as exc:
            record.update(status="api_error", error=type(exc).__name__,
                          http_status=exc.status_code if isinstance(exc, APIStatusError) else None,
                          elapsed_seconds=perf_counter() - start)
            self._record(record)
            raise LLMError(f"{prompt} 接口失败：{type(exc).__name__}，详见调用日志") from None

        elapsed = perf_counter() - start
        record["response"] = response.model_dump(mode="json")
        record["elapsed_seconds"] = elapsed
        try:
            if not response.choices or response.choices[0].finish_reason != "stop":
                raise ValueError("缺少完整输出，可能达到长度限制或被拒绝")
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise ValueError("缺少 JSON 文本")
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("输出必须是 JSON 对象")
            parsed = parse(data)
        except (ValueError, KeyError, TypeError) as exc:
            record.update(status="invalid_response", error=type(exc).__name__)
            self._record(record)
            raise LLMValidationError(f"{prompt} 输出未通过结构/范围校验，详见调用日志") from None
        record["status"] = "ok"
        self._record(record)
        usage = response.usage
        completion = Completion(content, response.id, response.model,
                                usage.prompt_tokens if usage else None,
                                usage.completion_tokens if usage else None, elapsed)
        return parsed, completion


def open_client(config: LLMConfig, api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=config.base_url,
                  timeout=config.timeout, max_retries=0)
