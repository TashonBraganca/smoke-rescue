"""Chat backends for the robot and commander agents.

All backends take OpenAI-style tool schemas and return a Reply with text and tool calls, so the agent loop does not
care which model runs. `ollama` serves a local Qwen model; `bedrock` calls Amazon Bedrock's Converse API; `replay`
returns the replies recorded in an earlier trace, so a run can be reproduced without any model.

    backend = make_backend("ollama", model="qwen3-vl:8b")
    reply = backend.chat(messages, tools)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_s: float = 0.0
    model: str = ""
    raw: dict = field(default_factory=dict)


def request_key(messages: list[dict], tools: list[dict]) -> str:
    """Hash of everything that decides a reply; the replay backend looks replies up by it."""
    blob = json.dumps({"m": messages, "t": tools}, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


class OllamaBackend:
    def __init__(self, model: str = "qwen3-vl:8b", host: str | None = None, seed: int = 7, timeout: float = 300.0):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.seed = seed
        self.client = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        body = {"model": self.model, "messages": [_ollama_message(m) for m in messages], "tools": tools, "stream": False,
                "think": False, "options": {"temperature": 0, "seed": self.seed, "num_ctx": 8192}}
        start = time.perf_counter()
        r = self.client.post(f"{self.host}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = _loads_or_empty(args)
            calls.append(ToolCall(fn.get("name", ""), args))
        return Reply(msg.get("content", "") or "", calls, time.perf_counter() - start, self.model,
                     {k: data.get(k) for k in ("eval_count", "prompt_eval_count", "total_duration")})


class BedrockBackend:
    """Amazon Bedrock Converse API with tool use. Needs AWS credentials with bedrock:InvokeModel on the model."""

    def __init__(self, model: str, region: str | None = None, max_tokens: int = 1024):
        import boto3

        self.model = model
        self.max_tokens = max_tokens
        self.client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        convo = _bedrock_messages([m for m in messages if m["role"] != "system"])
        tool_config = {"tools": [{"toolSpec": {"name": t["function"]["name"], "description": t["function"]["description"],
                                               "inputSchema": {"json": t["function"]["parameters"]}}} for t in tools]}
        start = time.perf_counter()
        out = self.client.converse(modelId=self.model, system=system, messages=convo, toolConfig=tool_config,
                                   inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0})
        content = out["output"]["message"]["content"]
        text = "".join(c.get("text", "") for c in content)
        calls = [ToolCall(c["toolUse"]["name"], c["toolUse"].get("input") or {}) for c in content if "toolUse" in c]
        return Reply(text, calls, time.perf_counter() - start, self.model, {"usage": out.get("usage")})


def _load_env_file(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


class AzureOpenAIBackend:
    """Azure OpenAI chat completions with tools. Credentials come from the environment or from
    ~/.config/smoke-rescue/azure.env (outside the repo, mode 600); they are never logged or written to traces."""

    def __init__(self, deployment: str | None = None, reasoning_effort: str | None = None, timeout: float = 180.0):
        env = {**_load_env_file(Path.home() / ".config" / "smoke-rescue" / "azure.env"), **os.environ}
        self.endpoint = env["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        self.key = env["AZURE_OPENAI_API_KEY"]
        self.version = env.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        self.model = deployment or env.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.6-terra")
        self.effort = reasoning_effort or env.get("AZURE_REASONING_EFFORT", "high")
        self.client = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        body = {"messages": _openai_messages(messages), "tools": tools, "tool_choice": "auto",
                "max_completion_tokens": 2000, "reasoning_effort": self.effort}
        url = f"{self.endpoint}/openai/deployments/{self.model}/chat/completions?api-version={self.version}"
        start = time.perf_counter()
        for attempt in range(4):
            r = self.client.post(url, headers={"api-key": self.key}, json=body)
            if r.status_code == 400 and "reasoning_effort" in r.text and "reasoning_effort" in body:
                body.pop("reasoning_effort")            # deployment does not take the parameter
                continue
            if r.status_code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
                continue
            break
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        calls = [ToolCall(c["function"]["name"], _loads_or_empty(c["function"].get("arguments") or "{}"))
                 for c in msg.get("tool_calls") or []]
        return Reply(msg.get("content") or "", calls, time.perf_counter() - start, self.model,
                     {"usage": data.get("usage")})


def _openai_messages(messages: list[dict]) -> list[dict]:
    """Our neutral messages to OpenAI's format: assistant tool calls get ids, tool results answer them in order."""
    out, pending = [], []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            calls = m.get("tool_calls") or []
            ids = [f"call_{i}_{j}" for j in range(len(calls))]
            pending = list(ids)
            entry = {"role": "assistant", "content": m.get("content") or None}
            if calls:
                entry["tool_calls"] = [{"id": cid, "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                       for cid, c in zip(ids, calls)]
            out.append(entry)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else "call_x", "content": m.get("content", "")})
        else:
            out.append({"role": m["role"], "content": m.get("content", "")})
    return out


class ReplayBackend:
    """Replays model replies from a trace written by an earlier run, keyed by the exact request."""

    def __init__(self, trace_path: Path):
        self.replies: dict[str, dict] = {}
        for line in Path(trace_path).read_text().splitlines():
            rec = json.loads(line)
            if rec.get("kind") == "model_reply":
                self.replies[rec["request_key"]] = rec
        self.model = f"replay:{Path(trace_path).name}"

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        rec = self.replies.get(request_key(messages, tools))
        if rec is None:
            raise KeyError("No recorded reply for this request; the inputs differ from the recorded run.")
        return Reply(rec["text"], [ToolCall(c["name"], c["arguments"]) for c in rec["tool_calls"]], 0.0, self.model)


def make_backend(kind: str | None = None, model: str | None = None, trace: str | None = None):
    kind = kind or os.environ.get("AGENT_BACKEND", "ollama")
    if kind == "ollama":
        return OllamaBackend(model or os.environ.get("AGENT_MODEL", "qwen3-vl:8b"))
    if kind == "azure":
        return AzureOpenAIBackend(model)
    if kind == "bedrock":
        return BedrockBackend(model or os.environ["AGENT_MODEL"])
    if kind == "replay":
        return ReplayBackend(Path(trace or os.environ["AGENT_REPLAY_TRACE"]))
    raise ValueError(f"unknown agent backend {kind!r}")


def image_b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _loads_or_empty(s: str) -> dict:
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def _ollama_message(m: dict) -> dict:
    out = {"role": m["role"], "content": m.get("content", "")}
    if m.get("images"):
        out["images"] = m["images"]
    if m.get("tool_calls"):
        out["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in m["tool_calls"]]
    if m["role"] == "tool":
        out["tool_name"] = m.get("name", "")
    return out


def _bedrock_messages(messages: list[dict]) -> list[dict]:
    """Converse wants alternating user/assistant turns; tool results ride in a user turn as toolResult blocks."""
    out: list[dict] = []
    pending_ids: list[str] = []
    for m in messages:
        if m["role"] == "user":
            blocks = [{"text": m["content"]}]
            for img in m.get("images") or []:
                blocks.append({"image": {"format": "png", "source": {"bytes": base64.b64decode(img)}}})
            out.append({"role": "user", "content": blocks})
        elif m["role"] == "assistant":
            blocks = [{"text": m["content"]}] if m.get("content") else []
            pending_ids = []
            for i, c in enumerate(m.get("tool_calls") or []):
                tid = f"t{len(out)}_{i}"
                pending_ids.append(tid)
                blocks.append({"toolUse": {"toolUseId": tid, "name": c["name"], "input": c["arguments"]}})
            out.append({"role": "assistant", "content": blocks or [{"text": "."}]})
        elif m["role"] == "tool":
            tid = pending_ids.pop(0) if pending_ids else "t0"
            block = {"toolResult": {"toolUseId": tid, "content": [{"text": m["content"]}]}}
            if out and out[-1]["role"] == "user" and any("toolResult" in b for b in out[-1]["content"]):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out
