"""Shared helpers for HMG LoCoMo retrieval runners (extracted from run_hmg_locomo.py)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DOMAIN_PACK = "software-engineering"
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
SCORED_CATEGORIES = {1, 2, 3, 4}


@dataclass
class McpClient:
    store_path: Path
    proc: subprocess.Popen[bytes] | None = None
    _next_id: int = 1

    def __enter__(self):
        self.store_path.mkdir(parents=True, exist_ok=True)
        server_bin = os.environ.get("HMG_SERVER_BIN", "hmg-server")
        self.proc = subprocess.Popen(
            [server_bin, str(self.store_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self._initialize()
        return self

    def __exit__(self, *a):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def _initialize(self):
        self._request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "hmg-locomo-eval", "version": "0.1"}})
        self._notify("notifications/initialized", {})

    def _notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method, params):
        req_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        resp = self._recv()
        return resp

    def call_tool(self, name, arguments):
        resp = self._request("tools/call", {"name": name, "arguments": arguments})
        result = resp.get("result", {})
        if result.get("isError"):
            raise RuntimeError(f"{name} failed: {_join_mcp_text(result.get('content', []))}")
        structured = result.get("structuredContent", {})
        if structured.get("ok") is False:
            raise RuntimeError(f"{name} failed: {structured.get('error')}")
        return structured.get("data") or {}

    def _send(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        frame = b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body
        self.proc.stdin.write(frame)
        self.proc.stdin.flush()

    def _recv(self, timeout=120.0):
        start = time.time()
        header = b""
        while b"\r\n\r\n" not in header:
            if time.time() - start > timeout:
                raise TimeoutError("Timed out waiting for MCP header")
            chunk = self.proc.stdout.read(1)
            if not chunk:
                stderr = self.proc.stderr.read1(4096) if self.proc.stderr else b""
                raise RuntimeError(f"MCP EOF before header. stderr={stderr.decode('utf-8','ignore')}")
            header += chunk
        head, body = header.split(b"\r\n\r\n", 1)
        content_length = None
        for line in head.decode("utf-8").split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
                break
        if content_length is None:
            raise RuntimeError("MCP response missing Content-Length")
        while len(body) < content_length:
            chunk = self.proc.stdout.read(content_length - len(body))
            if not chunk:
                raise RuntimeError("MCP EOF while reading body")
            body += chunk
        return json.loads(body.decode("utf-8"))


def _join_mcp_text(content):
    return "\n".join(item.get("text", "") for item in content if item.get("type") == "text")


def load_dataset(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_turns(conversation):
    turns = []
    for key, value in conversation.items():
        if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list):
            date_key = f"{key}_date_time"
            session_time = conversation.get(date_key, "")
            for turn in value:
                turns.append((key, session_time, turn))
    return turns


def build_memory_text(session_key, session_time, turn):
    dia_id = turn.get("dia_id", "")
    speaker = turn.get("speaker", "")
    text = turn.get("text", "").strip()
    blip = turn.get("blip_caption", "").strip()
    query = turn.get("query", "").strip()
    img_parts = []
    if query:
        img_parts.append(f"image-query: {query}")
    if blip:
        img_parts.append(f"image-caption: {blip}")
    suffix = ""
    if img_parts:
        suffix = " [" + " | ".join(img_parts) + "]"
    return f"[{dia_id}] ({session_key} @ {session_time}) {speaker}: {text}{suffix}".strip()


def parse_dia_ids(memory_text):
    return re.findall(r"\[(D\d+:\d+)\]", memory_text)
