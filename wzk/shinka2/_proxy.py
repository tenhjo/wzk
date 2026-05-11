"""Anthropic API proxy that routes through Claude Code CLI (Pro Max subscription).

Accepts standard Anthropic /v1/messages requests and forwards them to ``claude -p``
so ShinkaEvolve can use the Pro Max subscription instead of separate API credits.

Usage::

    python -m wzk.shinka2._proxy [--port 8082]

Then in another terminal::

    ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY=promax \\
        python your_runner.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["ProxyHandler", "main"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_PORT = 8082

# Map Anthropic API model IDs -> claude CLI --model values
MODEL_MAP: dict[str, str] = {
    "claude-opus-4-6": "opus",
    "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5-20251001": "haiku",
}


def _format_prompt(body: dict) -> str:
    """Combine system + messages into a single text prompt for claude -p."""
    parts: list[str] = []
    if system := body.get("system"):
        if isinstance(system, list):
            system = "\n".join(b.get("text", "") for b in system if b.get("type") == "text")
        parts.append(f"<system>\n{system}\n</system>\n")
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        parts.append(f"<{role}>\n{content}\n</{role}>\n")
    return "\n".join(parts)


def _call_claude(prompt: str, max_tokens: int, model: str | None = None) -> str:
    """Invoke claude -p and return the text result."""
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--max-turns",
        "1",
        "--tools",
        "",
    ]
    if model:
        cmd.extend(["--model", model])
    # Strip CLAUDECODE env var so claude -p doesn't refuse to run inside a session
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    log.info("Calling claude -p (%d char prompt, max_tokens=%d, model=%s)", len(prompt), max_tokens, model or "default")
    t0 = time.monotonic()
    # Pipe prompt via stdin to avoid OS arg-length limits on large prompts
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600, env=env)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        log.error("claude -p failed (rc=%d): %s", result.returncode, result.stderr[:500])
        raise RuntimeError(f"claude -p exit code {result.returncode}: {result.stderr[:500]}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Fall back to raw stdout if not JSON
        log.warning("claude -p returned non-JSON output, using raw text")
        return result.stdout.strip()
    # claude --output-format json returns {"result": "...", ...}
    text = data.get("result", result.stdout.strip())
    log.info("claude -p responded in %.1fs (%d chars)", elapsed, len(text))
    log.debug("Response preview: %s", text[:200])
    return text


def _make_response(text: str, model: str, input_chars: int = 0) -> dict:
    """Build an Anthropic API-compatible response with estimated token counts."""
    input_tokens = max(1, input_chars // 4)
    output_tokens = max(1, len(text) // 4)
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/v1/messages":
            self._send_error(404, f"Not found: {self.path}")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError) as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        model = body.get("model", "claude-sonnet-4-6")
        max_tokens = body.get("max_tokens", 4096)
        prompt = _format_prompt(body)

        # Map Anthropic model IDs to claude CLI model names
        cli_model = MODEL_MAP.get(model, model)

        try:
            text = _call_claude(prompt, max_tokens, model=cli_model)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            self._send_error(500, str(e))
            return

        resp = _make_response(text, model, input_chars=len(prompt))
        payload = json.dumps(resp).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except BrokenPipeError:
            log.warning("Client disconnected before response was sent (likely SDK timeout)")

    def _send_error(self, code: int, message: str) -> None:
        log.error("HTTP %d: %s", code, message)
        payload = json.dumps({"error": {"type": "proxy_error", "message": message}}).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except BrokenPipeError:
            log.warning("Client disconnected before error response was sent")

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Anthropic API proxy via Claude Code CLI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ProxyHandler)
    log.info("Proxy listening on http://127.0.0.1:%d/v1/messages", args.port)
    log.info("Set ANTHROPIC_BASE_URL=http://localhost:%d ANTHROPIC_API_KEY=promax", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
