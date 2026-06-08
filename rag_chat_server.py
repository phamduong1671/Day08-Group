"""Tiny HTML chatbot server for the RAG pipeline.

Run:
    python rag_chat_server.py

Then open:
    http://localhost:8000
"""

from __future__ import annotations

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

HOST = "127.0.0.1"
PORT = 8000
WEB_DIR = Path(__file__).parent / "web"
DEFAULT_SESSION_ID = "html-ui"

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length)
    for encoding in ("utf-8", "utf-8-sig", "cp1258", "cp1252"):
        try:
            return json.loads(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    return json.loads(raw.decode("utf-8", errors="replace"))


class ChatHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        path = unquote(self.path.split("?", 1)[0])
        if path == "/api/health":
            self.write_json(200, {"ok": True, "backend": "group_project.rag_chatbot_backend"})
            return

        if path == "/":
            path = "/index.html"

        file_path = (WEB_DIR / path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(WEB_DIR.resolve())) or not file_path.is_file():
            self.send_error(404, "Not found")
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(self.path.split("?", 1)[0]).rstrip("/")
        if path == "/api/reset":
            self.handle_reset()
            return

        if path != "/api/chat":
            self.send_error(404, "Not found")
            return

        try:
            payload = read_json_body(self)
            question = str(payload.get("question") or "").strip()
            top_k = int(payload.get("top_k") or 5)
            session_id = str(payload.get("session_id") or DEFAULT_SESSION_ID)
            exact_phrase = bool(payload.get("exact_phrase"))
            if not question:
                raise ValueError("Question is required.")

            from group_project.rag_chatbot_backend import chat

            result = chat(
                question=question,
                session_id=session_id,
                top_k=max(3, min(top_k, 8)),
                exact_phrase=exact_phrase,
            )
            response = {
                "answer": str(result.get("answer") or "").strip()
                or "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
                "retrieval_source": str(result.get("retrieval_source") or "unknown"),
                "generation_backend": str(result.get("generation_backend") or "unknown"),
                "citations": list(result.get("citations") or []),
                "session_id": str(result.get("session_id") or session_id),
                "search_mode": str(result.get("search_mode") or ("exact_phrase" if exact_phrase else "keyword")),
                "sources": list(result.get("source_documents") or result.get("sources") or []),
            }
            self.write_json(200, response)
        except Exception as exc:  # Keep the demo UI alive if a model/service is missing.
            self.write_json(
                500,
                {
                    "answer": (
                        "Chưa tạo được câu trả lời vì backend RAG đang lỗi hoặc LLM chưa chạy. "
                        f"Chi tiết: {exc}"
                    ),
                    "retrieval_source": "error",
                    "sources": [],
                },
            )

    def handle_reset(self) -> None:
        try:
            payload = read_json_body(self)
            session_id = str(payload.get("session_id") or DEFAULT_SESSION_ID)
            from group_project.rag_chatbot_backend import reset_session

            reset_session(session_id)
            self.write_json(200, {"ok": True, "session_id": session_id})
        except Exception as exc:
            self.write_json(500, {"ok": False, "error": str(exc)})

    def write_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), ChatHandler)
    print(f"RAG chatbot HTML server: http://localhost:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
