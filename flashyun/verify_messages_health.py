"""Offline regression of native diagnostic and serving route parity."""

import asyncio
import json
import logging
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import litellm

logging.disable(logging.CRITICAL)
litellm.suppress_debug_info = True
litellm.turn_off_message_logging = True


class MessagesHealthTest(unittest.TestCase):
    def test_diagnostic_uses_the_same_upstream_route_as_messages(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            responses_supported = False

            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body["model"]))
                if self.path == "/relay/v1/chat/completions":
                    status, result = (
                        200,
                        {
                            "id": "chatcmpl-probe",
                            "object": "chat.completion",
                            "created": 1,
                            "model": "route-probe",
                            "choices": [
                                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                            ],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                        },
                    )
                elif self.path == "/relay/v1/responses" and self.responses_supported:
                    status, result = (
                        200,
                        {
                            "id": "resp-probe",
                            "object": "response",
                            "created_at": 1,
                            "status": "completed",
                            "model": "route-probe",
                            "error": None,
                            "incomplete_details": None,
                            "output": [
                                {
                                    "type": "message",
                                    "id": "msg-probe",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                                }
                            ],
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        },
                    )
                elif self.path == "/relay/v1/messages":
                    status, result = (
                        200,
                        {
                            "id": "msg-probe",
                            "type": "message",
                            "role": "assistant",
                            "model": "route-probe",
                            "content": [{"type": "text", "text": "ok"}],
                            "stop_reason": "end_turn",
                            "stop_sequence": None,
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        },
                    )
                else:
                    status, result = 404, {"error": {"type": "not_found_error", "message": "Unsupported test route"}}
                data = json.dumps(result).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}/relay"

        async def run():
            for label, provider, suffix, extra, path, succeeds in [
                ("default", "openai", "/v1", {}, "/relay/v1/responses", False),
                ("chat", "openai", "/v1", {"use_chat_completions_api": True}, "/relay/v1/chat/completions", True),
                ("responses", "openai", "/v1", {"use_chat_completions_api": False}, "/relay/v1/responses", False),
                (
                    "messages",
                    "openai",
                    "/v1/messages",
                    {"model_info": {"supported_endpoints": ["/v1/messages"]}},
                    "/relay/v1/messages",
                    True,
                ),
                ("anthropic", "anthropic", "", {}, "/relay/v1/messages", True),
                (
                    "messages-base",
                    "openai",
                    "",
                    {"model_info": {"supported_endpoints": ["/v1/messages"]}},
                    "/relay/v1/messages",
                    True,
                ),
                (
                    "messages-v1",
                    "openai",
                    "/v1",
                    {"model_info": {"supported_endpoints": ["/v1/messages"]}},
                    "/relay/v1/messages",
                    True,
                ),
                (
                    "responses-supported",
                    "openai",
                    "/v1",
                    {"use_chat_completions_api": False},
                    "/relay/v1/responses",
                    True,
                ),
                ("default-supported", "openai", "/v1", {}, "/relay/v1/responses", True),
            ]:
                with self.subTest(label=label):
                    Handler.responses_supported = label.endswith("-supported")
                    params = {
                        "model": f"{provider}/route-probe",
                        "api_base": base + suffix,
                        "api_key": "local-probe",
                        "num_retries": 0,
                        "timeout": 5,
                        **extra,
                    }
                    requests.clear()
                    result = await litellm.ahealth_check(dict(params), mode="anthropic_messages")
                    self.assertEqual("error" not in result, succeeds, label)
                    self.assertEqual(requests, [(path, "route-probe")], label)
                    requests.clear()
                    try:
                        await litellm.anthropic.messages.acreate(
                            **params, messages=[{"role": "user", "content": "ping"}], max_tokens=16
                        )
                        serving_ok = True
                    except litellm.NotFoundError:
                        serving_ok = False
                    self.assertEqual(serving_ok, succeeds)
                    self.assertEqual(requests, [(path, "route-probe")])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
