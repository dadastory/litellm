"""Loopback acceptance of native supplier proxy routing, without upstream secrets."""

import asyncio
import base64
import http.client
import json
import logging
import select
import socket
import ssl
import tempfile
import threading
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

import litellm
from litellm.llms.custom_httpx.http_handler import get_httpx_client
from litellm.llms.custom_httpx.outbound_proxy import (
    PROXY_FIELDS,
    current_outbound_proxy,
    outbound_proxy_scope,
    parse_outbound_proxy,
)
from litellm.proxy.management_endpoints.native_model_discovery import _discover
from litellm.types.router import Deployment
from litellm.types.utils import CredentialItem

logging.disable(logging.CRITICAL)
litellm.suppress_debug_info = True
litellm.turn_off_message_logging = True


class Provider(BaseHTTPRequestHandler):
    calls: ClassVar[list[tuple[str, dict]]] = []

    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.reply({"data": [{"id": "proxy-fixture"}], "has_more": False})

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append((self.path, body))
        if self.path.endswith("/messages"):
            self.reply(
                {
                    "id": "msg_fixture",
                    "type": "message",
                    "role": "assistant",
                    "model": "proxy-fixture",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )
        elif self.path.endswith("/responses"):
            self.reply(
                {
                    "id": "resp_fixture",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "proxy-fixture",
                    "error": None,
                    "incomplete_details": None,
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_fixture",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
            )
        elif self.path.endswith("/embeddings"):
            self.reply(
                {
                    "object": "list",
                    "model": "proxy-fixture",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            )
        elif body.get("stream"):
            chunks = [
                {
                    "id": "chatcmpl-fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "proxy-fixture",
                    "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}],
                },
                {
                    "id": "chatcmpl-fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "proxy-fixture",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            payload = ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.reply(
                {
                    "id": "chatcmpl-fixture",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "proxy-fixture",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )


class ForwardProxy(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def authorize(self):
        self.server.calls.append((self.command, self.path, self.headers.get("Proxy-Authorization")))
        if self.server.authentication and self.headers.get("Proxy-Authorization") != self.server.authentication:
            self.send_error(407, "Proxy authentication required")
            return False
        return True

    def do_CONNECT(self):
        if not self.authorize():
            return
        host, port = self.path.rsplit(":", 1)
        with socket.create_connection((host, int(port)), timeout=5) as upstream:
            self.send_response(200)
            self.end_headers()
            for _ in range(200):
                readable, _, _ = select.select([self.connection, upstream], [], [], 1)
                for source in readable:
                    try:
                        data = source.recv(65536)
                    except OSError:
                        return
                    if not data:
                        return
                    (upstream if source is self.connection else self.connection).sendall(data)

    def forward(self):
        if not self.authorize():
            return
        target = urlsplit(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        with closing(http.client.HTTPConnection(target.hostname, target.port, timeout=5)) as connection:
            connection.request(
                self.command,
                target.path + ("?" + target.query if target.query else ""),
                body,
                {k: v for k, v in self.headers.items() if k.lower() not in ("proxy-authorization", "connection")},
            )
            response = connection.getresponse()
            data = response.read()
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    do_GET = forward
    do_POST = forward


class OutboundProxyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servers = []
        cls.temp = tempfile.TemporaryDirectory()
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=1))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1))
            .sign(key, hashes.SHA256())
        )
        keyfile = Path(cls.temp.name) / "key.pem"
        certfile = Path(cls.temp.name) / "cert.pem"
        keyfile.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
        )
        certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        cls.tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        cls.tls.load_cert_chain(certfile, keyfile)
        cls.target = cls.start(Provider)
        cls.secure_target = cls.start(Provider, secure=True)
        cls.proxy = cls.start(ForwardProxy, authentication="Basic " + base64.b64encode(b"alice:p@ss:word").decode())
        cls.other = cls.start(ForwardProxy)
        cls.secure_proxy = cls.start(ForwardProxy, secure=True)

    @classmethod
    def start(cls, handler, authentication=None, secure=False):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        server.authentication = authentication
        server.calls = []
        if secure:
            server.socket = cls.tls.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cls.servers.append(server)
        return server

    @classmethod
    def tearDownClass(cls):
        for server in cls.servers:
            server.shutdown()
            server.server_close()
        cls.temp.cleanup()

    def settings(self, proxy=None, **overrides):
        return {
            "proxy_enabled": "true",
            "proxy_url": f"http://127.0.0.1:{(proxy or self.proxy).server_port}",
            "proxy_username": "alice" if proxy is None else "",
            "proxy_password": "p@ss:word" if proxy is None else "",
            **overrides,
        }

    def params(self, **overrides):
        return {
            "model": "openai/proxy-fixture",
            "api_base": f"http://127.0.0.1:{self.target.server_port}/v1",
            "api_key": "fixture",
            "num_retries": 0,
            "timeout": 5,
            **self.settings(),
            **overrides,
        }

    def test_default_off_and_unavailable_proxy_no_direct_fallback(self):
        before = len(self.proxy.calls)
        result = litellm.completion(
            **self.params(proxy_enabled="false"), messages=[{"role": "user", "content": "ping"}]
        )
        self.assertEqual(result.choices[0].message.content, "ok")
        self.assertEqual(len(self.proxy.calls), before)
        calls = len(Provider.calls)
        with self.assertRaisesRegex(litellm.APIConnectionError, "Proxy authentication failed"):
            litellm.completion(**self.params(proxy_password="wrong"), messages=[{"role": "user", "content": "ping"}])
        self.assertEqual(len(Provider.calls), calls)
        self.assertIsNone(current_outbound_proxy.get())

    def test_native_discovery_health_and_serving(self):
        async def run():
            matrix = [
                ("openai", {"use_chat_completions_api": True}),
                ("openai", {"use_chat_completions_api": False}),
                ("openai", {"model_info": {"supported_endpoints": ["/v1/messages"]}}),
                ("anthropic", {}),
                ("azure", {"api_version": "2024-10-21"}),
            ]
            for provider, extra in matrix:
                params = self.params(model=f"{provider}/proxy-fixture", **extra)
                before = len(self.proxy.calls)
                health = await litellm.ahealth_check(dict(params), mode="anthropic_messages")
                self.assertNotIn("error", health, str(health))
                result = await litellm.anthropic.messages.acreate(
                    **params, messages=[{"role": "user", "content": "ping"}], max_tokens=16
                )
                self.assertTrue(result["content"], (provider, extra, result, Provider.calls[-1][0]))
                self.assertEqual(result["content"][0]["text"], "ok")
                self.assertEqual(len(self.proxy.calls) - before, 2, provider + str(extra))
            await litellm.aembedding(**self.params(), input=["ping"])
            self.assertIsNone(current_outbound_proxy.get())

        asyncio.run(run())
        for provider in ("openai", "anthropic"):
            before = len(self.proxy.calls)
            deployment = Deployment(
                model_name=f"fixture-{provider}/*",
                litellm_params=self.params(model=f"{provider}/*"),
                model_info={"id": f"fixture-{provider}"},
            )
            self.assertTrue(_discover(deployment)["data"])
            self.assertEqual(len(self.proxy.calls), before + 1)
        for _, body in Provider.calls:
            self.assertFalse(PROXY_FIELDS.intersection(body))

    def test_concurrent_streams_and_reconfiguration(self):
        async def run():
            async def request(proxy):
                stream = await litellm.acompletion(
                    **self.params(**self.settings(proxy)), messages=[{"role": "user", "content": "ping"}], stream=True
                )
                text = "".join([chunk.choices[0].delta.content or "" async for chunk in stream])
                self.assertEqual(text, "ok")

            first, second = len(self.proxy.calls), len(self.other.calls)
            await asyncio.gather(*(request(None if i % 2 else self.other) for i in range(6)))
            self.assertEqual(len(self.proxy.calls) - first, 3)
            self.assertEqual(len(self.other.calls) - second, 3)
            await request(self.other)
            self.assertEqual(len(self.other.calls) - second, 4)
            self.assertIsNone(current_outbound_proxy.get())

        asyncio.run(run())

    def test_https_connect_and_https_proxy_certificate_policy(self):
        for proxy, secure in ((self.other, False), (self.secure_proxy, True)):
            for verify in (True, False):
                settings = self.settings(
                    proxy,
                    proxy_url=f"{'https' if secure else 'http'}://127.0.0.1:{proxy.server_port}",
                    proxy_ssl_verify="true" if verify else "false",
                )
                with outbound_proxy_scope(parse_outbound_proxy(settings)):
                    client = get_httpx_client()
                    url = f"https://127.0.0.1:{self.secure_target.server_port}/v1/models"
                    if verify:
                        with self.assertRaises(httpx.ConnectError):
                            client.get(url, timeout=5)
                    else:
                        self.assertEqual(client.get(url, timeout=5).status_code, 200)

    def test_tls_errors_and_cancelled_request_restore_context(self):
        async def run():
            secure = self.params(
                **self.settings(self.other),
                api_base=f"https://127.0.0.1:{self.secure_target.server_port}/v1",
            )
            with self.assertRaisesRegex(litellm.APIConnectionError, "TLS certificate verification failed"):
                await litellm.acompletion(**secure, messages=[{"role": "user", "content": "ping"}])
            secure["proxy_ssl_verify"] = "false"
            result = await litellm.acompletion(**secure, messages=[{"role": "user", "content": "ping"}])
            self.assertEqual(result.choices[0].message.content, "ok")
            with self.assertRaises(asyncio.CancelledError):
                async with asyncio.timeout(5):
                    stream = await litellm.acompletion(
                        **self.params(), messages=[{"role": "user", "content": "ping"}], stream=True
                    )
                    try:
                        await stream.__anext__()
                        raise asyncio.CancelledError()
                    finally:
                        await stream.aclose()
            self.assertIsNone(current_outbound_proxy.get())
            before = len(self.proxy.calls)
            direct = await litellm.acompletion(
                **self.params(proxy_enabled="false"), messages=[{"role": "user", "content": "ping"}]
            )
            self.assertEqual(direct.choices[0].message.content, "ok")
            self.assertEqual(len(self.proxy.calls), before)

        asyncio.run(run())

    def test_native_saved_credential_uses_same_route(self):
        credential = CredentialItem(
            credential_name="proxy-credential-fixture",
            credential_info={},
            credential_values={"api_key": "fixture", **self.settings()},
        )
        previous = litellm.credential_list
        litellm.credential_list = [*previous, credential]
        try:
            before = len(self.proxy.calls)
            params = {
                key: value for key, value in self.params().items() if key not in PROXY_FIELDS and key != "api_key"
            }
            response = litellm.completion(
                **params,
                litellm_credential_name=credential.credential_name,
                messages=[{"role": "user", "content": "ping"}],
            )
            self.assertEqual(response.choices[0].message.content, "ok")
            self.assertEqual(len(self.proxy.calls), before + 1)
        finally:
            litellm.credential_list = previous


if __name__ == "__main__":
    unittest.main()
