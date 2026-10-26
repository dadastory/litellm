"""Offline regression tests executed against the actual pinned native image."""

import unittest
from unittest.mock import Mock, patch

import httpx
import litellm
from fastapi import HTTPException
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class ProviderDiscoveryTests(unittest.TestCase):
    def test_openai_preserves_base_path(self):
        for base, expected in (
            ("https://provider.example", "/v1/models"),
            ("https://provider.example/", "/v1/models"),
            ("https://provider.example/v1/", "/v1/models"),
            ("https://provider.example/api/coding/v3/", "/api/coding/v3/models"),
            ("http://[::1]:9090/custom/v1", "/custom/v1/models"),
        ):
            with self.subTest(base=base):

                def respond(request, expected=expected):
                    self.assertEqual(request.url.path, expected)
                    self.assertEqual(request.headers["authorization"], "Bearer fixture")
                    return httpx.Response(200, json={"data": [{"id": "org/exact-model"}]})

                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    with patch.object(
                        litellm.module_level_client.client,
                        "get",
                        side_effect=client.get,
                    ):
                        self.assertEqual(
                            OpenAIGPTConfig().get_models("fixture", base),
                            ["org/exact-model"],
                        )

    def test_anthropic_keeps_standard_auth_and_normalizes_version_suffix(self):
        for base in (
            "https://provider.example/gateway",
            "https://provider.example/gateway/v1/",
        ):
            with self.subTest(base=base):

                def respond(request):
                    self.assertEqual(request.url.path, "/gateway/v1/models")
                    self.assertEqual(request.headers["x-api-key"], "fixture")
                    self.assertNotIn("authorization", request.headers)
                    return httpx.Response(200, json={"data": [{"id": "claude-fixture"}]})

                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    with patch.object(
                        litellm.module_level_client.client,
                        "get",
                        side_effect=client.get,
                    ) as call:
                        self.assertEqual(
                            AnthropicModelInfo().get_models("fixture", base),
                            ["anthropic/claude-fixture"],
                        )
                        self.assertEqual(call.call_count, 1)

    def test_custom_anthropic_auth_fallback_stays_on_same_url(self):
        for rejected in (401, 403):
            with self.subTest(status=rejected):
                requests = []

                def respond(request, requests=requests, rejected=rejected):
                    requests.append(request)
                    if "x-api-key" in request.headers:
                        return httpx.Response(rejected)
                    self.assertEqual(request.headers["authorization"], "Bearer fixture")
                    return httpx.Response(200, json={"data": [{"id": "org/exact-model"}]})

                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    with patch.object(
                        litellm.module_level_client.client,
                        "get",
                        side_effect=client.get,
                    ):
                        self.assertEqual(
                            AnthropicModelInfo().get_models("fixture", "https://provider.example/custom"),
                            ["anthropic/org/exact-model"],
                        )
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0].url, requests[1].url)

    def test_authentication_fallback_is_bounded(self):
        for base, code, calls in (
            ("https://api.anthropic.com", 401, 1),
            ("https://provider.example", 429, 1),
            ("https://provider.example", 500, 1),
            ("https://provider.example", 401, 2),
        ):
            with self.subTest(base=base, code=code):

                def respond(request, code=code):
                    return httpx.Response(code, json={"error": "private-secret-and-address"})

                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    with patch.object(
                        litellm.module_level_client.client,
                        "get",
                        side_effect=client.get,
                    ) as call:
                        with self.assertRaises(httpx.HTTPStatusError):
                            AnthropicModelInfo().get_models("fixture", base)
                        self.assertEqual(call.call_count, calls)


class StrictDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_deployment_and_safe_errors(self):
        from litellm.proxy import proxy_server
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.native_model_discovery import (
            discover_model,
        )
        from litellm.types.router import Deployment
        from litellm.utils import _model_cache

        selected = Deployment(
            model_name="selected/*",
            litellm_params={
                "model": "openai/*",
                "api_key": "fixture",
                "api_base": "https://provider.example/custom/v1",
            },
            model_info={"id": "selected"},
        )
        router = Mock()
        router.get_deployment.return_value = selected
        router._is_deployment_blocked.return_value = False
        admin = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
        for code, items in (
            (200, [{"id": "org/model"}]),
            (200, []),
            (401, []),
            (404, []),
            (429, []),
            (503, []),
        ):
            with self.subTest(code=code, empty=not items):
                _model_cache.cache_dict.clear()

                def respond(request, code=code, items=items):
                    self.assertEqual(request.url.path, "/custom/v1/models")
                    return httpx.Response(
                        code,
                        json={"data": items, "error": "private-secret-and-address"},
                    )

                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    with (
                        patch.object(
                            litellm.module_level_client.client,
                            "get",
                            side_effect=client.get,
                        ) as call,
                        patch.object(proxy_server, "llm_router", router),
                        patch.object(litellm, "check_provider_endpoint", True),
                    ):
                        if code == 200:
                            result = await discover_model(model_id="selected", user_api_key_dict=admin)
                            self.assertEqual(
                                result,
                                {"data": [{"id": "selected/" + item["id"]} for item in items]},
                            )
                        else:
                            with self.assertRaises(HTTPException) as raised:
                                await discover_model(model_id="selected", user_api_key_dict=admin)
                            self.assertEqual(raised.exception.status_code, 502)
                            self.assertEqual(raised.exception.detail, {"status_code": code})
                        self.assertEqual(call.call_count, 1)
                router.get_deployment.assert_called_with("selected")

    async def test_non_admin_paused_and_missing_rejected_before_provider(self):
        from litellm.proxy import proxy_server
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.native_model_discovery import (
            discover_model,
        )
        from litellm.types.router import Deployment

        selected = Deployment(
            model_name="selected/*",
            litellm_params={"model": "openai/*", "api_key": "fixture"},
            model_info={"id": "selected"},
        )
        for role, deployment, blocked, expected in (
            (LitellmUserRoles.INTERNAL_USER, selected, False, 403),
            (LitellmUserRoles.PROXY_ADMIN, None, False, 404),
            (LitellmUserRoles.PROXY_ADMIN, selected, True, 409),
        ):
            with self.subTest(expected=expected):
                router = Mock()
                router.get_deployment.return_value = deployment
                router._is_deployment_blocked.return_value = blocked
                with (
                    patch.object(proxy_server, "llm_router", router),
                    patch.object(litellm.module_level_client, "get") as provider,
                ):
                    with self.assertRaises(HTTPException) as raised:
                        await discover_model(
                            model_id="selected",
                            user_api_key_dict=UserAPIKeyAuth(user_role=role),
                        )
                    self.assertEqual(raised.exception.status_code, expected)
                    provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
