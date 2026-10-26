"""Native management boundary regressions; no provider, database or real keys."""

import unittest
from unittest.mock import AsyncMock, Mock, patch

import litellm
from fastapi import HTTPException
from starlette.requests import Request
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.health_endpoints._health_endpoints import test_model_connection
from litellm.proxy.management_endpoints.model_management_endpoints import ModelManagementAuthChecks
from litellm.types.router import Deployment


class NativeHealthEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.deployment = Deployment(
            model_name="shared-name",
            litellm_params={
                "model": "openai/probe",
                "litellm_credential_name": "stored-credential",
                "use_chat_completions_api": True,
            },
            model_info={"id": "selected", "team_id": "owner-team", "supported_endpoints": ["/v1/chat/completions"]},
        )
        self.router = Mock()
        self.router.get_deployment.return_value = self.deployment
        self.router.get_model_list.return_value = [self.deployment.model_dump(exclude_none=True)]
        self.health = AsyncMock(return_value={})
        self.auth = AsyncMock()
        for target, name, value in [
            (proxy_server, "llm_router", self.router),
            (proxy_server, "prisma_client", Mock()),
            (proxy_server, "general_settings", {}),
            (litellm, "ahealth_check", self.health),
            (ModelManagementAuthChecks, "can_user_make_model_call", self.auth),
        ]:
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def check(self, params, info):
        return await test_model_connection(
            request=Request({"type": "http", "method": "POST", "path": "/health/test_connection", "headers": []}),
            mode="anthropic_messages",
            litellm_params=params,
            model_info=info,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    async def test_draft_endpoint_override_keeps_authoritative_ownership(self):
        await self.check(
            {"model": "shared-name", "use_chat_completions_api": False},
            {"id": "selected", "team_id": "forged-team", "supported_endpoints": ["/v1/messages"]},
        )
        params = self.health.call_args.kwargs["model_params"]
        self.assertEqual(self.health.call_args.kwargs["mode"], "anthropic_messages")
        self.assertEqual(params["model_info"]["supported_endpoints"], ["/v1/messages"])
        self.assertEqual(params["model_info"]["team_id"], "owner-team")
        self.assertEqual(params["litellm_credential_name"], "stored-credential")
        self.assertFalse(params["use_chat_completions_api"])
        self.assertEqual(self.auth.call_args.kwargs["model_params"].model_info.team_id, "owner-team")
        self.router.get_deployment.assert_called_once_with(model_id="selected")
        self.router.get_model_list.assert_not_called()

    async def test_different_credential_does_not_inherit_conversation_route(self):
        await self.check({"model": "openai/probe", "litellm_credential_name": "draft-credential"}, {"mode": "chat"})
        params = self.health.call_args.kwargs["model_params"]
        self.assertEqual(params["litellm_credential_name"], "draft-credential")
        self.assertNotIn("use_chat_completions_api", params)
        self.assertNotIn("supported_endpoints", params["model_info"])
        self.assertEqual(params["model_info"]["team_id"], "owner-team")

    async def test_saved_model_preserves_route_without_a_draft(self):
        await self.check({}, {"id": "selected"})
        params = self.health.call_args.kwargs["model_params"]
        self.assertTrue(params["use_chat_completions_api"])
        self.assertEqual(params["model_info"]["supported_endpoints"], ["/v1/chat/completions"])

    async def test_missing_explicit_member_cannot_fall_back_to_same_name(self):
        self.router.get_deployment.return_value = None
        with self.assertRaises(HTTPException) as failure:
            await self.check({"model": "shared-name"}, {"id": "missing"})
        self.assertEqual(failure.exception.status_code, 404)
        self.router.get_model_list.assert_not_called()
        self.health.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
