"""Exact-deployment native discovery; no provider transport or credential store."""

from typing import Annotated, Final

import httpx
from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.model_checks import (
    _hydrate_litellm_credential_name,
    get_known_models_from_wildcard,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.router import Deployment
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, _model_cache

router: Final = APIRouter()


def _discover(deployment: Deployment) -> dict[str, list[dict[str, str]]]:
    params: Final = _hydrate_litellm_credential_name(deployment.litellm_params)
    provider: Final = params.model.split("/", 1)[0]
    config: Final = ProviderConfigManager.get_provider_model_info(model=None, provider=LlmProviders(provider))
    if config is None:
        raise HTTPException(status_code=422, detail="Native discovery is unavailable")
    # Keep native TTL/key semantics across pagination and search requests. Failed
    # provider calls are never inserted as successful empty results.
    cached: Final = _model_cache.get_cached_model_info(provider, params)
    if cached is None:
        discovered: Final = config.discover_models(params.model_dump(exclude_none=True))
        _model_cache.set_cached_model_info(provider, params, discovered)
    return {"data": [{"id": model} for model in get_known_models_from_wildcard(deployment.model_name, params)]}


@router.get(
    "/model/discovery",
    dependencies=[Depends(user_api_key_auth)],
    tags=["model management"],
)
async def discover_model(
    model_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> dict[str, list[dict[str, str]]]:
    from litellm.proxy.proxy_server import llm_router

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Proxy administrator required")
    deployment: Final = llm_router.get_deployment(model_id) if llm_router is not None else None
    if deployment is None:
        raise HTTPException(status_code=404, detail="Discovery deployment not found")
    if llm_router._is_deployment_blocked(deployment):
        raise HTTPException(status_code=409, detail="Discovery deployment is paused")
    if not deployment.model_name.endswith("/*") or not deployment.litellm_params.model.endswith("/*"):
        raise HTTPException(status_code=422, detail="Wildcard deployment required")
    try:
        return await run_in_threadpool(_discover, deployment)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=502, detail={"status_code": error.response.status_code}) from None
    except httpx.TimeoutException:
        raise HTTPException(status_code=502, detail={"status_code": 504}) from None
    except Exception:  # noqa: BLE001  # Native provider failures must not expose credentials or raw bodies.
        raise HTTPException(status_code=502, detail={"status_code": 502}) from None
