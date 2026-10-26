# FlashYun native gateway fork

Based on BerriAI/litellm **v1.106.4**, synced before creating branch
`flashyun-native-gateway`. Upstream owns provider transport, protocol conversion,
credentials, routing, persistence, retries and accounting. No FlashYun callback
or independent provider implementation is introduced

Maintained differences:

- Preserve custom OpenAI model-discovery base paths and HTTP status failures.
- For custom Anthropic discovery, retry the same URL once with the native Bearer
  helper after standard authentication returns 401/403. Official Anthropic keeps
  its standard behavior. Keep upstream pagination and URL normalization.
- Expose selected-deployment discovery through a private administrator-only
  `/model/discovery` route, reusing native credentials, provider implementations,
  cache and wildcard normalization. Return safe numeric failure status only.
- Forward native Messages endpoint metadata through `/health/test_connection`.
  Permit endpoint drafts without overriding stored ownership metadata; do not
  inherit a different credential's conversation route. A missing explicit member
  ID cannot fall back to a same-name deployment.

Upstream v1.106.4 already supplies `anthropic_messages` health checks; there is no
custom health mode or protocol converter. FlashYun builds the complete Python
package from this checkout over the matching official runtime, retaining its
frozen dependencies and compiled Rust extension. No string-replacement patch
scripts run during build or startup

The parent project executes these regressions during its host Docker Compose build:

```sh
LITELLM_LOCAL_MODEL_COST_MAP=True python flashyun/verify_discovery.py
LITELLM_LOCAL_MODEL_COST_MAP=True python flashyun/verify_messages_health.py
LITELLM_LOCAL_MODEL_COST_MAP=True python flashyun/verify_health_endpoint.py
```

They use synthetic credentials and in-process HTTP transports or loopback only.
The route matrix compares native Messages serving and health checks for default,
Chat Completions, Responses, native Messages and Anthropic configurations. An
unsupported Responses endpoint must fail both calls instead of producing a false
successful connection check. Management regressions cover explicit member IDs,
draft endpoint selection and authoritative ownership metadata

For upgrades, fetch upstream and merge the reviewed stable release into this
branch first. Recheck/remove differences absorbed upstream, run these probes,
commit/push the branch, then update the parent submodule pointer, matching base
image and generated OpenAPI/client provenance together. Runtime credentials and
existing user configuration must remain unchanged by this workflow
