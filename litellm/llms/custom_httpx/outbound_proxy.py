import ssl
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from hashlib import sha256
from inspect import iscoroutinefunction
from typing import Final

import httpx

PROXY_FIELDS: Final = frozenset({"proxy_enabled", "proxy_url", "proxy_username", "proxy_password", "proxy_ssl_verify"})

PROXY_CREDENTIAL_FIELDS: Final = (
    {
        "key": "proxy_enabled",
        "label": "Use proxy",
        "field_type": "select",
        "options": ["false", "true"],
        "required": False,
    },
    {"key": "proxy_url", "label": "Proxy URL", "field_type": "text", "required": False},
    {"key": "proxy_username", "label": "Proxy username", "field_type": "text", "required": False},
    {"key": "proxy_password", "label": "Proxy password", "field_type": "password", "required": False},
    {
        "key": "proxy_ssl_verify",
        "label": "Verify certificates",
        "field_type": "select",
        "options": ["true", "false"],
        "required": False,
    },
)


@dataclass(frozen=True, slots=True)
class OutboundProxy:
    url: str
    username: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    verify: bool = True

    @property
    def identity(self) -> str:
        return sha256(repr((self.url, self.username, self.password, self.verify)).encode()).hexdigest()

    def httpx_proxy(self, ssl_config: bool | str | ssl.SSLContext) -> httpx.Proxy:
        return httpx.Proxy(
            self.url,
            auth=(self.username, self.password) if self.username or self.password else None,
            ssl_context=self._ssl_context(ssl_config),
        )

    def _ssl_context(self, ssl_config: bool | str | ssl.SSLContext) -> ssl.SSLContext | None:
        if not self.url.startswith("https://"):
            return None
        if isinstance(ssl_config, ssl.SSLContext):
            return ssl_config
        if ssl_config is not False:
            return None
        context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    def __str__(self) -> str:
        return self.identity


current_outbound_proxy: Final[ContextVar[OutboundProxy | None]] = ContextVar("litellm_outbound_proxy", default=None)
_requested_outbound_proxy: Final[ContextVar[OutboundProxy | None]] = ContextVar(
    "litellm_requested_outbound_proxy", default=None
)


def parse_outbound_proxy(values: Mapping[str, object]) -> OutboundProxy | None:
    for key in ("proxy_enabled", "proxy_ssl_verify"):
        if key in values and values[key] is not None and values[key] not in ("true", "false"):
            raise ValueError("Proxy switches must be true or false")
    if values.get("proxy_enabled") != "true":
        return None
    url: Final = values.get("proxy_url")
    username: Final = values.get("proxy_username") or ""
    password: Final = values.get("proxy_password") or ""
    if not isinstance(url, str) or not isinstance(username, str) or not isinstance(password, str):
        raise TypeError("Proxy address and credentials must be strings")
    try:
        parsed: Final = httpx.URL(url)
        valid: Final = (
            parsed.scheme in ("http", "https")
            and bool(parsed.host)
            and not parsed.userinfo
            and not parsed.query
            and not parsed.fragment
            and parsed.path in ("", "/")
            and (parsed.port is None or 0 < parsed.port <= 65535)
        )
    except (httpx.InvalidURL, ValueError):
        raise ValueError("Invalid HTTP/HTTPS proxy address") from None
    if not valid:
        raise ValueError("Invalid HTTP/HTTPS proxy address")
    return OutboundProxy(str(parsed), username, password, values.get("proxy_ssl_verify") != "false")


def resolve_outbound_proxy(values: Mapping[str, object]) -> OutboundProxy | None:
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor

    name: Final = values.get("litellm_credential_name")
    credential: Final = CredentialAccessor.find_credential(name) if isinstance(name, str) else None
    inherited: Final = {} if credential is None else credential.credential_values
    settings: Final = {**inherited, **{key: value for key, value in values.items() if key in PROXY_FIELDS}}
    if credential is None and not PROXY_FIELDS.intersection(settings):
        return _requested_outbound_proxy.get() or current_outbound_proxy.get()
    return parse_outbound_proxy(settings)


@contextmanager
def outbound_proxy_scope(policy: OutboundProxy | None) -> Iterator[None]:
    token: Final = current_outbound_proxy.set(policy)
    try:
        yield
    finally:
        current_outbound_proxy.reset(token)


def _error_chain(error: BaseException | None, seen: tuple[BaseException, ...] = ()) -> tuple[BaseException, ...]:
    if error is None or any(error is previous for previous in seen):
        return ()
    return (error, *_error_chain(error.__cause__ or error.__context__, (*seen, error)))


def _proxy_failure_message(error: Exception) -> str | None:
    chain: Final = _error_chain(error)
    if any(
        getattr(item, "status_code", None) == 407 or isinstance(item, httpx.ProxyError) and "407" in str(item)
        for item in chain
    ):
        return "Proxy authentication failed (407)"
    if any(
        isinstance(item, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(item) for item in chain
    ):
        return "TLS certificate verification failed"
    if any(isinstance(item, (httpx.ProxyError, httpx.ConnectError, httpx.ConnectTimeout)) for item in chain):
        return "Proxy connection failed"
    return None


def _proxy_error(error: Exception, model: object) -> Exception:
    from litellm.exceptions import APIConnectionError

    message: Final = _proxy_failure_message(error)
    if message is None:
        return error
    return APIConnectionError(message=message, model=model if isinstance(model, str) else "", llm_provider="")


def with_outbound_proxy(function: Callable[..., object], *, prepare: bool = False) -> Callable[..., object]:
    if iscoroutinefunction(function):

        @wraps(function)
        async def async_call(*args: object, **kwargs: object) -> object:
            policy: Final = resolve_outbound_proxy(kwargs) if prepare else _requested_outbound_proxy.get()
            clean: Final = {key: value for key, value in kwargs.items() if key not in PROXY_FIELDS}
            variable: Final = _requested_outbound_proxy if prepare else current_outbound_proxy
            token: Final = variable.set(policy)
            try:
                return await function(*args, **clean)
            except Exception as error:
                if not prepare and policy is not None:
                    mapped: Final = _proxy_error(error, kwargs.get("model"))
                    if mapped is not error:
                        raise mapped from error
                raise
            finally:
                variable.reset(token)

        return async_call

    @wraps(function)
    def sync_call(*args: object, **kwargs: object) -> object:
        policy: Final = resolve_outbound_proxy(kwargs) if prepare else _requested_outbound_proxy.get()
        clean: Final = {key: value for key, value in kwargs.items() if key not in PROXY_FIELDS}
        variable: Final = _requested_outbound_proxy if prepare else current_outbound_proxy
        token: Final = variable.set(policy)
        try:
            return function(*args, **clean)
        except Exception as error:
            if not prepare and policy is not None:
                mapped: Final = _proxy_error(error, kwargs.get("model"))
                if mapped is not error:
                    raise mapped from error
            raise
        finally:
            variable.reset(token)

    return sync_call
