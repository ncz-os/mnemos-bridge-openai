"""OpenAI Chat Completions adapter for MNEMOS MCP tools."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

try:  # pragma: no cover - exercised by integration tests when the core package is installed.
    from mnemos_bridge_core import AuthResolver, Dispatcher, McpClient, ResultRenderer, SchemaTranslator
except ImportError:  # pragma: no cover - offline tests monkeypatch these names.
    AuthResolver = None  # type: ignore[assignment]
    Dispatcher = None  # type: ignore[assignment]
    McpClient = None  # type: ignore[assignment]
    ResultRenderer = None  # type: ignore[assignment]
    SchemaTranslator = None  # type: ignore[assignment]


class MnemosOpenAIAdapter:
    """Bridge MNEMOS MCP tools into OpenAI Chat Completions tool calls."""

    def __init__(
        self,
        mcp_client: Any,
        *,
        dispatcher: Any | None = None,
        schema_translator: Any | None = None,
        result_renderer: Any | None = None,
    ) -> None:
        self._mcp_client = mcp_client
        self._dispatcher = dispatcher if dispatcher is not None else _make_dispatcher(mcp_client)
        self._schema_translator = (
            schema_translator if schema_translator is not None else _make_service(SchemaTranslator, mcp_client)
        )
        self._result_renderer = (
            result_renderer if result_renderer is not None else _make_service(ResultRenderer, mcp_client)
        )

    @classmethod
    async def connect(cls, mcp_url: str, mcp_token: str, *, timeout: float = 30) -> MnemosOpenAIAdapter:
        """Connect to a MNEMOS MCP HTTP/SSE endpoint and return an adapter."""

        if McpClient is None:
            raise RuntimeError(
                "mnemos-bridge-core is required to connect to MNEMOS. "
                "Install this package with its runtime dependencies first."
            )

        auth = _build_auth_resolver(mcp_token)
        # Prefer the v0.1.2+ open_from_url that opens the SSE session in one call;
        # fall back to from_url + __aenter__ when running against older core builds.
        opener = getattr(McpClient, "open_from_url", None)
        if opener is not None:
            client = await opener(mcp_url, token=mcp_token, timeout=timeout)
        else:
            client = await _maybe_await(_call_from_url(McpClient.from_url, mcp_url, mcp_token, auth, timeout))
            aenter = getattr(client, "__aenter__", None)
            if aenter is not None:
                await aenter()
        return cls(client)

    async def openai_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI Chat Completions function-tool definitions for MNEMOS tools."""

        mcp_tools = await self._list_mcp_tools()
        translated = await self._translate_tools(mcp_tools)
        return [_normalize_openai_tool(tool) for tool in _ensure_sequence(translated)]

    async def handle_tool_call(self, tool_call: Any) -> dict[str, str]:
        """Dispatch one OpenAI tool call to MNEMOS and return a Chat Completions tool message."""

        tool_call_id = _shape_get(tool_call, "id")
        if not tool_call_id:
            raise ValueError("OpenAI tool_call is missing an id.")

        function = _shape_get(tool_call, "function")
        name = _shape_get(function, "name")
        if not name:
            raise ValueError("OpenAI tool_call is missing function.name.")

        arguments = _parse_arguments(_shape_get(function, "arguments", default={}))
        result = await self._dispatch_tool(name, arguments, tool_call)
        rendered = await self._render_result(result, tool_call_id)
        return _normalize_tool_message(rendered, tool_call_id)

    async def handle_tool_call_stream(self, tool_call: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield one streaming-compatible chunk for a completed MNEMOS tool result.

        TODO(v0.2): replace this compatibility shim with true incremental streaming once
        the MNEMOS core dispatcher exposes streaming result events.
        """

        message = await self.handle_tool_call(tool_call)
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": message,
                    "finish_reason": None,
                }
            ]
        }

    async def aclose(self) -> None:
        """Close the underlying MCP client if it exposes an async or sync close hook."""

        close = getattr(self._mcp_client, "aclose", None) or getattr(self._mcp_client, "close", None)
        if close is not None:
            await _maybe_await(close())
            return

        exit_method = getattr(self._mcp_client, "__aexit__", None)
        if exit_method is not None:
            await _maybe_await(exit_method(None, None, None))

    async def __aenter__(self) -> MnemosOpenAIAdapter:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _list_mcp_tools(self) -> list[Any]:
        for name in ("list_tools", "tools", "get_tools", "list_available_tools"):
            candidate = getattr(self._mcp_client, name, None)
            if callable(candidate):
                tools = await _maybe_await(candidate())
                return list(_ensure_sequence(tools))

        tools_attr = getattr(self._mcp_client, "tools", None)
        if tools_attr is not None and not callable(tools_attr):
            return list(_ensure_sequence(await _maybe_await(tools_attr)))

        raise AttributeError("MCP client does not expose a supported tool-listing method.")

    async def _translate_tools(self, mcp_tools: list[Any]) -> Any:
        if self._schema_translator is None:
            return [_fallback_openai_tool(tool) for tool in mcp_tools]

        to_openai = getattr(self._schema_translator, "to_openai", None)
        if to_openai is None:
            raise AttributeError("SchemaTranslator does not expose to_openai().")

        try:
            return await _maybe_await(to_openai(mcp_tools))
        except (TypeError, AttributeError) as exc:
            # Core SchemaTranslator.to_openai() takes a single ToolSchema, not a
            # list. When that signature is in play, iterate per-tool. Older
            # batch-shaped translators continue to work via the try-branch above.
            per_tool: list[Any] = []
            for tool in mcp_tools:
                try:
                    per_tool.append(await _maybe_await(to_openai(tool)))
                except (TypeError, AttributeError):
                    raise exc from None
            return per_tool

    async def _dispatch_tool(self, name: str, arguments: dict[str, Any], tool_call: Any) -> Any:
        if self._dispatcher is not None:
            for method_name in ("dispatch", "dispatch_tool_call", "handle_tool_call", "call_tool"):
                dispatch = getattr(self._dispatcher, method_name, None)
                if dispatch is None:
                    continue

                candidates = (
                    (
                        (name, arguments),
                        (self._mcp_client, name, arguments),
                        (tool_call,),
                        (name,),
                        ({"name": name, "arguments": arguments},),
                    )
                    if method_name == "dispatch"
                    else ((name, arguments), (tool_call,), (name,), ({"name": name, "arguments": arguments},))
                )
                args = _bind_supported_args(dispatch, candidates)
                if args is not None:
                    return await _maybe_await(dispatch(*args))

        call_tool = getattr(self._mcp_client, "call_tool", None)
        if call_tool is None:
            raise AttributeError("Neither Dispatcher nor McpClient exposes a supported tool dispatch method.")

        args = _bind_supported_args(
            call_tool, ((name, arguments), (name,), ({"name": name, "arguments": arguments},))
        )
        if args is None:
            raise TypeError("McpClient.call_tool() exists but did not accept any supported argument shape.")
        return await _maybe_await(call_tool(*args))

    async def _render_result(self, result: Any, tool_call_id: str) -> Any:
        if self._result_renderer is None:
            return _fallback_render_result(result)

        to_message = getattr(self._result_renderer, "to_openai_message", None)
        if to_message is None:
            raise AttributeError("ResultRenderer does not expose to_openai_message().")

        for args in ((result, tool_call_id), (result,)):
            try:
                return await _maybe_await(to_message(*args))
            except TypeError:
                continue

        raise TypeError("ResultRenderer.to_openai_message() did not accept the result.")


def _call_from_url(from_url: Callable[..., Any], mcp_url: str, mcp_token: str, auth: Any | None, timeout: float) -> Any:
    candidates = _from_url_candidates(from_url, mcp_token, auth, timeout)
    last_error: TypeError | None = None

    for kwargs in candidates:
        try:
            return from_url(mcp_url, **kwargs)
        except TypeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return from_url(mcp_url)


def _from_url_candidates(
    from_url: Callable[..., Any], mcp_token: str, auth: Any | None, timeout: float
) -> list[dict[str, Any]]:
    try:
        parameters = inspect.signature(from_url).parameters
    except (TypeError, ValueError):
        parameters = {}

    kwargs: dict[str, Any] = {}
    if "timeout" in parameters:
        kwargs["timeout"] = timeout
    if auth is not None and "auth" in parameters:
        kwargs["auth"] = auth
    elif auth is not None and "auth_resolver" in parameters:
        kwargs["auth_resolver"] = auth
    elif "token" in parameters:
        kwargs["token"] = mcp_token
    elif "bearer_token" in parameters:
        kwargs["bearer_token"] = mcp_token
    elif "mcp_token" in parameters:
        kwargs["mcp_token"] = mcp_token
    elif "headers" in parameters:
        kwargs["headers"] = {"Authorization": f"Bearer {mcp_token}"}

    ordered: list[dict[str, Any]] = []
    if kwargs:
        ordered.append(kwargs)
    if auth is not None:
        ordered.extend(
            [
                {"auth": auth, "timeout": timeout},
                {"auth_resolver": auth, "timeout": timeout},
            ]
        )
    ordered.extend(
        [
            {"token": mcp_token, "timeout": timeout},
            {"bearer_token": mcp_token, "timeout": timeout},
            {"mcp_token": mcp_token, "timeout": timeout},
            {"headers": {"Authorization": f"Bearer {mcp_token}"}, "timeout": timeout},
            {"timeout": timeout},
            {},
        ]
    )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for item in ordered:
        key = tuple(sorted((name, repr(value)) for name, value in item.items()))
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def _build_auth_resolver(token: str) -> Any | None:
    if AuthResolver is None:
        return None

    for factory_name in ("bearer", "from_bearer_token", "for_bearer_token", "from_token"):
        factory = getattr(AuthResolver, factory_name, None)
        if factory is None:
            continue
        try:
            return factory(token)
        except TypeError:
            continue

    for args, kwargs in (((token,), {}), ((), {"token": token}), ((), {"bearer_token": token})):
        try:
            return AuthResolver(*args, **kwargs)
        except TypeError:
            continue

    return None


def _make_dispatcher(mcp_client: Any) -> Any | None:
    if Dispatcher is None:
        return None

    for args, kwargs in (
        ((mcp_client,), {}),
        ((), {"client": mcp_client}),
        ((), {"mcp_client": mcp_client}),
        ((), {}),
    ):
        try:
            return Dispatcher(*args, **kwargs)
        except TypeError:
            continue
    return None


def _make_service(service_type: Any | None, mcp_client: Any) -> Any | None:
    if service_type is None:
        return None

    for args, kwargs in (
        ((), {}),
        ((mcp_client,), {}),
        ((), {"client": mcp_client}),
        ((), {"mcp_client": mcp_client}),
    ):
        try:
            return service_type(*args, **kwargs)
        except TypeError:
            continue
    return service_type


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _bind_supported_args(
    function: Callable[..., Any], candidates: tuple[tuple[Any, ...], ...]
) -> tuple[Any, ...] | None:
    """Select a supported call shape without invoking a potentially mutating operation."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        # Some extension callables do not expose a signature. Select the canonical
        # shape so the caller still invokes the operation exactly once.
        return candidates[0]

    for args in candidates:
        try:
            signature.bind(*args)
        except TypeError:
            continue
        return args

    return None


def _ensure_sequence(value: Any) -> list[Any]:
    # NOTE: do NOT call _to_plain on the WHOLE value — pydantic ToolSchema
    # items must reach SchemaTranslator.to_openai() with their attributes
    # intact, not pre-dumped to plain dicts. We only unwrap container
    # shapes ({"tools":[...]}, etc.) here; per-item normalization is
    # done at the consumer (e.g., _normalize_openai_tool).
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        if "tools" in value:
            return _ensure_sequence(value["tools"])
        if "data" in value:
            return _ensure_sequence(value["data"])
    return [value]


def _normalize_openai_tool(tool: Any) -> dict[str, Any]:
    plain = _to_plain(tool)
    if not isinstance(plain, Mapping):
        raise TypeError(f"OpenAI tool definition must be a mapping, got {type(tool).__name__}.")

    if plain.get("type") == "function" and isinstance(plain.get("function"), Mapping):
        function = dict(plain["function"])
    else:
        function = dict(plain)

    name = function.get("name")
    if not name:
        raise ValueError("OpenAI function tool is missing function.name.")

    parameters = function.get("parameters") or function.get("input_schema") or function.get("inputSchema") or {}
    return {
        "type": "function",
        "function": {
            "name": str(name),
            "description": str(function.get("description") or ""),
            "parameters": _normalize_parameters(parameters),
        },
    }


def _fallback_openai_tool(tool: Any) -> dict[str, Any]:
    plain = _to_plain(tool)
    if not isinstance(plain, Mapping):
        name = str(getattr(tool, "name", "mnemos_tool"))
        return {"type": "function", "function": {"name": name, "description": "", "parameters": {"type": "object"}}}

    return {
        "type": "function",
        "function": {
            "name": str(plain.get("name") or plain.get("id")),
            "description": str(plain.get("description") or ""),
            "parameters": _normalize_parameters(plain.get("inputSchema") or plain.get("input_schema") or {}),
        },
    }


def _normalize_parameters(parameters: Any) -> dict[str, Any]:
    plain = _to_plain(parameters)
    if not isinstance(plain, Mapping):
        return {"type": "object"}

    normalized = dict(plain)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    return normalized


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    plain = _to_plain(arguments)
    if plain in (None, ""):
        return {}
    if isinstance(plain, str):
        parsed = json.loads(plain)
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI tool_call function.arguments must decode to a JSON object.")
        return parsed
    if isinstance(plain, Mapping):
        return dict(plain)
    raise TypeError("OpenAI tool_call function.arguments must be a JSON string or mapping.")


def _shape_get(value: Any, key: str, *, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _normalize_tool_message(rendered: Any, tool_call_id: str) -> dict[str, str]:
    plain = _to_plain(rendered)
    if isinstance(plain, Mapping):
        content = plain.get("content", "")
        return {
            "role": "tool",
            "tool_call_id": str(plain.get("tool_call_id") or tool_call_id),
            "content": _stringify_content(content),
        }

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": _stringify_content(plain),
    }


def _fallback_render_result(result: Any) -> str:
    plain = _to_plain(result)
    if isinstance(plain, Mapping):
        if "content" in plain:
            return _stringify_content(plain["content"])
        if "text" in plain:
            return _stringify_content(plain["text"])
    return _stringify_content(plain)


def _stringify_content(content: Any) -> str:
    plain = _to_plain(content)
    if isinstance(plain, str):
        return plain
    if plain is None:
        return ""
    return json.dumps(plain, ensure_ascii=True)


def _to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_plain(item) for item in value)
    return value
