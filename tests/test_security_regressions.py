from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

import mnemos_bridge_openai.adapter as adapter_module
from mnemos_bridge_openai import MnemosOpenAIAdapter


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search_memories",
                "description": "Search MNEMOS memories.",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, str]:
        self.calls.append((name, arguments))
        return {"content": "ok"}


class PassthroughRenderer:
    def to_openai_message(self, result: Any) -> Any:
        return result


def _tool_call(name: str, arguments: str = "{}") -> FakeToolCall:
    return FakeToolCall(id="call_1", function=FakeFunction(name=name, arguments=arguments))


@pytest.mark.asyncio
async def test_no_allowlist_configured_fails_closed_before_discovery() -> None:
    client = RecordingClient()
    adapter = MnemosOpenAIAdapter(client, dispatcher=False, result_renderer=PassthroughRenderer())

    with pytest.raises(ValueError, match=r"disabled until openai_tools\(\) completes discovery"):
        await adapter.handle_tool_call(_tool_call("delete_memory", '{"memory_id": "victim"}'))

    assert client.calls == []


@pytest.mark.asyncio
async def test_tool_dispatch_fails_closed_while_discovery_is_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_module, "SchemaTranslator", None)
    discovery_started = asyncio.Event()
    finish_discovery = asyncio.Event()

    class BlockingDiscoveryClient(RecordingClient):
        async def list_tools(self) -> list[dict[str, Any]]:
            discovery_started.set()
            await finish_discovery.wait()
            return await super().list_tools()

    client = BlockingDiscoveryClient()
    adapter = MnemosOpenAIAdapter(client, dispatcher=False, result_renderer=PassthroughRenderer())
    discovery = asyncio.create_task(adapter.openai_tools())
    await discovery_started.wait()

    with pytest.raises(ValueError, match=r"disabled until openai_tools\(\) completes discovery"):
        await adapter.handle_tool_call(_tool_call("delete_memory", '{"memory_id": "victim"}'))

    finish_discovery.set()
    await discovery
    assert client.calls == []


@pytest.mark.asyncio
async def test_overlapping_discoveries_remain_closed_until_both_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_module, "SchemaTranslator", None)
    discovery_started = [asyncio.Event(), asyncio.Event()]
    finish_discovery = [asyncio.Event(), asyncio.Event()]

    class OverlappingDiscoveryClient(RecordingClient):
        calls_started = 0

        async def list_tools(self) -> list[dict[str, Any]]:
            call_index = self.calls_started
            self.calls_started += 1
            discovery_started[call_index].set()
            await finish_discovery[call_index].wait()
            return await super().list_tools()

    client = OverlappingDiscoveryClient()
    adapter = MnemosOpenAIAdapter(client, dispatcher=False, result_renderer=PassthroughRenderer())
    first = asyncio.create_task(adapter.openai_tools())
    await discovery_started[0].wait()
    second = asyncio.create_task(adapter.openai_tools())
    await discovery_started[1].wait()

    finish_discovery[0].set()
    await first
    with pytest.raises(ValueError, match=r"disabled until openai_tools\(\) completes discovery"):
        await adapter.handle_tool_call(_tool_call("search_memories"))

    finish_discovery[1].set()
    await second
    message = await adapter.handle_tool_call(_tool_call("search_memories"))
    assert message == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}


@pytest.mark.asyncio
async def test_discovery_loads_allowlist_before_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_module, "SchemaTranslator", None)
    client = RecordingClient()
    adapter = MnemosOpenAIAdapter(client, dispatcher=False, result_renderer=PassthroughRenderer())

    tools = await adapter.openai_tools()
    assert [tool["function"]["name"] for tool in tools] == ["search_memories"]

    with pytest.raises(ValueError, match="allowed-tool set"):
        await adapter.handle_tool_call(_tool_call("delete_memory", '{"memory_id": "victim"}'))

    message = await adapter.handle_tool_call(_tool_call("search_memories", '{"query": "safe"}'))
    assert client.calls == [("search_memories", {"query": "safe"})]
    assert message == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}


@pytest.mark.asyncio
async def test_failed_discovery_refresh_invalidates_stale_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_module, "SchemaTranslator", None)

    class FailingRefreshClient(RecordingClient):
        fail_listing = False

        async def list_tools(self) -> list[dict[str, Any]]:
            if self.fail_listing:
                raise RuntimeError("discovery failed")
            return await super().list_tools()

    client = FailingRefreshClient()
    adapter = MnemosOpenAIAdapter(client, dispatcher=False, result_renderer=PassthroughRenderer())
    await adapter.openai_tools()

    client.fail_listing = True
    with pytest.raises(RuntimeError, match="discovery failed"):
        await adapter.openai_tools()
    with pytest.raises(ValueError, match=r"disabled until openai_tools\(\) completes discovery"):
        await adapter.handle_tool_call(_tool_call("search_memories"))

    assert client.calls == []


def test_call_from_url_forwards_credentials_for_every_supported_fallback_shape() -> None:
    auth = object()

    def with_token(url: str, *, token: str, timeout: float) -> tuple[Any, ...]:
        return url, "token", token, timeout

    def with_bearer_token(url: str, *, bearer_token: str) -> tuple[Any, ...]:
        return url, "bearer_token", bearer_token

    def with_mcp_token(url: str, *, mcp_token: str, timeout: float) -> tuple[Any, ...]:
        return url, "mcp_token", mcp_token, timeout

    def with_headers(url: str, *, headers: dict[str, str]) -> tuple[Any, ...]:
        return url, "headers", headers

    def with_auth(url: str, *, auth: Any, timeout: float) -> tuple[Any, ...]:
        return url, "auth", auth, timeout

    def with_auth_resolver(url: str, *, auth_resolver: Any) -> tuple[Any, ...]:
        return url, "auth_resolver", auth_resolver

    def with_explicit_auth_and_kwargs(url: str, *, auth: Any, **kwargs: Any) -> tuple[Any, ...]:
        return url, "explicit_auth", auth, kwargs

    call = adapter_module._call_from_url
    url = "https://mnemos.example/sse"
    token = "secret-token"
    assert call(with_token, url, token, auth, 17) == (url, "token", token, 17)
    assert call(with_bearer_token, url, token, auth, 17) == (url, "bearer_token", token)
    assert call(with_mcp_token, url, token, auth, 17) == (url, "mcp_token", token, 17)
    assert call(with_headers, url, token, auth, 17) == (
        url,
        "headers",
        {"Authorization": f"Bearer {token}"},
    )
    assert call(with_auth, url, token, auth, 17) == (url, "auth", auth, 17)
    assert call(with_auth_resolver, url, token, auth, 17) == (url, "auth_resolver", auth)
    assert call(with_explicit_auth_and_kwargs, url, token, auth, 17) == (
        url,
        "explicit_auth",
        auth,
        {"timeout": 17},
    )


def test_call_from_url_rejects_credentialless_legacy_signature_without_calling() -> None:
    calls: list[tuple[str, float]] = []

    def credentialless(url: str, *, timeout: float) -> object:
        calls.append((url, timeout))
        return object()

    with pytest.raises(TypeError, match="credential-bearing argument shape"):
        adapter_module._call_from_url(
            credentialless,
            "https://mnemos.example/sse",
            "secret-token",
            None,
            17,
        )

    assert calls == []


def test_call_from_url_propagates_internal_typeerror_without_retry() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fails_internally(url: str, **kwargs: Any) -> object:
        calls.append((url, kwargs))
        raise TypeError("connection setup failed internally")

    with pytest.raises(TypeError, match="connection setup failed internally"):
        adapter_module._call_from_url(
            fails_internally,
            "https://mnemos.example/sse",
            "secret-token",
            object(),
            17,
        )

    assert calls == [
        (
            "https://mnemos.example/sse",
            {"token": "secret-token", "timeout": 17},
        )
    ]
