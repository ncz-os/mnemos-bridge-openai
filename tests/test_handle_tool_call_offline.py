from __future__ import annotations

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


@dataclass
class FakeToolResult:
    content: str


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeToolResult:
        self.calls.append((name, arguments))
        return FakeToolResult(content="known MNEMOS result")


class FakeResultRenderer:
    def to_openai_message(self, result: FakeToolResult) -> str:
        return result.content


class CoreDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[FakeMcpClient, str, dict[str, Any]]] = []

    async def dispatch(self, client: FakeMcpClient, name: str, arguments: dict[str, Any]) -> FakeToolResult:
        self.calls.append((client, name, arguments))
        return await client.call_tool(name, arguments)


class PostDispatchTypeError:
    def __init__(self, client: FakeMcpClient) -> None:
        self.client = client
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> FakeToolResult:
        self.calls.append((name, arguments))
        await self.client.call_tool(name, arguments)
        raise TypeError("response decoding failed after dispatch")


@pytest.mark.asyncio
async def test_handle_tool_call_dispatches_and_returns_tool_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_module, "Dispatcher", None)
    monkeypatch.setattr(adapter_module, "ResultRenderer", FakeResultRenderer)

    client = FakeMcpClient()
    adapter = MnemosOpenAIAdapter(client)
    tool_call = FakeToolCall(
        id="call_123",
        function=FakeFunction(name="search_memories", arguments='{"query": "infrastructure"}'),
    )

    message = await adapter.handle_tool_call(tool_call)

    assert client.calls == [("search_memories", {"query": "infrastructure"})]
    assert message == {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "known MNEMOS result",
    }


@pytest.mark.asyncio
async def test_handle_tool_call_uses_core_dispatcher_signature() -> None:
    client = FakeMcpClient()
    dispatcher = CoreDispatcher()
    adapter = MnemosOpenAIAdapter(client, dispatcher=dispatcher, result_renderer=FakeResultRenderer())
    tool_call = FakeToolCall(
        id="call_123",
        function=FakeFunction(name="search_memories", arguments='{"query": "infrastructure"}'),
    )

    await adapter.handle_tool_call(tool_call)

    assert dispatcher.calls == [(client, "search_memories", {"query": "infrastructure"})]
    assert client.calls == [("search_memories", {"query": "infrastructure"})]


@pytest.mark.asyncio
async def test_post_dispatch_type_error_does_not_retry_tool_call() -> None:
    client = FakeMcpClient()
    dispatcher = PostDispatchTypeError(client)
    adapter = MnemosOpenAIAdapter(client, dispatcher=dispatcher, result_renderer=FakeResultRenderer())
    tool_call = FakeToolCall(
        id="call_123",
        function=FakeFunction(name="create_memory", arguments='{"content": "important"}'),
    )

    with pytest.raises(TypeError, match="response decoding failed after dispatch"):
        await adapter.handle_tool_call(tool_call)

    assert dispatcher.calls == [("create_memory", {"content": "important"})]
    assert client.calls == [("create_memory", {"content": "important"})]
