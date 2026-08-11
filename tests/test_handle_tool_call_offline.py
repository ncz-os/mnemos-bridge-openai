from __future__ import annotations

import json
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
    is_error: bool = False


@dataclass
class FakeTextContent:
    type: str
    text: str


@dataclass
class FakeStructuredToolResult:
    content: list[FakeTextContent]
    is_error: bool = False


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeToolResult:
        self.calls.append((name, arguments))
        return FakeToolResult(content="known MNEMOS result")


class FakeResultRenderer:
    def to_openai_message(self, result: FakeToolResult) -> str:
        return result.content


class FakeStructuredResultRenderer:
    def to_openai_message(self, result: FakeStructuredToolResult) -> str:
        return " ".join(block.text for block in result.content if block.text)


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


class AmbiguousFailureClient:
    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeToolResult:
        self.calls += 1
        raise TimeoutError("response was lost after dispatch")


class RetryConfigurableDispatcher:
    def __init__(self, *, retries: int = 2) -> None:
        self.retries = retries

    async def dispatch(
        self, client: AmbiguousFailureClient, name: str, arguments: dict[str, Any]
    ) -> FakeToolResult:
        for attempt in range(self.retries + 1):
            try:
                return await client.call_tool(name, arguments)
            except TimeoutError:
                if attempt == self.retries:
                    raise
        raise AssertionError("unreachable")


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


@pytest.mark.asyncio
async def test_default_dispatcher_disables_ambiguous_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_module, "Dispatcher", RetryConfigurableDispatcher)
    client = AmbiguousFailureClient()
    adapter = MnemosOpenAIAdapter(client, result_renderer=FakeResultRenderer())
    tool_call = FakeToolCall(
        id="call_123",
        function=FakeFunction(name="create_memory", arguments='{"content": "important"}'),
    )

    with pytest.raises(TimeoutError, match="response was lost after dispatch"):
        await adapter.handle_tool_call(tool_call)

    assert client.calls == 1


@pytest.mark.asyncio
async def test_handle_tool_call_preserves_mcp_error_state_as_json() -> None:
    class ErrorClient(FakeMcpClient):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeToolResult:
            self.calls.append((name, arguments))
            return FakeToolResult(content="permission denied", is_error=True)

    adapter = MnemosOpenAIAdapter(ErrorClient(), dispatcher=False, result_renderer=FakeResultRenderer())
    tool_call = FakeToolCall(
        id="call_123",
        function=FakeFunction(name="delete_memory", arguments='{"memory_id": "mem_123"}'),
    )

    message = await adapter.handle_tool_call(tool_call)

    assert message == {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": '{"error":"permission denied","is_error":true}',
    }


@pytest.mark.parametrize(
    "producer_payload",
    [
        '{"success": false, "error": "backend unavailable"}',
        '{"error": "503 Service Unavailable", "detail": {"retry": true}}',
    ],
)
@pytest.mark.asyncio
async def test_handle_tool_call_recognizes_mnemos_json_failure_envelope(producer_payload: str) -> None:
    class ProducerShapedClient(FakeMcpClient):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeStructuredToolResult:
            self.calls.append((name, arguments))
            return FakeStructuredToolResult(
                content=[FakeTextContent(type="text", text=producer_payload)],
                is_error=False,
            )

    adapter = MnemosOpenAIAdapter(
        ProducerShapedClient(),
        dispatcher=False,
        result_renderer=FakeStructuredResultRenderer(),
    )
    tool_call = FakeToolCall(
        id="call_123",
        function=FakeFunction(name="create_memory", arguments="{}"),
    )

    message = await adapter.handle_tool_call(tool_call)

    marked_error = json.loads(message["content"])
    assert marked_error["is_error"] is True
    assert json.loads(marked_error["error"]) == json.loads(producer_payload)


@pytest.mark.asyncio
async def test_handle_tool_call_does_not_mark_success_with_nested_error_field() -> None:
    producer_payload = '{"success": true, "result": {"error": "historical value"}}'

    class ProducerShapedClient(FakeMcpClient):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeStructuredToolResult:
            return FakeStructuredToolResult(
                content=[FakeTextContent(type="text", text=producer_payload)],
                is_error=False,
            )

    adapter = MnemosOpenAIAdapter(
        ProducerShapedClient(),
        dispatcher=False,
        result_renderer=FakeStructuredResultRenderer(),
    )
    tool_call = FakeToolCall(id="call_123", function=FakeFunction(name="search_memories", arguments="{}"))

    message = await adapter.handle_tool_call(tool_call)

    assert message["content"] == producer_payload


@pytest.mark.asyncio
async def test_handle_tool_call_stream_emits_custom_event_with_tool_message() -> None:
    client = FakeMcpClient()
    adapter = MnemosOpenAIAdapter(client, dispatcher=False, result_renderer=FakeResultRenderer())
    tool_call = FakeToolCall(
        id="call_123",
        function=FakeFunction(name="search_memories", arguments='{"query": "infrastructure"}'),
    )

    chunks = [
        chunk
        async for chunk in adapter.handle_tool_call_stream(
            tool_call,
            completion_id="chatcmpl_123",
            created=1_723_456_789,
            model="gpt-4o",
        )
    ]

    assert len(chunks) == 1
    assert chunks[0] == {
        "type": "mnemos.tool_result",
        "message": {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "known MNEMOS result",
        },
    }


@pytest.mark.asyncio
async def test_handle_tool_call_stream_default_call_returns_next_turn_message() -> None:
    client = FakeMcpClient()
    adapter = MnemosOpenAIAdapter(client, dispatcher=False, result_renderer=FakeResultRenderer())
    tool_call = FakeToolCall(id="call_123", function=FakeFunction(name="search_memories", arguments="{}"))

    chunks = [chunk async for chunk in adapter.handle_tool_call_stream(tool_call)]

    assert chunks == [
        {
            "type": "mnemos.tool_result",
            "message": {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "known MNEMOS result",
            },
        }
    ]
