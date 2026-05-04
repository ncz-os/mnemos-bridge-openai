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
