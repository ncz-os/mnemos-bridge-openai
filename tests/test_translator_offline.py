from __future__ import annotations

from typing import Any

import pytest

import mnemos_bridge_openai.adapter as adapter_module
from mnemos_bridge_openai import MnemosOpenAIAdapter


class FakeMcpClient:
    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search_memories",
                "description": "Search MNEMOS memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]


class FakeSchemaTranslator:
    def to_openai(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"],
                },
            }
            for tool in tools
        ]


@pytest.mark.asyncio
async def test_openai_tools_uses_translator_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_module, "SchemaTranslator", FakeSchemaTranslator)

    adapter = MnemosOpenAIAdapter(FakeMcpClient())
    tools = await adapter.openai_tools()

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "search_memories",
                "description": "Search MNEMOS memories.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    tool = tools[0]
    assert tool["type"] == "function"
    assert set(tool["function"]) == {"name", "description", "parameters"}
    assert set(tool["function"]["parameters"]) >= {"type", "properties", "required"}
