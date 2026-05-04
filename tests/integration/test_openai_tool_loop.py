from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (
            os.environ.get("OPENAI_API_KEY")
            and os.environ.get("MNEMOS_TEST_BASE")
            and os.environ.get("MNEMOS_MCP_TOKEN")
        ),
        reason="set OPENAI_API_KEY, MNEMOS_TEST_BASE, and MNEMOS_MCP_TOKEN to run live integration tests",
    ),
]

openai = pytest.importorskip("openai")
pytest.importorskip("mnemos_bridge_core")

from mnemos_bridge_openai import MnemosOpenAIAdapter


def _message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return dict(message)


@pytest.mark.asyncio
async def test_openai_tool_loop_search_memories() -> None:
    client = openai.AsyncOpenAI()

    async with await MnemosOpenAIAdapter.connect(
        os.environ["MNEMOS_TEST_BASE"],
        os.environ["MNEMOS_MCP_TOKEN"],
    ) as adapter:
        tools = await adapter.openai_tools()
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": "Search MNEMOS for memories about infrastructure",
            }
        ]

        first = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        assistant_message = first.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        assert any(tool_call.function.name == "search_memories" for tool_call in tool_calls)

        messages.append(_message_to_dict(assistant_message))
        for tool_call in tool_calls:
            messages.append(await adapter.handle_tool_call(tool_call))

        final = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        content = final.choices[0].message.content or ""

    assert content
    assert any(term in content.lower() for term in ("infrastructure", "memory", "memories", "mnemos"))
