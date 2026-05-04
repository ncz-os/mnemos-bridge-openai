# mnemos-bridge-openai

OpenAI Python SDK adapter for the MNEMOS bridge abstraction.

This package connects a MNEMOS MCP HTTP/SSE endpoint to OpenAI Chat Completions tools. It uses
`mnemos-bridge-core` for MCP transport, schema translation, dispatch, result rendering, and auth resolution.

## Install

```bash
uv pip install mnemos-bridge-openai
```

For local development before `mnemos-bridge-core` is published on PyPI:

```bash
uv sync --extra dev
```

The `pyproject.toml` includes a `tool.uv.sources` entry that points `mnemos-bridge-core` at
`git+https://gitlab.com/mnemos-os/mnemos-bridge-core`.

## API

```python
from mnemos_bridge_openai import MnemosOpenAIAdapter
```

`MnemosOpenAIAdapter.connect(mcp_url, mcp_token, timeout=30)` connects to a MNEMOS MCP endpoint.

`await adapter.openai_tools()` returns Chat Completions tool definitions:

```python
[{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
```

`await adapter.handle_tool_call(tool_call)` accepts an OpenAI `ChatCompletionMessageToolCall` or a compatible object
and returns a tool message:

```python
{"role": "tool", "tool_call_id": "...", "content": "..."}
```

`await adapter.aclose()` closes the MCP client. The adapter also works as an async context manager.

## Stateless One-Turn Example

```python
import asyncio
import os

from openai import AsyncOpenAI

from mnemos_bridge_openai import MnemosOpenAIAdapter


async def main() -> None:
    client = AsyncOpenAI()
    messages = [{"role": "user", "content": "Search MNEMOS for memories about infrastructure"}]

    async with await MnemosOpenAIAdapter.connect(
        os.environ["MNEMOS_TEST_BASE"],
        os.environ["MNEMOS_MCP_TOKEN"],
    ) as adapter:
        first = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=await adapter.openai_tools(),
        )

        assistant = first.choices[0].message
        messages.append(assistant.model_dump(exclude_none=True))
        for tool_call in assistant.tool_calls or []:
            messages.append(await adapter.handle_tool_call(tool_call))

        final = await client.chat.completions.create(model="gpt-4o", messages=messages)
        print(final.choices[0].message.content)


asyncio.run(main())
```

## Multi-Turn Loop With Tool Dispatch

```python
from typing import Any


async def run_turn(client: Any, adapter: MnemosOpenAIAdapter, messages: list[dict[str, Any]]) -> str:
    tools = await adapter.openai_tools()

    while True:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content or ""

        for tool_call in message.tool_calls:
            messages.append(await adapter.handle_tool_call(tool_call))
```

## Agents SDK Integration Recipe

Use the adapter at the Chat Completions boundary used by your agent runtime:

```python
async with await MnemosOpenAIAdapter.connect(mcp_url, mcp_token) as adapter:
    mnemos_tools = await adapter.openai_tools()

    # Supply mnemos_tools anywhere your agent runtime accepts OpenAI Chat Completions tool schemas.
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=agent_messages,
        tools=[*agent_native_tools, *mnemos_tools],
    )

    # Route OpenAI tool calls whose names belong to MNEMOS back through the adapter.
    for tool_call in response.choices[0].message.tool_calls or []:
        if any(tool["function"]["name"] == tool_call.function.name for tool in mnemos_tools):
            agent_messages.append(await adapter.handle_tool_call(tool_call))
```

If your Agents SDK layer exposes a custom tool-dispatch hook, call `adapter.handle_tool_call(tool_call)` in that hook
and append the returned tool message before continuing the run.

## Integration Tests

Live integration tests are skipped unless all required environment variables are set:

```bash
export OPENAI_API_KEY=...
export MNEMOS_TEST_BASE=http://192.168.207.67:5003
export MNEMOS_MCP_TOKEN=...
pytest tests/integration
```

Tokens are never hardcoded.

## What Does Not Work Yet

`handle_tool_call_stream()` currently emits a single completed tool-result chunk. True incremental streaming is planned
for v0.2 once MNEMOS core exposes streaming dispatcher events.
