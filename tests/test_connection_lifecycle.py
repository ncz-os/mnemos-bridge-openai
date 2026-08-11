from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Self

import pytest

import mnemos_bridge_openai.adapter as adapter_module
from mnemos_bridge_openai import MnemosOpenAIAdapter


class TaskAffineClient:
    instance: TaskAffineClient | None = None

    def __init__(self) -> None:
        self.enter_task: asyncio.Task[object] | None = None
        self.operation_task: asyncio.Task[object] | None = None
        self.exit_task: asyncio.Task[object] | None = None
        self.closed = False
        type(self).instance = self

    @classmethod
    def from_url(cls, url: str, *, token: str, timeout: float) -> Self:
        return cls()

    async def __aenter__(self) -> Self:
        self.enter_task = asyncio.current_task()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exit_task = asyncio.current_task()
        if self.exit_task is not self.enter_task:
            raise RuntimeError("Attempted to exit cancel scope in a different task than it was entered in")
        self.closed = True

    async def list_tools(self) -> list[dict[str, object]]:
        self.operation_task = asyncio.current_task()
        return []


@pytest.mark.asyncio
async def test_connect_and_close_keep_task_affine_transport_on_owner_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_module, "McpClient", TaskAffineClient)
    monkeypatch.setattr(adapter_module, "AuthResolver", None)
    monkeypatch.setattr(adapter_module, "SchemaTranslator", None)

    adapter = await MnemosOpenAIAdapter.connect("https://mnemos.example/sse", "secret")
    assert await adapter.openai_tools() == []
    await asyncio.create_task(adapter.aclose())

    client = TaskAffineClient.instance
    assert client is not None
    assert client.closed is True
    assert client.operation_task is client.enter_task
    assert client.enter_task is client.exit_task


def test_transitive_core_import_error_is_not_reported_as_missing_core(tmp_path: Path) -> None:
    fake_package = tmp_path / "mnemos_bridge_core"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text("import dependency_missing_inside_core\n")
    source = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(source)))

    result = subprocess.run(
        [sys.executable, "-c", "import mnemos_bridge_openai.adapter"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "dependency_missing_inside_core" in result.stderr
