from __future__ import annotations

import tomllib
from pathlib import Path


def test_core_dependency_requires_dispatcher_release() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())

    assert "mnemos-bridge-core>=0.1.1" in project["project"]["dependencies"]
