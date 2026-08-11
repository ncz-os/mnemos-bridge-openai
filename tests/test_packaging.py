from __future__ import annotations

import tomllib
from pathlib import Path


def test_core_dependency_requires_dispatcher_release() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())

    assert "mnemos-bridge-core>=0.1.1" in project["project"]["dependencies"]


def test_mcp_dependency_excludes_incompatible_major_release() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())

    assert "mcp>=1.27,<2" in project["project"]["dependencies"]


def test_development_core_source_uses_canonical_namespace() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())

    assert project["tool"]["uv"]["sources"]["mnemos-bridge-core"]["git"] == (
        "https://gitlab.com/ncz-os/mnemos-bridge-core.git"
    )
