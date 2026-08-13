from __future__ import annotations

import tomllib
from pathlib import Path


def _core_floor(deps: list[str]) -> tuple[int, ...]:
    """Parsed lower bound of the mnemos-bridge-core pin.

    Asserting the exact pin string made every legitimate floor raise fail this
    test, which is backwards: the requirement is a MINIMUM, so a higher floor
    satisfies it by definition.
    """
    for dep in deps:
        if dep.replace(" ", "").startswith("mnemos-bridge-core>="):
            spec = dep.replace(" ", "").split(">=", 1)[1].split(",")[0]
            return tuple(int(p) for p in spec.split(".") if p.isdigit())
    raise AssertionError(f"no mnemos-bridge-core lower bound in {deps}")

def test_core_dependency_requires_dispatcher_release() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())

    # The dispatcher fix landed in core 0.1.1; anything at or above it is fine.
    assert _core_floor(project["project"]["dependencies"]) >= (0, 1, 1)


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
