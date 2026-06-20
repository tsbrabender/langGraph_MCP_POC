"""Sandbox path resolution — enforces all tool access stays within a root directory."""

from pathlib import Path

from app.utils.errors import SandboxViolationError


def resolve_safe_path(sandbox_root: Path, relative: str) -> Path:
    """Resolve a user-supplied path relative to sandbox_root and verify containment.

    Args:
        sandbox_root: The absolute, resolved root directory all tools are confined to.
        relative: A path string supplied by the tool caller (may include '..' etc.).

    Returns:
        The fully resolved absolute Path, guaranteed to be inside sandbox_root.

    Raises:
        SandboxViolationError: If the resolved path escapes the sandbox root.
    """
    root = sandbox_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise SandboxViolationError(
            f"Path '{relative}' resolves outside sandbox root '{root}'"
        )
    return candidate
