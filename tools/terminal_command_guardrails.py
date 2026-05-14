#!/usr/bin/env python3
"""Shared command classification for terminal launch guardrails.

Keep long-lived process detection in one lightweight module so both the
terminal tool and the agent-level loop guardrails reason about the same
command shapes.
"""

from __future__ import annotations

import re


SHELL_LEVEL_BACKGROUND_RE = re.compile(r"\b(?:nohup|disown|setsid)\b", re.IGNORECASE)
INLINE_BACKGROUND_AMP_RE = re.compile(r"\s&\s")
TRAILING_BACKGROUND_AMP_RE = re.compile(r"\s&\s*(?:#.*)?$")
LONG_LIVED_FOREGROUND_PATTERNS = (
    re.compile(r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|start|serve|watch)\b", re.IGNORECASE),
    re.compile(r"\bdocker\s+compose\s+up\b", re.IGNORECASE),
    re.compile(r"\bnext\s+dev\b", re.IGNORECASE),
    re.compile(r"\bvite(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bnodemon\b", re.IGNORECASE),
    re.compile(r"\buvicorn\b", re.IGNORECASE),
    re.compile(r"\bgunicorn\b", re.IGNORECASE),
    re.compile(r"\bpython(?:3)?\s+-m\s+http\.server\b", re.IGNORECASE),
)


def looks_like_help_or_version_command(command: str) -> bool:
    """Return True for informational invocations that should never be blocked."""
    normalized = " ".join((command or "").lower().split())
    return (
        " --help" in normalized
        or normalized.endswith(" -h")
        or " --version" in normalized
        or normalized.endswith(" -v")
    )


def looks_like_long_lived_launch(command: str) -> bool:
    """Return True when *command* appears to launch a server/watcher process."""
    if not isinstance(command, str) or looks_like_help_or_version_command(command):
        return False
    if SHELL_LEVEL_BACKGROUND_RE.search(command):
        return True
    if INLINE_BACKGROUND_AMP_RE.search(command) or TRAILING_BACKGROUND_AMP_RE.search(command):
        return True
    return any(pattern.search(command) for pattern in LONG_LIVED_FOREGROUND_PATTERNS)


def foreground_background_guidance(command: str) -> str | None:
    """Suggest background mode when a foreground command looks long-lived."""
    if looks_like_help_or_version_command(command):
        return None

    if SHELL_LEVEL_BACKGROUND_RE.search(command):
        return (
            "Foreground command uses shell-level background wrappers (nohup/disown/setsid). "
            "Use terminal(background=true) so Hermes can track the process, then run "
            "readiness checks and tests in separate commands."
        )

    if INLINE_BACKGROUND_AMP_RE.search(command) or TRAILING_BACKGROUND_AMP_RE.search(command):
        return (
            "Foreground command uses '&' backgrounding. Use terminal(background=true) for long-lived "
            "processes, then run health checks and tests in follow-up terminal calls."
        )

    if any(pattern.search(command) for pattern in LONG_LIVED_FOREGROUND_PATTERNS):
        return (
            "This foreground command appears to start a long-lived server/watch process. "
            "Run it with background=true, verify readiness (health endpoint/log signal), "
            "then execute tests in a separate command."
        )

    return None


def managed_background_conflict_guidance(command: str) -> str | None:
    """Reject shell-level backgrounding when Hermes already manages the process."""
    if looks_like_help_or_version_command(command):
        return None

    if SHELL_LEVEL_BACKGROUND_RE.search(command):
        return (
            "background=true already launches and tracks the process for you. "
            "Remove shell-level background wrappers (nohup/disown/setsid) and pass "
            "the long-lived command directly."
        )

    if INLINE_BACKGROUND_AMP_RE.search(command) or TRAILING_BACKGROUND_AMP_RE.search(command):
        return (
            "background=true already launches and tracks the process for you. "
            "Remove '&' from the command and pass the server/watch command directly so Hermes "
            "can manage its lifecycle, output, and cleanup."
        )

    return None