from __future__ import annotations

from pathlib import Path


# Synthetic profiles commonly used throughout kanban tests. These are test-only
# stand-ins for valid Hermes assignees so create_task() can exercise the real
# profile-existence contract without every individual test rebuilding the same
# boilerplate profile tree.
COMMON_TEST_PROFILES = {
    "worker",
    "worker-2",
    "reviewer",
    "writer",
    "alice",
    "bob",
    "carol",
    "librarian",
    "a",
    "b",
    "x",
    "y",
    "w",
    "r",
    "peer",
    "orig",
    "dev",
    "teknium",
    "docs",
    "researcher",
    "linguist",
    "daily",
    "test-worker",
    "shared",
    "bench-worker",
    "broken",
    "c",
    "coder",
    "crashy",
    "fixed",
    "p",
    "slow-worker",
    "orion-cc",
    "orion-research",
    "jules",
    "qa",
    "old",
    "new",
    "some-profile",
    "someone",
}


def seed_test_profiles(home: Path, extra: set[str] | None = None) -> None:
    """Create a standard synthetic profiles tree under ``home``.

    Tests that need genuinely non-profile assignees should bypass create_task()
    intentionally (for example by creating a task first, then updating the row),
    rather than depending on implicit acceptance of nonexistent assignees.
    """
    profiles = home / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    for name in sorted(COMMON_TEST_PROFILES | (extra or set())):
        profile_dir = profiles / name
        profile_dir.mkdir(parents=True, exist_ok=True)
        config = profile_dir / "config.yaml"
        if not config.exists():
            config.write_text("model: {}\n", encoding="utf-8")