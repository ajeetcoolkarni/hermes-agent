from __future__ import annotations

import importlib.util
import json
import os
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(os.path.expanduser("~/.hermes/scripts/kanban_deadlock_resolver.py"))


@pytest.fixture
def resolver_module():
    assert SCRIPT_PATH.exists(), f"missing resolver script: {SCRIPT_PATH}"
    module_name = f"kanban_deadlock_resolver_test_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)


def _task(mod, **overrides):
    data = dict(
        id="t_task",
        title="task",
        status="todo",
        assignee="worker",
        body="",
        workspace_kind="scratch",
        workspace_path="",
        consecutive_failures=0,
        max_retries=None,
        result=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        parents=[],
        children=[],
        runs=[],
        events=[],
        comments=[],
        block_reason=None,
        last_run_summary=None,
    )
    data.update(overrides)
    return mod.Task(**data)


def test_reviewer_remediation_pipeline_preserves_source_workspace(resolver_module):
    workspace = "/tmp/shared repo"
    review = _task(
        resolver_module,
        id="t_review",
        title="Review auth flow",
        status="blocked",
        assignee="reviewer",
        body="still broken",
        parents=["t_worker"],
        workspace_kind="dir",
        workspace_path=workspace,
    )
    worker = _task(
        resolver_module,
        id="t_worker",
        assignee="worker",
        status="done",
        workspace_kind="dir",
        workspace_path=workspace,
    )

    cmds = resolver_module.PatternReviewerBlockedNoRemediation()._create_pipeline(
        review,
        {review.id: review, worker.id: worker},
    )

    create_cmds = [cmd for cmd in cmds if cmd.startswith("hermes kanban create ")]
    assert len(create_cmds) == 2
    expected = shlex.quote(f"dir:{workspace}")
    assert all(f"--workspace {expected}" in cmd for cmd in create_cmds)


def test_reviewer_remediation_pipeline_falls_back_to_scratch_without_source_workspace(resolver_module):
    review = _task(
        resolver_module,
        id="t_review",
        title="Review auth flow",
        status="blocked",
        assignee="reviewer",
        body="still broken",
        parents=["t_worker"],
    )
    worker = _task(resolver_module, id="t_worker", assignee="worker", status="done")

    cmds = resolver_module.PatternReviewerBlockedNoRemediation()._create_pipeline(
        review,
        {review.id: review, worker.id: worker},
    )

    create_cmds = [cmd for cmd in cmds if cmd.startswith("hermes kanban create ")]
    assert len(create_cmds) == 2
    assert all("--workspace scratch" in cmd for cmd in create_cmds)
    assert not any("/mnt/win-workspace/nebula-drift" in cmd for cmd in create_cmds)


def test_worker_rereview_pipeline_falls_back_to_scratch_without_source_workspace(resolver_module):
    fix_task = _task(
        resolver_module,
        id="t_fix",
        title="Fix auth flow",
        status="done",
        assignee="worker-2",
    )

    cmds = resolver_module.PatternWorkerFixCompletedNeedsRereview().resolve(
        fix_task,
        {fix_task.id: fix_task},
    )

    create_cmd = next(cmd for cmd in cmds if cmd.startswith("hermes kanban create "))
    assert "--workspace scratch" in create_cmd
    assert "/mnt/win-workspace/nebula-drift" not in create_cmd


def test_reviewer_remediation_pipeline_recovers_real_workspace_from_dir_ancestor_when_current_task_is_scratch(resolver_module):
    workspace = "/mnt/win-workspace/nebula-drift"
    root_review = _task(
        resolver_module,
        id="t_root_review",
        title="Initial strict review",
        status="done",
        assignee="reviewer",
        workspace_kind="dir",
        workspace_path=workspace,
    )
    remediation = _task(
        resolver_module,
        id="t_fix_1",
        title="First remediation",
        status="done",
        assignee="worker-2",
        parents=[root_review.id],
        children=["t_rereview_1"],
        workspace_kind="scratch",
        workspace_path="/home/test/.hermes/kanban/workspaces/t_fix_1",
    )
    rereview = _task(
        resolver_module,
        id="t_rereview_1",
        title="Follow-up review",
        status="blocked",
        assignee="reviewer",
        parents=[remediation.id],
        workspace_kind="scratch",
        workspace_path="/home/test/.hermes/kanban/workspaces/t_rereview_1",
    )
    root_review.children = [remediation.id]

    cmds = resolver_module.PatternReviewerBlockedNoRemediation()._create_pipeline(
        rereview,
        {
            root_review.id: root_review,
            remediation.id: remediation,
            rereview.id: rereview,
        },
    )

    create_cmds = [cmd for cmd in cmds if cmd.startswith("hermes kanban create ")]
    expected = shlex.quote(f"dir:{workspace}")
    assert len(create_cmds) == 2
    assert all(f"--workspace {expected}" in cmd for cmd in create_cmds)


def test_worker_rereview_pipeline_recovers_real_workspace_from_dir_ancestor_when_fix_task_is_scratch(resolver_module):
    workspace = "/mnt/win-workspace/nebula-drift"
    upstream_review = _task(
        resolver_module,
        id="t_upstream_review",
        title="Review gate",
        status="done",
        assignee="reviewer",
        workspace_kind="dir",
        workspace_path=workspace,
    )
    fix_task = _task(
        resolver_module,
        id="t_fix",
        title="Fix auth flow",
        status="done",
        assignee="worker-2",
        parents=[upstream_review.id],
        workspace_kind="scratch",
        workspace_path="/home/test/.hermes/kanban/workspaces/t_fix",
    )
    upstream_review.children = [fix_task.id]

    cmds = resolver_module.PatternWorkerFixCompletedNeedsRereview().resolve(
        fix_task,
        {upstream_review.id: upstream_review, fix_task.id: fix_task},
    )

    create_cmd = next(cmd for cmd in cmds if cmd.startswith("hermes kanban create "))
    expected = shlex.quote(f"dir:{workspace}")
    assert f"--workspace {expected}" in create_cmd


def test_worker_rereview_detect_blocks_non_terminal_worker_with_worker_descendant(resolver_module):
    workspace = "/tmp/shared repo"
    worker_a = _task(
        resolver_module,
        id="t_worker_a",
        title="phase a",
        status="done",
        assignee="worker",
        workspace_kind="dir",
        workspace_path=workspace,
        children=["t_worker_b"],
    )
    worker_b = _task(
        resolver_module,
        id="t_worker_b",
        title="phase b",
        status="ready",
        assignee="worker-2",
        workspace_kind="dir",
        workspace_path=workspace,
        parents=["t_worker_a"],
    )
    pattern = resolver_module.PatternWorkerFixCompletedNeedsRereview()
    assert pattern.detect(worker_a, {worker_a.id: worker_a, worker_b.id: worker_b}) is False


def test_worker_rereview_detect_blocks_when_active_reviewer_exists_in_lineage(resolver_module):
    workspace = "/tmp/shared repo"
    worker = _task(
        resolver_module,
        id="t_worker",
        title="fix",
        status="done",
        assignee="worker-2",
        workspace_kind="dir",
        workspace_path=workspace,
        children=["t_review"],
    )
    reviewer = _task(
        resolver_module,
        id="t_review",
        title="review",
        status="ready",
        assignee="reviewer",
        workspace_kind="dir",
        workspace_path=workspace,
        parents=["t_worker"],
    )
    pattern = resolver_module.PatternWorkerFixCompletedNeedsRereview()
    assert pattern.detect(worker, {worker.id: worker, reviewer.id: reviewer}) is False


def test_worker_rereview_resolve_parents_reviewer_to_terminal_worker(resolver_module):
    workspace = "/tmp/shared repo"
    fix_task = _task(
        resolver_module,
        id="t_fix",
        title="Fix auth flow",
        status="done",
        assignee="worker-2",
        workspace_kind="dir",
        workspace_path=workspace,
    )

    cmds = resolver_module.PatternWorkerFixCompletedNeedsRereview().resolve(
        fix_task,
        {fix_task.id: fix_task},
    )

    create_cmd = next(cmd for cmd in cmds if cmd.startswith("hermes kanban create "))
    assert "--assignee reviewer --parent t_fix" in create_cmd


def test_execute_commands_preserves_workspace_with_spaces(resolver_module, tmp_path, monkeypatch):
    body_path = tmp_path / "body.txt"
    body_path.write_text("details")
    workspace = tmp_path / "repo with spaces"

    cmd = (
        "hermes kanban create $'Fix: spacing bug' "
        "--assignee worker-2 --parent t_parent "
        f"--workspace {shlex.quote(f'dir:{workspace}')} "
        f"--body $(cat {body_path}) "
        "--idempotency-key remediation_t_parent"
    )

    seen: dict[str, list[str]] = {}

    def _fake_run(args, **kwargs):
        seen["args"] = list(args)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(resolver_module.subprocess, "run", _fake_run)

    resolver_module.execute_commands([cmd], dry_run=False)

    workspace_idx = seen["args"].index("--workspace")
    assert seen["args"][workspace_idx + 1] == f"dir:{workspace}"


def test_execute_commands_preserves_workspace_without_spaces(resolver_module, tmp_path, monkeypatch):
    body_path = tmp_path / "body.txt"
    body_path.write_text("details")
    workspace = tmp_path / "repo"

    cmd = (
        "hermes kanban create $'Fix: plain path bug' "
        "--assignee worker-2 --parent t_parent "
        f"--workspace {shlex.quote(f'dir:{workspace}')} "
        f"--body $(cat {body_path}) "
        "--idempotency-key remediation_t_plain"
    )

    seen: dict[str, list[str]] = {}

    def _fake_run(args, **kwargs):
        seen["args"] = list(args)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(resolver_module.subprocess, "run", _fake_run)

    resolver_module.execute_commands([cmd], dry_run=False)

    workspace_idx = seen["args"].index("--workspace")
    assert seen["args"][workspace_idx + 1] == f"dir:{workspace}"


def test_parse_create_cmd_preserves_title_workspace_and_body_file(resolver_module, tmp_path):
    body_path = tmp_path / "body.txt"
    body_path.write_text("hello body")
    workspace = tmp_path / "repo with spaces"

    cmd = (
        "hermes kanban create $'Fix: parser bug' "
        "--assignee worker-2 --parent t_parent "
        f"--workspace {shlex.quote(f'dir:{workspace}')} "
        f"--body $(cat {body_path}) "
        "--idempotency-key remediation_parser"
    )

    args = resolver_module._parse_create_cmd(cmd)
    assert args[:4] == ["hermes", "kanban", "create", "Fix: parser bug"]
    workspace_idx = args.index("--workspace")
    assert args[workspace_idx + 1] == f"dir:{workspace}"
    body_idx = args.index("--body")
    assert args[body_idx + 1] == "hello body"
    assert args[-2:] == ["--idempotency-key", "remediation_parser"]
