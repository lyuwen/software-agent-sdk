"""Tests for orphan reconciliation."""

import json
import os
import time
from unittest.mock import Mock, patch

import pytest

from openhands.workspace.docker import egress_runtime
from openhands.workspace.docker.egress_runtime import (
    controller_is_alive,
    reconcile_orphans,
)


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    monkeypatch.setattr(egress_runtime, "STATE_ROOT", tmp_path)
    return tmp_path


def _write_manifest(root, workspace_id, controller_id, lease_expires_at):
    directory = root / workspace_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "workspace_id": workspace_id,
                "controller_id": controller_id,
                "network_id": f"{workspace_id}-net",
                "sidecar_id": f"{workspace_id}-side",
                "rules_path": str(directory / "rules.nft"),
                "policy_digest": "deadbeef",
                "status": "active",
                "lease_expires_at": lease_expires_at,
            }
        )
    )
    (directory / "rules.nft").write_text("table inet workspace_egress {}\n")
    return directory


def test_live_pid_is_alive():
    assert controller_is_alive(f"boot1234-{os.getpid()}-abcd1234") is True


def test_dead_pid_is_not_alive():
    # PID 0 is never a real user process.
    assert controller_is_alive("boot1234-0-abcd1234") is False


def test_different_boot_id_is_dead():
    assert controller_is_alive(f"ffffffff-{os.getpid()}-abcd1234") is False


def test_malformed_controller_id_is_treated_as_alive():
    """Undeterminable liveness must NOT delete: leaking beats destroying."""
    assert controller_is_alive("garbage") is True


def test_stale_manifest_from_dead_controller_is_reclaimed(state_root):
    _write_manifest(state_root, "ws-dead", "boot1234-0-abcd1234", time.time() - 999)
    with patch.object(egress_runtime, "execute_command") as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        reclaimed = reconcile_orphans()
    assert "ws-dead" in reclaimed
    assert not (state_root / "ws-dead").exists()


def test_live_lease_is_preserved(state_root):
    _write_manifest(
        state_root, "ws-live", f"boot1234-{os.getpid()}-abcd1234", time.time() + 999
    )
    with patch.object(egress_runtime, "execute_command") as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        reclaimed = reconcile_orphans()
    assert reclaimed == []
    assert (state_root / "ws-live").exists()


def test_expired_lease_but_live_controller_is_preserved(state_root):
    """A busy controller may let its lease lapse; the process is authoritative."""
    _write_manifest(
        state_root, "ws-busy", f"boot1234-{os.getpid()}-abcd1234", time.time() - 999
    )
    with patch.object(egress_runtime, "execute_command") as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        assert reconcile_orphans() == []
    assert (state_root / "ws-busy").exists()


def test_unreadable_manifest_is_left_alone(state_root):
    directory = state_root / "ws-corrupt"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{ not json")
    with patch.object(egress_runtime, "execute_command") as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        assert reconcile_orphans() == []
    assert directory.exists()


def test_reconcile_removes_container_before_network(state_root):
    _write_manifest(state_root, "ws-order", "boot1234-0-abcd1234", time.time() - 999)
    with patch.object(egress_runtime, "execute_command") as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        reconcile_orphans()
        issued = [" ".join(c.args[0]) for c in mock_exec.call_args_list]
    container_idx = next(i for i, c in enumerate(issued) if "ws-order-side" in c)
    network_idx = next(i for i, c in enumerate(issued) if "ws-order-net" in c)
    assert container_idx < network_idx


def test_one_failure_does_not_stop_the_scan(state_root):
    _write_manifest(state_root, "ws-a", "boot1234-0-abcd1234", time.time() - 999)
    _write_manifest(state_root, "ws-b", "boot1234-0-abcd1234", time.time() - 999)
    with patch.object(egress_runtime, "execute_command") as mock_exec:

        def flaky(cmd, *a, **kw):
            if "ws-a-side" in " ".join(cmd):
                raise RuntimeError("docker exploded")
            return Mock(returncode=0, stdout="", stderr="")

        mock_exec.side_effect = flaky
        reclaimed = reconcile_orphans()
    assert "ws-b" in reclaimed
