"""Tests for the egress sidecar runtime (docker calls faked)."""

import threading
from ipaddress import ip_network
from unittest.mock import Mock, patch

import pytest

from openhands.workspace import DockerWorkspace
from openhands.workspace.docker.egress_runtime import (
    EgressRuntime,
    subnets_overlap_allowlist,
)
from openhands.workspace.docker.network_policy import (
    AllowedEndpoint,
    WorkspaceNetworkPolicy,
)


def test_public_policy_is_default_and_needs_no_sidecar():
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(server_image="test:latest")
    assert ws.network_policy.mode == "public"
    assert not ws.network_policy.requires_sidecar


def test_env_var_sets_policy_without_explicit_argument(monkeypatch):
    monkeypatch.setenv("OH_NETWORK_MODE", "no-network")
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(server_image="test:latest")
    assert ws.network_policy.mode == "no-network"


def test_invalid_env_var_raises_rather_than_defaulting_public(monkeypatch):
    monkeypatch.setenv("OH_NETWORK_MODE", "nonsense")
    with pytest.raises(ValueError):
        with patch.object(DockerWorkspace, "_start_container"):
            DockerWorkspace(server_image="test:latest")


def test_workspace_drops_net_admin_and_net_raw_in_sidecar_mode():
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(
            server_image="test:latest",
            network_policy=WorkspaceNetworkPolicy(mode="no-network"),
        )
    ws.host_port = 30000
    ws._egress = Mock(sidecar_id="side123")
    args = ws._build_run_args("img:latest", container_name="c")
    assert args[args.index("--cap-drop") : args.index("--cap-drop") + 2] == [
        "--cap-drop",
        "NET_ADMIN",
    ]
    assert "NET_RAW" in args
    assert "--network" in args
    assert "container:side123" in args
    assert "no-new-privileges=true" in args


def test_workspace_publishes_no_ports_when_sidecar_owns_namespace():
    """Docker rejects -p on a container using container: network mode."""
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(
            server_image="test:latest",
            network_policy=WorkspaceNetworkPolicy(mode="no-network"),
        )
    ws.host_port = 30000
    ws._egress = Mock(sidecar_id="side123")
    assert ws._build_port_args() == []


def test_build_port_args_extra_ports_public_mode_publishes_all_three():
    """In public (non-sidecar) mode with extra_ports, publishes 8000/8001/8002."""
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(server_image="test:latest", extra_ports=True)
    ws.host_port = 30000
    args = ws._build_port_args()
    assert "30000:8000" in args
    assert "30001:8001" in args
    assert "30002:8002" in args


def test_build_port_args_extra_ports_sidecar_mode_publishes_nothing():
    """In sidecar mode, _build_port_args returns empty regardless of extra_ports."""
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(
            server_image="test:latest",
            extra_ports=True,
            network_policy=WorkspaceNetworkPolicy(mode="no-network"),
        )
    ws.host_port = 30000
    ws._egress = Mock(sidecar_id="side123")
    assert ws._build_port_args() == []


def test_subnet_overlap_detected():
    policy = WorkspaceNetworkPolicy(mode="no-network")
    assert subnets_overlap_allowlist([ip_network("10.5.0.0/16")], policy) is True
    assert subnets_overlap_allowlist([ip_network("172.18.0.0/16")], policy) is False


def test_subnet_overlap_checks_caller_endpoints_too():
    policy = WorkspaceNetworkPolicy(
        mode="static-allowlist",
        allowed_endpoints=(AllowedEndpoint(destination=ip_network("192.0.2.0/24")),),
    )
    assert subnets_overlap_allowlist([ip_network("192.0.2.128/25")], policy) is True


def test_cleanup_is_idempotent():
    runtime = EgressRuntime(
        workspace_id="ws1", controller_id="ctrl1", sidecar_id="side1", network_id="net1"
    )
    with patch(
        "openhands.workspace.docker.egress_runtime.execute_command"
    ) as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        runtime.cleanup()
        first = mock_exec.call_count
        runtime.cleanup()
        runtime.cleanup()
    assert mock_exec.call_count == first, "repeat cleanup must be a no-op"


def test_cleanup_is_threadsafe():
    runtime = EgressRuntime(
        workspace_id="ws1", controller_id="ctrl1", sidecar_id="side1", network_id="net1"
    )
    with patch(
        "openhands.workspace.docker.egress_runtime.execute_command"
    ) as mock_exec:
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")
        threads = [threading.Thread(target=runtime.cleanup) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        calls = mock_exec.call_count
    assert calls <= 4, f"expected one cleanup pass, saw {calls} docker calls"


def test_cleanup_continues_after_one_resource_fails():
    """Best-effort, not fail-fast: a failed stop must not skip network removal."""
    runtime = EgressRuntime(
        workspace_id="ws1", controller_id="ctrl1", sidecar_id="side1", network_id="net1"
    )
    with patch(
        "openhands.workspace.docker.egress_runtime.execute_command"
    ) as mock_exec:

        def fail_on_stop(cmd, *a, **kw):
            if "stop" in cmd or "rm" in cmd:
                return Mock(returncode=1, stdout="", stderr="boom")
            return Mock(returncode=0, stdout="", stderr="")

        mock_exec.side_effect = fail_on_stop
        runtime.cleanup()
        issued = [" ".join(c.args[0]) for c in mock_exec.call_args_list]
    assert any("network" in cmd and "rm" in cmd for cmd in issued)


def test_rules_file_is_world_readable_and_directory_is_not(tmp_path, monkeypatch):
    """Rules file must be 0o644; directory must remain 0o700.

    The sidecar runs with --cap-drop ALL --cap-add NET_ADMIN, which strips
    CAP_DAC_OVERRIDE.  Without that capability uid-0 cannot read a file it
    does not own that is mode 0o600.  0o644 allows the read while the 0o700
    directory still blocks host users who lack directory-traverse permission.

    This test will fail if the chmod in start_egress_sidecar is reverted to
    0o600 or if the directory is accidentally loosened to 0o755.
    """
    import openhands.workspace.docker.egress_runtime as _rt

    monkeypatch.setattr(_rt, "STATE_ROOT", tmp_path)

    # Stub every docker call so no daemon is needed.
    fake_network_id = "fakenetid123"
    call_count = {"n": 0}

    def fake_exec(cmd, *args, **kwargs):
        call_count["n"] += 1
        # docker network create -> return a network id
        if "network" in cmd and "create" in cmd:
            return Mock(returncode=0, stdout=fake_network_id, stderr="")
        # docker network inspect (subnets) -> return a non-overlapping range
        if "network" in cmd and "inspect" in cmd:
            return Mock(returncode=0, stdout="172.30.0.0/16 ", stderr="")
        # docker run -> pretend the container started
        if "run" in cmd:
            return Mock(returncode=0, stdout="fakesidecarid", stderr="")
        # docker exec (readiness probe) -> signal ready immediately
        if "exec" in cmd:
            return Mock(returncode=0, stdout="", stderr="")
        # anything else (stop, rm, network rm) -> success
        return Mock(returncode=0, stdout="", stderr="")

    with patch(
        "openhands.workspace.docker.egress_runtime.execute_command",
        side_effect=fake_exec,
    ):
        runtime = _rt.start_egress_sidecar(
            WorkspaceNetworkPolicy(mode="no-network"),
            host_port=39999,
        )

    try:
        assert runtime.rules_path is not None
        file_mode = runtime.rules_path.stat().st_mode & 0o777
        dir_mode = runtime.rules_path.parent.stat().st_mode & 0o777
        assert file_mode == 0o644, (
            f"rules file mode is {oct(file_mode)}, want 0o644; "
            "the sidecar runs --cap-drop ALL which removes CAP_DAC_OVERRIDE "
            "and uid-0 cannot read a 0o600 file it does not own"
        )
        assert dir_mode == 0o700, (
            f"rules directory mode is {oct(dir_mode)}, want 0o700; "
            "loosening the directory would expose the rules file to other host users"
        )
    finally:
        with patch(
            "openhands.workspace.docker.egress_runtime.execute_command",
            return_value=Mock(returncode=0, stdout="", stderr=""),
        ):
            runtime.cleanup()
