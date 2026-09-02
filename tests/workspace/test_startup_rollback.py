"""A failed startup must not leave the main container running.

`--rm` does not help here: it fires when a container exits, and a container
whose health check never passed has not exited. While it lives it also holds
the egress sidecar's network namespace, so the sidecar and network cannot be
removed either.
"""

from unittest.mock import Mock, patch

import pytest

from openhands.workspace import DockerWorkspace, FlexWorkspace
from openhands.workspace.docker.network_policy import WorkspaceNetworkPolicy


def _fake_exec(cmd, *args, **kwargs):
    if "create" in cmd:
        return Mock(returncode=0, stdout="plugincontainerid", stderr="")
    if cmd[:2] == ["docker", "run"]:
        return Mock(returncode=0, stdout="maincontainerid", stderr="")
    return Mock(returncode=0, stdout="", stderr="")


def _start_and_fail_health(cls, **kwargs):
    """Start a workspace whose health check fails; return the issued argv."""
    egress = Mock(sidecar_id="sidecar123")
    with (
        patch(
            "openhands.workspace.docker.workspace.execute_command",
            side_effect=_fake_exec,
        ) as main_exec,
        patch(
            "openhands.workspace.docker.flex_workspace.execute_command",
            side_effect=_fake_exec,
        ) as flex_exec,
        patch(
            "openhands.workspace.docker.workspace.start_egress_sidecar",
            return_value=egress,
        ),
        patch.object(
            DockerWorkspace,
            "_wait_for_health",
            side_effect=RuntimeError("container never became healthy"),
        ),
        patch("openhands.sdk.workspace.RemoteWorkspace.model_post_init"),
    ):
        with pytest.raises(RuntimeError, match="never became healthy"):
            cls(
                network_policy=WorkspaceNetworkPolicy(mode="no-network"),
                detach_logs=False,
                **kwargs,
            )
        issued = [
            c.args[0] for mock in (main_exec, flex_exec) for c in mock.call_args_list
        ]
    return issued, egress


def test_failed_health_check_removes_the_main_container():
    issued, _ = _start_and_fail_health(DockerWorkspace, server_image="srv:latest")
    removals = [c for c in issued if c[:3] == ["docker", "rm", "-f"]]
    assert ["docker", "rm", "-f", "maincontainerid"] in removals, (
        "the main container was left running after a failed startup"
    )
    assert ["docker", "stop", "maincontainerid"] in issued


def test_rollback_removes_container_before_releasing_egress():
    """Same order as cleanup: the container holds the sidecar's namespace."""
    issued, egress = _start_and_fail_health(DockerWorkspace, server_image="srv:latest")
    assert egress.cleanup.called, "the egress sidecar was leaked"
    removal_idx = max(i for i, c in enumerate(issued) if "maincontainerid" in c)
    # cleanup() on the sidecar happens through the mock, after the last
    # container command was issued.
    assert removal_idx == len(issued) - 1


def test_rollback_is_best_effort_across_steps():
    """A failure removing the container must not skip egress cleanup."""
    egress = Mock(sidecar_id="sidecar123")

    def explode_on_container(cmd, *args, **kwargs):
        if "maincontainerid" in cmd:
            raise RuntimeError("docker exploded")
        return _fake_exec(cmd, *args, **kwargs)

    with (
        patch(
            "openhands.workspace.docker.workspace.execute_command",
            side_effect=explode_on_container,
        ),
        patch(
            "openhands.workspace.docker.workspace.start_egress_sidecar",
            return_value=egress,
        ),
        patch.object(
            DockerWorkspace, "_wait_for_health", side_effect=RuntimeError("unhealthy")
        ),
        patch("openhands.sdk.workspace.RemoteWorkspace.model_post_init"),
    ):
        with pytest.raises(RuntimeError):
            DockerWorkspace(
                server_image="srv:latest",
                network_policy=WorkspaceNetworkPolicy(mode="no-network"),
                detach_logs=False,
            )
    assert egress.cleanup.called, "a failed container removal skipped egress cleanup"


def test_flex_rollback_also_removes_the_plugin_container():
    issued, egress = _start_and_fail_health(FlexWorkspace, base_image="base:latest")
    assert egress.cleanup.called
    flex_removals = [
        c for c in issued if c[:3] == ["docker", "rm", "-f"] and "plugin" in c[3]
    ]
    assert flex_removals, "the plugin data container was leaked on rollback"
