"""Every DockerWorkspace launcher must go through the same egress enforcement.

The bug this guards against: a subclass that overrode ``_start_container``
silently skipped sidecar startup, so a non-public policy produced a fully
unrestricted container. These tests are parametrized over every launcher so a
future subclass cannot regress the invariant unnoticed.
"""

from unittest.mock import Mock, patch

import pytest

from openhands.workspace import DockerDevWorkspace, DockerWorkspace, FlexWorkspace
from openhands.workspace.docker.network_policy import WorkspaceNetworkPolicy
from openhands.workspace.docker.workspace import ContainerLaunchSpec


LAUNCHERS = [
    pytest.param(DockerWorkspace, {"server_image": "srv:latest"}, id="DockerWorkspace"),
    pytest.param(FlexWorkspace, {"base_image": "base:latest"}, id="FlexWorkspace"),
    pytest.param(
        DockerDevWorkspace, {"server_image": "srv:latest"}, id="DockerDevWorkspace"
    ),
]


def _fake_exec(cmd, *args, **kwargs):
    """Stand in for every docker invocation. No daemon, no containers."""
    if "create" in cmd:
        return Mock(returncode=0, stdout="plugincontainerid", stderr="")
    if "run" in cmd:
        return Mock(returncode=0, stdout="maincontainerid", stderr="")
    return Mock(returncode=0, stdout="", stderr="")


def _launch(cls, kwargs, policy):
    """Construct a workspace with all I/O faked; return the docker run argv."""
    sidecar = Mock(sidecar_id="sidecar123", workspace_container_id=None)
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
            return_value=sidecar,
        ) as start_sidecar,
        patch.object(DockerWorkspace, "_wait_for_health"),
        patch("openhands.sdk.workspace.RemoteWorkspace.model_post_init"),
    ):
        ws = cls(network_policy=policy, detach_logs=False, **kwargs)
        ws._container_id = None  # do not let __del__ issue a real docker stop
        ws._egress = None
        # Collect from both modules: the invariant is about what argv reaches
        # docker, not about which module happens to issue it.
        runs = [
            c.args[0]
            for mock in (main_exec, flex_exec)
            for c in mock.call_args_list
            if c.args[0][:2] == ["docker", "run"] and "-d" in c.args[0]
        ]
    assert len(runs) == 1, f"expected exactly one docker run, saw {len(runs)}"
    return runs[0], start_sidecar


@pytest.mark.parametrize(("cls", "kwargs"), LAUNCHERS)
def test_non_public_policy_always_attaches_to_a_sidecar(cls, kwargs):
    """A non-public policy MUST start a sidecar and join its namespace."""
    run_cmd, start_sidecar = _launch(
        cls, kwargs, WorkspaceNetworkPolicy(mode="no-network")
    )
    assert start_sidecar.called, (
        f"{cls.__name__} never started an egress sidecar: the requested policy "
        "was silently ignored and the container has unrestricted networking"
    )
    assert "--network" in run_cmd, f"{cls.__name__} did not join the sidecar namespace"
    assert run_cmd[run_cmd.index("--network") + 1] == "container:sidecar123"
    for cap in ("NET_ADMIN", "NET_RAW"):
        pairs = [
            run_cmd[i : i + 2] for i, tok in enumerate(run_cmd) if tok == "--cap-drop"
        ]
        assert ["--cap-drop", cap] in pairs, f"{cls.__name__} did not drop {cap}"


@pytest.mark.parametrize(("cls", "kwargs"), LAUNCHERS)
def test_non_public_policy_publishes_no_ports_directly(cls, kwargs):
    """Ports are published by the sidecar; docker rejects -p in container: mode."""
    run_cmd, _ = _launch(cls, kwargs, WorkspaceNetworkPolicy(mode="no-network"))
    assert "-p" not in run_cmd, f"{cls.__name__} published ports past the sidecar"


@pytest.mark.parametrize(("cls", "kwargs"), LAUNCHERS)
def test_public_policy_keeps_unrestricted_networking(cls, kwargs):
    """public must behave exactly as before: no sidecar, ports published."""
    run_cmd, start_sidecar = _launch(cls, kwargs, WorkspaceNetworkPolicy(mode="public"))
    assert not start_sidecar.called, f"{cls.__name__} started a needless sidecar"
    assert "--network" not in run_cmd
    assert "-p" in run_cmd
    assert any(a.endswith(":8000") for a in run_cmd), (
        f"{cls.__name__} did not publish the agent-server port"
    )


def test_a_subclass_cannot_bypass_sidecar_startup():
    """The invariant is structural, not a convention a subclass may forget."""
    with pytest.raises(TypeError, match="may not override"):

        class SneakyWorkspace(DockerWorkspace):
            def _start_container(self, image, context):
                """Start a container without any egress enforcement."""

    with pytest.raises(TypeError, match="may not override"):

        class SneakierWorkspace(DockerWorkspace):
            def _build_run_args(self, image, **kwargs) -> list[str]:
                """Build run args that quietly omit --network."""
                return []


def test_subclasses_customise_through_the_launch_hook():
    """The sanctioned extension point still works and cannot reach the network."""

    class CustomWorkspace(DockerWorkspace):
        def _prepare_launch(self, image):
            return ContainerLaunchSpec(extra_flags=["-e", "CUSTOM=1"])

    run_cmd, start_sidecar = _launch(
        CustomWorkspace,
        {"server_image": "srv:latest"},
        WorkspaceNetworkPolicy(mode="no-network"),
    )
    assert "CUSTOM=1" in run_cmd
    assert start_sidecar.called
    assert run_cmd[run_cmd.index("--network") + 1] == "container:sidecar123"
