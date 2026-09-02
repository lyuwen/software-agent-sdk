"""The workspace must never be handed a container-runtime control socket.

A workspace with the docker socket can stop its own egress sidecar or start a
privileged, host-networked container, so the boundary would be advisory only.
"""

from unittest.mock import patch

import pytest

from openhands.workspace import DockerWorkspace, FlexWorkspace
from openhands.workspace.docker.network_policy import WorkspaceNetworkPolicy


SOCKET_MOUNTS = [
    "/var/run/docker.sock:/var/run/docker.sock",
    "/var/run/docker.sock:/var/run/docker.sock:ro",
    "/run/docker.sock:/run/docker.sock",
    "/run/podman/podman.sock:/run/podman/podman.sock",
    "/run/containerd/containerd.sock:/run/containerd/containerd.sock",
    "~/.docker/run/docker.sock:/var/run/docker.sock",
]


@pytest.mark.parametrize("volume", SOCKET_MOUNTS)
def test_socket_bind_volume_is_rejected(volume):
    with patch.object(DockerWorkspace, "_start_container"):
        with pytest.raises(ValueError, match="container-runtime socket"):
            DockerWorkspace(server_image="test:latest", bind_volumes=[volume])


def test_socket_mount_dir_is_rejected():
    with patch.object(DockerWorkspace, "_start_container"):
        with pytest.raises(ValueError, match="container-runtime socket"):
            DockerWorkspace(
                server_image="test:latest", mount_dir="/var/run/docker.sock"
            )


def test_rejected_in_public_mode_too():
    """The guard is mode-independent: a config that validates under 'public'
    must not become an unenforced boundary when OH_NETWORK_MODE changes."""
    with patch.object(DockerWorkspace, "_start_container"):
        with pytest.raises(ValueError, match="container-runtime socket"):
            DockerWorkspace(
                server_image="test:latest",
                network_policy=WorkspaceNetworkPolicy(mode="public"),
                bind_volumes=["/var/run/docker.sock:/var/run/docker.sock"],
            )


def test_subclasses_are_guarded_too():
    with patch.object(FlexWorkspace, "_start_container"):
        with pytest.raises(ValueError, match="container-runtime socket"):
            FlexWorkspace(
                base_image="base:latest",
                bind_volumes=["/var/run/docker.sock:/var/run/docker.sock"],
            )


def test_docker_host_socket_is_rejected_under_any_name(tmp_path, monkeypatch):
    """A socket reached by a non-standard path is still the daemon."""
    sock = tmp_path / "weird-name"
    sock.write_text("")
    monkeypatch.setenv("DOCKER_HOST", f"unix://{sock}")
    with patch.object(DockerWorkspace, "_start_container"):
        with pytest.raises(ValueError, match="container-runtime socket"):
            DockerWorkspace(server_image="test:latest", bind_volumes=[f"{sock}:/s"])


def test_symlink_to_the_socket_is_rejected(tmp_path, monkeypatch):
    """Resolution defeats an alias: /var/run is itself a symlink on Linux."""
    real = tmp_path / "docker.sock"
    real.write_text("")
    link = tmp_path / "innocent-looking-dir"
    link.symlink_to(real)
    with patch.object(DockerWorkspace, "_start_container"):
        with pytest.raises(ValueError, match="container-runtime socket"):
            DockerWorkspace(server_image="test:latest", bind_volumes=[f"{link}:/s"])


def test_ordinary_bind_volumes_are_untouched():
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(
            server_image="test:latest",
            bind_volumes=["/home/user/data:/data", "named-volume:/cache"],
            mount_dir="/home/user/project",
        )
    ws.host_port = 30000
    args = ws._build_run_args("img:latest", container_name="c")
    assert "/home/user/data:/data" in args
    assert "named-volume:/cache" in args


def test_guard_also_runs_at_argv_construction():
    """Defence in depth: the fields are mutable after validation."""
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(server_image="test:latest")
    ws.host_port = 30000
    ws.bind_volumes = ["/var/run/docker.sock:/var/run/docker.sock"]
    with pytest.raises(ValueError, match="container-runtime socket"):
        ws._build_run_args("img:latest", container_name="c")
