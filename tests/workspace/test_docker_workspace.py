"""Test DockerWorkspace import and basic functionality."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture
def mock_docker_workspace():
    """Fixture to create a mocked DockerWorkspace with minimal setup."""
    from openhands.workspace import DockerWorkspace

    with patch("openhands.workspace.docker.workspace.execute_command") as mock_exec:
        # Mock execute_command to return success
        mock_exec.return_value = Mock(returncode=0, stdout="", stderr="")

        def _create_workspace(cleanup_image=False):
            # Create workspace without triggering initialization
            with patch.object(DockerWorkspace, "_start_container"):
                workspace = DockerWorkspace(
                    server_image="test:latest",
                    cleanup_image=cleanup_image,
                    bind_volumes=[],
                )

            # Manually set up state that would normally be set during startup
            workspace._container_id = "container_id_123"
            workspace._image_name = "test:latest"
            workspace._stop_logs = MagicMock()
            workspace._logs_thread = None

            return workspace, mock_exec

        yield _create_workspace


def test_docker_workspace_import():
    """Test that DockerWorkspace can be imported from the new package."""
    from openhands.workspace import DockerWorkspace

    assert DockerWorkspace is not None
    assert hasattr(DockerWorkspace, "__init__")


def test_docker_workspace_inheritance():
    """Test that DockerWorkspace inherits from RemoteWorkspace."""
    from openhands.sdk.workspace import RemoteWorkspace
    from openhands.workspace import DockerWorkspace

    assert issubclass(DockerWorkspace, RemoteWorkspace)


def test_docker_dev_workspace_import():
    """Test that DockerDevWorkspace can be imported from the new package."""
    from openhands.workspace import DockerDevWorkspace

    assert DockerDevWorkspace is not None
    assert hasattr(DockerDevWorkspace, "__init__")


def test_docker_dev_workspace_inheritance():
    """Test that DockerDevWorkspace inherits from DockerWorkspace."""
    from openhands.workspace import DockerDevWorkspace, DockerWorkspace

    assert issubclass(DockerDevWorkspace, DockerWorkspace)


def test_docker_workspace_no_build_import():
    """DockerWorkspace import should not pull in build-time dependencies."""
    code = (
        "import importlib, sys\n"
        "importlib.import_module('openhands.workspace')\n"
        "print('1' if 'openhands.agent_server.docker.build' in sys.modules else '0')\n"
    )

    env = os.environ.copy()
    root = Path(__file__).resolve().parents[2]
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(root / "openhands-sdk"),
            str(root / "openhands-tools"),
            str(root / "openhands-workspace"),
            str(root / "openhands-agent-server"),
            *([pythonpath] if pythonpath else []),
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=root,
    )
    assert result.stdout.strip() == "0"

    from openhands.workspace import DockerWorkspace

    assert "server_image" in DockerWorkspace.model_fields
    assert "base_image" not in DockerWorkspace.model_fields


def test_docker_dev_workspace_has_build_fields():
    """Test that DockerDevWorkspace has both base_image and server_image fields."""
    from openhands.workspace import DockerDevWorkspace

    # DockerDevWorkspace should have both fields for flexibility
    assert "server_image" in DockerDevWorkspace.model_fields
    assert "base_image" in DockerDevWorkspace.model_fields
    assert "target" in DockerDevWorkspace.model_fields


def test_docker_workspace_uses_default_memory_limit_when_env_is_unset():
    """DockerWorkspace should default to 14g when no override is configured."""
    from openhands.workspace import DockerWorkspace

    with patch.dict(os.environ, {}, clear=True):
        with patch.object(DockerWorkspace, "_start_container"):
            workspace = DockerWorkspace(server_image="test:latest", bind_volumes=[])

    assert workspace.memory_limit == "14g"


def test_docker_workspace_reads_memory_limit_from_environment():
    """DockerWorkspace should read memory limit from OH_WORKSPACE_MEMORY_LIMIT."""
    from openhands.workspace import DockerWorkspace

    with patch.dict(os.environ, {"OH_WORKSPACE_MEMORY_LIMIT": "20g"}, clear=True):
        with patch.object(DockerWorkspace, "_start_container"):
            workspace = DockerWorkspace(server_image="test:latest", bind_volumes=[])

    assert workspace.memory_limit == "20g"


def test_docker_workspace_passes_memory_limit_to_docker_run():
    """DockerWorkspace should pass configured memory limit to docker run."""
    from openhands.workspace import DockerWorkspace

    execute_results = [
        Mock(returncode=0, stdout="", stderr=""),
        Mock(returncode=0, stdout="container-123\n", stderr=""),
        Mock(returncode=0, stdout="", stderr=""),
    ]

    with patch(
        "openhands.workspace.docker.workspace.find_available_tcp_port",
        return_value=34567,
    ):
        with patch(
            "openhands.workspace.docker.workspace.check_port_available",
            return_value=True,
        ):
            with patch(
                "openhands.workspace.docker.workspace.execute_command",
                side_effect=execute_results,
            ) as mock_exec:
                with patch.object(DockerWorkspace, "_wait_for_health"):
                    with patch(
                        "openhands.sdk.workspace.remote.RemoteWorkspace.model_post_init"
                    ):
                        DockerWorkspace(
                            server_image="test:latest",
                            memory_limit="17g",
                            detach_logs=False,
                            bind_volumes=[],
                        )

    run_cmd = mock_exec.call_args_list[1].args[0]
    assert "--memory" in run_cmd
    assert "17g" in run_cmd


def _flex_docker_run_argv():
    """Launch a FlexWorkspace with docker faked; return its `docker run` argv.

    Container startup lives in DockerWorkspace while the plugin/glibc calls
    live in FlexWorkspace, so both modules are stubbed and the run command is
    matched by content rather than by call index.
    """
    from openhands.workspace import FlexWorkspace

    def fake_exec(cmd, *args, **kwargs):
        if "create" in cmd:
            return Mock(returncode=0, stdout="plugin-container\n", stderr="")
        if cmd[:2] == ["docker", "run"] and "-d" in cmd:
            return Mock(returncode=0, stdout="workspace-container\n", stderr="")
        # docker version / pull / run ldd (glibc unparseable -> None) /
        # image inspect (base PATH -> fallback)
        return Mock(returncode=0, stdout="", stderr="")

    with (
        patch(
            "openhands.workspace.docker.workspace.find_available_tcp_port",
            return_value=34567,
        ),
        patch(
            "openhands.workspace.docker.workspace.check_port_available",
            return_value=True,
        ),
        patch(
            "openhands.workspace.docker.workspace.execute_command",
            side_effect=fake_exec,
        ) as main_exec,
        patch(
            "openhands.workspace.docker.flex_workspace.execute_command",
            side_effect=fake_exec,
        ) as flex_exec,
        patch.object(FlexWorkspace, "_wait_for_health"),
        patch("openhands.sdk.workspace.remote.RemoteWorkspace.model_post_init"),
    ):
        ws = FlexWorkspace(
            base_image="base:latest",
            detach_logs=False,
            bind_volumes=[],
        )
        ws._container_id = None  # keep __del__ from issuing a real docker stop
        ws._plugin_container_name = None
        runs = [
            c.args[0]
            for mock in (main_exec, flex_exec)
            for c in mock.call_args_list
            if c.args[0][:2] == ["docker", "run"] and "-d" in c.args[0]
        ]
    assert len(runs) == 1, f"expected exactly one docker run, saw {len(runs)}"
    return runs[0]


def test_flex_workspace_path_excludes_agent_server_venv_bin():
    """FlexWorkspace should not expose the agent server venv on PATH."""
    run_cmd = _flex_docker_run_argv()
    path_arg = next(arg for arg in run_cmd if arg.startswith("PATH="))
    assert "/agent-server/.venv/bin" not in path_arg
    assert "/agent-server/bin" in path_arg


def test_flex_workspace_still_uses_agent_server_venv_python_entrypoint():
    """FlexWorkspace should keep launching the server with its venv python."""
    run_cmd = _flex_docker_run_argv()
    entrypoint_index = run_cmd.index("--entrypoint")
    assert run_cmd[entrypoint_index + 1] == "/agent-server/.venv/bin/python"


def test_cleanup_without_image_deletion(mock_docker_workspace):
    """Test that cleanup with cleanup_image=False does not delete the image."""
    workspace, mock_exec = mock_docker_workspace(cleanup_image=False)

    # Call cleanup
    workspace.cleanup()

    # Verify docker rmi was NOT called
    calls = mock_exec.call_args_list
    rmi_calls = [c for c in calls if c[0] and "rmi" in str(c[0])]
    assert len(rmi_calls) == 0


def test_cleanup_with_image_deletion(mock_docker_workspace):
    """Test that cleanup with cleanup_image=True deletes the Docker image."""
    workspace, mock_exec = mock_docker_workspace(cleanup_image=True)

    # Call cleanup
    workspace.cleanup()

    # Verify docker rmi was called with correct arguments
    calls = mock_exec.call_args_list
    rmi_calls = [c for c in calls if c[0] and "rmi" in str(c[0])]
    assert len(rmi_calls) == 1

    # Verify the command includes -f flag and correct image name
    rmi_call_args = rmi_calls[0][0][0]
    assert "docker" in rmi_call_args
    assert "rmi" in rmi_call_args
    assert "-f" in rmi_call_args
    assert "test:latest" in rmi_call_args
