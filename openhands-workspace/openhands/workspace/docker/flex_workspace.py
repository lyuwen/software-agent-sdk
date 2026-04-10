"""Flex workspace using the agent-plugin volume mount pattern."""

import os
import threading
import uuid
from typing import Any

from pydantic import Field, PrivateAttr

from openhands.sdk.logger import get_logger
from openhands.sdk.utils.command import execute_command
from openhands.sdk.workspace import RemoteWorkspace

from .workspace import DockerWorkspace, check_port_available, find_available_tcp_port


logger = get_logger(__name__)


class FlexWorkspace(DockerWorkspace):
    """Workspace that launches a base image with agent-plugin volume mount.

    Instead of requiring a pre-built agent server image, this workspace:
    1. Creates a data container from the agent-plugin image
    2. Launches the base image with ``--volumes-from`` to mount ``/agent-server/``
    3. Sets PATH, LD_LIBRARY_PATH, and UV_PYTHON_INSTALL_DIR for the agent server

    This eliminates per-instance image builds — the agent-plugin is built once
    and shared across all base images via Docker volumes.

    Example:
        with FlexWorkspace(
            base_image="docker.io/swebench/sweb.eval.x86_64.django_1776_django-11333:latest",
            agent_plugin_image="openhands/agent-plugin",
        ) as workspace:
            result = workspace.execute_command("ls -la")
    """

    base_image: str = Field(
        description="Base Docker image (e.g., SWE-bench environment image).",
    )
    agent_plugin_image: str = Field(
        default="openhands/agent-plugin",
        description="Agent plugin image providing /agent-server/ volume.",
    )

    _plugin_container_name: str | None = PrivateAttr(default=None)

    def get_image(self) -> str:
        """Return the base image directly — no build step needed."""
        return self.base_image

    def _start_container(self, image: str, context: Any) -> None:
        """Start container with agent-plugin volume mounted.

        Same lifecycle as parent but adds:
        - A plugin data container (``--volumes-from``)
        - Agent-server environment variables
        - Entrypoint override to launch the agent server
        """
        self._image_name = image

        # Determine port
        if self.host_port is None:
            self.host_port = find_available_tcp_port()
        else:
            self.host_port = int(self.host_port)

        if not check_port_available(self.host_port):
            raise RuntimeError(f"Port {self.host_port} is not available")

        if self.extra_ports:
            if not check_port_available(self.host_port + 1):
                raise RuntimeError(
                    f"Port {self.host_port + 1} is not available for VSCode"
                )
            if not check_port_available(self.host_port + 2):
                raise RuntimeError(
                    f"Port {self.host_port + 2} is not available for VNC"
                )

        # Ensure docker is available
        docker_ver = execute_command(["docker", "version"]).returncode
        if docker_ver != 0:
            raise RuntimeError(
                "Docker is not available. Please install and start "
                "Docker Desktop/daemon."
            )

        # Create plugin data container
        self._plugin_container_name = f"agent-plugin-{uuid.uuid4()}"
        create_proc = execute_command(
            [
                "docker",
                "create",
                "--name",
                self._plugin_container_name,
                self.agent_plugin_image,
                "true",
            ]
        )
        if create_proc.returncode != 0:
            raise RuntimeError(
                f"Failed to create plugin data container: {create_proc.stderr}"
            )
        logger.info(f"Created plugin data container: {self._plugin_container_name}")

        # Prepare Docker run flags
        flags: list[str] = []

        # Agent-plugin volume
        flags += ["--volumes-from", self._plugin_container_name]

        # Agent-server environment variables
        flags += [
            "-e",
            "PATH=/agent-server/bin:/agent-server/.venv/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "-e",
            "LD_LIBRARY_PATH=/agent-server/lib",
            "-e",
            "UV_PYTHON_INSTALL_DIR=/agent-server/uv-managed-python",
        ]

        # Forward environment variables
        for key in self.forward_env:
            if key in os.environ:
                flags += ["-e", f"{key}={os.environ[key]}"]

        if self.mount_dir:
            mount_path = "/workspace"
            flags += ["-v", f"{self.mount_dir}:{mount_path}"]
            logger.info(
                f"Mounting host dir {self.mount_dir} to container path {mount_path}"
            )

        if self.bind_volumes:
            for volume in self.bind_volumes:
                flags += ["-v", volume]

        ports = ["-p", f"{self.host_port}:8000"]
        if self.extra_ports:
            ports += [
                "-p",
                f"{self.host_port + 1}:8001",  # VSCode
                "-p",
                f"{self.host_port + 2}:8002",  # Desktop VNC
            ]
        flags += ports

        # Add GPU support if enabled
        if self.enable_gpu:
            flags += ["--gpus", "all"]

        # Run container with entrypoint override for agent server
        run_cmd = [
            "docker",
            "run",
            "-d",
            "--platform",
            self.platform,
            "--rm",
            "--name",
            f"agent-server-{uuid.uuid4()}",
            *flags,
            "--entrypoint",
            "/agent-server/.venv/bin/python",
            image,
            "-m",
            "openhands.agent_server",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        proc = execute_command(run_cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to run docker container: {proc.stderr}")

        self._container_id = proc.stdout.strip()
        logger.info(f"Started container: {self._container_id}")

        # Optionally stream logs in background
        if self.detach_logs:
            self._logs_thread = threading.Thread(
                target=self._stream_docker_logs, daemon=True
            )
            self._logs_thread.start()

        # Set host for RemoteWorkspace to use
        object.__setattr__(self, "host", f"http://localhost:{self.host_port}")
        object.__setattr__(self, "api_key", None)

        # Wait for container to be healthy
        self._wait_for_health()
        logger.info(f"Docker workspace is ready at {self.host}")

        # Initialize the RemoteWorkspace (grandparent) with the container URL
        RemoteWorkspace.model_post_init(self, context)

    def cleanup(self) -> None:
        """Stop container and remove the plugin data container."""
        super().cleanup()
        if self._plugin_container_name:
            logger.info(
                f"Removing plugin data container: {self._plugin_container_name}"
            )
            execute_command(["docker", "rm", "-f", self._plugin_container_name])
            self._plugin_container_name = None
