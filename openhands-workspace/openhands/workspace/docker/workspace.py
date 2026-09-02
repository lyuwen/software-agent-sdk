"""Docker-based remote workspace implementation."""

import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from pydantic import Field, PrivateAttr, model_validator

from openhands.sdk.logger import get_logger
from openhands.sdk.utils.command import execute_command
from openhands.sdk.workspace import PlatformType, RemoteWorkspace

from .egress_runtime import EgressRuntime, start_egress_sidecar
from .network_policy import WorkspaceNetworkPolicy, policy_from_env


logger = get_logger(__name__)


def check_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        time.sleep(0.1)
        return False
    finally:
        sock.close()


def find_available_tcp_port(
    min_port: int = 30000, max_port: int = 39999, max_attempts: int = 50
) -> int:
    """Find an available TCP port in a specified range."""
    import random

    rng = random.SystemRandom()
    ports = list(range(min_port, max_port + 1))
    rng.shuffle(ports)

    for port in ports[:max_attempts]:
        if check_port_available(port):
            return port
    return -1


#: Basenames of container-runtime control sockets. Mounting any of these into
#: the workspace hands it the daemon, and with it the ability to dismantle its
#: own egress boundary.
DAEMON_SOCKET_NAMES = frozenset(
    {"docker.sock", "dockerd.sock", "podman.sock", "containerd.sock", "crio.sock"}
)


def _bind_source(volume: str) -> str:
    """Host-side source of a ``docker run -v`` argument."""
    return volume.split(":", 1)[0]


def _daemon_host_socket() -> Path | None:
    """The socket DOCKER_HOST points at, when it points at one."""
    host = os.getenv("DOCKER_HOST", "")
    prefix = "unix://"
    if not host.startswith(prefix):
        return None
    return Path(host[len(prefix) :])


def _is_daemon_socket(source: str) -> bool:
    """Whether a bind source names a container-runtime control socket.

    Both the literal and the fully resolved path are examined: /var/run is a
    symlink to /run on most distributions, and Docker Desktop points
    ~/.docker/run/docker.sock at a per-user path, so matching either form alone
    is evadable. A source that is not a host path at all (a named volume)
    cannot be a socket.
    """
    if not source.startswith(("/", "~", ".")):
        return False  # named volume, not a host path
    path = Path(source).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if resolved.name in DAEMON_SOCKET_NAMES or path.name in DAEMON_SOCKET_NAMES:
        return True
    configured = _daemon_host_socket()
    if configured is None:
        return False
    try:
        return resolved == configured.resolve()
    except OSError:
        return resolved == configured


@dataclass(frozen=True)
class ContainerLaunchSpec:
    """The image-specific pieces of one container launch.

    Deliberately narrow: everything security-relevant -- network attachment,
    capability drops, port publication -- is decided by DockerWorkspace and is
    not expressible here, so a subclass cannot influence it.
    """

    extra_flags: list[str] = field(default_factory=list)
    entrypoint: list[str] | None = None
    command: list[str] | None = None


#: Methods that carry the egress boundary. Subclasses may not override them;
#: see DockerWorkspace.__init_subclass__.
_SEALED_METHODS = ("_start_container", "_build_run_args", "_build_port_args")


class DockerWorkspace(RemoteWorkspace):
    """Remote workspace that sets up and manages a Docker container.

    This workspace creates a Docker container running a pre-built OpenHands agent
    server image, waits for it to become healthy, and then provides remote workspace
    operations through the container's HTTP API.

    Note: This class only works with pre-built images. To build images on-the-fly
    from a base image, use DockerDevWorkspace instead.

    Example:
        with DockerWorkspace(
            server_image="ghcr.io/openhands/agent-server:latest"
        ) as workspace:
            result = workspace.execute_command("ls -la")
    """

    # Override parent fields with defaults
    working_dir: str = Field(
        default="/workspace",
        description="Working directory inside the container.",
    )
    host: str = Field(
        default="",
        description=("Remote host URL (set automatically during container startup)."),
    )

    # Docker-specific configuration
    server_image: str | None = Field(
        default=None,
        description="Pre-built agent server image to use.",
    )
    host_port: int | None = Field(
        default=None,
        description="Port to bind the container to. If None, finds available port.",
    )
    forward_env: list[str] = Field(
        default_factory=lambda: ["DEBUG"],
        description="Environment variables to forward to the container.",
    )
    mount_dir: str | None = Field(
        default=None,
        description="Optional host directory to mount into the container.",
    )
    detach_logs: bool = Field(
        default=True, description="Whether to stream Docker logs in background."
    )
    platform: PlatformType = Field(
        default="linux/amd64", description="Platform for the Docker image."
    )
    extra_ports: bool = Field(
        default=False,
        description="Whether to expose additional ports (VSCode, VNC).",
    )
    enable_gpu: bool = Field(
        default=False,
        description="Whether to enable GPU support with --gpus all.",
    )
    cleanup_image: bool = Field(
        default=False,
        description="Whether to delete the Docker image when cleaning up workspace.",
    )
    bind_volumes: list[str] = Field(
        default_factory=list,
        description="Bind extra directories to container workspace.",
    )
    memory_limit: str = Field(
        default_factory=lambda: os.getenv("OH_WORKSPACE_MEMORY_LIMIT", "14g"),
        description="Docker container memory limit.",
    )

    network_policy: WorkspaceNetworkPolicy = Field(
        default_factory=policy_from_env,
        description=(
            "Egress policy. Defaults from OH_NETWORK_MODE; 'public' preserves "
            "unrestricted networking. Non-public modes start an nftables "
            "sidecar that owns the network namespace."
        ),
    )

    _container_id: str | None = PrivateAttr(default=None)
    _image_name: str | None = PrivateAttr(default=None)
    _logs_thread: threading.Thread | None = PrivateAttr(default=None)
    _stop_logs: threading.Event = PrivateAttr(default_factory=threading.Event)
    _egress: EgressRuntime | None = PrivateAttr(default=None)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Keep egress enforcement on exactly one code path.

        FlexWorkspace once overrode ``_start_container`` and, in doing so,
        skipped sidecar startup entirely: a non-public policy silently became
        unrestricted networking, with no error anywhere. Subclasses customise
        the launch through ``_prepare_launch()`` and
        ``_release_launch_artifacts()`` instead, so the methods that decide
        whether a sidecar is started and joined cannot be bypassed.
        """
        super().__init_subclass__(**kwargs)
        for name in _SEALED_METHODS:
            if name in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} may not override DockerWorkspace.{name}(): "
                    "the egress boundary is enforced there. Override "
                    "_prepare_launch() to customise the container launch."
                )

    @model_validator(mode="after")
    def _validate_server_image(self):
        """Ensure server_image is set when using DockerWorkspace directly."""
        if self.__class__ is DockerWorkspace and self.server_image is None:
            raise ValueError("server_image must be provided")
        return self

    @model_validator(mode="after")
    def _validate_bind_mounts(self):
        """Reject a daemon socket mount as early as construction."""
        self._reject_daemon_socket_mounts()
        return self

    def model_post_init(self, context: Any) -> None:
        """Set up the Docker container and initialize the remote workspace."""
        # Subclasses should call get_image() to get the image to use
        # This allows them to build or prepare the image before container startup
        image = self.get_image()
        self._start_container(image, context)

    def get_image(self) -> str:
        """Get the Docker image to use for the container.

        Subclasses can override this to provide custom image resolution logic
        (e.g., building images on-the-fly).

        Returns:
            The Docker image tag to use.
        """
        if self.server_image is None:
            raise ValueError("server_image must be set")
        return self.server_image

    def _build_run_args(
        self,
        image: str,
        *,
        extra_flags: list[str] | None = None,
        entrypoint: list[str] | None = None,
        command: list[str] | None = None,
        container_name: str,
    ) -> list[str]:
        """Build the full `docker run` argv for this workspace.

        Shared by DockerWorkspace and FlexWorkspace so there is exactly one
        place where run arguments -- and therefore network enforcement -- are
        constructed. Always returns an argv list; never a shell string.
        """
        self._reject_daemon_socket_mounts()

        flags: list[str] = list(extra_flags or [])

        for key in self.forward_env:
            if key in os.environ:
                flags += ["-e", f"{key}={os.environ[key]}"]
        for key, val in os.environ.items():
            if key.startswith("OH_") and key not in self.forward_env:
                flags += ["-e", f"{key}={val}"]

        if self.mount_dir:
            flags += ["-v", f"{self.mount_dir}:/workspace"]
            logger.info(
                f"Mounting host dir {self.mount_dir} to container path /workspace"
            )
        for volume in self.bind_volumes:
            flags += ["-v", volume]

        if self.memory_limit:
            flags += ["--memory", self.memory_limit]

        flags += self._build_port_args()

        if self.enable_gpu:
            flags += ["--gpus", "all"]

        if self._egress is not None and self._egress.sidecar_id:
            flags += [
                "--network",
                f"container:{self._egress.sidecar_id}",
                "--cap-drop",
                "NET_ADMIN",
                "--cap-drop",
                "NET_RAW",
                "--security-opt",
                "no-new-privileges=true",
            ]

        run_cmd = [
            "docker",
            "run",
            "-d",
            "--platform",
            self.platform,
            "--rm",
            "--name",
            container_name,
            *flags,
        ]
        if entrypoint:
            run_cmd += ["--entrypoint", *entrypoint]
        run_cmd.append(image)
        if command:
            run_cmd += command
        return run_cmd

    def _build_port_args(self) -> list[str]:
        """Publish the agent-server port (and optional VSCode/VNC ports).

        Returns an empty list when a sidecar owns the namespace: docker rejects
        -p on a container using container: network mode, and the sidecar itself
        publishes the ports on loopback.
        """
        if self._egress is not None:
            return []
        ports = ["-p", f"{self.host_port}:8000"]
        if self.extra_ports and self.host_port is not None:
            host_port = self.host_port
            ports += [
                "-p",
                f"{host_port + 1}:8001",  # VSCode
                "-p",
                f"{host_port + 2}:8002",  # Desktop VNC
            ]
        return ports

    def _prepare_launch(self, image: str) -> ContainerLaunchSpec:
        """Hook: image-specific preparation, run before the sidecar starts.

        Subclasses may pull images, create helper containers and contribute
        extra docker flags here. Anything acquired that needs undoing after a
        failed startup must be released in ``_release_launch_artifacts()``.
        """
        logger.debug("preparing launch for image %s", image)
        return ContainerLaunchSpec(command=["--host", "0.0.0.0", "--port", "8000"])

    def _release_launch_artifacts(self) -> None:
        """Hook: release whatever ``_prepare_launch()`` acquired.

        Called both on rollback and on cleanup, so it must be idempotent.
        """

    def _allocate_host_ports(self) -> int:
        """Pick and validate the host ports this workspace will occupy."""
        if self.host_port is None:
            self.host_port = find_available_tcp_port()
        else:
            self.host_port = int(self.host_port)
        host_port = self.host_port

        if not check_port_available(host_port):
            raise RuntimeError(f"Port {host_port} is not available")
        if self.extra_ports:
            if not check_port_available(host_port + 1):
                raise RuntimeError(f"Port {host_port + 1} is not available for VSCode")
            if not check_port_available(host_port + 2):
                raise RuntimeError(f"Port {host_port + 2} is not available for VNC")
        return host_port

    @staticmethod
    def _require_docker() -> None:
        """Fail early and clearly when there is no daemon to talk to."""
        if execute_command(["docker", "version"]).returncode != 0:
            raise RuntimeError(
                "Docker is not available. Please install and start "
                "Docker Desktop/daemon."
            )

    def _start_container(self, image: str, context: Any) -> None:
        """Start the workspace container. Sealed -- see ``__init_subclass__``.

        This is the ONE place that decides whether the requested policy needs
        an egress sidecar, starts it, and attaches the workspace container to
        its namespace. Every DockerWorkspace subclass reaches docker through
        here by construction, so no launcher can be added that silently skips
        the boundary. Subclasses contribute only image-specific launch details
        via ``_prepare_launch()``.

        Args:
            image: The Docker image tag to use.
            context: The Pydantic context from model_post_init.
        """
        self._image_name = image
        host_port = self._allocate_host_ports()
        self._require_docker()

        try:
            spec = self._prepare_launch(image)

            if self.network_policy.requires_sidecar:
                self._egress = start_egress_sidecar(
                    self.network_policy,
                    host_port=host_port,
                    extra_ports=self.extra_ports,
                )

            run_cmd = self._build_run_args(
                image,
                container_name=f"agent-server-{uuid.uuid4()}",
                extra_flags=spec.extra_flags,
                entrypoint=spec.entrypoint,
                command=spec.command,
            )
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
            # The container exposes port 8000, mapped to self.host_port
            # Override parent's host initialization
            object.__setattr__(self, "host", f"http://localhost:{host_port}")
            object.__setattr__(self, "api_key", None)

            # Wait for container to be healthy
            self._wait_for_health()
            logger.info(f"Docker workspace is ready at {self.host}")

            # Now initialize the parent RemoteWorkspace with the container URL
            super().model_post_init(context)
        except BaseException:
            self._rollback_start()
            raise

    def _rollback_start(self) -> None:
        """Undo a partial startup. Best-effort, container before network.

        Once the main container exists, a later failure (an unhealthy server,
        a RemoteWorkspace that will not initialise) must not leave it running:
        ``--rm`` never fires because the container never exits, and while it
        lives it holds the sidecar's network namespace, so the sidecar and the
        network cannot be removed either. Each step runs even if an earlier
        one failed -- skipping the rest would leak exactly what this exists to
        reclaim.
        """
        for step in (
            self._stop_main_container,
            self._release_launch_artifacts,
            self._release_egress,
        ):
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - best effort; keep going
                logger.warning("rollback step %s failed: %s", step.__name__, exc)

    def _stop_main_container(self) -> None:
        """Stop and remove the workspace container, if one was created."""
        container_id = self._container_id
        if not container_id:
            return

        # Stop logs streaming
        self._stop_logs.set()
        if self._logs_thread and self._logs_thread.is_alive():
            self._logs_thread.join(timeout=2)
        self._logs_thread = None

        logger.info(f"Stopping container: {container_id}")
        execute_command(["docker", "stop", container_id])
        # `--rm` cleans up on exit, but a container that never exited is still
        # present -- and still holding the sidecar's network namespace.
        execute_command(["docker", "rm", "-f", container_id])
        self._container_id = None

    def _release_egress(self) -> None:
        """Release the egress sidecar and its network."""
        if self._egress is not None:
            self._egress.cleanup()
            self._egress = None

    def _stream_docker_logs(self) -> None:
        """Stream Docker logs to stdout in the background."""
        if not self._container_id:
            return
        try:
            p = subprocess.Popen(
                ["docker", "logs", "-f", self._container_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if p.stdout is None:
                return
            for line in iter(p.stdout.readline, ""):
                if self._stop_logs.is_set():
                    break
                if line:
                    sys.stdout.write(f"[DOCKER] {line}")
                    sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Error streaming docker logs: {e}\n")
        finally:
            try:
                self._stop_logs.set()
            except Exception:
                pass

    def _wait_for_health(self, timeout: float = 120.0) -> None:
        """Wait for the Docker container to become healthy."""
        start = time.time()
        health_url = f"http://127.0.0.1:{self.host_port}/health"

        while time.time() - start < timeout:
            try:
                with urlopen(health_url, timeout=1.0) as resp:
                    if 200 <= getattr(resp, "status", 200) < 300:
                        return
            except Exception:
                pass

            # Check if container is still running
            if self._container_id:
                ps = execute_command(
                    [
                        "docker",
                        "inspect",
                        "-f",
                        "{{.State.Running}}",
                        self._container_id,
                    ]
                )
                if ps.stdout.strip() != "true":
                    logs = execute_command(["docker", "logs", self._container_id])
                    msg = (
                        "Container stopped unexpectedly. Logs:\n"
                        f"{logs.stdout}\n{logs.stderr}"
                    )
                    raise RuntimeError(msg)
            time.sleep(1)
        raise RuntimeError("Container failed to become healthy in time")

    def _reject_daemon_socket_mounts(self) -> None:
        """Refuse to mount a container-runtime control socket into the workspace.

        A workspace holding the docker socket can stop its own egress sidecar,
        or start a privileged host-networked container, which defeats the
        boundary completely -- and it is root-equivalent on the host whatever
        the network mode is. ``bind_volumes`` and ``mount_dir`` are supplied by
        the operator, never by the agent, so this is a misconfiguration guard
        and it applies in EVERY mode: a config that validates under 'public'
        must not turn into an unenforced boundary the moment OH_NETWORK_MODE
        changes.
        """
        sources = [_bind_source(volume) for volume in self.bind_volumes]
        if self.mount_dir:
            sources.append(_bind_source(self.mount_dir))
        for source in sources:
            if _is_daemon_socket(source):
                raise ValueError(
                    f"refusing to mount container-runtime socket {source!r} into "
                    "the workspace: it grants control of the container daemon, "
                    "which bypasses the egress boundary and is root-equivalent "
                    "on the host. If the workspace genuinely needs docker "
                    "access, put a filtered socket proxy in front of it."
                )

    def __enter__(self) -> "DockerWorkspace":
        """Context manager entry - returns the workspace itself."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Context manager exit - cleans up the Docker container."""
        self.cleanup()

    def __del__(self) -> None:
        """Clean up the Docker container when the workspace is destroyed."""
        self.cleanup()

    def cleanup(self) -> None:
        """Stop and remove the Docker container.

        Same order as rollback: the container holds the sidecar's network
        namespace, so it goes first.
        """
        self._stop_main_container()
        self._release_launch_artifacts()
        # Release the egress sidecar after the workspace container is gone
        self._release_egress()

        # Optionally delete the Docker image
        if self.cleanup_image and self._image_name:
            logger.info(f"Deleting Docker image: {self._image_name}")
            result = execute_command(["docker", "rmi", "-f", self._image_name])
            if result.returncode == 0:
                logger.info(f"Successfully deleted image: {self._image_name}")
            else:
                logger.warning(
                    f"Failed to delete image {self._image_name}: {result.stderr}"
                )
            self._image_name = None

    def pause(self) -> None:
        """Pause the Docker container to conserve resources.

        Uses `docker pause` to freeze all processes in the container without
        stopping it. The container can be resumed later with `resume()`.

        Raises:
            RuntimeError: If the container is not running or pause fails.
        """
        if not self._container_id:
            raise RuntimeError("Cannot pause: container is not running")

        logger.info(f"Pausing container: {self._container_id}")
        result = execute_command(["docker", "pause", self._container_id])
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pause container: {result.stderr}")
        logger.info(f"Container paused: {self._container_id}")

    def resume(self) -> None:
        """Resume a paused Docker container.

        Uses `docker unpause` to resume all processes in the container.

        Raises:
            RuntimeError: If the container is not running or resume fails.
        """
        if not self._container_id:
            raise RuntimeError("Cannot resume: container is not running")

        logger.info(f"Resuming container: {self._container_id}")
        result = execute_command(["docker", "unpause", self._container_id])
        if result.returncode != 0:
            raise RuntimeError(f"Failed to resume container: {result.stderr}")

        # Wait for health after resuming (use same timeout as initial startup)
        self._wait_for_health(timeout=120.0)
        logger.info(f"Container resumed: {self._container_id}")
