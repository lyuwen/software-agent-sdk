"""Flex workspace using the agent-plugin volume mount pattern."""

import re
import uuid

from pydantic import Field, PrivateAttr

from openhands.sdk.logger import get_logger
from openhands.sdk.utils.command import execute_command

from .workspace import ContainerLaunchSpec, DockerWorkspace


logger = get_logger(__name__)

# glibc version → variant directory name.  Sorted newest-first so the first
# match wins when comparing with >=.
_GLIBC_VARIANTS = [
    (2.36, "bookworm"),
    (2.31, "bullseye"),
    (2.28, "buster"),
]

_DEFAULT_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _detect_glibc_version(image: str, platform: str = "linux/amd64") -> float | None:
    """Detect the glibc version inside a Docker image.

    Pulls the image first (if not local), then runs ``ldd --version`` in a
    throwaway container.  Returns the glibc major.minor as a float (e.g. 2.31),
    or None on failure.
    """
    # Pull separately so the detection command itself doesn't time out
    pull_proc = execute_command(
        ["docker", "pull", "--platform", platform, image],
        timeout=600,
    )
    if pull_proc.returncode != 0:
        logger.warning(
            "docker pull failed for %s (exit %d): %s",
            image,
            pull_proc.returncode,
            pull_proc.stderr,
        )

    proc = execute_command(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            platform,
            "--entrypoint",
            "",
            image,
            "ldd",
            "--version",
        ],
        timeout=30,
    )
    if proc.returncode != 0:
        logger.warning(
            "glibc detection failed for %s (exit %d): %s",
            image,
            proc.returncode,
            proc.stderr,
        )
        return None

    output = proc.stdout + proc.stderr
    m = re.search(r"(?:GLIBC|GNU libc)[^\d]*(\d+\.\d+)", output, re.IGNORECASE)
    if not m:
        logger.warning("Could not parse glibc version from: %s", output[:200])
        return None

    version = float(m.group(1))
    logger.info("Detected glibc %.2f in image %s", version, image)
    return version


def _select_variant(glibc_version: float | None) -> str | None:
    """Select the best matching variant directory for a glibc version.

    Returns the variant name (e.g. "buster") or None to use the default
    bin/lib (bookworm).
    """
    if glibc_version is None:
        return None

    for threshold, name in _GLIBC_VARIANTS:
        if glibc_version >= threshold:
            if name == "bookworm":
                return None
            return name

    return _GLIBC_VARIANTS[-1][1]


def _get_base_image_path(image: str) -> str:
    """Read the base image's original PATH via docker image inspect.

    Falls back to the standard system PATH if inspection fails.
    """
    proc = execute_command(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{range .Config.Env}}{{println .}}{{end}}",
            image,
        ],
        timeout=15,
    )
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if line.startswith("PATH="):
                path = line[5:]
                logger.debug("Base image PATH: %s", path)
                return path

    return _DEFAULT_SYSTEM_PATH


class FlexWorkspace(DockerWorkspace):
    """Workspace that launches a base image with agent-plugin volume mount.

    Instead of requiring a pre-built agent server image, this workspace:
    1. Creates a data container from the agent-plugin image
    2. Launches the base image with ``--volumes-from`` to mount ``/agent-server/``
    3. Detects the base image's glibc version and selects the matching
       variant (buster/bullseye/bookworm) for bin/ and lib/
    4. Preserves the base image's PATH entries (prepends agent-server paths)

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

    def _prepare_launch(self, image: str) -> ContainerLaunchSpec:
        """Stage the agent-plugin volume and the glibc-matched environment.

        Only image-specific work happens here. Container startup, egress
        sidecar attachment and rollback stay in DockerWorkspace, so this
        launcher cannot drift away from the network boundary again.
        """
        # Detect glibc version and select matching variant
        glibc_ver = _detect_glibc_version(image, self.platform)
        variant = _select_variant(glibc_ver)

        if variant:
            bin_path = f"/agent-server/variants/{variant}/bin"
            lib_path = f"/agent-server/variants/{variant}/lib"
            # Include default paths as fallback for old plugin images that
            # lack variant directories — non-existent PATH entries are harmless.
            fallback_bin = "/agent-server/bin"
            fallback_lib = "/agent-server/lib"
            logger.info(
                "Using glibc variant '%s' (glibc %.2f) for bin/lib paths",
                variant,
                glibc_ver,
            )
        else:
            bin_path = "/agent-server/bin"
            lib_path = "/agent-server/lib"
            fallback_bin = None
            fallback_lib = None

        # Preserve the base image's PATH — prepend agent-server paths
        base_path = _get_base_image_path(image)
        path_parts = [bin_path]
        if fallback_bin:
            path_parts.append(fallback_bin)
        path_parts.append(base_path)
        combined_path = ":".join(path_parts)

        lib_parts = [lib_path]
        if fallback_lib:
            lib_parts.append(fallback_lib)
        combined_lib = ":".join(lib_parts)

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

        return ContainerLaunchSpec(
            extra_flags=[
                "--volumes-from",
                self._plugin_container_name,
                "-e",
                f"PATH={combined_path}",
                "-e",
                f"LD_LIBRARY_PATH={combined_lib}",
                "-e",
                "UV_PYTHON_INSTALL_DIR=/agent-server/uv-managed-python",
                "-w",
                "/",
            ],
            entrypoint=["/agent-server/.venv/bin/python"],
            command=[
                "-m",
                "openhands.agent_server",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
        )

    def _release_launch_artifacts(self) -> None:
        """Remove the plugin data container. Idempotent."""
        if self._plugin_container_name:
            logger.info(
                f"Removing plugin data container: {self._plugin_container_name}"
            )
            execute_command(["docker", "rm", "-f", self._plugin_container_name])
            self._plugin_container_name = None

    def cleanup(self) -> None:
        """Stop the container and remove the plugin data container.

        The base cleanup already calls ``_release_launch_artifacts()``; this
        override remains only to document the behaviour and stays correct
        because that hook is idempotent.
        """
        super().cleanup()
        self._release_launch_artifacts()
