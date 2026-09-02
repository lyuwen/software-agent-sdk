"""Lifecycle for the per-workspace nftables egress sidecar."""

import json
import os
import secrets
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from ipaddress import IPv4Network, IPv6Network, ip_network
from pathlib import Path

from openhands.sdk.logger import get_logger
from openhands.sdk.utils.command import execute_command

from .network_policy import WorkspaceNetworkPolicy
from .nftables_renderer import policy_digest, render_rules


logger = get_logger(__name__)

STATE_ROOT = Path(os.getenv("OH_EGRESS_STATE_ROOT", "/var/tmp/openhands-egress"))
READY_PATH = "/tmp/workspace-egress.ready"
RULES_MOUNT = "/etc/workspace-egress/rules.nft"
DEFAULT_SIDECAR_IMAGE = os.getenv("OH_EGRESS_IMAGE", "openhands-egress-static:dev")
READY_TIMEOUT_SECONDS = 30.0
STOP_TIMEOUT_SECONDS = 10
LEASE_STALE_SECONDS = 120.0

LABEL_MANAGED = "workspace.managed=true"


def controller_id() -> str:
    """Stable-per-process controller identity: boot id, pid, and a nonce.

    A differing boot id means the host rebooted, so the controller is dead.
    The nonce guards against pid reuse within one boot.
    """
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = "noboot"
    return f"{boot[:8]}-{os.getpid()}-{secrets.token_hex(4)}"


def subnets_overlap_allowlist(
    subnets: list[IPv4Network | IPv6Network], policy: WorkspaceNetworkPolicy
) -> bool:
    """True if any bridge subnet intersects an allowed destination.

    An overlapping bridge would place the gateway and host services inside the
    allowlist and shadow the internal service range, so startup must abort.
    """
    for endpoint in policy.resolved_endpoints():
        for subnet in subnets:
            if subnet.version != endpoint.destination.version:
                continue
            if subnet.overlaps(endpoint.destination):
                return True
    return False


@dataclass
class EgressRuntime:
    """Resources acquired for one workspace's egress boundary."""

    workspace_id: str
    controller_id: str
    network_id: str | None = None
    sidecar_id: str | None = None
    rules_path: Path | None = None
    policy_digest: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cleaned: bool = field(default=False, repr=False)

    @property
    def manifest_path(self) -> Path:
        return STATE_ROOT / self.workspace_id / "manifest.json"

    def write_manifest(self, status: str = "active") -> None:
        """Atomically record acquired resources so a later pass can reconcile."""
        directory = self.manifest_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "workspace_id": self.workspace_id,
            "controller_id": self.controller_id,
            "network_id": self.network_id,
            "sidecar_id": self.sidecar_id,
            "rules_path": str(self.rules_path) if self.rules_path else None,
            "policy_digest": self.policy_digest,
            "status": status,
            "lease_expires_at": time.time() + LEASE_STALE_SECONDS,
        }
        fd, tmp = tempfile.mkstemp(dir=str(directory), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp, self.manifest_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def is_alive(self) -> bool:
        """Whether the sidecar container is still running."""
        if not self.sidecar_id:
            return False
        proc = execute_command(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.sidecar_id],
            print_output=False,
        )
        return proc.stdout.strip() == "true"

    def cleanup(self) -> None:
        """Release every acquired resource. Idempotent and thread-safe.

        Best-effort across resources: a failure removing one resource must not
        prevent the removal of the others.
        """
        with self._lock:
            if self._cleaned:
                return
            self._cleaned = True

        errors: list[str] = []

        def attempt(label: str, cmd: list[str]) -> None:
            try:
                proc = execute_command(cmd, print_output=False)
                if proc.returncode != 0:
                    errors.append(f"{label}: {proc.stderr.strip()}")
            except Exception as exc:  # noqa: BLE001 - aggregate and continue
                errors.append(f"{label}: {exc}")

        if self.sidecar_id:
            attempt(
                f"stop sidecar {self.sidecar_id}",
                ["docker", "stop", "-t", str(STOP_TIMEOUT_SECONDS), self.sidecar_id],
            )
            attempt(
                f"remove sidecar {self.sidecar_id}",
                ["docker", "rm", "-f", self.sidecar_id],
            )
        if self.network_id:
            attempt(
                f"remove network {self.network_id}",
                ["docker", "network", "rm", self.network_id],
            )
        if self.rules_path:
            try:
                self.rules_path.unlink(missing_ok=True)
                parent = self.rules_path.parent
                if parent != STATE_ROOT and parent.is_dir():
                    for leftover in parent.iterdir():
                        leftover.unlink(missing_ok=True)
                    parent.rmdir()
            except OSError as exc:
                errors.append(f"remove rules {self.rules_path}: {exc}")

        if errors:
            logger.warning(
                "egress cleanup for %s completed with errors: %s",
                self.workspace_id,
                "; ".join(errors),
            )
        else:
            self.manifest_path.unlink(missing_ok=True)


def _network_subnets(network_id: str) -> list[IPv4Network | IPv6Network]:
    """Read the IPAM subnets docker assigned to a network."""
    proc = execute_command(
        [
            "docker",
            "network",
            "inspect",
            network_id,
            "--format",
            "{{range .IPAM.Config}}{{.Subnet}} {{end}}",
        ],
        print_output=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to inspect network {network_id}: {proc.stderr}")
    return [ip_network(token) for token in proc.stdout.split() if token]


def start_egress_sidecar(
    policy: WorkspaceNetworkPolicy,
    *,
    host_port: int,
    extra_ports: bool = False,
    image: str = DEFAULT_SIDECAR_IMAGE,
) -> EgressRuntime:
    """Create the network and sidecar that will own the workspace namespace.

    Fails closed: any error tears down whatever was already acquired and
    re-raises. The caller must never proceed to start a workspace after this
    raises.
    """
    workspace_id = f"ws-{uuid.uuid4().hex[:12]}"
    runtime = EgressRuntime(workspace_id=workspace_id, controller_id=controller_id())

    try:
        # 1. Rules file, private to this controller.
        rules_text = render_rules(policy)
        runtime.policy_digest = policy_digest(rules_text)
        directory = STATE_ROOT / workspace_id
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        rules_path = directory / "rules.nft"
        rules_path.write_text(rules_text, encoding="utf-8")
        os.chmod(rules_path, 0o600)
        runtime.rules_path = rules_path
        runtime.write_manifest()

        # 2. Dedicated bridge.
        net_proc = execute_command(
            [
                "docker",
                "network",
                "create",
                "--label",
                LABEL_MANAGED,
                "--label",
                f"workspace.id={workspace_id}",
                "--label",
                f"workspace.controller={runtime.controller_id}",
                f"{workspace_id}-network",
            ],
            print_output=False,
        )
        if net_proc.returncode != 0:
            raise RuntimeError(f"failed to create network: {net_proc.stderr}")
        network_id = net_proc.stdout.strip()
        runtime.network_id = network_id
        runtime.write_manifest()

        # 3. A bridge inside the allowlist would put the gateway and host
        #    services inside the policy. Abort rather than weaken the boundary.
        subnets = _network_subnets(network_id)
        if subnets_overlap_allowlist(subnets, policy):
            raise RuntimeError(
                f"docker allocated bridge subnets {subnets} which overlap the "
                "resolved allowlist; refusing to start. Configure the daemon's "
                "default-address-pools to a range outside the allowlist."
            )

        # 4. Sidecar owns the namespace and publishes ports on loopback only.
        publish = ["-p", f"127.0.0.1:{host_port}:8000"]
        if extra_ports:
            publish += [
                "-p",
                f"127.0.0.1:{host_port + 1}:8001",
                "-p",
                f"127.0.0.1:{host_port + 2}:8002",
            ]
        run_proc = execute_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                f"{workspace_id}-egress",
                "--init",
                "--restart",
                "no",
                "--stop-timeout",
                str(STOP_TIMEOUT_SECONDS),
                "--network",
                f"{workspace_id}-network",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "NET_ADMIN",
                "--security-opt",
                "no-new-privileges=true",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=1m",
                *publish,
                "--label",
                LABEL_MANAGED,
                "--label",
                f"workspace.id={workspace_id}",
                "--label",
                f"workspace.controller={runtime.controller_id}",
                "--label",
                "workspace.role=egress",
                "--mount",
                f"type=bind,src={rules_path},dst={RULES_MOUNT},readonly",
                image,
            ],
            print_output=False,
        )
        if run_proc.returncode != 0:
            raise RuntimeError(f"failed to start egress sidecar: {run_proc.stderr}")
        sidecar_id = run_proc.stdout.strip()
        runtime.sidecar_id = sidecar_id
        runtime.write_manifest()

        # 5. Readiness is published only after the sidecar verifies its own
        #    ruleset, so this is also policy verification.
        deadline = time.time() + READY_TIMEOUT_SECONDS
        while time.time() < deadline:
            probe = execute_command(
                ["docker", "exec", sidecar_id, "test", "-f", READY_PATH],
                print_output=False,
            )
            if probe.returncode == 0:
                logger.info(
                    "egress sidecar %s ready for %s (mode=%s digest=%s)",
                    sidecar_id[:12],
                    workspace_id,
                    policy.mode,
                    runtime.policy_digest[:12] if runtime.policy_digest else "-",
                )
                return runtime
            if not runtime.is_alive():
                logs = execute_command(
                    ["docker", "logs", sidecar_id], print_output=False
                )
                raise RuntimeError(
                    f"egress sidecar exited before readiness:\n"
                    f"{logs.stdout}\n{logs.stderr}"
                )
            time.sleep(0.2)

        logs = execute_command(["docker", "logs", sidecar_id], print_output=False)
        raise RuntimeError(
            f"egress sidecar not ready within {READY_TIMEOUT_SECONDS}s:\n"
            f"{logs.stdout}\n{logs.stderr}"
        )
    except BaseException:
        runtime.cleanup()
        raise
