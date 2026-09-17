"""Lifecycle for the per-workspace nftables egress sidecar."""

import json
import os
import secrets
import stat
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

POLICY_DIGEST_ENV = "OH_EGRESS_POLICY_DIGEST"
"""Env var carrying the expected policy digest into the sidecar.

The entrypoint recomputes the sha256 of the mounted rules file and refuses to
apply it -- failing closed, without publishing readiness -- unless it matches.
"""


def _within_state_root(path: Path) -> bool:
    """Whether ``path`` resolves to a location strictly beneath STATE_ROOT.

    Both sides are fully resolved so that ``..`` segments and symlinks cannot
    be used to escape; containment is then decided structurally with
    ``is_relative_to`` rather than by string prefix, which would accept a
    sibling such as ``/var/tmp/openhands-egress-evil``. STATE_ROOT itself is
    not "beneath" itself, so the root can never be wiped.
    """
    try:
        root = STATE_ROOT.resolve()
        target = path.resolve()
    except OSError:
        return False
    return target != root and target.is_relative_to(root)


def ensure_state_root() -> Path:
    """Create STATE_ROOT private to this user and return it."""
    STATE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    return STATE_ROOT


def _state_root_is_trustworthy() -> bool:
    """Whether manifests found in STATE_ROOT may be acted upon at all.

    Reconciliation deletes resources named by files it reads, so the directory
    those files come from must not be writable by other users. The default
    root lives under world-writable /var/tmp, where any local user could
    pre-create it and plant manifests. Refuse rather than repair: silently
    chmod-ing a directory owned by somebody else would be its own bug.
    """
    try:
        info = STATE_ROOT.stat()
    except OSError as exc:
        logger.warning("cannot stat egress state root %s: %s", STATE_ROOT, exc)
        return False
    if info.st_uid != os.getuid():
        logger.warning(
            "refusing to reconcile: egress state root %s is owned by uid %d, "
            "not the current user (uid %d)",
            STATE_ROOT,
            info.st_uid,
            os.getuid(),
        )
        return False
    if info.st_mode & stat.S_IWOTH:
        logger.warning(
            "refusing to reconcile: egress state root %s is world-writable "
            "(mode %o); another local user could plant manifests there",
            STATE_ROOT,
            stat.S_IMODE(info.st_mode),
        )
        return False
    return True


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
    #: The workspace container itself. Recorded so a reconciliation pass can
    #: reclaim it: it holds the sidecar's network namespace, so without it a
    #: leaked workspace container blocks removal of the sidecar and network.
    workspace_container_id: str | None = None
    rules_path: Path | None = None
    policy_digest: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cleaned: bool = field(default=False, repr=False)
    _cleanup_errors: list[str] = field(default_factory=list, repr=False)

    @property
    def manifest_path(self) -> Path:
        return STATE_ROOT / self.workspace_id / "manifest.json"

    def write_manifest(
        self,
        status: str = "active",
        *,
        lease_seconds: float = LEASE_STALE_SECONDS,
        errors: list[str] | None = None,
    ) -> None:
        """Atomically record acquired resources so a later pass can reconcile.

        ``lease_seconds`` of 0 writes an already-expired lease, which is how a
        failed cleanup asks the next pass to retry immediately.
        """
        directory = self.manifest_path.parent
        ensure_state_root()
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "workspace_id": self.workspace_id,
            "controller_id": self.controller_id,
            "network_id": self.network_id,
            "sidecar_id": self.sidecar_id,
            "workspace_container_id": self.workspace_container_id,
            "rules_path": str(self.rules_path) if self.rules_path else None,
            "policy_digest": self.policy_digest,
            "status": status,
            "cleanup_errors": errors or [],
            "lease_expires_at": time.time() + lease_seconds,
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

    def cleanup(self) -> list[str]:
        """Release every acquired resource. Idempotent and thread-safe.

        Best-effort across resources: a failure removing one resource must not
        prevent the removal of the others.

        Returns:
            The aggregated failure messages -- empty when every resource was
            genuinely removed. A caller holding the only record of these
            resources (reconciliation) must check this before discarding it.
            Repeat calls return the outcome of the pass that ran.
        """
        with self._lock:
            if self._cleaned:
                return list(self._cleanup_errors)
            self._cleaned = True

        errors: list[str] = []

        def attempt(label: str, cmd: list[str]) -> None:
            try:
                proc = execute_command(cmd, print_output=False)
                if proc.returncode != 0:
                    detail = proc.stderr.strip()
                    # A resource that is already gone is the desired end state,
                    # not a failure -- otherwise a retry after a partial pass
                    # could never succeed.
                    if _already_absent(detail):
                        return
                    errors.append(f"{label}: {detail}")
            except Exception as exc:  # noqa: BLE001 - aggregate and continue
                errors.append(f"{label}: {exc}")

        # The workspace container holds the sidecar's network namespace, so it
        # must go before the sidecar and the network.
        if self.workspace_container_id:
            attempt(
                f"stop workspace container {self.workspace_container_id}",
                [
                    "docker",
                    "stop",
                    "-t",
                    str(STOP_TIMEOUT_SECONDS),
                    self.workspace_container_id,
                ],
            )
            attempt(
                f"remove workspace container {self.workspace_container_id}",
                ["docker", "rm", "-f", self.workspace_container_id],
            )
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
            # Bound the blast radius here, not only in the caller: cleanup()
            # removes every file in the rules directory, so a rules_path that
            # came from an untrusted manifest must never be honoured.
            if not _within_state_root(self.rules_path):
                errors.append(
                    f"refusing to remove rules {self.rules_path}: "
                    f"outside state root {STATE_ROOT}"
                )
            else:
                try:
                    self.rules_path.unlink(missing_ok=True)
                except OSError as exc:
                    errors.append(f"remove rules {self.rules_path}: {exc}")

        self._cleanup_errors = errors
        if errors:
            logger.warning(
                "egress cleanup for %s completed with errors: %s",
                self.workspace_id,
                "; ".join(errors),
            )
            # Deliberately keep the manifest and its directory: something
            # survived, and this is the only record of what.
            return list(errors)

        # Everything is gone, so the recovery record can go too.
        if _within_state_root(self.manifest_path):
            self.manifest_path.unlink(missing_ok=True)
            directory = self.manifest_path.parent
            if _within_state_root(directory) and directory.is_dir():
                for leftover in directory.iterdir():
                    leftover.unlink(missing_ok=True)
                directory.rmdir()
        return []


_ABSENT_MARKERS = ("no such container", "no such network", "not found")


def _already_absent(stderr: str) -> bool:
    """Whether docker failed only because the resource is already gone."""
    lowered = stderr.lower()
    return any(marker in lowered for marker in _ABSENT_MARKERS)


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
        directory = ensure_state_root() / workspace_id
        directory.mkdir(exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        rules_path = directory / "rules.nft"
        rules_path.write_text(rules_text, encoding="utf-8")
        # 0o644, not 0o600: the sidecar runs with --cap-drop ALL --cap-add NET_ADMIN,
        # which removes CAP_DAC_OVERRIDE — the capability that normally lets root
        # bypass permission checks.  Without it, uid-0 inside the container cannot
        # read a file it does not own that is mode 0o600.  0o644 (world-readable)
        # is safe because the parent directory is 0o700, so host users without
        # access to that directory cannot reach the file at all.
        os.chmod(rules_path, 0o644)
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
                # The sidecar recomputes this over the rules file it was given
                # and refuses to apply anything else, so it verifies the policy
                # that was REQUESTED rather than merely that some ruleset loaded.
                "-e",
                f"{POLICY_DIGEST_ENV}={runtime.policy_digest}",
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


_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _is_boot_prefix(text: str) -> bool:
    """Whether a controller id's first field is a real boot-id prefix.

    ``controller_id()`` falls back to ``"noboot"`` where /proc is unavailable,
    and ids minted elsewhere may carry anything at all. Only an 8-character
    hex prefix -- what a boot uuid actually looks like -- can be compared.
    """
    return len(text) == 8 and all(char in _HEX_DIGITS for char in text)


def controller_is_alive(cid: str) -> bool:
    """Whether the controller that created a resource still exists.

    Returns True whenever liveness cannot be determined. Reclaiming a live
    controller's workspace destroys a running evaluation; leaving a leaked
    container costs disk until the next pass. Bias to leaving it alone.
    """
    parts = cid.split("-")
    if len(parts) != 3:
        return True  # unparseable: do not touch
    boot, pid_text, _nonce = parts
    try:
        current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()[:8]
    except OSError:
        return True  # cannot compare: do not touch
    # A differing boot id is decisive only when it is genuinely a boot-id
    # prefix. Anything else says nothing about which boot minted the id, so
    # fall through to the pid check rather than declaring the controller dead.
    if _is_boot_prefix(boot) and boot != current_boot:
        return False  # host rebooted: the process cannot exist
    try:
        pid = int(pid_text)
    except ValueError:
        return True
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def reconcile_orphans(now: float | None = None) -> list[str]:
    """Remove managed resources whose controller is gone. Returns reclaimed IDs.

    Runs before a worker accepts work. Only acts on a complete, readable
    manifest whose controller is provably dead AND whose lease has expired.
    Continues scanning after individual failures.
    """
    current = time.time() if now is None else now
    reclaimed: list[str] = []
    if not STATE_ROOT.is_dir():
        return reclaimed
    if not _state_root_is_trustworthy():
        return reclaimed

    for directory in sorted(STATE_ROOT.iterdir()):
        manifest_file = directory / "manifest.json"
        if not manifest_file.is_file():
            continue
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            workspace_id = str(manifest["workspace_id"])
            cid = str(manifest["controller_id"])
            lease = float(manifest["lease_expires_at"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("skipping unreadable manifest %s: %s", manifest_file, exc)
            continue

        if lease > current or controller_is_alive(cid):
            continue

        logger.info(
            "reconciling orphaned workspace %s (controller %s)", workspace_id, cid
        )
        raw_rules = manifest.get("rules_path")
        rules_path = Path(str(raw_rules)) if raw_rules else None
        # A manifest is untrusted input: it names paths that cleanup() will
        # delete. Honour only paths that stay beneath the state root, and skip
        # the whole entry otherwise -- do not reclaim it, do not delete for it.
        if rules_path is not None and not _within_state_root(rules_path):
            logger.warning(
                "skipping manifest %s: rules_path %s escapes state root %s",
                manifest_file,
                rules_path,
                STATE_ROOT,
            )
            continue
        # workspace_id is likewise attacker-controlled and feeds manifest_path.
        if not _within_state_root(STATE_ROOT / workspace_id / "manifest.json"):
            logger.warning(
                "skipping manifest %s: workspace_id %r escapes state root %s",
                manifest_file,
                workspace_id,
                STATE_ROOT,
            )
            continue

        runtime = EgressRuntime(
            workspace_id=workspace_id,
            controller_id=cid,
            network_id=manifest.get("network_id"),
            sidecar_id=manifest.get("sidecar_id"),
            workspace_container_id=manifest.get("workspace_container_id"),
            rules_path=rules_path,
        )
        try:
            # container before network, best-effort; errors are reported, not
            # raised, so the outcome has to be inspected rather than assumed.
            errors = runtime.cleanup()
            if errors:
                # Something survived and can still block network removal.
                # Discarding the manifest here would destroy the only record
                # needed to retry, so keep it -- with the failure detail and an
                # already-expired lease, so the next pass picks it straight up.
                logger.warning(
                    "reconciliation for %s did not remove everything: %s",
                    workspace_id,
                    "; ".join(errors),
                )
                runtime.write_manifest(
                    "cleanup_failed", lease_seconds=0.0, errors=errors
                )
                continue
            manifest_file.unlink(missing_ok=True)
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
            reclaimed.append(workspace_id)
        except Exception as exc:  # noqa: BLE001 - keep scanning
            logger.warning("reconciliation failed for %s: %s", workspace_id, exc)

    return reclaimed
