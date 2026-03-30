"""LSP workspace setup helpers — reusable across benchmarks.

Upload LSP scripts into a Docker/remote workspace, install Pyright,
and start the LSP daemon.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.sdk.workspace import RemoteWorkspace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths to LSP scripts that get uploaded into the container
# ---------------------------------------------------------------------------
LSP_DIR = Path(__file__).parent
LSP_DAEMON_SCRIPT = LSP_DIR / "lsp_daemon.py"
LSP_TOOL_SCRIPT = LSP_DIR / "lsp_tool.py"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def setup_lsp_in_workspace(workspace: "RemoteWorkspace") -> str:
    """Upload LSP scripts and install dependencies. Returns the pyright-langserver path.

    Call :func:`start_lsp_daemon` separately after the repo is copied to ``/workspace/``.
    """
    logger.info("Setting up LSP tool in workspace...")

    # Upload daemon and tool scripts
    workspace.file_upload(str(LSP_DAEMON_SCRIPT), "/tmp/lsp_daemon.py")
    workspace.file_upload(str(LSP_TOOL_SCRIPT), "/tmp/lsp_tool.py")

    # Install dependencies.  The [nodejs] extra bundles Node.js via
    # nodejs-wheel so pyright doesn't need to download it at runtime.
    res = workspace.execute_command(
        "pip install orjson 'pyright[nodejs]'",
        timeout=120.0,
    )
    if res.exit_code != 0:
        logger.warning(f"pip install failed (exit {res.exit_code}): {res.stderr}")
    else:
        logger.info("pip install orjson pyright succeeded")

    # Find pyright-langserver binary — pip may install it to a scripts dir
    # that isn't on PATH (e.g. /agent-server/.venv/bin/).
    res = workspace.execute_command(
        "python -c \""
        "import sysconfig, os; "
        "scripts = sysconfig.get_path('scripts'); "
        "p = os.path.join(scripts, 'pyright-langserver'); "
        "print(p if os.path.isfile(p) else 'NOT_FOUND')\"",
        timeout=10.0,
    )
    pyright_path = res.stdout.strip()
    if pyright_path == "NOT_FOUND":
        # Fallback: broader search
        res = workspace.execute_command(
            "find / -name pyright-langserver -type f 2>/dev/null | head -1 || echo 'NOT_FOUND'",
            timeout=15.0,
        )
        pyright_path = res.stdout.strip()
    if not pyright_path or pyright_path == "NOT_FOUND":
        logger.error(
            "pyright-langserver not found anywhere after pip install. "
            "LSP daemon will fail to start."
        )
        pyright_path = "pyright-langserver"  # fallback; will likely fail
    else:
        logger.info(f"pyright-langserver found at: {pyright_path}")

    # Pre-warm: run pyright --version to ensure Node.js is ready.
    # With pyright[nodejs] this should be instant; without it, this triggers
    # the Node.js download so the daemon doesn't have to wait.
    pyright_bin = pyright_path.replace("pyright-langserver", "pyright")
    res = workspace.execute_command(
        f"{pyright_bin} --version",
        timeout=120.0,
    )
    if res.exit_code == 0:
        logger.info(f"pyright pre-warm succeeded: {res.stdout.strip()}")
    else:
        logger.warning(f"pyright pre-warm failed (exit {res.exit_code}): {res.stderr}")

    return pyright_path


def start_lsp_daemon(
    workspace: "RemoteWorkspace",
    pyright_path: str,
    project_root: str,
) -> None:
    """Start the LSP daemon pointing at the given project root.

    Must be called after the repo is copied to *project_root*.
    """
    # Start LSP daemon in background with explicit path to pyright-langserver.
    # The daemon reads LSP_COMMAND and LSP_PROJECT_ROOT env vars.
    lsp_command = f"{pyright_path} --stdio"
    res = workspace.execute_command(
        f"cd {project_root} && "
        f"LSP_COMMAND='{lsp_command}' "
        f"LSP_PROJECT_ROOT='{project_root}' "
        "nohup python /tmp/lsp_daemon.py "
        "> /tmp/lsp_daemon_stdout.log 2> /tmp/lsp_daemon.log &",
        timeout=30.0,
    )
    logger.info(f"LSP daemon start result: exit_code={res.exit_code}")

    # Poll for the port file — Pyright initialization on large codebases
    # can take 60+ seconds.  The daemon writes the port file only after
    # Pyright is fully initialized and the TCP server is listening.
    PORT_FILE = "/var/tmp/lsp_port_session_abc.pid"
    port_info = None
    for attempt in range(1, 19):  # up to 90 seconds total
        res = workspace.execute_command(
            f"cat {PORT_FILE} 2>/dev/null || echo 'NO_PORT'",
            timeout=10.0,
        )
        content = res.stdout.strip()
        if content != "NO_PORT" and content.isdigit():
            port_info = content
            break
        logger.info(f"Waiting for LSP daemon (attempt {attempt}/18)...")
        workspace.execute_command("sleep 5", timeout=10.0)

    if port_info:
        logger.info(f"LSP daemon listening on port {port_info}")
    else:
        # Dump logs so we can diagnose
        log_res = workspace.execute_command(
            "echo '=== DAEMON LOG ===' && cat /tmp/lsp_daemon.log 2>/dev/null; "
            "echo '=== DAEMON STDOUT ===' && cat /tmp/lsp_daemon_stdout.log 2>/dev/null",
            timeout=10.0,
        )
        logger.error(
            f"LSP daemon failed to start after 90s. Logs:\n{log_res.stdout}"
        )
