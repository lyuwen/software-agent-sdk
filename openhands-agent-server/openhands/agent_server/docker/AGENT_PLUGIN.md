# Agent Plugin: Design & Implementation Guide

## Overview

The agent plugin is a **single, portable Docker image** (`openhands/agent-plugin`) that packages
the entire agent server runtime — Python venv, managed Python, source code, CLI tools, and shared
libraries — under one mountable directory: `/agent-server/`. It is built once and injected into
any number of base environment images at runtime via Docker volumes, with **zero per-image
rebuilds** and **zero layer duplication**.

## Design Philosophy

### Problem

We maintain many different base images (Java, Node, Go, Ruby, etc.) that each need the same
agent server plugin. Baking the plugin into every base image via `COPY --from` duplicates
hundreds of megabytes per image and requires rebuilding all images whenever the plugin changes.

### Solution: Build Once, Mount Everywhere

The plugin image declares `VOLUME /agent-server`, which creates a named Docker volume populated
with the image contents on first use. Any container can then access this volume via
`--volumes-from`, gaining the full agent server runtime without any image-layer coupling.

```
┌──────────────────────────────────────────────┐
│  openhands/agent-plugin (built once)         │
│                                              │
│  /agent-server/                              │
│    ├── .venv/             Python venv        │
│    ├── uv-managed-python/ Managed Python     │
│    ├── bin/               uv, uvx, rg, ...   │
│    ├── lib/               Shared libraries   │
│    ├── openhands-sdk/     Source              │
│    ├── openhands-tools/   Source              │
│    ├── openhands-workspace/ Source            │
│    └── openhands-agent-server/ Source         │
└──────────────────────────────────────────────┘
         │
         │  docker create --name=ap ... true
         │  docker run --volumes-from=ap ...
         ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  Java env image     │  │  Node env image     │  │  Ruby env image     │
│  + /agent-server/   │  │  + /agent-server/   │  │  + /agent-server/   │
│    (volume mount)   │  │    (volume mount)   │  │    (volume mount)   │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

### Why `/agent-server/` as a Single Mount Point

Apt-installed packages scatter files across `/usr/bin/`, `/usr/lib/`, `/etc/`, etc. These system
paths cannot be volume-mounted without shadowing the base image's own files. The plugin solves
this by **flattening** all binaries into `/agent-server/bin/` and all shared libraries into
`/agent-server/lib/`, then using `PATH` and `LD_LIBRARY_PATH` to make them discoverable.

## Dockerfile.agent-plugin Structure

### Stage 1: Builder

Builds the Python venv with all openhands packages from source using `uv`:

```
python:3.12-bullseye → COPY source → uv sync → /agent-server/.venv/
```

The venv is built with `--no-editable` (installed as copies, not symlinks). This makes the
venv self-contained and relocatable, but means source bind mounts during development require
a `PYTHONPATH` override (see Development Mode below).

### Stage 2: Pluggable Package Stages

Each optional package group (agent-tools, ripgrep, build-tools) is built via a filesystem-diff
pattern on `debian:bookworm-slim`:

1. Snapshot all files before install (`find / ... > /tmp/before.txt`)
2. Run `apt-get install`
3. Snapshot all files after install (`find / ... > /tmp/after.txt`)
4. Extract only the delta (`comm -13 before after`) into a `FROM scratch` image

This captures exactly the binaries + transitive shared library dependencies that each package
group adds, with zero base-image bloat.

### Stage 3: Conditional Selectors

ARG-driven stage name interpolation enables zero-overhead toggling:

```dockerfile
FROM scratch AS pkg-ripgrep-true    # contains ripgrep files
COPY --from=pkg-ripgrep / /
FROM scratch AS pkg-ripgrep-false   # empty

FROM pkg-ripgrep-${INSTALL_RIPGREP} AS selected-ripgrep
```

When `INSTALL_RIPGREP=false`, the selected stage is empty scratch — BuildKit never builds the
ripgrep builder stage at all (lazy evaluation).

### Stage 4: Agent Plugin (Final)

Consolidates everything under `/agent-server/`:

1. `COPY --from=builder` — venv, managed python, source
2. `COPY --from=ghcr.io/astral-sh/uv` — uv, uvx binaries
3. `COPY --from=selected-*` — package files into a staging area
4. **Flatten step** — moves binaries from `usr/bin/`, `bin/`, etc. into `/agent-server/bin/`
   and libraries from `usr/lib/`, `lib/` into `/agent-server/lib/`
5. `VOLUME /agent-server` — declares the volume mount point

### Build Args

| ARG                   | Default | Purpose                                      |
|-----------------------|---------|----------------------------------------------|
| `INSTALL_AGENT_TOOLS` | `true`  | curl, wget, jq, gnupg, lsb-release          |
| `INSTALL_RIPGREP`     | `false` | rg (used by grep/glob tools with fallback)   |
| `INSTALL_BUILD_TOOLS` | `false` | build-essential (gcc, make, etc.)             |
| `USE_CN_MIRRORS`      | `false` | Use Chinese apt mirrors                      |

## Runtime: Volume Mount Pattern

### Step 1: Build the Plugin Image (Once)

```bash
# From repo root
docker build \
  -f openhands-agent-server/openhands/agent_server/docker/Dockerfile.agent-plugin \
  -t openhands/agent-plugin .
```

### Step 2: Create a Data Container

```bash
docker create --name=agent-plugin openhands/agent-plugin true
```

This creates a stopped container whose sole purpose is to own the `/agent-server` volume.
The volume lives on **local Docker storage** (fast SSD), not shared NFS.

### Step 3: Run Any Base Image with the Plugin

```bash
docker run --volumes-from=agent-plugin \
  -e PATH="/agent-server/bin:/agent-server/.venv/bin:$PATH" \
  -e UV_PYTHON_INSTALL_DIR=/agent-server/uv-managed-python \
  -e LD_LIBRARY_PATH="/agent-server/lib" \
  <any-base-image> \
  /agent-server/.venv/bin/python -m openhands.agent_server
```

### Required Environment Variables

When launching a container with the agent-plugin volume, the inference code **must** set:

| Variable                 | Value                                                 | Why                                      |
|--------------------------|-------------------------------------------------------|------------------------------------------|
| `PATH`                   | `/agent-server/bin:/agent-server/.venv/bin:$PATH`     | Finds uv, uvx, rg, curl, jq, etc.       |
| `LD_LIBRARY_PATH`        | `/agent-server/lib`                                   | Finds shared libs for relocated binaries |
| `UV_PYTHON_INSTALL_DIR`  | `/agent-server/uv-managed-python`                     | uv finds its managed Python runtime      |

### Entrypoint

Source mode:
```
/agent-server/.venv/bin/python -m openhands.agent_server
```

Binary mode (if using `agent-plugin-binary` target):
```
/agent-server/bin/openhands-agent-server
```

## Development Mode: Bind Mounts Override Volumes

Docker's mount precedence: **bind mount > volume mount**. This means you can overlay your
local source code on top of the plugin volume for live development:

```bash
docker run --volumes-from=agent-plugin \
  -v /path/to/openhands-sdk:/agent-server/openhands-sdk \
  -v /path/to/openhands-tools:/agent-server/openhands-tools \
  -v /path/to/openhands-workspace:/agent-server/openhands-workspace \
  -v /path/to/openhands-agent-server:/agent-server/openhands-agent-server \
  -e PATH="/agent-server/bin:/agent-server/.venv/bin:$PATH" \
  -e UV_PYTHON_INSTALL_DIR=/agent-server/uv-managed-python \
  -e LD_LIBRARY_PATH="/agent-server/lib" \
  -e PYTHONPATH="/agent-server/openhands-agent-server:/agent-server/openhands-sdk:/agent-server/openhands-tools:/agent-server/openhands-workspace" \
  <any-base-image> \
  /agent-server/.venv/bin/python -m openhands.agent_server
```

### How It Works

```
/agent-server/                          ← from volume (--volumes-from)
  ├── .venv/                            ← from volume (installed packages)
  ├── uv-managed-python/                ← from volume
  ├── bin/                              ← from volume (uv, rg, curl, ...)
  ├── lib/                              ← from volume (shared libraries)
  ├── openhands-sdk/                    ← OVERRIDDEN by bind mount (live code)
  ├── openhands-tools/                  ← OVERRIDDEN by bind mount (live code)
  ├── openhands-workspace/              ← OVERRIDDEN by bind mount (live code)
  └── openhands-agent-server/           ← OVERRIDDEN by bind mount (live code)
```

### Why PYTHONPATH Is Needed in Dev Mode

The venv was built with `--no-editable`, so Python imports resolve to installed copies inside
`.venv/lib/python3.12/site-packages/`, not from the source directories. The bind mounts place
fresh source at `/agent-server/openhands-*/`, but Python doesn't know to look there.

Setting `PYTHONPATH` adds these directories **before** `site-packages` in the import path,
so Python picks up the bind-mounted live code over the installed copies.

In production, `PYTHONPATH` is not needed — the installed copies in `.venv/` are correct.

### Inference Code Contract

When the inference/orchestration code launches an agent server container, it must:

1. **Ensure the data container exists** — create it if it doesn't:
   ```
   docker create --name=agent-plugin openhands/agent-plugin true
   ```

2. **Attach the volume** — add `--volumes-from=agent-plugin` to `docker run`

3. **Set environment variables** — PATH, LD_LIBRARY_PATH, UV_PYTHON_INSTALL_DIR (see table above)

4. **Optionally add bind mounts** — for development mode, mount source directories and set
   PYTHONPATH

5. **Set the entrypoint/command** — use the full path to the venv Python:
   `/agent-server/.venv/bin/python -m openhands.agent_server`

The base image itself needs no modification. It only needs basic OS packages (ca-certificates,
sudo, git, tmux) which most software engineering environment images already have.

## Cross-Distro Compatibility

The plugin is built on `debian:bookworm-slim`. The pre-built binaries (curl, rg, etc.) link
against bookworm's glibc (2.36). They are compatible with:

- Debian 12+ (exact match)
- Ubuntu 22.04+ (glibc >= 2.35, forward-compatible)
- Any glibc-based distro with glibc >= 2.36

They are **not** compatible with:
- Alpine Linux (musl libc)
- CentOS 7 or older (glibc too old)

One plugin image per CPU architecture (amd64, arm64) covers all supported distros.
