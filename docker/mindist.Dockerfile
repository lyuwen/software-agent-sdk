# Build the OpenHands env image FROM SOURCE -- no prebuilt tarball required.
#
# This is the from-scratch counterpart to tools/oh-env-image.Dockerfile. That
# image unpacks a tarball produced by tools/build_oh_env.sh; this one performs
# the same work (fetch python-build-standalone CPython 3.12, editable-install the
# 4 OpenHands SDK packages + third-party deps at the fixed prefix /opt/oh-env)
# entirely inside the Docker build, then ships the result as a bare `scratch`
# image whose rootfs IS the prefix contents.
#
# The output contract is identical to tools/oh-env-image.Dockerfile: the prefix
# is stripped so `/opt/oh-env/bin/python` lands at the image root `/bin/python`.
# Consume it exactly the same way -- as an *image volume* (containerd / k8s
# `image` volume, or `docker run -v <img>:...`) mounted at /opt/oh-env inside the
# sandbox, so the contents reappear at the fixed prefix the harness
# (slime/agent/harness/openhands.py) expects.
#
# Because the editable install is performed at the REAL absolute prefix
# /opt/oh-env inside the builder, the recorded editable paths and console-script
# shebangs (/opt/oh-env/bin/python) are correct the moment the image is mounted
# back at /opt/oh-env -- do not change PREFIX without updating the harness.
#
# Build (BuildKit required -- default on modern Docker), from the SDK repo root:
#   DOCKER_BUILDKIT=1 docker build \
#     -f docker/mindist.Dockerfile \
#     -t openhands-sdk-min:latest .
#
# Override the SDK source, python build, or prefix via --build-arg if needed:
#   --build-arg OH_SDK_SRC=path/to/software-agent-sdk   (relative to context)
#   --build-arg PYTHON_URL=https://.../cpython-...tar.gz
#   --build-arg PREFIX=/opt/oh-env

# syntax=docker/dockerfile:1.7

# ---- stage 1: build the prefix from source ----------------------------------
FROM debian:bookworm-slim AS builder

# python-build-standalone: self-contained CPython 3.12 that bundles its own
# libpython, so the result is independent of any base image's Python/glibc.
ARG PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20250808/cpython-3.12.11+20250808-x86_64-unknown-linux-gnu-install_only.tar.gz
# Fixed install prefix. MUST match the harness (slime/agent/harness/openhands.py)
# and the mount target -- editable-install paths depend on it.
ARG PREFIX=/opt/oh-env
# SDK source dir, relative to the build context (repo root).
ARG OH_SDK_SRC=.
# Optional PyPI index override (e.g. a local mirror). Empty => pip's default.
ARG PIP_INDEX_URL=
ENV PREFIX=${PREFIX} \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# curl+ca-certificates to fetch python; build-essential so any source-only dep
# can compile from scratch (wheels are used when available).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates tar build-essential && \
    rm -rf /var/lib/apt/lists/*

# 1. Materialize python-build-standalone CPython 3.12 at the fixed prefix.
#    The archive contains a top-level `python/` dir; strip it onto the prefix.
RUN mkdir -p "${PREFIX}" && \
    curl -fSL "${PYTHON_URL}" -o /tmp/python.tar.gz && \
    tar xzf /tmp/python.tar.gz -C "${PREFIX}" --strip-components=1 && \
    rm -f /tmp/python.tar.gz && \
    test -x "${PREFIX}/bin/python" && \
    "${PREFIX}/bin/python" --version

# 2. Stage the 4 OpenHands packages as source at the fixed in-prefix path. A
#    bind mount keeps the (large) SDK tree out of the image layers. The root
#    workspace files are copied when present for consistent metadata.
RUN --mount=type=bind,source=.,target=/ctx,ro \
    mkdir -p "${PREFIX}/src/software-agent-sdk" && \
    for pkg in openhands-sdk openhands-tools openhands-workspace openhands-agent-server; do \
        test -d "/ctx/${OH_SDK_SRC}/${pkg}" || { echo "error: missing package: ${pkg}" >&2; exit 1; }; \
        cp -a "/ctx/${OH_SDK_SRC}/${pkg}" "${PREFIX}/src/software-agent-sdk/${pkg}"; \
    done && \
    for f in pyproject.toml uv.lock README.md LICENSE; do \
        if [ -e "/ctx/${OH_SDK_SRC}/${f}" ]; then \
            cp -a "/ctx/${OH_SDK_SRC}/${f}" "${PREFIX}/src/software-agent-sdk/${f}"; \
        fi; \
    done

# 3. Editable-install the 4 packages FROM the in-prefix path (records the fixed
#    /opt/oh-env/src/... path), then verify the SDK imports out of the prefix --
#    the same check the harness runs at boot (install_cli).
RUN "${PREFIX}/bin/python" -m pip install --upgrade pip && \
    for pkg in openhands-sdk openhands-tools openhands-workspace openhands-agent-server; do \
        echo ">> pip install -e ${PREFIX}/src/software-agent-sdk/${pkg}"; \
        "${PREFIX}/bin/python" -m pip install -e "${PREFIX}/src/software-agent-sdk/${pkg}"; \
    done && \
    "${PREFIX}/bin/python" -c 'import openhands.sdk; import openhands.tools; print("import OK")'

# 4. Collect the prefix contents into a clean rootfs the scratch stage can copy.
RUN mkdir -p /rootfs && \
    cp -a "${PREFIX}/." /rootfs/ && \
    test -x /rootfs/bin/python

# ---- stage 2: pack the prefix contents into a bare scratch image ------------
FROM scratch
COPY --from=builder /rootfs/ /
