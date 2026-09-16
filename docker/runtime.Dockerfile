# Copies the openhands-sdk-min prefix into a real OS base image.
# Requires a glibc-based OS (ubuntu, debian); musl-based images (alpine) are
# not supported because the mindist CPython is built against glibc.
#
# Build from the SDK repo root:
#   DOCKER_BUILDKIT=1 docker build \
#     -f docker/runtime.Dockerfile \
#     --build-arg BASE_IMAGE=openhands-sdk-min:latest \
#     --build-arg BASE_OS=ubuntu:22.04 \
#     -t openhands-sdk:latest-ubuntu2204 .

# syntax=docker/dockerfile:1.7

# Source oh-env/mindist image (rootfs is the prefix: /bin/python, /lib/, /src/, ...)
ARG BASE_IMAGE=openhands-sdk-min:latest
# Target OS base. Must be glibc-based.
ARG BASE_OS=ubuntu:22.04

# ---- stage 0: materialize the mindist prefix ------------------------------------
FROM ${BASE_IMAGE} AS base

# ---- stage 1: copy prefix into the real OS --------------------------------------
FROM ${BASE_OS}
RUN mkdir -p /opt/oh-env
COPY --from=base / /opt/oh-env/
