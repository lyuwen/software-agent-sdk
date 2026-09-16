# DOCKER_BUILDKIT=1 docker build -f docker/ubuntu2204.Dockerfile --build-arg BASE_IMAGE=openhands-sdk-min:v20260915 -t openhands-sdk:v20260915-ubuntu2204 .

# syntax=docker/dockerfile:1.7

# Base oh-env image whose SDK source we'll replace. Must be a valid oh-env image
# (rootfs is the prefix contents: /bin/python, /lib/, /src/, ...).
ARG BASE_IMAGE=oh-env:latest

# ---- stage 0: materialize base image as a named stage --------------------------
FROM ${BASE_IMAGE} AS base

# ---- stage 2: pack the result into a fresh scratch image --------------------
FROM ubuntu:22.04
# ENV PATH=/opt/oh-env/bin:$PATH
RUN mkdir -p /opt/oh-env
COPY --from=base / /opt/oh-env/
