"""Tests for shared docker run-argument construction."""

from unittest.mock import patch

import pytest

from openhands.workspace import DockerWorkspace, FlexWorkspace


@pytest.fixture
def docker_ws():
    with patch.object(DockerWorkspace, "_start_container"):
        ws = DockerWorkspace(server_image="test:latest")
    ws.host_port = 30000
    return ws


@pytest.fixture
def flex_ws():
    with patch.object(FlexWorkspace, "_start_container"):
        ws = FlexWorkspace(base_image="base:latest")
    ws.host_port = 30000
    return ws


def _memory_flags(args: list[str]) -> list[str]:
    return [
        args[i + 1] if args[i] == "--memory" else args[i].split("=", 1)[1]
        for i in range(len(args))
        if args[i] == "--memory" or args[i].startswith("--memory=")
    ]


def test_memory_limit_default_is_14g(docker_ws):
    assert docker_ws.memory_limit == "14g"


def test_exactly_one_memory_flag_is_emitted(docker_ws):
    args = docker_ws._build_run_args("img:latest", container_name="c")
    assert _memory_flags(args) == ["14g"]


def test_memory_limit_is_honoured(docker_ws):
    docker_ws.memory_limit = "7g"
    args = docker_ws._build_run_args("img:latest", container_name="c")
    assert _memory_flags(args) == ["7g"]


def test_flex_emits_one_memory_flag(flex_ws):
    args = flex_ws._build_run_args("img:latest", container_name="c")
    assert _memory_flags(args) == ["14g"]


def test_args_are_argv_list_not_shell_string(docker_ws):
    args = docker_ws._build_run_args("img:latest", container_name="c")
    assert isinstance(args, list)
    assert all(isinstance(a, str) for a in args)
    assert args[:3] == ["docker", "run", "-d"]


def test_ports_published_on_all_interfaces_in_public_mode(docker_ws):
    args = docker_ws._build_run_args("img:latest", container_name="c")
    assert "30000:8000" in args


def test_container_name_is_used(docker_ws):
    args = docker_ws._build_run_args("img:latest", container_name="my-name")
    assert args[args.index("--name") + 1] == "my-name"


def test_entrypoint_and_command_appended_after_image(docker_ws):
    args = docker_ws._build_run_args(
        "img:latest",
        container_name="c",
        entrypoint=["/bin/python"],
        command=["-m", "server"],
    )
    assert args[args.index("--entrypoint") + 1] == "/bin/python"
    image_idx = args.index("img:latest")
    assert args[image_idx + 1 :] == ["-m", "server"]
