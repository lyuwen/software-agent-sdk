"""Tests for the workspace network policy model."""

from ipaddress import ip_network

import pytest
from pydantic import ValidationError

from openhands.workspace.docker.network_policy import (
    INTERNAL_BASELINE,
    AllowedEndpoint,
    WorkspaceNetworkPolicy,
    parse_network_mode,
    policy_from_env,
)


def test_baseline_is_ten_slash_eight():
    assert INTERNAL_BASELINE == ip_network("10.0.0.0/8")


def test_unset_mode_is_public():
    assert parse_network_mode(None) == "public"
    assert parse_network_mode("") == "public"
    assert parse_network_mode("   ") == "public"


def test_invalid_mode_is_rejected_not_coerced():
    with pytest.raises(ValueError, match="OH_NETWORK_MODE"):
        parse_network_mode("no-netwrok")


def test_step2_modes_rejected_naming_step2():
    for mode in ("host-allowlist", "public-bootstrap"):
        with pytest.raises(ValueError, match="Step 2"):
            parse_network_mode(mode)


def test_no_network_resolves_to_exactly_the_baseline():
    policy = WorkspaceNetworkPolicy(mode="no-network")
    resolved = policy.resolved_endpoints()
    assert len(resolved) == 1
    assert resolved[0].destination == INTERNAL_BASELINE
    assert resolved[0].protocol is None
    assert resolved[0].ports == ()


def test_strict_no_network_resolves_to_no_destinations():
    assert WorkspaceNetworkPolicy(mode="strict-no-network").resolved_endpoints() == ()


def test_caller_entries_union_with_baseline_and_cannot_replace_it():
    policy = WorkspaceNetworkPolicy(
        mode="static-allowlist",
        allowed_endpoints=(AllowedEndpoint(destination=ip_network("192.0.2.0/24")),),
    )
    destinations = [e.destination for e in policy.resolved_endpoints()]
    assert INTERNAL_BASELINE in destinations
    assert ip_network("192.0.2.0/24") in destinations


def test_narrowing_the_baseline_still_yields_the_full_baseline():
    """A caller entry for a 10.x subnet must not shrink the /8."""
    policy = WorkspaceNetworkPolicy(
        mode="static-allowlist",
        allowed_endpoints=(
            AllowedEndpoint(destination=ip_network("10.1.2.0/24"), protocol="tcp", ports=(443,)),
        ),
    )
    unrestricted = [
        e.destination
        for e in policy.resolved_endpoints()
        if e.protocol is None and e.ports == ()
    ]
    assert INTERNAL_BASELINE in unrestricted


def test_no_network_rejects_caller_endpoints():
    with pytest.raises(ValidationError, match="no-network"):
        WorkspaceNetworkPolicy(
            mode="no-network",
            allowed_endpoints=(AllowedEndpoint(destination=ip_network("192.0.2.0/24")),),
        )


def test_public_rejects_allowlist_fields():
    with pytest.raises(ValidationError, match="public"):
        WorkspaceNetworkPolicy(
            mode="public",
            allowed_endpoints=(AllowedEndpoint(destination=ip_network("192.0.2.0/24")),),
        )


def test_protocol_requires_nonempty_ports():
    with pytest.raises(ValidationError, match="ports"):
        AllowedEndpoint(destination=ip_network("192.0.2.0/24"), protocol="tcp", ports=())


def test_ports_require_protocol():
    with pytest.raises(ValidationError, match="protocol"):
        AllowedEndpoint(destination=ip_network("192.0.2.0/24"), ports=(443,))


@pytest.mark.parametrize("port", [0, 65536, -1])
def test_port_range_enforced(port):
    with pytest.raises(ValidationError):
        AllowedEndpoint(destination=ip_network("192.0.2.0/24"), protocol="tcp", ports=(port,))


@pytest.mark.parametrize(
    "dest", ["224.0.0.0/4", "169.254.0.0/16", "0.0.0.0/0", "255.255.255.255/32"]
)
def test_unsafe_destinations_rejected(dest):
    with pytest.raises(ValidationError):
        AllowedEndpoint(destination=ip_network(dest))


def test_env_unset_gives_public():
    assert policy_from_env({}).mode == "public"


def test_env_invalid_raises():
    with pytest.raises(ValueError):
        policy_from_env({"OH_NETWORK_MODE": "bogus"})
