"""Tests for nftables rule rendering."""

from ipaddress import ip_network

from openhands.workspace.docker.network_policy import (
    AllowedEndpoint,
    WorkspaceNetworkPolicy,
)
from openhands.workspace.docker.nftables_renderer import (
    DOCKER_EMBEDDED_RESOLVER,
    TABLE_NAME,
    policy_digest,
    render_rules,
)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def test_renders_inet_table_with_drop_policy():
    text = render_rules(WorkspaceNetworkPolicy(mode="no-network"))
    assert f"table inet {TABLE_NAME}" in text
    assert "type filter hook output priority filter;" in text
    assert "policy drop;" in text


def test_resolver_drop_precedes_loopback_accept():
    """Ordering is load-bearing: see spec 2.7."""
    lines = _lines(render_rules(WorkspaceNetworkPolicy(mode="no-network")))
    resolver_idx = next(
        i for i, ln in enumerate(lines) if DOCKER_EMBEDDED_RESOLVER in ln
    )
    loopback_idx = next(i for i, ln in enumerate(lines) if 'oifname "lo"' in ln)
    assert resolver_idx < loopback_idx


def test_resolver_rule_matches_address_only_not_port():
    """A dport 53 match is defeated by docker's nat-OUTPUT DNAT."""
    lines = _lines(render_rules(WorkspaceNetworkPolicy(mode="no-network")))
    resolver_rule = next(ln for ln in lines if DOCKER_EMBEDDED_RESOLVER in ln)
    assert "dport" not in resolver_rule
    assert "53" not in resolver_rule
    assert resolver_rule.endswith("drop")


def test_no_network_emits_baseline_accept():
    text = render_rules(WorkspaceNetworkPolicy(mode="no-network"))
    assert "ip daddr 10.0.0.0/8 accept" in text


def test_strict_no_network_has_no_baseline():
    text = render_rules(WorkspaceNetworkPolicy(mode="strict-no-network"))
    assert "10.0.0.0/8" not in text
    assert 'oifname "lo" accept' in text
    assert "ct state established,related accept" in text


def test_final_reject_present():
    text = render_rules(WorkspaceNetworkPolicy(mode="no-network"))
    assert "reject with icmpx type admin-prohibited" in _lines(text)[-3]


def test_protocol_and_ports_rendered():
    policy = WorkspaceNetworkPolicy(
        mode="static-allowlist",
        allowed_endpoints=(
            AllowedEndpoint(
                destination=ip_network("192.0.2.0/24"), protocol="tcp", ports=(443, 80)
            ),
        ),
    )
    text = render_rules(policy)
    assert "ip daddr 192.0.2.0/24 tcp dport { 80, 443 } accept" in text


def test_ipv6_endpoint_uses_ip6_keyword():
    policy = WorkspaceNetworkPolicy(
        mode="static-allowlist",
        allowed_endpoints=(AllowedEndpoint(destination=ip_network("2001:db8::/32")),),
    )
    assert "ip6 daddr 2001:db8::/32 accept" in render_rules(policy)


def test_rendering_is_deterministic_regardless_of_input_order():
    a = AllowedEndpoint(destination=ip_network("192.0.2.0/24"))
    b = AllowedEndpoint(destination=ip_network("198.51.100.0/24"))
    first = render_rules(
        WorkspaceNetworkPolicy(mode="static-allowlist", allowed_endpoints=(a, b))
    )
    second = render_rules(
        WorkspaceNetworkPolicy(mode="static-allowlist", allowed_endpoints=(b, a))
    )
    assert first == second
    assert policy_digest(first) == policy_digest(second)


def test_duplicate_endpoints_deduplicated():
    e = AllowedEndpoint(destination=ip_network("192.0.2.0/24"))
    text = render_rules(
        WorkspaceNetworkPolicy(mode="static-allowlist", allowed_endpoints=(e, e))
    )
    assert text.count("ip daddr 192.0.2.0/24 accept") == 1


def test_digest_changes_when_policy_changes():
    one = render_rules(WorkspaceNetworkPolicy(mode="no-network"))
    two = render_rules(WorkspaceNetworkPolicy(mode="strict-no-network"))
    assert policy_digest(one) != policy_digest(two)


def test_public_mode_renders_nothing():
    assert render_rules(WorkspaceNetworkPolicy(mode="public")) == ""
