"""Render a validated network policy into an nftables ruleset.

The rules text is built only from values that have already passed the
validation in network_policy.py. No caller-provided string is ever
interpolated as a raw nftables fragment.
"""

import hashlib
from ipaddress import IPv6Network

from .network_policy import AllowedEndpoint, WorkspaceNetworkPolicy


TABLE_NAME = "workspace_egress"

DOCKER_EMBEDDED_RESOLVER = "127.0.0.11"
"""Docker's embedded DNS resolver.

Traffic to this address must be dropped by ADDRESS ONLY, before the loopback
accept. Docker DNATs 127.0.0.11:53 in the nat OUTPUT chain (priority -100),
which runs before this filter OUTPUT hook (priority 0), so a `dport 53` match
never fires and external names remain resolvable despite `policy drop`.
Verified empirically; see the design spec section 2.7.
"""


def _render_endpoint(endpoint: AllowedEndpoint) -> str:
    family = "ip6" if isinstance(endpoint.destination, IPv6Network) else "ip"
    parts = [f"{family} daddr {endpoint.destination}"]
    if endpoint.protocol is not None:
        ports = ", ".join(str(p) for p in endpoint.ports)
        parts.append(f"{endpoint.protocol} dport {{ {ports} }}")
    parts.append("accept")
    return " ".join(parts)


def render_rules(policy: WorkspaceNetworkPolicy) -> str:
    """Render the canonical ruleset. Returns "" for public mode (no sidecar)."""
    if not policy.requires_sidecar:
        return ""

    lines = [
        f"table inet {TABLE_NAME} {{",
        "    chain output {",
        "        type filter hook output priority filter;",
        "        policy drop;",
        "",
        "        # Docker's embedded resolver forwards external queries from the",
        "        # host namespace. Match on address only -- a dport 53 match is",
        "        # defeated by docker's nat-OUTPUT DNAT. See spec 2.7.",
        f"        ip daddr {DOCKER_EMBEDDED_RESOLVER} drop",
        "",
        '        oifname "lo" accept',
        "        ct state established,related accept",
        "",
    ]
    lines.extend(f"        {_render_endpoint(e)}" for e in policy.resolved_endpoints())
    lines.extend(
        [
            "",
            "        reject with icmpx type admin-prohibited",
            "    }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def policy_digest(rules_text: str) -> str:
    """SHA-256 of the canonical rules text, used to verify the applied ruleset."""
    return hashlib.sha256(rules_text.encode("utf-8")).hexdigest()
