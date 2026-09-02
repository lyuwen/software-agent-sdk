"""Typed network policy for Docker workspace egress control.

Pure data and validation: no Docker calls and no filesystem I/O, so this
module is cheap to unit test.
"""

import os
from collections.abc import Mapping
from ipaddress import IPv4Network, IPv6Network, ip_network
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


INTERNAL_BASELINE: IPv4Network = ip_network("10.0.0.0/8")
"""Mandatory internal destination: LLM proxy and package mirrors live here."""

NetworkMode = Literal[
    "public", "static-allowlist", "no-network", "strict-no-network"
]

_STEP2_MODES = frozenset({"host-allowlist", "public-bootstrap"})
_VALID_MODES = frozenset({"public", "static-allowlist", "no-network", "strict-no-network"})

MAX_ENDPOINTS = 64
MAX_PORTS_PER_ENDPOINT = 32

ENV_VAR = "OH_NETWORK_MODE"


def parse_network_mode(raw: str | None) -> NetworkMode:
    """Parse an OH_NETWORK_MODE value.

    Unset/empty means "public" (preserving historical unrestricted behavior).
    Anything unrecognized is an error — never silently coerced to public.
    """
    if raw is None or not raw.strip():
        return "public"
    value = raw.strip().lower()
    if value in _STEP2_MODES:
        raise ValueError(
            f"{ENV_VAR}={value!r} requires Step 2 (GOST hostname allowlisting), "
            "which is not implemented. Use one of: "
            f"{', '.join(sorted(_VALID_MODES))}."
        )
    if value not in _VALID_MODES:
        raise ValueError(
            f"Invalid {ENV_VAR}={raw!r}. Expected one of: "
            f"{', '.join(sorted(_VALID_MODES))}."
        )
    return value  # type: ignore[return-value]


class AllowedEndpoint(BaseModel):
    """One allowed destination, optionally narrowed to a protocol and ports."""

    model_config = ConfigDict(frozen=True)

    destination: IPv4Network | IPv6Network
    protocol: Literal["tcp", "udp"] | None = None
    ports: tuple[int, ...] = ()
    description: str | None = None

    @field_validator("destination")
    @classmethod
    def _reject_unsafe_destinations(
        cls, v: IPv4Network | IPv6Network
    ) -> IPv4Network | IPv6Network:
        if v.is_multicast:
            raise ValueError(f"multicast destination not allowed: {v}")
        if v.is_link_local:
            raise ValueError(f"link-local destination not allowed: {v}")
        if v.is_unspecified or int(v.network_address) == 0:
            raise ValueError(f"unspecified destination not allowed: {v}")
        if isinstance(v, IPv4Network) and v.broadcast_address == v.network_address:
            if str(v.network_address) == "255.255.255.255":
                raise ValueError(f"broadcast destination not allowed: {v}")
        return v

    @field_validator("ports")
    @classmethod
    def _validate_ports(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if len(v) > MAX_PORTS_PER_ENDPOINT:
            raise ValueError(f"too many ports: {len(v)} > {MAX_PORTS_PER_ENDPOINT}")
        for port in v:
            if not 1 <= port <= 65535:
                raise ValueError(f"port out of range 1..65535: {port}")
        return tuple(sorted(set(v)))

    @model_validator(mode="after")
    def _protocol_and_ports_agree(self) -> "AllowedEndpoint":
        if self.protocol is not None and not self.ports:
            raise ValueError(
                "a non-empty 'ports' list is required when 'protocol' is set"
            )
        if self.protocol is None and self.ports:
            raise ValueError("'ports' requires an explicit 'protocol'")
        return self

    def sort_key(self) -> tuple[int, str, str, tuple[int, ...]]:
        """Deterministic ordering key for stable rule rendering."""
        return (
            self.destination.version,
            str(self.destination),
            self.protocol or "",
            self.ports,
        )


class WorkspaceNetworkPolicy(BaseModel):
    """The egress policy applied to one workspace."""

    model_config = ConfigDict(frozen=True)

    mode: NetworkMode = "public"
    allowed_endpoints: tuple[AllowedEndpoint, ...] = Field(default=())

    @model_validator(mode="after")
    def _mode_accepts_fields(self) -> "WorkspaceNetworkPolicy":
        if self.allowed_endpoints:
            if self.mode == "public":
                raise ValueError(
                    "mode 'public' must not carry allowed_endpoints; it applies "
                    "no policy at all"
                )
            if self.mode == "no-network":
                raise ValueError(
                    "mode 'no-network' must not carry caller-supplied "
                    "allowed_endpoints; it resolves to the fixed 10.0.0.0/8 baseline"
                )
            if self.mode == "strict-no-network":
                raise ValueError(
                    "mode 'strict-no-network' must not carry allowed_endpoints; "
                    "it resolves to no external destinations"
                )
        if len(self.allowed_endpoints) > MAX_ENDPOINTS:
            raise ValueError(
                f"too many endpoints: {len(self.allowed_endpoints)} > {MAX_ENDPOINTS}"
            )
        return self

    @property
    def requires_sidecar(self) -> bool:
        return self.mode != "public"

    def resolved_endpoints(self) -> tuple[AllowedEndpoint, ...]:
        """Final destination set: baseline unioned with caller entries.

        Caller entries are additive only. The unrestricted 10.0.0.0/8 baseline
        is always present except in 'strict-no-network'.
        """
        if self.mode in ("public", "strict-no-network"):
            return ()

        baseline = AllowedEndpoint(
            destination=INTERNAL_BASELINE,
            description="internal-llm-proxy-and-package-mirrors",
        )
        merged: dict[tuple, AllowedEndpoint] = {baseline.sort_key(): baseline}
        for endpoint in self.allowed_endpoints:
            merged.setdefault(endpoint.sort_key(), endpoint)
        return tuple(sorted(merged.values(), key=lambda e: e.sort_key()))


def policy_from_env(
    env: Mapping[str, str] | None = None,
) -> WorkspaceNetworkPolicy:
    """Build a policy from OH_NETWORK_MODE. Raises on an invalid value."""
    source = os.environ if env is None else env
    return WorkspaceNetworkPolicy(mode=parse_network_mode(source.get(ENV_VAR)))
