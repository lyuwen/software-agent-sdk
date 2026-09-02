#!/bin/sh
# Workspace egress sidecar entrypoint.
#
# Owns the network namespace the workspace container joins. Verifies the
# host-rendered nftables policy against the digest the host requested, applies
# it, verifies the APPLIED ruleset against that same requested policy, then
# publishes readiness and stays alive. Any failure exits non-zero WITHOUT
# publishing readiness, so the controller fails closed.
set -eu

RULES_FILE="/etc/workspace-egress/rules.nft"
READY_FILE="/tmp/workspace-egress.ready"
TABLE_NAME="workspace_egress"

cleanup() {
    rm -f "$READY_FILE" 2>/dev/null || true
}
trap cleanup EXIT

on_signal() {
    exit 0
}
trap on_signal TERM INT HUP

# Emit one normalized destination per line for every accept rule carrying a
# `daddr` match, read from stdin. Comment lines are skipped, and the `ip daddr
# 127.0.0.11 drop` resolver rule is excluded because it is not an accept.
# nft lists a host route as a bare address, so /32 and /128 are stripped on
# both sides to make the requested and applied forms comparable.
daddr_accepts() {
    awk '
        /^[[:space:]]*#/ { next }
        {
            dest = ""
            is_accept = 0
            for (i = 1; i < NF; i++) {
                if ($i == "daddr") {
                    dest = $(i + 1)
                }
            }
            for (i = 1; i <= NF; i++) {
                if ($i == "accept") {
                    is_accept = 1
                }
            }
            if (dest != "" && is_accept) {
                sub(/\/32$/, "", dest)
                sub(/\/128$/, "", dest)
                print dest
            }
        }
    ' | sort
}

count_lines() {
    awk 'NF { n++ } END { print n + 0 }'
}

if [ ! -f "$RULES_FILE" ]; then
    echo "egress: missing rules file $RULES_FILE" >&2
    exit 1
fi

# The host passes the sha256 of the exact rules text it rendered. Without it we
# cannot know which policy was requested, so there is nothing to verify against.
expected_digest="$(printf '%s' "${OH_EGRESS_POLICY_DIGEST:-}" | tr 'A-Z' 'a-z')"
if [ -z "$expected_digest" ]; then
    echo "egress: OH_EGRESS_POLICY_DIGEST is unset or empty; refusing to apply an unverified policy" >&2
    exit 1
fi

actual_digest="$(sha256sum "$RULES_FILE" | cut -d' ' -f1 | tr 'A-Z' 'a-z')"
if [ "$actual_digest" != "$expected_digest" ]; then
    echo "egress: rules file digest mismatch: expected $expected_digest, got $actual_digest" >&2
    exit 1
fi

if ! nft -f "$RULES_FILE"; then
    echo "egress: failed to apply ruleset" >&2
    exit 1
fi

# Verify the applied ruleset rather than trusting that nft exited 0.
applied="$(nft list table inet "$TABLE_NAME" 2>/dev/null || true)"
if [ -z "$applied" ]; then
    echo "egress: table inet $TABLE_NAME absent after apply" >&2
    exit 1
fi

for required in \
    "policy drop" \
    "ip daddr 127.0.0.11 drop" \
    "ct state established,related accept"
do
    if ! printf '%s\n' "$applied" | grep -qF "$required"; then
        echo "egress: verification failed, missing: $required" >&2
        exit 1
    fi
done

# Verify the applied ruleset against the REQUESTED policy, not just against a
# fixed set of fragments: every allow destination the rules file asks for must
# be present as an accept, and there must be no extra ones.
requested_dests="$(daddr_accepts < "$RULES_FILE")"
applied_dests="$(printf '%s\n' "$applied" | daddr_accepts)"

for dest in $requested_dests; do
    if ! printf '%s\n' "$applied_dests" | grep -qxF -- "$dest"; then
        echo "egress: verification failed, applied ruleset has no accept for requested destination: $dest" >&2
        exit 1
    fi
done

requested_count="$(printf '%s\n' "$requested_dests" | count_lines)"
applied_count="$(printf '%s\n' "$applied_dests" | count_lines)"
if [ "$requested_count" != "$applied_count" ]; then
    echo "egress: verification failed, applied ruleset has $applied_count daddr accept rules, requested $requested_count" >&2
    exit 1
fi

# The resolver drop must precede the loopback accept, or DNS escapes.
resolver_line="$(printf '%s\n' "$applied" | grep -n 'daddr 127.0.0.11' | head -1 | cut -d: -f1)"
loopback_line="$(printf '%s\n' "$applied" | grep -n 'oifname "lo"' | head -1 | cut -d: -f1)"
# Fail closed: if the resolver drop is absent here the ordering check cannot run.
# (The required-rule loop above should have already caught this, but this guards
# against future relaxations of that literal check.)
if [ -z "$resolver_line" ]; then
    echo "egress: ordering check failed, resolver drop rule not found" >&2
    exit 1
fi
if [ -n "$loopback_line" ] && [ "$resolver_line" -ge "$loopback_line" ]; then
    echo "egress: resolver drop must precede loopback accept" >&2
    exit 1
fi

echo "egress: policy applied and verified ($requested_count allow destinations, digest $expected_digest)"
touch "$READY_FILE"

# Stay alive as the namespace owner, remaining responsive to signals.
while true; do
    sleep 1 &
    wait $!
done
