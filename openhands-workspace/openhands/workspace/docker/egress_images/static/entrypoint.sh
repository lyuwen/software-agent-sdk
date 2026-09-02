#!/bin/sh
# Workspace egress sidecar entrypoint.
#
# Owns the network namespace the workspace container joins. Applies the
# host-rendered nftables policy, verifies it, then publishes readiness and
# stays alive. Any failure exits non-zero WITHOUT publishing readiness, so
# the controller fails closed.
set -eu

RULES_FILE="/etc/workspace-egress/rules.nft"
READY_FILE="/tmp/workspace-egress.ready"
TABLE_NAME="workspace_egress"

cleanup() {
    rm -f "$READY_FILE" 2>/dev/null || true
}
trap cleanup EXIT

on_signal() {
    cleanup
    exit 0
}
trap on_signal TERM INT HUP

if [ ! -f "$RULES_FILE" ]; then
    echo "egress: missing rules file $RULES_FILE" >&2
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

# The resolver drop must precede the loopback accept, or DNS escapes.
resolver_line="$(printf '%s\n' "$applied" | grep -n 'daddr 127.0.0.11' | head -1 | cut -d: -f1)"
loopback_line="$(printf '%s\n' "$applied" | grep -n 'oifname "lo"' | head -1 | cut -d: -f1)"
if [ -n "$loopback_line" ] && [ "$resolver_line" -ge "$loopback_line" ]; then
    echo "egress: resolver drop must precede loopback accept" >&2
    exit 1
fi

echo "egress: policy applied and verified"
touch "$READY_FILE"

# Stay alive as the namespace owner, remaining responsive to signals.
while true; do
    sleep 1 &
    wait $!
done
