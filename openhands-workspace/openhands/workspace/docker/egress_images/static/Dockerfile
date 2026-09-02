# Minimal nftables sidecar. All packages are installed at BUILD time; the
# runtime entrypoint must never install anything.
FROM alpine@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc

RUN apk add --no-cache nftables

COPY --chmod=755 entrypoint.sh /usr/local/bin/egress-entrypoint

ENTRYPOINT ["/usr/local/bin/egress-entrypoint"]
