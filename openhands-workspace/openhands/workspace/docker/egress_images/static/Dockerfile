# Minimal nftables sidecar. All packages are installed at BUILD time; the
# runtime entrypoint must never install anything.
FROM alpine@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc

RUN apk add --no-cache nftables

# The entrypoint recomputes the policy digest at startup and must never install
# a tool to do it. Fail the build here if busybox does not provide sha256sum.
RUN command -v sha256sum >/dev/null 2>&1 || { \
        echo "sha256sum is required by the entrypoint but is missing from this image" >&2; \
        exit 1; \
    }

COPY --chmod=755 entrypoint.sh /usr/local/bin/egress-entrypoint

ENTRYPOINT ["/usr/local/bin/egress-entrypoint"]
