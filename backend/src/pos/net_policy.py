"""Which URLs the server may be pointed at on a user's behalf.

A user can save their own "local server" URL (LM Studio, Ollama, ...), and the
backend then makes requests to it from its own network position -- listing models
and sending chats. Unchecked, that reaches whatever the backend can reach and the
user cannot: other containers, services on the host, and on a cloud VM the metadata
endpoint that hands out the machine's credentials (server-side request forgery).

The rules (OWASP SSRF Prevention Cheat Sheet):
  * only http/https, with a host and no credentials embedded in the URL;
  * EVERY address the host resolves to is checked, so a name with a public and a
    private record is refused, and numeric spellings of 127.0.0.1 (decimal, hex,
    octal, short) are judged by what they really are;
  * link-local, cloud-metadata (including IPv6 and the CGNAT range Alibaba uses),
    unspecified and multicast addresses are ALWAYS refused;
  * private and loopback addresses are refused unless the operator sets
    ALLOW_PRIVATE_LLM_URLS (right for a personal install with LM Studio on the same
    machine, wrong for a shared one);
  * redirects are never followed (callers pass allow_redirects=False).

Limits, stated plainly: this checks at save time and again when a request is built,
but a hostname can be re-pointed in between (DNS rebinding). The complete fix is
network-level egress control, which this cannot replace.

A URL the operator set in their own environment (LOCAL_BASE_URL) is trusted: it is
their configuration, not user input.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Callable
from urllib.parse import urlsplit

from pos import config


class UnsafeUrl(ValueError):
    """The URL may not be used. The message says why, without echoing the URL."""


# Never reachable, whatever the operator allows. ip.is_link_local covers 169.254/16 and fe80::/10.
_ALWAYS_BLOCKED = [
    ipaddress.ip_network("100.64.0.0/10"),     # shared/CGNAT space: Alibaba Cloud metadata lives here
    ipaddress.ip_network("fd00:ec2::/32"),     # AWS metadata over IPv6
]


def _resolve(host: str, port: int) -> list[str]:
    """Every address `host` resolves to, using the platform's own parsing."""
    return sorted({info[4][0] for info in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)})


def _classify(address: str) -> str | None:
    """None if the address is fine, else why not: 'blocked' (never) or 'private' (operator may allow)."""
    ip = ipaddress.ip_address(address.split("%")[0])          # drop an IPv6 zone id
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped                                     # ::ffff:169.254.169.254 is 169.254.169.254
    if ip.is_loopback:
        return "private"            # checked first: Python also files ::1 under "reserved"
    if (
        ip.is_link_local or ip.is_unspecified or ip.is_multicast or ip.is_reserved
        or any(ip in net for net in _ALWAYS_BLOCKED)
    ):
        return "blocked"
    if ip.is_private:
        return "private"
    return None


def check_url(
    url: str,
    *,
    allow_private: bool | None = None,
    https_only: bool = False,
    resolver: Callable[[str, int], list[str]] | None = None,
) -> str:
    """Return `url` (stripped) if it may be used, else raise UnsafeUrl."""
    if allow_private is None:
        allow_private = config.ALLOW_PRIVATE_LLM_URLS
    resolve = resolver or _resolve
    url = (url or "").strip()
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise UnsafeUrl("that is not a valid URL") from None
    if parts.scheme not in ("http", "https"):
        raise UnsafeUrl("the URL must start with http:// or https://")
    if https_only and parts.scheme != "https":
        raise UnsafeUrl("the URL must use https://")
    host = parts.hostname
    if not host:
        raise UnsafeUrl("the URL has no host")
    if parts.username or parts.password:
        raise UnsafeUrl("the URL must not contain credentials")

    try:
        addresses = [str(ipaddress.ip_address(host))]      # an IP literal needs no DNS: judge it as written
    except ValueError:
        try:   # a name, or a numeric spelling like 2130706433 that only the platform's resolver understands
            addresses = resolve(host, port or (443 if parts.scheme == "https" else 80))
        except (OSError, ValueError):
            raise UnsafeUrl("that host could not be resolved") from None
    if not addresses:
        raise UnsafeUrl("that host could not be resolved")

    for address in addresses:
        verdict = _classify(address)
        if verdict == "blocked":
            raise UnsafeUrl("that address is not allowed (link-local, metadata or reserved range)")
        if verdict == "private" and not allow_private:
            raise UnsafeUrl(
                "that is a private or loopback address; the server operator has not allowed "
                "private LLM server URLs (ALLOW_PRIVATE_LLM_URLS)"
            )
    return url


def check_user_url(env_name: str, url: str, *, https_only: bool = False, allow_private: bool | None = None) -> str:
    """check_url for a URL that may have come from a user. The operator's own
    setting of the same variable in the server environment is trusted as is."""
    if url and os.environ.get(env_name) == url:
        return url
    return check_url(url, https_only=https_only, allow_private=allow_private)


# Credential fields that hold a URL the server will call, and how strict each must be.
# An Azure endpoint is always a public https host, so it never gets the private-address exemption.
URL_FIELDS: dict[str, dict] = {
    "LOCAL_BASE_URL": {},
    "AZURE_OPENAI_ENDPOINT": {"https_only": True, "allow_private": False},
}


def validate_credential_urls(payload: dict[str, str]) -> None:
    """Raise UnsafeUrl (naming the field) if any URL in a credential about to be saved is unsafe."""
    for name, options in URL_FIELDS.items():
        if payload.get(name):
            try:
                check_user_url(name, payload[name], **options)
            except UnsafeUrl as e:
                raise UnsafeUrl(f"{name}: {e}") from None
