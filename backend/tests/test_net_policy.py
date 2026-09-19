import pytest

from pos.net_policy import UnsafeUrl, check_url, check_user_url
from pos.net_policy import _resolve as REAL_RESOLVE  # imported before the autouse fixture replaces it

PUBLIC = "93.184.216.34"


def _resolver(*ips):
    return lambda host, port: list(ips)


# Never allowed, even when the operator permits private addresses: these are how
# a cloud VM's credentials are stolen, and no LLM server lives there.
@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://[fe80::1]/v1",
    "http://0.0.0.0:8000/v1",
    "http://224.0.0.1/v1",
    "http://100.100.100.200/v1",                 # Alibaba metadata, in the CGNAT range
    "http://[::ffff:169.254.169.254]/v1",        # IPv4-mapped IPv6 spelling of the same address
    "http://[fd00:ec2::254]/v1",                 # AWS metadata over IPv6
])
def test_metadata_and_link_local_addresses_are_always_refused(url):
    for allow in (False, True):
        with pytest.raises(UnsafeUrl):
            check_url(url, allow_private=allow)


def test_a_hostname_that_resolves_to_the_metadata_address_is_refused():
    with pytest.raises(UnsafeUrl):
        check_url("http://metadata.google.internal/v1", allow_private=True, resolver=_resolver("169.254.169.254"))


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:1234/v1", "http://10.0.0.5/v1", "http://172.16.3.4/v1", "http://192.168.1.5:1234/v1",
    "http://[::1]:1234/v1", "http://[fd12:3456::1]/v1",
])
def test_private_and_loopback_addresses_need_the_operator_to_allow_them(url):
    with pytest.raises(UnsafeUrl, match="private"):
        check_url(url, allow_private=False)
    assert check_url(url, allow_private=True) == url


@pytest.mark.parametrize("host,ip", [("localhost", "127.0.0.1"), ("host.docker.internal", "172.17.0.1"),
                                     ("lm-studio.lan", "192.168.1.9")])
def test_names_are_judged_by_what_they_resolve_to(host, ip):
    url = f"http://{host}:1234/v1"
    with pytest.raises(UnsafeUrl):
        check_url(url, allow_private=False, resolver=_resolver(ip))
    assert check_url(url, allow_private=True, resolver=_resolver(ip)) == url


def test_public_addresses_and_names_are_fine():
    assert check_url(f"http://{PUBLIC}/v1", allow_private=False)
    assert check_url("https://api.example.com/v1", allow_private=False, resolver=_resolver(PUBLIC))


def test_one_private_answer_among_public_ones_is_enough_to_refuse():
    """Otherwise a name with several A records can be steered at an internal one."""
    with pytest.raises(UnsafeUrl):
        check_url("http://mixed.example/v1", allow_private=False, resolver=_resolver(PUBLIC, "10.1.1.1"))


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/x", "gopher://example.com/", "javascript:alert(1)",
    "http:///nohost", "//example.com/v1", "", "   ", "not a url",
])
def test_only_http_and_https_with_a_host_are_accepted(url):
    with pytest.raises(UnsafeUrl):
        check_url(url, allow_private=True, resolver=_resolver(PUBLIC))


def test_credentials_in_the_url_are_refused():
    with pytest.raises(UnsafeUrl, match="credentials"):
        check_url("http://user:secret@example.com/v1", allow_private=True, resolver=_resolver(PUBLIC))


def test_https_only_rejects_plain_http():
    with pytest.raises(UnsafeUrl, match="https"):
        check_url("http://example.com/v1", allow_private=False, https_only=True, resolver=_resolver(PUBLIC))
    assert check_url("https://example.com/v1", allow_private=False, https_only=True, resolver=_resolver(PUBLIC))


def test_a_host_that_does_not_resolve_is_refused():
    def nothing(host, port):
        raise OSError("no such host")

    with pytest.raises(UnsafeUrl, match="resolve"):
        check_url("http://nope.invalid/v1", allow_private=True, resolver=nothing)


@pytest.mark.parametrize("url", ["http://2130706433/v1", "http://0x7f000001/v1", "http://127.1/v1", "http://0177.0.0.1/v1"])
def test_numeric_spellings_of_loopback_are_caught_by_the_real_resolver(url, monkeypatch):
    """Decimal, hex, short and octal forms of 127.0.0.1 are how filters get bypassed."""
    monkeypatch.setattr("pos.net_policy._resolve", REAL_RESOLVE)
    with pytest.raises(UnsafeUrl):
        check_url(url, allow_private=False)          # no fake resolver: the platform's own parsing decides


def test_the_error_names_the_reason_but_not_the_full_url_or_answers():
    with pytest.raises(UnsafeUrl) as info:
        check_url("http://10.0.0.5/secret-path?token=abc", allow_private=False)
    assert "secret-path" not in str(info.value) and "token" not in str(info.value)


def test_a_url_the_operator_configured_themselves_is_trusted(monkeypatch):
    monkeypatch.setenv("LOCAL_BASE_URL", "http://host.docker.internal:1234/v1")
    monkeypatch.setattr("pos.net_policy._resolve", lambda host, port: ["172.17.0.1"])
    from pos import config

    monkeypatch.setattr(config, "ALLOW_PRIVATE_LLM_URLS", False)

    same = "http://host.docker.internal:1234/v1"
    assert check_user_url("LOCAL_BASE_URL", same) == same                       # operator's own setting: fine
    with pytest.raises(UnsafeUrl):
        check_user_url("LOCAL_BASE_URL", "http://host.docker.internal:9999/v1")  # a user's different one: not
