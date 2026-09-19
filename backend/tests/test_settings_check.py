import base64
import os

from pos.settings_check import check_settings

_GOOD = {
    "JWT_SECRET": "j" * 40,
    "ENCRYPTION_KEY": base64.b64encode(os.urandom(32)).decode(),
    "COOKIE_SECURE": "false",
    "CORS_ORIGINS": "http://localhost:3000",
}


def _check(**overrides):
    return check_settings({**_GOOD, **overrides})


def test_a_good_local_configuration_has_no_errors_or_warnings():
    assert _check() == ([], [])


def test_a_short_or_missing_jwt_secret_is_fatal():
    errors, _ = _check(JWT_SECRET="short")
    assert any("JWT_SECRET" in e for e in errors)
    errors, _ = _check(JWT_SECRET="")
    assert any("JWT_SECRET" in e for e in errors)


def test_the_encryption_key_must_be_base64_for_exactly_32_bytes():
    for bad in ("", "not base64 !!", base64.b64encode(b"x" * 16).decode()):
        errors, _ = _check(ENCRYPTION_KEY=bad)
        assert any("ENCRYPTION_KEY" in e for e in errors), bad


def test_an_https_origin_without_secure_cookies_is_fatal():
    errors, _ = _check(CORS_ORIGINS="https://app.example.com", COOKIE_SECURE="false")
    assert any("COOKIE_SECURE" in e for e in errors)
    assert _check(CORS_ORIGINS="https://app.example.com", COOKIE_SECURE="true") == ([], [])


def test_a_plain_http_non_local_origin_only_warns():
    errors, warnings = _check(CORS_ORIGINS="http://192.168.1.20:3000")
    assert errors == []
    assert any("COOKIE_SECURE" in w for w in warnings)


def test_localhost_variants_do_not_warn():
    for origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
        assert _check(CORS_ORIGINS=origin) == ([], [])


def test_messages_never_contain_the_secret_values():
    errors, warnings = _check(JWT_SECRET="tiny-secret-value", ENCRYPTION_KEY="key-value-xyz")
    assert all("tiny-secret-value" not in m and "key-value-xyz" not in m for m in errors + warnings)
