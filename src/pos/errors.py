"""Turns provider failures into something worth reading.

A misconfigured key surfaces as a wall of SDK traceback ending in a URL and
a request id, which says nothing about what to change. These map the few
failures that actually happen -- wrong key, unknown deployment, wrong
endpoint, no quota -- onto a sentence naming the `.env` variable at fault.

Matching is on status code first and message text second, because the
exception *types* differ between the OpenAI SDK, LangChain's wrappers and
plain HTTP errors, while the status codes do not.
"""

from __future__ import annotations

from pos import config

# (status code, text that must appear, what to tell the user). Checked in
# order, so put the specific cases above the general ones.
_RULES: list[tuple[int | None, str | None, str]] = [
    (
        404,
        "deploymentnotfound",
        'No deployment named "{model}" exists on this Azure resource. '
        "Azure deployment names are chosen when the deployment is created and are "
        "often not the model's name -- check the portal, then set POS_MODEL in .env.",
    ),
    (
        404,
        None,
        'The provider could not find "{model}". On Azure this is a deployment name, '
        "not a model name -- check POS_MODEL in .env.",
    ),
    (
        401,
        None,
        "The provider rejected the API key. Check {key_var} in .env.",
    ),
    (
        403,
        None,
        "The provider refused the request. The key in {key_var} may lack access to "
        'the "{model}" deployment, or the resource may be restricted by network rules.',
    ),
    (
        429,
        "quota",
        "Out of quota for this deployment. Check its rate limits in the Azure portal.",
    ),
    (
        429,
        None,
        "Rate limited by the provider. Wait a moment and try again.",
    ),
    (
        None,
        "getaddrinfo failed",
        "Could not reach the provider. Check AZURE_OPENAI_ENDPOINT in .env, and that "
        "this machine is online.",
    ),
    (
        None,
        "api_key",
        "No API key was found. Set {key_var} in .env -- see .env.example.",
    ),
]


def _status_of(error: BaseException) -> int | None:
    """Digs the HTTP status out of whatever the SDK raised."""
    for attr in ("status_code", "code", "http_status"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def explain(error: BaseException, model: str | None = None) -> str:
    """Describes a provider failure in terms of what to change.

    Args:
        error: Whatever the call raised.
        model: The model or deployment being called, named in the message.

    Returns:
        A sentence for the user. Falls back to the original message when
        the failure isn't one of the known ones -- an unfamiliar error is
        better shown than swallowed.
    """
    spec = model or config.MODEL_NAME
    provider, name = config.split_model(spec)
    fields = {"model": name, "key_var": config.KEY_VARIABLES.get(provider, "the provider's API key")}

    status = _status_of(error)
    text = str(error).lower()

    for rule_status, needle, message in _RULES:
        if rule_status is not None and rule_status != status:
            continue
        if needle is not None and needle.lower() not in text:
            continue
        return message.format(**fields)

    original = str(error).strip() or type(error).__name__
    return original[:400]
