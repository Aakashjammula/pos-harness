"""Magic-link email delivery via fastapi-mail.

Config here reads MAIL_TLS/MAIL_SSL (this deployment's env var names) and
translates them to the library's own MAIL_STARTTLS/MAIL_SSL_TLS -- the
library renamed these between versions; keeping our env var names stable
avoids a breaking config change if it renames them again."""

from __future__ import annotations

from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType

from pos import config


def _connection_config() -> ConnectionConfig:
    return ConnectionConfig(
        MAIL_USERNAME=config.MAIL_USERNAME,
        MAIL_PASSWORD=config.MAIL_PASSWORD,
        MAIL_FROM=config.MAIL_FROM,
        MAIL_PORT=config.MAIL_PORT,
        MAIL_SERVER=config.MAIL_SERVER,
        MAIL_STARTTLS=config.MAIL_TLS,
        MAIL_SSL_TLS=config.MAIL_SSL,
        USE_CREDENTIALS=bool(config.MAIL_USERNAME),
        VALIDATE_CERTS=True,
    )


async def send_magic_link_email(email: str, link_url: str) -> None:
    message = MessageSchema(
        subject="Sign in to POS",
        recipients=[email],
        body=(
            "<p>Click below to sign in. This link works once and expires in 15 minutes.</p>"
            f'<p><a href="{link_url}">{link_url}</a></p>'
            "<p>If you didn't request this, you can ignore this email.</p>"
        ),
        subtype=MessageType.html,
    )
    await FastMail(_connection_config()).send_message(message)
