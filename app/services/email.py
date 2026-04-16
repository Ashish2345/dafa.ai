"""
Email service — sends transactional emails via Zoho SMTP (aiosmtplib).

Used for:
  - Email verification (6-digit OTP on signup)
  - Password reset (6-digit OTP)
"""

from email.message import EmailMessage

import aiosmtplib
from loguru import logger

from app.settings import settings
from app.services.email_templates import password_reset_email, verification_email


async def _send(to: str, subject: str, html_body: str) -> None:
    """Low-level send via SMTP. Raises on failure."""
    if not settings.smtp_username or not settings.smtp_password:
        logger.warning("SMTP not configured — skipping email send")
        return

    msg = EmailMessage()
    msg["From"] = f"{settings.smtp_from_name} <{settings.smtp_from_email}>"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(html_body, subtype="html")

    await aiosmtplib.send(
        msg,
        hostname=settings.smtp_host,
        port=settings.smtp_port,
        username=settings.smtp_username,
        password=settings.smtp_password,
        start_tls=True,
    )
    logger.info(f"Email sent to {to}: {subject}")


async def send_verification_code(to: str, code: str, name: str = "") -> None:
    """Send 6-digit email verification code."""
    await _send(
        to=to,
        subject=f"{code} — Verify your MeroDafa account",
        html_body=verification_email(
            code=code,
            name=name,
            expire_minutes=settings.verification_code_expire_minutes,
        ),
    )


async def send_password_reset_code(to: str, code: str) -> None:
    """Send 6-digit password reset code."""
    await _send(
        to=to,
        subject=f"{code} — Reset your MeroDafa password",
        html_body=password_reset_email(
            code=code,
            expire_minutes=settings.verification_code_expire_minutes,
        ),
    )
