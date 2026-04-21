"""
Pluggable email service — Phase 16d.

Wraps the transactional email concerns behind a small interface so the team-
invite flow (and future transactional emails) can swap between a real SMTP
sender in production and a no-op console logger in development.

Selection rule (see ``get_email_service``):
    * If ``SMTP_HOST`` is set and credentials are present, use SMTPEmailService.
    * Otherwise fall back to ConsoleEmailService, which just logs the message.

Kept in its own module separate from ``app/services/email.py`` so the older
email helpers (verification code, password reset) can keep working during
migration while the new invite flow routes through this service.
"""

from __future__ import annotations

import abc
import html
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Optional

import aiosmtplib
from fastapi import Depends
from loguru import logger

from app.settings import settings


@dataclass(frozen=True)
class EmailMessageSpec:
    """Lightweight envelope — keeps send call sites short."""
    to: str
    subject: str
    html_body: str
    from_email: Optional[str] = None
    from_name: Optional[str] = None


class EmailService(abc.ABC):
    """Minimal interface — both implementations must be awaitable."""

    @abc.abstractmethod
    async def send(self, message: EmailMessageSpec) -> None: ...

    async def send_invite(
        self,
        *,
        to: str,
        token: str,
        owner_name: str,
        team_name: str,
        note: Optional[str],
        frontend_url: str,
        role: str,
    ) -> None:
        """
        Render a simple invite HTML + send. The frontend URL is required so
        callers can point at either prod (https://merodafa.com) or a preview
        deploy; we don't hard-code it.
        """
        subject = f"{owner_name} invited you to join {team_name} on MeroDafa"
        accept_url = f"{frontend_url.rstrip('/')}/invite/accept?token={token}"
        body = _render_invite_html(
            owner_name=owner_name,
            team_name=team_name,
            note=note,
            accept_url=accept_url,
            role=role,
            to=to,
        )
        await self.send(EmailMessageSpec(to=to, subject=subject, html_body=body))


class SMTPEmailService(EmailService):
    """Production sender — talks to Zoho (or any SMTP) over TLS."""

    async def send(self, message: EmailMessageSpec) -> None:
        if not settings.smtp_username or not settings.smtp_password:
            # Shouldn't hit this in prod (we only pick SMTPEmailService when
            # creds are set) but guard anyway.
            logger.warning(
                "SMTPEmailService.send: credentials missing; dropping message"
            )
            return

        msg = EmailMessage()
        from_email = message.from_email or settings.smtp_from_email
        from_name = message.from_name or settings.smtp_from_name
        msg["From"] = f"{from_name} <{from_email}>"
        msg["To"] = message.to
        msg["Subject"] = message.subject
        msg.set_content(message.html_body, subtype="html")

        try:
            await aiosmtplib.send(
                msg,
                hostname=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_username,
                password=settings.smtp_password,
                start_tls=True,
            )
            logger.info(f"Email sent to {message.to}: {message.subject}")
        except Exception as exc:
            # Caller decides whether a send failure is fatal. The invite flow
            # treats it as non-fatal so the invite row still persists.
            logger.error(f"Email send failed: {exc}")
            raise


class ConsoleEmailService(EmailService):
    """Dev fallback — logs the message so you can click the accept link from a
    terminal during local development without needing SMTP credentials."""

    async def send(self, message: EmailMessageSpec) -> None:
        logger.info(
            "── CONSOLE EMAIL ──\n"
            f"To: {message.to}\n"
            f"Subject: {message.subject}\n\n"
            f"{message.html_body}\n"
            "── END ──"
        )


# ── Dependency wiring ──────────────────────────────────────────────────────

_singleton: EmailService | None = None


def get_email_service() -> EmailService:
    """
    FastAPI dependency + module-global getter. Reads SMTP credentials at first
    call and caches the chosen implementation for the life of the process.
    """
    global _singleton
    if _singleton is not None:
        return _singleton
    if settings.smtp_host and settings.smtp_username and settings.smtp_password:
        _singleton = SMTPEmailService()
        logger.info("EmailService: using SMTPEmailService")
    else:
        _singleton = ConsoleEmailService()
        logger.info("EmailService: using ConsoleEmailService (no SMTP creds set)")
    return _singleton


def email_service_dep() -> EmailService:
    """FastAPI dependency wrapper — lets endpoints write Depends(email_service_dep)."""
    return get_email_service()


# ── HTML renderers ─────────────────────────────────────────────────────────

def _render_invite_html(
    *,
    owner_name: str,
    team_name: str,
    note: Optional[str],
    accept_url: str,
    role: str,
    to: str,
) -> str:
    """
    f-string template — plain HTML, inline styles so Gmail / Outlook render it
    consistently. Every variable is `html.escape`d to avoid breaking the tags
    if a user name or note contains `<`.
    """
    safe_owner = html.escape(owner_name or "Someone")
    safe_team = html.escape(team_name or "a MeroDafa team")
    safe_role = html.escape(role)
    safe_to = html.escape(to)
    safe_note = f"<p style=\"color:#4a5568;font-size:14px;margin:16px 0\"><em>“{html.escape(note)}”</em></p>" if note else ""
    # Don't escape accept_url — it's a URL we generated. It's already URL-safe
    # because the token comes from `secrets.token_urlsafe`.
    return (
        "<!DOCTYPE html>"
        "<html><body style=\"font-family:system-ui,-apple-system,Segoe UI,sans-serif;"
        "background:#f7fafc;margin:0;padding:24px;\">"
        "<table width=\"100%\" cellspacing=\"0\" cellpadding=\"0\"><tr><td align=\"center\">"
        "<table width=\"560\" cellspacing=\"0\" cellpadding=\"0\" style=\"background:#fff;"
        "border-radius:12px;padding:40px;border:1px solid #e2e8f0;\">"
        "<tr><td>"
        "<h1 style=\"font-family:Georgia,serif;font-size:22px;color:#0f172a;margin:0 0 8px;\">"
        "You're invited to MeroDafa</h1>"
        f"<p style=\"color:#4a5568;font-size:14px;line-height:1.6;margin:0 0 16px;\">"
        f"Hi {safe_to},<br><br>"
        f"<strong>{safe_owner}</strong> has invited you to join <strong>{safe_team}</strong> "
        f"on MeroDafa as a <strong>{safe_role}</strong>. Accept this invitation to start "
        "collaborating on legal research, shared pinned threads, and private documents.</p>"
        f"{safe_note}"
        "<div style=\"margin:24px 0;\">"
        f"<a href=\"{accept_url}\" style=\"display:inline-block;padding:12px 24px;"
        "background:#09383e;color:#ffffff;text-decoration:none;border-radius:8px;"
        "font-weight:600;font-size:14px;\">Accept invitation</a></div>"
        "<p style=\"color:#718096;font-size:12px;line-height:1.6;margin:16px 0 0;\">"
        f"Or copy this link: <a href=\"{accept_url}\" style=\"color:#09383e;\">{accept_url}</a>"
        "<br><br>This invite expires in 7 days. If you didn't expect this email you can "
        "safely ignore it.</p>"
        "</td></tr></table>"
        "</td></tr></table></body></html>"
    )


def _inject(dep=Depends(email_service_dep)) -> EmailService:  # pragma: no cover - alias for clarity
    return dep
