"""Transactional e-mail via Resend.

Uses Resend's REST endpoint directly rather than the `resend` SDK. `requests`
is already a dependency (see requirements-deploy.txt, trimmed to fit Render's
512 MB free tier); adding an SDK for one POST would put that back at risk for
no benefit.

NO-KEY BEHAVIOUR IS DELIBERATE. With no RESEND_API_KEY configured, messages are
appended to a JSONL outbox instead of being sent, and the call still succeeds.
That keeps local development and the test-suite working with no network, and
makes "was the mail sent, and what did it say?" a checkable fact rather than a
log line. It never silently drops mail.

SENDING MUST NOT BREAK THE CALLER. A registration that hashed the password,
wrote the user and then raised because an SMTP provider was briefly down would
leave an account nobody can verify. Send failures are logged and reported to
the caller as a boolean; the surrounding request decides what to do.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..core.config import Settings

logger = logging.getLogger(__name__)

RESEND_ENDPOINT = "https://api.resend.com/emails"
_TIMEOUT_SECONDS = 10


class MailService:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.resend_api_key)

    def send(self, *, to: str, subject: str, html: str, text: str = "") -> bool:
        """Returns True when the message was accepted (or written to the outbox)."""
        if not self.enabled:
            return self._write_outbox(to=to, subject=subject, html=html, text=text,
                                      reason="no RESEND_API_KEY configured")

        payload: dict[str, Any] = {
            "from": self.settings.mail_from,
            "to": [to],
            "subject": subject,
            "html": html,
        }
        if text:
            payload["text"] = text
        if self.settings.mail_reply_to:
            payload["reply_to"] = self.settings.mail_reply_to

        try:
            response = requests.post(
                RESEND_ENDPOINT,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.settings.resend_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            logger.error("Resend request failed for %s: %s", _mask(to), exc)
            self._write_outbox(to=to, subject=subject, html=html, text=text,
                               reason=f"transport error: {exc}")
            return False

        if response.status_code >= 400:
            # Body is logged because Resend's 4xx messages are the only way to
            # diagnose a bad sender domain, and it never contains the API key.
            logger.error(
                "Resend rejected mail to %s: %s %s",
                _mask(to), response.status_code, response.text[:400],
            )
            self._write_outbox(to=to, subject=subject, html=html, text=text,
                               reason=f"rejected {response.status_code}")
            return False

        logger.info("Sent %r to %s", subject, _mask(to))
        return True

    def _write_outbox(self, *, to: str, subject: str, html: str, text: str, reason: str) -> bool:
        try:
            path = Path(self.settings.mail_outbox_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "at": datetime.now(timezone.utc).isoformat(),
                    "to": to,
                    "subject": subject,
                    "reason": reason,
                    "text": text,
                    "html": html,
                }) + "\n")
            logger.warning("Mail to %s written to outbox (%s)", _mask(to), reason)
            return True
        except OSError as exc:
            logger.error("Could not write mail outbox: %s", exc)
            return False


def _mask(address: str) -> str:
    """Log addresses partially. Audit records hold the full value; application
    logs are read far more casually and do not need it."""
    local, _, domain = address.partition("@")
    if not domain:
        return "***"
    head = local[:2] if len(local) > 2 else local[:1]
    return f"{head}***@{domain}"
