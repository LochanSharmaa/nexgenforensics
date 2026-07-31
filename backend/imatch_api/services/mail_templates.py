"""Responsive HTML e-mail templates, NexGen Forensics branding.

Table-based layout with inline styles, which looks dated but is what actually
renders in Outlook and Gmail -- both strip <style> blocks and neither supports
flexbox or grid reliably. Every message ships a plain-text alternative, because
a mail client showing only the HTML part is not a safe assumption and OTP codes
must stay readable.

All interpolated values pass through `esc()`. A display name is user-controlled
input, and an unescaped one would put arbitrary markup into an e-mail we send
under our own domain.
"""

from __future__ import annotations

from html import escape as _escape

BRAND = "NexGen Forensics"
_INK = "#24170f"
_MUTED = "#6b5c4f"
_CRIMSON = "#9a2f42"
_NAVY = "#1d3557"
_PAPER = "#fbfaf6"
_CARD = "#fffdf8"
_BORDER = "#e6ddcf"


def esc(value: str) -> str:
    return _escape(str(value or ""), quote=True)


def _shell(*, title: str, body: str, preheader: str) -> str:
    return f"""\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
</head>
<body style="margin:0;padding:0;background:{_PAPER};">
<div style="display:none;max-height:0;overflow:hidden;opacity:0;">{esc(preheader)}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{_PAPER};padding:28px 12px;">
<tr><td align="center">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:560px;">
    <tr><td style="padding:0 4px 18px;">
      <span style="font:600 13px/1 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
                   letter-spacing:.18em;text-transform:uppercase;color:{_CRIMSON};">{BRAND}</span>
    </td></tr>
    <tr><td style="background:{_CARD};border:1px solid {_BORDER};border-radius:16px;padding:34px 30px;">
      {body}
    </td></tr>
    <tr><td style="padding:18px 4px 0;">
      <p style="margin:0;font:400 12px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:{_MUTED};">
        This is an automated message from {BRAND}. Replies are not monitored.
      </p>
    </td></tr>
  </table>
</td></tr>
</table>
</body>
</html>"""


def _h1(text: str) -> str:
    return (f'<h1 style="margin:0 0 14px;font:600 22px/1.25 -apple-system,Segoe UI,Roboto,'
            f'Helvetica,Arial,sans-serif;color:{_INK};">{esc(text)}</h1>')


def _p(text: str, *, muted: bool = False) -> str:
    colour = _MUTED if muted else _INK
    return (f'<p style="margin:0 0 14px;font:400 15px/1.65 -apple-system,Segoe UI,Roboto,'
            f'Helvetica,Arial,sans-serif;color:{colour};">{text}</p>')


def _code_block(code: str) -> str:
    return f"""\
<table role="presentation" cellpadding="0" cellspacing="0" style="margin:22px 0;">
<tr><td style="background:{_PAPER};border:1px solid {_BORDER};border-radius:12px;padding:18px 26px;">
  <span style="font:700 32px/1 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
               letter-spacing:.28em;color:{_NAVY};">{esc(code)}</span>
</td></tr></table>"""


def _button(url: str, label: str) -> str:
    return f"""\
<table role="presentation" cellpadding="0" cellspacing="0" style="margin:22px 0;">
<tr><td style="background:{_CRIMSON};border-radius:10px;">
  <a href="{esc(url)}" style="display:inline-block;padding:13px 26px;font:600 15px/1 -apple-system,
     Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#fff;text-decoration:none;">{esc(label)}</a>
</td></tr></table>
<p style="margin:0 0 14px;font:400 12px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
   color:{_MUTED};word-break:break-all;">If the button does not work, paste this into your browser:<br>{esc(url)}</p>"""


def _ignore(action: str) -> str:
    return _p(f"If you did not {esc(action)}, you can ignore this e-mail and nothing will change.",
              muted=True)


# ------------------------------------------------------------------ verify --

def verify_email(*, name: str, otp: str, ttl_minutes: int) -> tuple[str, str, str]:
    subject = f"Verify your {BRAND} Account"
    body = (
        _h1(f"Welcome, {esc(name or 'there')}")
        + _p("Use this code to verify your e-mail address and activate your account:")
        + _code_block(otp)
        + _p(f"This code expires in <strong>{ttl_minutes} minutes</strong> and can be used once.")
        + _ignore("create this account")
    )
    text = (f"Welcome, {name or 'there'}\n\n"
            f"Your {BRAND} verification code is: {otp}\n\n"
            f"It expires in {ttl_minutes} minutes and can be used once.\n\n"
            "If you did not create this account, ignore this e-mail.\n")
    return subject, _shell(title=subject, body=body, preheader=f"Your code is {otp}"), text


def resend_otp(*, name: str, otp: str, ttl_minutes: int) -> tuple[str, str, str]:
    subject = f"Your new {BRAND} verification code"
    body = (
        _h1("Here is your new code")
        + _p("You asked for another verification code. Any previous code is now invalid.")
        + _code_block(otp)
        + _p(f"This code expires in <strong>{ttl_minutes} minutes</strong>.")
        + _ignore("request this code")
    )
    text = (f"Hello {name or 'there'}\n\nYour new {BRAND} verification code is: {otp}\n\n"
            f"It expires in {ttl_minutes} minutes. Any previous code is now invalid.\n")
    return subject, _shell(title=subject, body=body, preheader=f"Your new code is {otp}"), text


# ------------------------------------------------------------------- reset --

def forgot_password(*, name: str, reset_url: str, ttl_minutes: int) -> tuple[str, str, str]:
    subject = f"Reset your {BRAND} password"
    body = (
        _h1("Reset your password")
        + _p(f"Hello {esc(name or 'there')}, we received a request to reset your password.")
        + _button(reset_url, "Choose a new password")
        + _p(f"This link expires in <strong>{ttl_minutes} minutes</strong> and can be used once.")
        + _ignore("request a password reset")
    )
    text = (f"Hello {name or 'there'}\n\nReset your {BRAND} password:\n{reset_url}\n\n"
            f"This link expires in {ttl_minutes} minutes and can be used once.\n\n"
            "If you did not request this, ignore this e-mail.\n")
    return subject, _shell(title=subject, body=body, preheader="Password reset link"), text


def password_changed(*, name: str, ip_address: str = "") -> tuple[str, str, str]:
    subject = f"Your {BRAND} password was changed"
    where = f" from {esc(ip_address)}" if ip_address else ""
    body = (
        _h1("Your password was changed")
        + _p(f"Hello {esc(name or 'there')}, your password was just changed{where}.")
        + _p("You have been signed out everywhere and will need to sign in again.")
        + _p("<strong>If this was not you, contact your administrator immediately.</strong>")
    )
    text = (f"Hello {name or 'there'}\n\nYour {BRAND} password was changed{where}.\n"
            "You have been signed out everywhere.\n\n"
            "If this was not you, contact your administrator immediately.\n")
    return subject, _shell(title=subject, body=body, preheader="Password changed"), text


def welcome(*, name: str, login_url: str) -> tuple[str, str, str]:
    subject = f"Your {BRAND} account is verified"
    body = (
        _h1("You're verified")
        + _p(f"Thanks {esc(name or 'there')} — your e-mail is confirmed and your account is active.")
        + _button(login_url, "Sign in")
        + _p("Every biometric operation you perform is recorded against your account.", muted=True)
    )
    text = (f"Thanks {name or 'there'} - your {BRAND} account is verified.\n\n"
            f"Sign in: {login_url}\n\n"
            "Every biometric operation you perform is recorded against your account.\n")
    return subject, _shell(title=subject, body=body, preheader="Account verified"), text
