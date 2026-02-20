#!/usr/bin/env python3
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> int:
    smtp_host = require_env("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = require_env("SMTP_USERNAME")
    smtp_password = require_env("SMTP_PASSWORD")
    smtp_from = require_env("SMTP_FROM")
    smtp_to = require_env("SMTP_TO")
    smtp_security = os.getenv("SMTP_SECURITY", "starttls").strip().lower()

    msg = EmailMessage()
    msg["Subject"] = "Daily Hi"
    msg["From"] = smtp_from
    msg["To"] = smtp_to
    msg.set_content("hi")

    if smtp_security == "ssl":
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

    print("Email sent successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
