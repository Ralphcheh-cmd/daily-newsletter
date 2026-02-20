#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import smtplib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


USER_AGENT = "daily-medtech-headline-bot/1.0 (contact: recdosec@gmail.com)"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "over",
    "more",
    "after",
    "before",
    "under",
    "your",
    "world",
    "today",
    "report",
    "reports",
    "news",
}

KEYWORDS = {
    "medtech",
    "biomedical",
    "biotech",
    "medical",
    "device",
    "fda",
    "trial",
    "clinical",
    "diagnostic",
    "diagnostics",
    "therapeutic",
    "therapeutics",
    "imaging",
    "surgical",
    "robotic",
}


@dataclass(frozen=True)
class Outlet:
    name: str
    domain: str
    weight: float


OUTLETS: list[Outlet] = [
    Outlet(name="MedTech Dive", domain="medtechdive.com", weight=1.25),
    Outlet(name="MassDevice", domain="massdevice.com", weight=1.20),
    Outlet(name="Fierce Biotech", domain="fiercebiotech.com", weight=1.15),
    Outlet(name="STAT", domain="statnews.com", weight=1.10),
    Outlet(name="MedCity News", domain="medcitynews.com", weight=1.05),
]


@dataclass
class NewsItem:
    title: str
    url: str
    outlet_name: str
    outlet_domain: str
    published_at: datetime
    score: float = 0.0


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=25) as response:
        return response.read().decode("utf-8", errors="replace")


def safe_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    raw = value.strip()
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def strip_source_suffix(title: str, source: str) -> str:
    clean_title = title.strip()
    clean_source = source.strip()
    suffix = f" - {clean_source}"
    if clean_source and clean_title.endswith(suffix):
        return clean_title[: -len(suffix)].strip()
    return clean_title


def normalize_tokens(text: str) -> set[str]:
    value = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = [
        token
        for token in value.split()
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    ]
    return set(tokens)


def normalize_title_key(title: str) -> str:
    value = re.sub(r"[^a-z0-9\s]", " ", title.lower())
    value = re.sub(r"\s+", " ", value).strip()
    tokens = [tok for tok in value.split() if tok not in STOPWORDS]
    return " ".join(tokens[:16])


def relevance_score(title: str) -> float:
    tokens = normalize_tokens(title)
    hits = sum(1 for token in tokens if token in KEYWORDS)
    return min(4.0, hits * 0.8)


def recency_score(published_at: datetime) -> float:
    age_hours = max(
        0.0, (datetime.now(timezone.utc) - published_at).total_seconds() / 3600.0
    )
    return max(0.0, (96.0 - age_hours) / 24.0)


def build_outlet_query(outlet: Outlet) -> str:
    return (
        f'site:{outlet.domain} '
        '("medical device" OR medtech OR biomedical OR biotech)'
    )


def fetch_outlet_items(outlet: Outlet, max_items: int = 8) -> list[NewsItem]:
    query = build_outlet_query(outlet)
    url = f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        xml_text = fetch_text(url)
        root = ET.fromstring(xml_text)
    except Exception as exc:
        print(f"Warning: {outlet.name} fetch failed: {exc}")
        return []

    rows: list[NewsItem] = []
    for item_el in root.findall("./channel/item"):
        raw_title = (item_el.findtext("title") or "").strip()
        link = (item_el.findtext("link") or "").strip()
        source = (item_el.findtext("source") or "").strip() or outlet.name
        pub_date = safe_datetime(item_el.findtext("pubDate"))
        if not raw_title or not link:
            continue

        title = strip_source_suffix(raw_title, source)
        item = NewsItem(
            title=title,
            url=link,
            outlet_name=outlet.name,
            outlet_domain=outlet.domain,
            published_at=pub_date,
        )
        item.score = outlet.weight + relevance_score(title) + recency_score(pub_date)
        rows.append(
            item
        )
    return sorted(rows, key=lambda row: row.score, reverse=True)[:max_items]


def desired_headline_count() -> int:
    raw = os.getenv("HEADLINE_COUNT", "4").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 4
    if value < 3:
        return 3
    if value > 4:
        return 4
    return value


def collect_top_headlines() -> list[NewsItem]:
    collected: list[NewsItem] = []
    for outlet in OUTLETS:
        collected.extend(fetch_outlet_items(outlet))

    deduped: list[NewsItem] = []
    seen: set[str] = set()
    for item in sorted(collected, key=lambda row: (row.score, row.published_at), reverse=True):
        key = normalize_title_key(item.title)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped[: desired_headline_count()]


def build_top_headline_message() -> tuple[str, str]:
    top_items = collect_top_headlines()
    if not top_items:
        subject = (
            f"Daily MedTech/Biomedical Headlines | "
            f"{datetime.now(timezone.utc).date().isoformat()}"
        )
        body = (
            "No headlines were found today from the selected medtech/biomedical outlets.\n"
            "Checked outlets: "
            + ", ".join(outlet.name for outlet in OUTLETS)
            + "\n"
        )
        return subject, body

    subject = (
        f"Top {len(top_items)} MedTech/Biomedical Headlines | "
        f"{datetime.now(timezone.utc).date().isoformat()}"
    )
    lines: list[str] = []
    lines.append(
        f"Top {len(top_items)} headlines from selected outlets "
        f"({', '.join(outlet.name for outlet in OUTLETS)}):"
    )
    lines.append("")
    for index, item in enumerate(top_items, start=1):
        lines.append(f"{index}. {item.title}")
        lines.append(f"   Outlet: {item.outlet_name}")
        lines.append(f"   Published (UTC): {item.published_at.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"   Link: {item.url}")
        lines.append("")
    lines.append(f"Generated at (UTC): {datetime.now(timezone.utc).isoformat()}")
    body = "\n".join(lines).strip() + "\n"
    return subject, body


def main() -> int:
    smtp_host = require_env("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = require_env("SMTP_USERNAME")
    smtp_password = require_env("SMTP_PASSWORD")
    smtp_from = require_env("SMTP_FROM")
    smtp_to = require_env("SMTP_TO")
    smtp_security = os.getenv("SMTP_SECURITY", "starttls").strip().lower()
    subject, body = build_top_headline_message()

    if os.getenv("DRY_RUN", "").strip().lower() in {"1", "true", "yes"}:
        print("Subject:", subject)
        print("")
        print(body)
        return 0

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = smtp_to
    msg.set_content(body)

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

    print("Top headline email sent successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
