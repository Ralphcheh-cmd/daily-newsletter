#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import smtplib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


USER_AGENT = "daily-medtech-headline-bot/1.0 (contact: recdosec@gmail.com)"
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
DEFAULT_QUERY = 'medtech OR biomedical OR "medical device"'
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


@dataclass
class NewsItem:
    title: str
    url: str
    domain: str
    published_at: datetime
    source_type: str


@dataclass
class HeadlineCluster:
    items: list[NewsItem] = field(default_factory=list)
    tokens: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)
    source_types: set[str] = field(default_factory=set)

    def add(self, item: NewsItem, item_tokens: set[str]) -> None:
        self.items.append(item)
        if not self.tokens:
            self.tokens = set(item_tokens)
        else:
            self.tokens |= item_tokens
        self.domains.add(item.domain)
        self.source_types.add(item.source_type)

    def most_recent(self) -> datetime:
        return max(item.published_at for item in self.items)


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


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def fetch_google_news_items(query: str, max_items: int = 35) -> list[NewsItem]:
    url = (
        f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        xml_text = fetch_text(url)
        root = ET.fromstring(xml_text)
    except Exception as exc:
        print(f"Warning: Google News fetch failed: {exc}")
        return []

    rows: list[NewsItem] = []
    for item_el in root.findall("./channel/item"):
        raw_title = (item_el.findtext("title") or "").strip()
        link = (item_el.findtext("link") or "").strip()
        source = (item_el.findtext("source") or "").strip() or "Google News"
        pub_date = safe_datetime(item_el.findtext("pubDate"))
        if not raw_title or not link:
            continue

        rows.append(
            NewsItem(
                title=strip_source_suffix(raw_title, source),
                url=link,
                domain=source,
                published_at=pub_date,
                source_type="google_news",
            )
        )
    return rows[:max_items]


def fetch_gdelt_items(query: str, max_items: int = 45) -> list[NewsItem]:
    url = (
        f"{GDELT_DOC_API}?query={quote_plus(query)}&mode=ArtList"
        f"&format=json&sort=DateDesc&maxrecords={max_items}"
    )
    try:
        payload = json.loads(fetch_text(url))
    except Exception as exc:
        print(f"Warning: GDELT fetch failed: {exc}")
        return []
    rows: list[NewsItem] = []
    for article in payload.get("articles", []):
        title = (article.get("title") or "").strip()
        link = (article.get("url") or "").strip()
        domain = (article.get("domain") or "").strip()
        if not domain and link:
            domain = urlparse(link).netloc or "web"
        if not title or not link:
            continue
        rows.append(
            NewsItem(
                title=title,
                url=link,
                domain=domain or "web",
                published_at=safe_datetime(article.get("seendate")),
                source_type="gdelt",
            )
        )
    return rows


def cluster_headlines(items: Iterable[NewsItem]) -> list[HeadlineCluster]:
    clusters: list[HeadlineCluster] = []
    for item in sorted(items, key=lambda row: row.published_at, reverse=True):
        item_tokens = normalize_tokens(item.title)
        if not item_tokens:
            continue

        best_idx = -1
        best_sim = 0.0
        for idx, cluster in enumerate(clusters):
            sim = jaccard_similarity(item_tokens, cluster.tokens)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx >= 0 and best_sim >= 0.52:
            clusters[best_idx].add(item, item_tokens)
        else:
            new_cluster = HeadlineCluster()
            new_cluster.add(item, item_tokens)
            clusters.append(new_cluster)
    return clusters


def cluster_score(cluster: HeadlineCluster) -> float:
    now = datetime.now(timezone.utc)
    recency_hours = max(
        0.0, (now - cluster.most_recent()).total_seconds() / 3600.0
    )
    recency_factor = max(0.0, (72.0 - recency_hours) / 72.0)
    mentions = len(cluster.items)
    unique_domains = len(cluster.domains)
    cross_source_bonus = 2.0 if len(cluster.source_types) >= 2 else 0.0
    return mentions * 3.0 + unique_domains * 2.0 + cross_source_bonus + recency_factor


def pick_representative(cluster: HeadlineCluster) -> NewsItem:
    source_priority = {"gdelt": 2, "google_news": 1}
    return sorted(
        cluster.items,
        key=lambda row: (
            source_priority.get(row.source_type, 0),
            row.published_at,
        ),
        reverse=True,
    )[0]


def build_top_headline_message() -> tuple[str, str]:
    query = os.getenv("TOP_NEWS_QUERY", DEFAULT_QUERY).strip() or DEFAULT_QUERY
    google_items = fetch_google_news_items(query=query)
    gdelt_items = fetch_gdelt_items(query=query)
    all_items = google_items + gdelt_items

    clusters = cluster_headlines(all_items)
    if not clusters:
        subject = f"Daily MedTech Top Headline | {datetime.now(timezone.utc).date().isoformat()}"
        body = (
            "No biomedical/medtech headline could be confidently selected today.\n"
            "Sources checked: Google News RSS, GDELT.\n"
        )
        return subject, body

    best_cluster = sorted(
        clusters,
        key=lambda c: (cluster_score(c), c.most_recent()),
        reverse=True,
    )[0]
    top = pick_representative(best_cluster)
    cross_ref_status = (
        "Cross-referenced across multiple sources"
        if len(best_cluster.source_types) >= 2 or len(best_cluster.domains) >= 2
        else "Single-source fallback"
    )

    unique_links: list[str] = []
    seen_links: set[str] = set()
    for item in sorted(best_cluster.items, key=lambda row: row.published_at, reverse=True):
        if item.url in seen_links:
            continue
        seen_links.add(item.url)
        unique_links.append(f"- {item.domain}: {item.url}")
        if len(unique_links) >= 5:
            break

    subject = f"Daily MedTech Top Headline | {datetime.now(timezone.utc).date().isoformat()}"
    body = (
        f"Top Headline:\n{top.title}\n\n"
        f"Primary Link:\n{top.url}\n\n"
        f"Selection Notes:\n"
        f"- Status: {cross_ref_status}\n"
        f"- Matching headlines found: {len(best_cluster.items)}\n"
        f"- Unique domains: {len(best_cluster.domains)}\n"
        f"- Sources used: {', '.join(sorted(best_cluster.source_types))}\n\n"
        "Other matching links:\n"
        f"{os.linesep.join(unique_links)}\n\n"
        f"Generated at (UTC): {datetime.now(timezone.utc).isoformat()}\n"
    )
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
