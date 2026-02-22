#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import smtplib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from urllib.parse import quote, quote_plus
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

ACTION_VERBS = [
    "announces",
    "launches",
    "wins",
    "receives",
    "acquires",
    "buys",
    "invests",
    "resolves",
    "posts",
    "reports",
    "partners",
    "raises",
    "outlines",
    "hits",
    "doses",
]

ROLE_TOKENS = {"CTO", "CEO", "CFO", "COO", "CMO", "President"}


@dataclass
class NewsItem:
    title: str
    url: str
    outlet_name: str
    outlet_domain: str
    published_at: datetime
    score: float = 0.0


_COMPANY_PROFILE_CACHE: dict[str, str] = {}
_WIKIPEDIA_SUMMARY_CACHE: dict[str, dict[str, str] | None] = {}


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


def to_roman(number: int) -> str:
    values = {1: "I", 2: "II", 3: "III", 4: "IV"}
    return values.get(number, str(number))


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


def clean_company_name(name: str) -> str:
    value = re.sub(r"\s+", " ", name).strip(" -,:;.")
    parts = value.split()
    while parts and parts[-1] in ROLE_TOKENS:
        parts.pop()
    cleaned = " ".join(parts).strip()
    return cleaned or value


def infer_company_name(title: str) -> str:
    possessive_match = re.search(
        r"^([A-Za-z][A-Za-z0-9&.\-]*(?:\s+[A-Za-z][A-Za-z0-9&.\-]*){0,4})[’']s\b",
        title,
    )
    if possessive_match:
        return clean_company_name(possessive_match.group(1))

    verbs = "|".join(ACTION_VERBS)
    action_match = re.search(
        rf"^([A-Za-z][A-Za-z0-9&.\-]*(?:\s+[A-Za-z][A-Za-z0-9&.\-]*){{0,4}})\s+(?:{verbs})\b",
        title,
        flags=re.IGNORECASE,
    )
    if action_match:
        return clean_company_name(action_match.group(1))

    # Fallback: first capitalized phrase.
    fallback = re.search(
        r"([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,3})",
        title,
    )
    if fallback:
        return clean_company_name(fallback.group(1))
    return "The company"


def infer_product_name(title: str, company_name: str) -> str | None:
    clean_title = title
    if company_name and company_name != "The company":
        clean_title = re.sub(re.escape(company_name), "", clean_title, flags=re.IGNORECASE).strip()
    clean_title = clean_title.replace("’s", "").replace("'s", "")

    for pattern in [
        r"(?:for|with|using|on)\s+([A-Za-z0-9][^,;:.]{3,90})",
        r"([A-Za-z0-9][A-Za-z0-9\-\s]{2,60}\s+device)",
        r"([A-Za-z0-9][A-Za-z0-9\-\s]{2,60}\s+therapy)",
        r"([A-Za-z0-9][A-Za-z0-9\-\s]{2,60}\s+platform)",
    ]:
        match = re.search(pattern, clean_title, flags=re.IGNORECASE)
        if match:
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" -,:;.")
            candidate = re.sub(
                r"^(?:announces?|launches?|wins?|receives?|acquires?|buys?|invests?|"
                r"resolves?|posts?|reports?|partners?|raises?|outlines?|hits?|doses?)\s+",
                "",
                candidate,
                flags=re.IGNORECASE,
            )
            candidate = re.sub(r"^(?:new|its|their)\s+", "", candidate, flags=re.IGNORECASE)
            candidate = candidate.strip(" -,:;.")
            lowered = candidate.lower()
            token_count = len(candidate.split())
            noisy_tokens = {
                "success",
                "strategy",
                "relationship",
                "capabilities",
                "quarter",
                "global",
                "firsts",
            }
            if (
                token_count >= 2
                and token_count <= 8
                and not any(token in lowered for token in noisy_tokens)
            ):
                return candidate
    return None


def parse_json(text: str) -> dict | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def first_sentences(text: str, count: int = 2) -> str:
    cleaned = normalize_spaces(text.replace("\n", " "))
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", cleaned) if chunk.strip()]
    if not chunks:
        return ""
    return " ".join(chunks[:count]).strip()


def wikipedia_summary(term: str) -> dict[str, str] | None:
    key = term.strip().lower()
    if not key:
        return None
    if key in _WIKIPEDIA_SUMMARY_CACHE:
        return _WIKIPEDIA_SUMMARY_CACHE[key]

    def fetch_summary_direct(title: str) -> dict[str, str] | None:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
        try:
            payload = parse_json(fetch_text(url))
        except Exception:
            return None
        if not payload:
            return None
        if "extract" not in payload:
            return None
        extract = normalize_spaces(str(payload.get("extract", "")))
        if not extract:
            return None
        page_url = (
            payload.get("content_urls", {})
            .get("desktop", {})
            .get("page", "")
        )
        return {
            "title": str(payload.get("title", title)),
            "extract": extract,
            "url": str(page_url),
        }

    direct = fetch_summary_direct(term)
    if direct:
        _WIKIPEDIA_SUMMARY_CACHE[key] = direct
        return direct

    search_url = (
        "https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&format=json&srlimit=1&srsearch={quote(term)}"
    )
    try:
        search_payload = parse_json(fetch_text(search_url))
    except Exception:
        search_payload = None

    best_title = (
        (search_payload or {})
        .get("query", {})
        .get("search", [{}])[0]
        .get("title")
    )
    if isinstance(best_title, str) and best_title.strip():
        looked_up = fetch_summary_direct(best_title.strip())
        _WIKIPEDIA_SUMMARY_CACHE[key] = looked_up
        return looked_up

    _WIKIPEDIA_SUMMARY_CACHE[key] = None
    return None


def build_company_research(company_name: str) -> str:
    if company_name in _COMPANY_PROFILE_CACHE:
        return _COMPANY_PROFILE_CACHE[company_name]

    if not company_name or company_name == "The company":
        fallback = (
            "The headline does not include a clear company name, so deeper profile lookup was skipped "
            "to avoid guessing."
        )
        _COMPANY_PROFILE_CACHE[company_name] = fallback
        return fallback

    summary = wikipedia_summary(company_name)
    if summary:
        detail = first_sentences(summary["extract"], count=2)
        if summary["url"]:
            detail = f"{detail} Source: {summary['url']}"
        _COMPANY_PROFILE_CACHE[company_name] = detail
        return detail

    fallback = (
        f"Public profile data for {company_name} was limited from open sources in this run. "
        "From the headline context, it appears active in medtech/biomedical innovation."
    )
    _COMPANY_PROFILE_CACHE[company_name] = fallback
    return fallback


def product_category(product_name: str, headline: str) -> str:
    value = f"{product_name} {headline}".lower()
    if any(token in value for token in ["robotic", "surgery", "surgical"]):
        return "robotic_surgery"
    if any(token in value for token in ["glucose", "sensor", "monitor", "wearable"]):
        return "monitoring"
    if any(token in value for token in ["imaging", "ultrasound", "mri", "ct", "scan"]):
        return "imaging"
    if any(token in value for token in ["stent", "valve", "implant", "pacemaker"]):
        return "implant"
    if any(token in value for token in ["diagnostic", "test", "assay", "sequencing"]):
        return "diagnostic"
    if any(token in value for token in ["therapy", "drug", "biologic", "gene", "treatment"]):
        return "therapy"
    return "general"


def product_explanations(product_name: str, headline: str, product_summary: str) -> tuple[str, str]:
    category = product_category(product_name, headline)
    display_name = product_name[:1].upper() + product_name[1:]

    easy_templates = {
        "robotic_surgery": (
            f"{display_name} helps surgeons perform procedures with steadier and more precise movements. "
            "Think of it like power steering for the surgeon's hands."
        ),
        "monitoring": (
            f"{display_name} tracks important body signals continuously so issues can be spotted early. "
            "Think of it like a live dashboard for the human body."
        ),
        "imaging": (
            f"{display_name} helps clinicians see inside the body without large incisions. "
            "Think of it like switching from a rough map to a high-resolution GPS view."
        ),
        "implant": (
            f"{display_name} works like a hardware upgrade placed inside the body to support a failing function. "
            "Think of it as replacing a worn mechanical part in a machine."
        ),
        "diagnostic": (
            f"{display_name} is used to detect disease-related signals so treatment decisions can be made faster. "
            "Think of it like a highly specific smoke detector for biology."
        ),
        "therapy": (
            f"{display_name} is designed to intervene directly in a disease process to improve outcomes. "
            "Think of it like fixing a control loop instead of only observing it."
        ),
        "general": (
            f"{display_name} is a medical technology designed to improve how clinicians diagnose, monitor, or treat patients. "
            "Think of it as an engineering tool that turns weak biological signals into useful clinical action."
        ),
    }
    technical_templates = {
        "robotic_surgery": (
            "Engineering view: this is a mechatronic control system where surgeon inputs are mapped to robotic "
            "end-effectors using kinematics, tremor filtering, and motion scaling for precision at the tissue interface."
        ),
        "monitoring": (
            "Engineering view: this follows a sensor pipeline: transduction of a physiological signal, analog/digital "
            "noise filtering, algorithmic estimation, then threshold-based clinical alerting."
        ),
        "imaging": (
            "Engineering view: the system captures energy interactions with tissue and reconstructs an image through "
            "signal processing, similar to solving an inverse problem from noisy measurements."
        ),
        "implant": (
            "Engineering view: this combines biocompatible materials with mechanical and sometimes electronic subsystems "
            "to restore function while maintaining stable long-term interaction with tissue."
        ),
        "diagnostic": (
            "Engineering view: the test converts a biological event into a measurable output signal, then applies "
            "calibration and decision thresholds to separate true disease signals from background noise."
        ),
        "therapy": (
            "Engineering view: the intervention targets a specific biological pathway and aims to shift system behavior "
            "toward a healthier equilibrium, similar to tuning a feedback controller."
        ),
        "general": (
            "Engineering view: treat it as a full system chain from sensing or intervention, through processing and control, "
            "to a measurable clinical endpoint."
        ),
    }

    easy = easy_templates[category]
    technical = technical_templates[category]
    if product_summary:
        summary_sentence = first_sentences(product_summary, count=1)
        easy = f"{summary_sentence} {easy}"
    return easy, technical


def build_product_research(product_name: str, headline: str) -> tuple[str, str, str]:
    summary = wikipedia_summary(f"{product_name} medical") or wikipedia_summary(product_name)
    summary_extract = summary["extract"] if summary else ""
    source_url = summary["url"] if summary else ""
    easy, technical = product_explanations(product_name, headline, summary_extract)
    return easy, technical, source_url


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
    lines: list[str] = ["General News:", ""]
    for index, item in enumerate(top_items, start=1):
        company_name = infer_company_name(item.title)
        product_name = infer_product_name(item.title, company_name)
        company_research = build_company_research(company_name)

        lines.append(f"{to_roman(index)} - {item.title}")
        lines.append(f"Company Research: {company_research}")
        if product_name:
            product_easy, product_technical, product_source = build_product_research(
                product_name, item.title
            )
            lines.append(f"Product Research (easy): {product_easy}")
            lines.append(f"Product Research (tech): {product_technical}")
            if product_source:
                lines.append(f"Product Source: {product_source}")
        lines.append(
            "Link / Date: "
            f"{item.url} | {item.published_at.strftime('%Y-%m-%d %H:%M UTC')}"
        )
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
