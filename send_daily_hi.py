#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import smtplib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from urllib.parse import quote, quote_plus, urlencode
from urllib.request import Request, urlopen


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


USER_AGENT = "daily-medtech-headline-bot/1.0 (contact: recdosec@gmail.com)"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
OPENFDA_PMA_API = "https://api.fda.gov/device/pma.json"
OPENFDA_510K_API = "https://api.fda.gov/device/510k.json"
CTGOV_STUDIES_API = "https://clinicaltrials.gov/api/v2/studies"
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


@dataclass
class RegulatoryItem:
    title: str
    company_name: str
    product_name: str | None
    source_name: str
    source_url: str
    event_date: datetime
    status_label: str
    detail: str
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


def desired_regulatory_count() -> int:
    raw = os.getenv("REGULATORY_COUNT", "3").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 3
    if value < 2:
        return 2
    if value > 4:
        return 4
    return value


def fetch_json(url: str, params: dict[str, str]) -> dict | None:
    query = urlencode(params, quote_via=quote_plus)
    full_url = f"{url}?{query}"
    try:
        return parse_json(fetch_text(full_url))
    except Exception as exc:
        print(f"Warning: API fetch failed for {url}: {exc}")
        return None


def safe_date(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    raw = value.strip()
    if not raw:
        return datetime.now(timezone.utc)

    normalized = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y%m%d"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def regulatory_recency_score(event_date: datetime) -> float:
    age_days = max(0.0, (datetime.now(timezone.utc) - event_date).total_seconds() / 86400.0)
    return max(0.0, (120.0 - age_days) / 30.0)


def regulatory_status_weight(status_label: str) -> float:
    lower = status_label.lower()
    if "rejection" in lower:
        return 3.2
    if "approval" in lower or "clearance" in lower or "grant" in lower:
        return 2.8
    if "trial results" in lower:
        return 2.3
    return 1.6


def pma_status_label(decision_code: str) -> str:
    code = decision_code.strip().upper()
    if code in {"APPR", "APRL", "OK30"}:
        return "FDA PMA approval"
    if code.startswith("DEN") or "REJ" in code:
        return "FDA PMA rejection"
    if code:
        return f"FDA PMA decision ({code})"
    return "FDA PMA decision"


def k510_status_label(decision_code: str, decision_description: str) -> str:
    code = decision_code.strip().upper()
    description = decision_description.strip().lower()
    if code.startswith("NSE") or "not substantially equivalent" in description:
        return "FDA 510(k) rejection"
    if code.startswith("SES") or "substantially equivalent" in description:
        return "FDA 510(k) clearance"
    if code == "DENG":
        return "FDA De Novo grant"
    if code:
        return f"FDA 510(k) decision ({code})"
    return "FDA 510(k) decision"


def pma_source_url(pma_number: str) -> str:
    if not pma_number:
        return "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpma/pma.cfm"
    return (
        "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpma/pma.cfm"
        f"?id={quote_plus(pma_number)}"
    )


def k510_source_url(k_number: str) -> str:
    if not k_number:
        return "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm"
    return (
        "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm"
        f"?ID={quote_plus(k_number)}"
    )


def fetch_openfda_pma_updates(limit: int = 24) -> list[RegulatoryItem]:
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=120)).strftime("%Y%m%d")
    end = now.strftime("%Y%m%d")
    payload = fetch_json(
        OPENFDA_PMA_API,
        {
            "search": f"decision_date:[{start} TO {end}]",
            "limit": str(limit),
            "sort": "decision_date:desc",
        },
    )
    if not payload:
        return []

    updates: list[RegulatoryItem] = []
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        company_name = normalize_spaces(str(row.get("applicant", ""))) or "Unknown applicant"
        product_name = (
            normalize_spaces(str(row.get("trade_name", "")))
            or normalize_spaces(str(row.get("openfda", {}).get("device_name", "")))
            or normalize_spaces(str(row.get("generic_name", "")))
            or None
        )
        pma_number = normalize_spaces(str(row.get("pma_number", "")))
        decision_code = normalize_spaces(str(row.get("decision_code", "")))
        status_label = pma_status_label(decision_code)
        event_date = safe_date(str(row.get("decision_date", "")))
        supplement_reason = normalize_spaces(str(row.get("supplement_reason", "")))
        ao_statement = first_sentences(str(row.get("ao_statement", "")), count=1)

        detail_parts = [f"Decision code: {decision_code or 'N/A'}."]
        if supplement_reason:
            detail_parts.append(f"Reason: {supplement_reason}.")
        if ao_statement:
            detail_parts.append(f"FDA note: {ao_statement}")
        detail = " ".join(part for part in detail_parts if part).strip()
        title_core = product_name or pma_number or "PMA device update"

        item = RegulatoryItem(
            title=f"{status_label}: {title_core}",
            company_name=company_name,
            product_name=product_name,
            source_name="openFDA PMA",
            source_url=pma_source_url(pma_number),
            event_date=event_date,
            status_label=status_label,
            detail=detail,
        )
        item.score = regulatory_status_weight(status_label) + regulatory_recency_score(event_date)
        updates.append(item)
    return updates


def fetch_openfda_510k_updates(limit: int = 30) -> list[RegulatoryItem]:
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=120)).strftime("%Y%m%d")
    end = now.strftime("%Y%m%d")
    payload = fetch_json(
        OPENFDA_510K_API,
        {
            "search": f"decision_date:[{start} TO {end}]",
            "limit": str(limit),
            "sort": "decision_date:desc",
        },
    )
    if not payload:
        return []

    updates: list[RegulatoryItem] = []
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        company_name = normalize_spaces(str(row.get("applicant", ""))) or "Unknown applicant"
        product_name = (
            normalize_spaces(str(row.get("device_name", "")))
            or normalize_spaces(str(row.get("openfda", {}).get("device_name", "")))
            or None
        )
        decision_code = normalize_spaces(str(row.get("decision_code", "")))
        decision_description = normalize_spaces(str(row.get("decision_description", "")))
        status_label = k510_status_label(decision_code, decision_description)
        event_date = safe_date(str(row.get("decision_date", "")))
        k_number = normalize_spaces(str(row.get("k_number", "")))
        clearance_type = normalize_spaces(str(row.get("clearance_type", "")))
        detail_parts = [f"Decision code: {decision_code or 'N/A'}."]
        if decision_description and decision_description.lower() != "unknown":
            detail_parts.append(f"Decision: {decision_description}.")
        if clearance_type:
            detail_parts.append(f"Pathway: {clearance_type}.")

        title_core = product_name or k_number or "510(k) device update"
        item = RegulatoryItem(
            title=f"{status_label}: {title_core}",
            company_name=company_name,
            product_name=product_name,
            source_name="openFDA 510(k)",
            source_url=k510_source_url(k_number),
            event_date=event_date,
            status_label=status_label,
            detail=" ".join(detail_parts).strip(),
        )
        item.score = regulatory_status_weight(status_label) + regulatory_recency_score(event_date)
        updates.append(item)
    return updates


def trial_sponsor_bias(sponsor_name: str) -> float:
    lowered = sponsor_name.lower()
    academic_tokens = ("university", "hospital", "medical center", "college", "institute")
    return -0.3 if any(token in lowered for token in academic_tokens) else 0.4


def fetch_recent_trial_result_updates(limit: int = 20) -> list[RegulatoryItem]:
    now = datetime.now(timezone.utc)
    start_iso = (now - timedelta(days=120)).strftime("%Y-%m-%d")
    query = (
        "AREA[HasResults]true AND AREA[InterventionType]DEVICE "
        f"AND AREA[ResultsFirstPostDate]RANGE[{start_iso},MAX]"
    )
    payload = fetch_json(
        CTGOV_STUDIES_API,
        {
            "query.term": query,
            "pageSize": str(limit),
            "fields": (
                "NCTId,BriefTitle,LeadSponsorName,InterventionName,HasResults,"
                "ResultsFirstPostDateStruct,PrimaryCompletionDateStruct,Condition"
            ),
        },
    )
    if not payload:
        return []

    updates: list[RegulatoryItem] = []
    for study in payload.get("studies", []):
        if not isinstance(study, dict):
            continue
        protocol = study.get("protocolSection", {})
        if not isinstance(protocol, dict):
            continue

        ident = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        sponsor_mod = protocol.get("sponsorCollaboratorsModule", {})
        conditions_mod = protocol.get("conditionsModule", {})
        arms_mod = protocol.get("armsInterventionsModule", {})

        nct_id = normalize_spaces(str((ident or {}).get("nctId", "")))
        brief_title = normalize_spaces(str((ident or {}).get("briefTitle", "")))
        sponsor_name = normalize_spaces(
            str((sponsor_mod or {}).get("leadSponsor", {}).get("name", ""))
        ) or "Study sponsor"
        results_date = normalize_spaces(
            str((status or {}).get("resultsFirstPostDateStruct", {}).get("date", ""))
        )
        if not results_date:
            continue

        interventions = (arms_mod or {}).get("interventions", [])
        product_name = None
        if isinstance(interventions, list) and interventions:
            first_intervention = interventions[0]
            if isinstance(first_intervention, dict):
                product_name = normalize_spaces(str(first_intervention.get("name", ""))) or None

        conditions = (conditions_mod or {}).get("conditions", [])
        condition_text = ""
        if isinstance(conditions, list) and conditions:
            named_conditions = [normalize_spaces(str(value)) for value in conditions[:2] if str(value).strip()]
            if named_conditions:
                condition_text = ", ".join(named_conditions)

        event_date = safe_date(results_date)
        detail = f"Clinical trial results were posted on {results_date}."
        if condition_text:
            detail += f" Primary conditions: {condition_text}."
        source_url = (
            f"https://clinicaltrials.gov/study/{quote_plus(nct_id)}"
            if nct_id
            else "https://clinicaltrials.gov/"
        )
        item = RegulatoryItem(
            title=f"Clinical trial results posted: {brief_title or nct_id or 'Device study'}",
            company_name=sponsor_name,
            product_name=product_name,
            source_name="ClinicalTrials.gov",
            source_url=source_url,
            event_date=event_date,
            status_label="Clinical trial results posted",
            detail=detail,
        )
        item.score = (
            regulatory_status_weight(item.status_label)
            + regulatory_recency_score(event_date)
            + trial_sponsor_bias(sponsor_name)
        )
        updates.append(item)
    return updates


def regulatory_dedupe_key(item: RegulatoryItem) -> str:
    return normalize_title_key(f"{item.title} {item.company_name} {item.source_name}")


def collect_regulatory_updates() -> list[RegulatoryItem]:
    collected: list[RegulatoryItem] = []
    collected.extend(fetch_openfda_pma_updates())
    collected.extend(fetch_openfda_510k_updates())
    collected.extend(fetch_recent_trial_result_updates())
    if not collected:
        return []

    sorted_items = sorted(
        collected,
        key=lambda item: (item.score, item.event_date),
        reverse=True,
    )

    chosen: list[RegulatoryItem] = []
    seen: set[str] = set()

    # Keep source diversity by attempting to include at least one trial update when available.
    trial_item = next(
        (item for item in sorted_items if item.source_name == "ClinicalTrials.gov"),
        None,
    )
    if trial_item:
        key = regulatory_dedupe_key(trial_item)
        seen.add(key)
        chosen.append(trial_item)

    for item in sorted_items:
        key = regulatory_dedupe_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        chosen.append(item)
        if len(chosen) >= desired_regulatory_count():
            break
    return chosen[: desired_regulatory_count()]


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


def lookup_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 3}


def wiki_title_matches_query(query: str, title: str) -> bool:
    query_tokens = lookup_tokens(query)
    title_tokens = lookup_tokens(title)
    if not query_tokens or not title_tokens:
        return False
    if query_tokens.intersection(title_tokens):
        return True
    return False


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

    search_rows = (search_payload or {}).get("query", {}).get("search", [])
    best_title = None
    if isinstance(search_rows, list) and search_rows:
        first_row = search_rows[0]
        if isinstance(first_row, dict):
            best_title = first_row.get("title")
    if isinstance(best_title, str) and best_title.strip():
        candidate_title = best_title.strip()
        if not wiki_title_matches_query(term, candidate_title):
            _WIKIPEDIA_SUMMARY_CACHE[key] = None
            return None
        looked_up = fetch_summary_direct(candidate_title)
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
    tokens = set(re.findall(r"[a-z0-9]+", value))
    if any(token in tokens for token in ["robotic", "surgery", "surgical"]):
        return "robotic_surgery"
    if any(token in tokens for token in ["glucose", "sensor", "monitor", "wearable"]):
        return "monitoring"
    if any(token in tokens for token in ["imaging", "ultrasound", "mri", "ct", "scan"]):
        return "imaging"
    if any(token in tokens for token in ["stent", "valve", "implant", "pacemaker"]):
        return "implant"
    if any(token in tokens for token in ["diagnostic", "test", "assay", "sequencing"]):
        return "diagnostic"
    if any(token in tokens for token in ["therapy", "drug", "biologic", "gene", "treatment"]):
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
    regulatory_items = collect_regulatory_updates()
    if not top_items and not regulatory_items:
        subject = f"Daily MedTech/Biomedical Report | {datetime.now(timezone.utc).date().isoformat()}"
        body = (
            "No headline or regulatory updates were found today from configured sources.\n"
            "Checked headline outlets: "
            + ", ".join(outlet.name for outlet in OUTLETS)
            + "\nChecked regulatory feeds: openFDA PMA, openFDA 510(k), ClinicalTrials.gov\n"
        )
        return subject, body

    subject = (
        f"MedTech Daily Report | {len(top_items)} Headlines + {len(regulatory_items)} FDA/Trial Updates | "
        f"{datetime.now(timezone.utc).date().isoformat()}"
    )
    lines: list[str] = ["Headlines:", ""]
    if not top_items:
        lines.append("No recent medtech headlines were selected from the configured outlets today.")
        lines.append("")
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

    lines.append("FDA & Trial Updates:")
    lines.append("")
    if not regulatory_items:
        lines.append("No recent FDA/trial events were selected from openFDA and ClinicalTrials.gov today.")
        lines.append("")
    else:
        for index, item in enumerate(regulatory_items, start=1):
            lines.append(f"{to_roman(index)} - {item.title}")
            lines.append(f"Status: {item.status_label}")
            lines.append(f"Company Research: {build_company_research(item.company_name)}")
            if item.product_name:
                product_easy, product_technical, product_source = build_product_research(
                    item.product_name, item.title
                )
                lines.append(f"Product Research (easy): {product_easy}")
                lines.append(f"Product Research (tech): {product_technical}")
                if product_source:
                    lines.append(f"Product Source: {product_source}")
            lines.append(f"Event Detail: {item.detail}")
            lines.append(
                "Link / Date: "
                f"{item.source_url} | {item.event_date.strftime('%Y-%m-%d %H:%M UTC')}"
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
