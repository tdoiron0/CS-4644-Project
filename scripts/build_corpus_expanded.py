#!/usr/bin/env python3
"""
Build aircraft domain text corpus from Wikipedia for DAPT of a VLM.

Uses the MediaWiki API (action=parse, prop=wikitext) for structured retrieval
and mwparserfromhell for deterministic wikitext parsing.

Pipeline:
  1. Fetch raw wikitext via API
  2. Parse with mwparserfromhell
  3. Extract infobox (expand convert/cvt/list templates to preserve values)
  4. Extract lead + target sections (Design, Development, Variants, Specifications)
  5. Apply taxonomy-aware sentence filtering
  6. Write text_corpus.jsonl + text_corpus.txt

Outputs:
  text_corpus.jsonl  — one JSON object per aircraft
  text_corpus.txt    — plain text version for inspection
"""

import json
import logging
import os
import re
import time
from urllib.parse import unquote

import mwparserfromhell
import requests


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_LINKS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "wikipediastuff", "Wiki_links_expanded.txt")
TAXONOMY_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "fgvc-aircraft-2013b", "data", "taxonomy")
OUTPUT_JSONL = os.path.join(PROJECT_ROOT, "data", "processed", "wikitext", "text_corpus_expanded.jsonl")
OUTPUT_TXT = os.path.join(PROJECT_ROOT, "data", "processed", "wikitext", "text_corpus_expanded.txt")
API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_DELAY = 0.5


TARGET_SECTION_KEYWORDS = [
    "variant", "design", "development", "specification", "description",
    "history", "production", "operational", "operator", "service",
    "overview", "general characteristics", "performance", "orders",
    "deliveries", "assembly", "manufacture", "technical", "feature",
    "introduction", "background", "origin", "configuration", "model",
    "aircraft", "product",
]

SKIP_SECTIONS = frozenset({
    "references", "external links", "see also", "bibliography",
    "further reading", "notes", "citations",
})


MAX_SECTION_CHARS = 15000

TECHNICAL_KEYWORDS = frozenset({
    "wingspan", "fuselage", "engine", "turbofan", "turboprop", "cockpit",
    "landing gear", "wing", "tail", "avionics", "thrust", "range",
    "ceiling", "speed", "altitude", "passengers", "payload", "length",
    "height", "weight", "mtow", "configuration", "variant", "model",
    "series", "derivative", "stretched", "narrow-body", "wide-body",
    "twin-engine", "four-engine", "propeller", "jet", "cabin",
    "capacity", "crew", "diameter", "aerodynamic", "airframe",
    "undercarriage", "nacelle", "pylon", "stabilizer", "rudder",
    "aileron", "flap", "slat", "spoiler", "winglet", "powerplant",
    "horsepower", "turboshaft", "swept", "canard", "biplane",
    "monoplane", "retractable", "tricycle", "tailwheel", "pressurized",
    "unpressurized", "composite", "aluminum", "alloy", "mach",
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def load_taxonomy():
    terms = set()
    for filename in ("families.txt", "variants.txt", "manufacturers.txt"):
        path = os.path.join(TAXONOMY_DIR, filename)
        if not os.path.exists(path):
            log.warning("Taxonomy file not found: %s", path)
            continue
        with open(path) as f:
            for line in f:
                term = line.strip()
                if not term:
                    continue
                terms.add(term.lower())
    return frozenset(terms)


# ---------------------------------------------------------------------------
# Wiki_links.txt
# ---------------------------------------------------------------------------

def load_wiki_links(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            family, url = line.split(": ", 1)
            family = family.strip()
            page_type = "aircraft"
            type_match = re.match(r"(.+?)\s*\[(manufacturer|concept)\]$", family)
            if type_match:
                family = type_match.group(1)
                page_type = type_match.group(2)
            page_title = unquote(url.split("/wiki/")[-1]).replace("_", " ")
            entries.append((family, page_title, page_type))
    return entries


# ---------------------------------------------------------------------------
# MediaWiki API
# ---------------------------------------------------------------------------

def fetch_page(page_title, session):
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "format": "json",
        "formatversion": 2,
        "redirects": 1,
    }
    resp = session.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise ValueError(data["error"].get("info", "unknown API error"))
    return data["parse"]


# ---------------------------------------------------------------------------
# Template expansion — run BEFORE strip_code to preserve numeric values
# ---------------------------------------------------------------------------

def expand_known_templates(wikicode):
    """Expand convert/cvt/list/formatting templates in-place so their
    numeric content survives strip_code()."""
    for template in list(wikicode.filter_templates(recursive=True)):
        name = str(template.name).strip().lower()
        try:
            if name in ("convert", "cvt"):
                _expand_convert(wikicode, template)
            elif name in ("plainlist", "ubl", "unbulleted list", "flatlist",
                          "bulleted list", "hlist"):
                _expand_list(wikicode, template)
            elif name in ("nowrap", "nobr", "nobreak", "mvar"):
                _replace_first_positional(wikicode, template)
            elif name in ("small", "smaller", "big", "large", "larger", "resize"):
                _replace_first_positional(wikicode, template)
            elif name in ("nbsp", "spaces"):
                wikicode.replace(template, " ")
            elif name in ("snd", "spaced ndash", "snds"):
                wikicode.replace(template, " - ")
            elif name in ("ndash", "en dash"):
                wikicode.replace(template, "-")
            elif name in ("mdash", "em dash"):
                wikicode.replace(template, " - ")
            elif name == "formatnum":
                _replace_first_positional(wikicode, template)
            elif name in ("lang", "transl"):
                _replace_last_positional(wikicode, template)
            elif name in ("val",):
                _expand_val(wikicode, template)
            elif name in ("abbr",):
                _replace_first_positional(wikicode, template)
        except (IndexError, ValueError):
            continue


def _positional_params(template):
    return [p for p in template.params if not p.showkey]


def _replace_first_positional(wikicode, template):
    pos = _positional_params(template)
    val = str(pos[0].value).strip() if pos else ""
    wikicode.replace(template, val)


def _replace_last_positional(wikicode, template):
    pos = _positional_params(template)
    val = str(pos[-1].value).strip() if pos else ""
    wikicode.replace(template, val)


def _expand_convert(wikicode, template):
    """{{convert|37.57|m|ft}} → '37.57 m'"""
    pos = _positional_params(template)
    if len(pos) >= 2:
        wikicode.replace(template, f"{str(pos[0].value).strip()} {str(pos[1].value).strip()}")
    elif pos:
        wikicode.replace(template, str(pos[0].value).strip())


def _expand_val(wikicode, template):
    """{{val|12345|u=km}} → '12345 km'"""
    pos = _positional_params(template)
    num = str(pos[0].value).strip() if pos else ""
    unit = ""
    for p in template.params:
        if p.showkey and str(p.name).strip().lower() in ("u", "ul"):
            unit = str(p.value).strip()
            break
    wikicode.replace(template, f"{num} {unit}".strip())


def _expand_list(wikicode, template):
    """{{plainlist|* A\\n* B}} or {{ubl|A|B}} → 'A, B'"""
    name = str(template.name).strip().lower()
    if name in ("plainlist", "flatlist"):
        content = str(template.params[0].value) if template.params else ""
        items = [m.strip() for m in re.findall(r"\*\s*(.+)", content)]
        wikicode.replace(template, ", ".join(items))
    else:
        items = [str(p.value).strip() for p in template.params if not p.showkey]
        wikicode.replace(template, ", ".join(i for i in items if i))


# ---------------------------------------------------------------------------
# Infobox extraction — uses mwparserfromhell, expands templates per-value
# ---------------------------------------------------------------------------

def extract_infobox(parsed_wikitext):
    """Find the Infobox template and extract cleaned key-value pairs.
    Each value gets template expansion before strip_code so numbers/units
    inside {{convert}}, {{cvt}}, etc. are preserved."""
    for template in parsed_wikitext.filter_templates(recursive=False):
        tname = str(template.name).strip().lower()
        if not tname.startswith("infobox"):
            continue

        fields = {}
        for param in template.params:
            key = str(param.name).strip().lower().replace(" ", "_")
            if not key:
                continue
            val_code = mwparserfromhell.parse(str(param.value))
            expand_known_templates(val_code)
            cleaned = val_code.strip_code().strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            if cleaned:
                fields[key] = cleaned
        return fields
    return {}


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def extract_lead(parsed_wikitext):
    """Lead = everything before the first heading, minus the infobox template."""
    flat = parsed_wikitext.get_sections(include_lead=True, flat=True)
    if not flat:
        return ""
    lead = flat[0]
    if lead.filter_headings():
        return ""
    lead_code = mwparserfromhell.parse(str(lead))
    for t in list(lead_code.filter_templates(recursive=False)):
        if str(t.name).strip().lower().startswith("infobox"):
            lead_code.remove(t)
        elif str(t.name).strip().lower().startswith("short description"):
            lead_code.remove(t)
        elif str(t.name).strip().lower().startswith("about"):
            lead_code.remove(t)
    expand_known_templates(lead_code)
    text = lead_code.strip_code().strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def extract_target_sections(parsed_wikitext):
    """Get level-2 sections (with all their subsections) that match target keywords."""
    sections_out = {}
    for section in parsed_wikitext.get_sections(levels=[2], include_headings=True):
        headings = section.filter_headings()
        if not headings:
            continue
        title = headings[0].title.strip_code().strip()
        title_lower = title.lower()

        if title_lower in SKIP_SECTIONS:
            continue
        if not any(kw in title_lower for kw in TARGET_SECTION_KEYWORDS):
            continue

        sec_code = mwparserfromhell.parse(str(section))
        top_heading = sec_code.filter_headings()
        if top_heading:
            try:
                sec_code.remove(top_heading[0])
            except ValueError:
                pass

        expand_known_templates(sec_code)
        content = sec_code.strip_code().strip()
        content = re.sub(r"\n{3,}", "\n\n", content)

        if content and len(content) > 30:
            if len(content) > MAX_SECTION_CHARS:
                content = content[:MAX_SECTION_CHARS].rsplit(" ", 1)[0] + "..."
            sections_out[title] = content

    return sections_out


# ---------------------------------------------------------------------------
# Taxonomy-aware filtering
# ---------------------------------------------------------------------------

def filter_by_taxonomy(text, taxonomy_terms):
    """Relaxed filter: keep ALL sentences in sections that have at least one
    taxonomy or technical-keyword hit. Only discard sections that are entirely
    off-topic (no taxonomy or technical terms at all).
    Applied to section text only — lead and infobox are kept in full."""
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        s_lower = sentence.lower()
        if any(term in s_lower for term in taxonomy_terms):
            return text
        if any(kw in s_lower for kw in TECHNICAL_KEYWORDS):
            return text

    return ""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_full_text(family, lead, infobox, sections):
    parts = [family]
    if lead:
        parts.append(lead)
    if infobox:
        parts.append("\n".join(f"{k}: {v}" for k, v in infobox.items()))
    for title, body in sections.items():
        parts.append(f"{title}\n{body}")
    return "\n\n".join(parts)


def process_page(family, page_title, session, taxonomy_terms, page_type="aircraft"):
    log.info("Fetching: %s  (family: %s, type: %s)", page_title, family, page_type)

    api_data = fetch_page(page_title, session)
    raw_wikitext = api_data["wikitext"]
    parsed = mwparserfromhell.parse(raw_wikitext)

    infobox = extract_infobox(parsed)
    lead = extract_lead(parsed)
    sections = extract_target_sections(parsed)


    # Only apply taxonomy filtering to aircraft pages
    if page_type == "aircraft" and taxonomy_terms:
        filtered = {}
        for title, content in sections.items():
            result = filter_by_taxonomy(content, taxonomy_terms)
            if result:
                filtered[title] = result
        sections = filtered

    full_text = build_full_text(family, lead, infobox, sections)

    entry = {
        "family": family,
        "aircraft_name": api_data.get("title", page_title),
        "page_type": page_type,
        "lead": lead,
        "infobox": infobox,
        "sections": sections,
        "full_clean_text": full_text,
    }

    log.info(
        "  lead=%d chars, infobox=%d fields, sections=%d, total=%d chars",
        len(lead), len(infobox), len(sections), len(full_text),
    )
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    entries = load_wiki_links(WIKI_LINKS_PATH)
    log.info("Loaded %d aircraft from %s", len(entries), WIKI_LINKS_PATH)

    taxonomy_terms = load_taxonomy()
    log.info("Loaded %d taxonomy terms for filtering", len(taxonomy_terms))

    session = requests.Session()
    session.headers["User-Agent"] = (
        "AircraftCorpusBuilder/1.0 (academic research; FGVC-Aircraft DAPT)"
    )

    results = []
    failed = []


    for i, (family, page_title, page_type) in enumerate(entries):
        try:
            entry = process_page(family, page_title, session, taxonomy_terms, page_type)
            results.append(entry)
        except Exception as e:
            log.error("FAILED: %s — %s", page_title, e)
            failed.append((family, page_title, str(e)))
        if i < len(entries) - 1:
            time.sleep(REQUEST_DELAY)

    with open(OUTPUT_JSONL, "w") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("Written: %s  (%d entries)", OUTPUT_JSONL, len(results))

    with open(OUTPUT_TXT, "w") as f:
        for entry in results:
            f.write(entry["full_clean_text"])
            f.write("\n\n=== END AIRCRAFT ENTRY ===\n\n")
    log.info("Written: %s", OUTPUT_TXT)

    total_chars = sum(len(e["full_clean_text"]) for e in results)
    log.info("Corpus: %d pages, %s characters", len(results), f"{total_chars:,}")

    if failed:
        log.warning("Failed pages (%d):", len(failed))
        for fam, title, err in failed:
            log.warning("  %s: %s — %s", fam, title, err)

    print(f"\n{'Family':<25} {'Lead':>6} {'Infobox':>8} {'Sections':>9} {'Total':>8}")
    print("-" * 60)
    for e in results:
        print(
            f"{e['family']:<25} {len(e['lead']):>6} "
            f"{len(e['infobox']):>8} {len(e['sections']):>9} "
            f"{len(e['full_clean_text']):>8}"
        )


if __name__ == "__main__":
    main()
