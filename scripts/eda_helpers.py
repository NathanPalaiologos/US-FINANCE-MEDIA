from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import zipfile
from ast import literal_eval
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from wordcloud import WordCloud


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw_extracted" / "a5006USFinance_economicsBusinessMedia"
EXTENDED_ZIP = DATA_DIR / "extended.csv.zip"
EXTENDED_CSV = "extended.csv"
ANALYSIS_DIR = ROOT / "analysis"
OUTPUT_DIR = ANALYSIS_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FINAL_DIR = ANALYSIS_DIR / "final"
REPORT_PATH = ANALYSIS_DIR / "metadata_quality_report.md"
REPORT_PDF_PATH = ANALYSIS_DIR / "metadata_quality_report.pdf"
SUMMARY_XLSX = OUTPUT_DIR / "summary_statistics.xlsx"
SUMMARY_XLSX_FALLBACK = OUTPUT_DIR / "summary_statistics_refreshed.xlsx"
PROCESSING_NOTES_PATH = OUTPUT_DIR / "processing_notes.json"

PIP_NOTE = (
    "Packages were installed with --user because normal site-packages was not writeable; "
    "script paths are not on PATH. The default Jupyter kernel used a different Python, "
    "so a dedicated tdm-studio-python kernel was registered for this notebook."
)
TEXT_LIMITATION = (
    "The export contains metadata fields but no full article body; the revised topic model "
    "uses article titles only after filtering non-North-American publishers."
)
MEDIA_FILTER_RULE = (
    "Retain North-American publishers, including North-American publishers with international "
    "or regional edition titles; exclude only publishers that appear outside North America. "
    "Edition-like titles are flagged for separate exploration rather than removed."
)


KEY_COLUMNS = [
    "GOID",
    "Title",
    "Date",
    "Source Type",
    "Authors",
    "Publication ID",
    "Publication Title",
    "Publisher City",
    "Publisher Province",
    "Publisher Name",
    "Object Type",
    "Language",
    "Pages",
    "Company Name",
    "Class Terms",
    "Subject Terms",
]


XML_FIELD_MAP = [
    {
        "panel_field": "goid",
        "csv_columns": ["GOID"],
        "xml_paths": ["GOID", "DFS/GOID"],
        "role": "document key",
        "notes": "Stable document identifier used as the unit-of-observation key.",
    },
    {
        "panel_field": "title",
        "csv_columns": ["Title"],
        "xml_paths": ["Title", "DFS/Title", "DFS/PubFrosting/Title"],
        "role": "title/topic text",
        "notes": "Primary title for metadata exploration and title-only topic modeling.",
    },
    {
        "panel_field": "abstract",
        "csv_columns": [],
        "xml_paths": ["Abstract/AbsText", "Abstract/Short/AbsText", "AbsText"],
        "role": "optional topic text",
        "notes": "Not present in the current CSV export; scan XML exports before using it.",
    },
    {
        "panel_field": "genre_source_type",
        "csv_columns": ["Source Type"],
        "xml_paths": ["DFS/PubFrosting/SourceType", "SourceType"],
        "role": "genre/source type",
        "notes": "Broad source family such as Newspapers, Magazines, or Trade Journals.",
    },
    {
        "panel_field": "genre_object_type",
        "csv_columns": ["Object Type"],
        "xml_paths": ["DFS/PubFrosting/ObjectType", "ObjectType"],
        "role": "genre/document type",
        "notes": "Document-level object type; Undefined often signals weaker metadata/full-text evidence.",
    },
    {
        "panel_field": "journal_title",
        "csv_columns": ["Publication Title"],
        "xml_paths": ["DFS/PubFrosting/PublicationTitle", "PublicationTitle", "PubTitle"],
        "role": "journal/publication",
        "notes": "Publication title; normalized downstream into a stable journal/media key.",
    },
    {
        "panel_field": "journal_id",
        "csv_columns": ["Publication ID"],
        "xml_paths": ["DFS/PubFrosting/PublicationID", "PublicationID"],
        "role": "journal/publication",
        "notes": "Provider publication identifier when available.",
    },
    {
        "panel_field": "publication_date",
        "csv_columns": ["Date"],
        "xml_paths": ["Date", "NumericPubDate", "AlphaPubDate"],
        "role": "time",
        "notes": "Document publication date; used to derive year, decade, and observed coverage range.",
    },
    {
        "panel_field": "authors",
        "csv_columns": ["Authors"],
        "xml_paths": ["Author", "Authors", "DFS/Author"],
        "role": "author",
        "notes": "List-like author metadata; current export has substantial missingness.",
    },
    {
        "panel_field": "publisher_name",
        "csv_columns": ["Publisher Name"],
        "xml_paths": ["DFS/PubFrosting/Publisher/PublisherName", "PublisherName"],
        "role": "publisher",
        "notes": "Publisher name for source-level diagnostics.",
    },
    {
        "panel_field": "publisher_location",
        "csv_columns": ["Publisher City", "Publisher Province", "Publisher ZipCode"],
        "xml_paths": [
            "DFS/PubFrosting/Publisher/City",
            "DFS/PubFrosting/Publisher/State",
            "DFS/PubFrosting/Publisher/ZipCode",
        ],
        "role": "publisher geography",
        "notes": "Used for the North-American publisher scope and location normalization.",
    },
    {
        "panel_field": "pages",
        "csv_columns": ["Pages", "Start Page"],
        "xml_paths": ["Pages", "StartPage"],
        "role": "coverage proxy",
        "notes": "Article length/start page proxy; not a serial issue-completeness field.",
    },
    {
        "panel_field": "subject_terms",
        "csv_columns": ["Subject Terms"],
        "xml_paths": ["SubjectTerms", "SubjectTerm", "Terms/Subject"],
        "role": "topic metadata",
        "notes": "Controlled or provider terms useful for metadata-enriched topic exploration.",
    },
    {
        "panel_field": "class_terms",
        "csv_columns": ["Class Terms"],
        "xml_paths": ["ClassTerms", "ClassTerm", "Terms/Class"],
        "role": "topic metadata",
        "notes": "Classification metadata useful for topical coverage diagnostics.",
    },
    {
        "panel_field": "full_text_status",
        "csv_columns": [],
        "xml_paths": ["FullText", "FullTextAvailable", "Availability", "ObjectAvailability"],
        "role": "full-text availability",
        "notes": "No such column exists in the current CSV export; XML scans record whether a provider tag exists.",
    },
]


STOPWORDS = set(ENGLISH_STOP_WORDS).union(
    {
        "article",
        "articles",
        "type",
        "types",
        "page",
        "pages",
        "news",
        "newspaper",
        "newspapers",
        "magazine",
        "magazines",
        "journal",
        "journals",
        "trade",
        "online",
        "edition",
        "report",
        "reports",
        "english",
        "proquest",
    }
)


US_STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
    "DC",
}

CANADA_PROVINCES = {
    "AB": "AB",
    "ALBERTA": "AB",
    "BC": "BC",
    "BRITISH COLUMBIA": "BC",
    "MB": "MB",
    "MANITOBA": "MB",
    "NB": "NB",
    "NEW BRUNSWICK": "NB",
    "NL": "NL",
    "NEWFOUNDLAND AND LABRADOR": "NL",
    "NS": "NS",
    "NOVA SCOTIA": "NS",
    "NT": "NT",
    "NORTHWEST TERRITORIES": "NT",
    "NU": "NU",
    "NUNAVUT": "NU",
    "ON": "ON",
    "ONTARIO": "ON",
    "PE": "PE",
    "PRINCE EDWARD ISLAND": "PE",
    "QC": "QC",
    "QUEBEC": "QC",
    "SK": "SK",
    "SASKATCHEWAN": "SK",
    "YT": "YT",
    "YUKON": "YT",
}

CITY_STATE_HINTS = {
    "new york n y": ("New York", "NY"),
    "new york": ("New York", None),
    "washington d c": ("Washington", "DC"),
    "washington": ("Washington", None),
}


INTERNATIONAL_TITLE_PATTERNS = [
    "asian wall street journal",
    "wall street journal asia",
    "international herald tribune",
    "international new york times",
    "national post",
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)


def record_processing_note(note: str) -> None:
    existing = []
    if PROCESSING_NOTES_PATH.exists():
        try:
            existing = json.loads(PROCESSING_NOTES_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    existing.append(note)
    PROCESSING_NOTES_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def read_extended_chunks(chunksize: int = 200_000):
    with zipfile.ZipFile(EXTENDED_ZIP) as zf:
        with zf.open(EXTENDED_CSV) as f:
            yield from pd.read_csv(
                f,
                dtype=str,
                chunksize=chunksize,
                keep_default_na=False,
                low_memory=False,
            )


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def compact_key(value: object) -> str:
    text = clean_text(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_publication(value: object) -> str:
    text = clean_text(value)
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"['’]s\b", "s", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(the)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(inc|llc|ltd|corp|corporation|company|co)\b\.?", " ", text, flags=re.IGNORECASE)
    text = compact_key(text)
    if not text:
        return "Unknown"
    text = re.sub(r"\bu s\b", "us", text)
    return text.title().replace("Us ", "US ")


def publication_display_name(raw_value: object, normalized_value: str) -> str:
    text = clean_text(raw_value)
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"^\s*the\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text or normalized_value


def normalize_city_state(city: object, province: object) -> tuple[str, str, str, str]:
    city_text = clean_text(city)
    province_text = clean_text(province)
    city_key = compact_key(city_text.replace(".", " "))
    province_key = compact_key(province_text).upper()

    hinted_city = None
    hinted_state = None
    if city_key in CITY_STATE_HINTS:
        hinted_city, hinted_state = CITY_STATE_HINTS[city_key]

    normalized_city = hinted_city or city_text.title()
    normalized_state = province_key
    if not normalized_state and hinted_state:
        normalized_state = hinted_state
    if normalized_state in CANADA_PROVINCES:
        normalized_state = CANADA_PROVINCES[normalized_state]

    if normalized_state in US_STATE_CODES:
        region = "United States"
    elif normalized_state in set(CANADA_PROVINCES.values()):
        region = "Canada"
    elif "oxfordshire" in city_key or "abingdon" in city_key:
        region = "Non-North America"
    else:
        region = "Unknown/Needs Review"

    location = f"{normalized_city}, {normalized_state}" if normalized_state else normalized_city
    return normalized_city or "Unknown", normalized_state or "", region, location or "Unknown"


def edition_flag(publication: object) -> str:
    pub_key = compact_key(publication)
    for pattern in INTERNATIONAL_TITLE_PATTERNS:
        if pattern in pub_key:
            return pattern
    return ""


def media_exclusion_reason(publication: object, city: object, province: object) -> str:
    _, _, region, _ = normalize_city_state(city, province)
    if region == "Non-North America":
        return "publisher appears outside North America"
    return ""


def parse_term_value(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [clean_text(item) for item in parsed if clean_text(item)]
        except (ValueError, SyntaxError):
            pass
    return [clean_text(part) for part in re.split(r"\s*[;|]\s*|\s{2,}", text) if clean_text(part)]


def canonical_term(term: str) -> str:
    term = clean_text(term).strip("'\"")
    if not term:
        return ""
    return term if not term.isupper() else term.title()


def split_terms(series: pd.Series) -> Counter:
    counts: Counter[str] = Counter()
    for value in series.fillna("").astype(str):
        for part in parse_term_value(value):
            term = canonical_term(part)
            if term:
                counts[term] += 1
    return counts


def split_authors(series: pd.Series) -> Counter:
    counts: Counter[str] = Counter()
    for value in series.fillna("").astype(str):
        for author in parse_term_value(value):
            cleaned = canonical_term(author)
            if cleaned:
                counts[cleaned] += 1
    return counts


def build_topic_text(df: pd.DataFrame) -> pd.Series:
    return df["Title"].fillna("").astype(str).map(clean_text)


def sample_rows(df: pd.DataFrame, frac: float, random_state: int) -> pd.DataFrame:
    if len(df) == 0:
        return df
    return df.sample(frac=frac, random_state=random_state) if frac < 1 else df


def profile_metadata(topic_sample_target: int = 100_000) -> dict:
    ensure_dirs()
    if PROCESSING_NOTES_PATH.exists():
        PROCESSING_NOTES_PATH.unlink()

    raw_row_count = 0
    row_count = 0
    missing = Counter()
    unique_values: dict[str, set[str]] = {
        "Source Type": set(),
        "Object Type": set(),
        "Language": set(),
        "Publication Title": set(),
        "Publication Title Normalized": set(),
        "Publisher Name": set(),
        "Publisher City": set(),
        "Publisher Province": set(),
        "Publisher City Normalized": set(),
        "Publisher Region": set(),
        "Edition Flag": set(),
    }
    value_counts = {
        "source_type": Counter(),
        "object_type": Counter(),
        "language": Counter(),
        "publication_raw": Counter(),
        "publication_normalized": Counter(),
        "publisher_name": Counter(),
        "publisher_city_raw": Counter(),
        "publisher_province_raw": Counter(),
        "publisher_city": Counter(),
        "publisher_province": Counter(),
        "publisher_region": Counter(),
        "edition_flag": Counter(),
        "year": Counter(),
        "decade": Counter(),
    }
    source_decade = Counter()
    media_decade = Counter()
    media_source = Counter()
    media_terms = Counter()
    pub_norm_raw = Counter()
    excluded_media = Counter()
    goid_counts = Counter()
    same_media_key_counts = Counter()
    title_date_key_counts = Counter()
    same_media_key_examples = {}
    title_date_key_examples = {}
    subject_counts = Counter()
    class_counts = Counter()
    author_counts = Counter()
    media_author_counts = Counter()
    media_rows = Counter()
    media_dates: dict[str, list[pd.Timestamp | None]] = {}
    media_locations: dict[str, Counter] = {}
    media_regions: dict[str, Counter] = {}
    media_edition_flags: dict[str, Counter] = {}
    media_raw_titles: dict[str, Counter] = {}
    media_display_titles: dict[str, Counter] = {}
    media_publishers: dict[str, Counter] = {}
    page_values = []
    min_date = None
    max_date = None
    invalid_dates = 0
    sample_frames = []
    chunk_index = 0

    for chunk in read_extended_chunks():
        chunk_index += 1
        chunk = chunk.reindex(columns=KEY_COLUMNS, fill_value="")
        raw_row_count += len(chunk)
        normalized_pub_all = chunk["Publication Title"].map(normalize_publication)
        location_info_all = [
            normalize_city_state(city, province)
            for city, province in zip(chunk["Publisher City"], chunk["Publisher Province"])
        ]
        edition_flag_all = chunk["Publication Title"].map(edition_flag)
        exclusion_reason = [
            media_exclusion_reason(pub, city, province)
            for pub, city, province in zip(
                chunk["Publication Title"],
                chunk["Publisher City"],
                chunk["Publisher Province"],
            )
        ]
        excluded_mask = pd.Series(exclusion_reason, index=chunk.index).ne("")
        if excluded_mask.any():
            excluded = chunk.loc[excluded_mask].copy()
            excluded["Publication Title Normalized"] = normalized_pub_all.loc[excluded_mask]
            excluded["Exclusion Reason"] = pd.Series(exclusion_reason, index=chunk.index).loc[excluded_mask]
            excluded["Publisher City Normalized"] = [location_info_all[i][0] for i, idx in enumerate(chunk.index) if excluded_mask.loc[idx]]
            excluded["Publisher Province Normalized"] = [location_info_all[i][1] for i, idx in enumerate(chunk.index) if excluded_mask.loc[idx]]
            excluded["Publisher Region"] = [location_info_all[i][2] for i, idx in enumerate(chunk.index) if excluded_mask.loc[idx]]
            excluded["Edition Flag"] = edition_flag_all.loc[excluded_mask]
            grouped_excluded = (
                excluded.groupby(
                    [
                        "Publication Title Normalized",
                        "Publication Title",
                        "Publisher City Normalized",
                        "Publisher Province Normalized",
                        "Publisher Region",
                        "Publisher Name",
                        "Edition Flag",
                        "Exclusion Reason",
                    ],
                    dropna=False,
                )
                .size()
                .reset_index(name="Documents")
            )
            for row in grouped_excluded.itertuples(index=False):
                excluded_media[tuple(row[:-1])] += int(row.Documents)

        chunk = chunk.loc[~excluded_mask].copy()
        normalized_pub = normalized_pub_all.loc[chunk.index]
        retained_locations = [location_info_all[i] for i, idx in enumerate(normalized_pub_all.index) if not excluded_mask.loc[idx]]
        retained_edition_flags = edition_flag_all.loc[chunk.index].map(lambda x: clean_text(x) or "main_or_unspecified")
        row_count += len(chunk)
        print(f"Scanned chunk {chunk_index}: {raw_row_count:,} raw rows, {row_count:,} retained North-America publisher rows")

        for col in chunk.columns:
            missing[col] += int(chunk[col].fillna("").astype(str).str.strip().eq("").sum())

        dates = pd.to_datetime(chunk["Date"], errors="coerce")
        invalid_dates += int(dates.isna().sum())
        valid_dates = dates.dropna()
        if len(valid_dates):
            current_min = valid_dates.min()
            current_max = valid_dates.max()
            min_date = current_min if min_date is None else min(min_date, current_min)
            max_date = current_max if max_date is None else max(max_date, current_max)

        years = dates.dt.year
        decades = (years // 10 * 10).astype("Int64")

        source_type = chunk["Source Type"].map(lambda x: clean_text(x) or "Unknown")
        object_type = chunk["Object Type"].map(lambda x: clean_text(x) or "Unknown")
        language = chunk["Language"].map(lambda x: clean_text(x) or "Unknown")
        publisher = chunk["Publisher Name"].map(lambda x: clean_text(x) or "Unknown")
        publisher_city_raw = chunk["Publisher City"].map(lambda x: clean_text(x) or "Unknown")
        publisher_province_raw = chunk["Publisher Province"].map(lambda x: clean_text(x) or "Unknown")
        publisher_city = pd.Series([item[0] or "Unknown" for item in retained_locations], index=chunk.index)
        publisher_province = pd.Series([item[1] or "Unknown" for item in retained_locations], index=chunk.index)
        publisher_region = pd.Series([item[2] or "Unknown" for item in retained_locations], index=chunk.index)
        publisher_location = pd.Series([item[3] or "Unknown" for item in retained_locations], index=chunk.index)

        value_counts["source_type"].update(source_type)
        value_counts["object_type"].update(object_type)
        value_counts["language"].update(language)
        value_counts["publication_raw"].update(chunk["Publication Title"].map(lambda x: clean_text(x) or "Unknown"))
        value_counts["publication_normalized"].update(normalized_pub)
        value_counts["publisher_name"].update(publisher)
        value_counts["publisher_city_raw"].update(publisher_city_raw)
        value_counts["publisher_province_raw"].update(publisher_province_raw)
        value_counts["publisher_city"].update(publisher_city)
        value_counts["publisher_province"].update(publisher_province)
        value_counts["publisher_region"].update(publisher_region)
        value_counts["edition_flag"].update(retained_edition_flags)
        value_counts["year"].update([int(y) for y in years.dropna()])
        value_counts["decade"].update([int(d) for d in decades.dropna()])

        for st, dec in zip(source_type, decades):
            if pd.notna(dec):
                source_decade[(st, int(dec))] += 1
        for media, dec in zip(normalized_pub, decades):
            if pd.notna(dec):
                media_decade[(media, int(dec))] += 1
        for media, st in zip(normalized_pub, source_type):
            media_source[(media, st)] += 1

        for col, series in [
            ("Source Type", source_type),
            ("Object Type", object_type),
            ("Language", language),
            ("Publication Title", chunk["Publication Title"].map(lambda x: clean_text(x) or "Unknown")),
            ("Publication Title Normalized", normalized_pub),
            ("Publisher Name", publisher),
            ("Publisher City", publisher_city_raw),
            ("Publisher Province", publisher_province_raw),
            ("Publisher City Normalized", publisher_city),
            ("Publisher Region", publisher_region),
            ("Edition Flag", retained_edition_flags),
        ]:
            unique_values[col].update(series.unique().tolist())

        for raw, norm in zip(chunk["Publication Title"], normalized_pub):
            pub_norm_raw[(norm, clean_text(raw) or "Unknown")] += 1
            media_raw_titles.setdefault(norm, Counter())[clean_text(raw) or "Unknown"] += 1
            media_display_titles.setdefault(norm, Counter())[publication_display_name(raw, norm)] += 1

        for norm, st, location, region, flag, pub_name in zip(normalized_pub, source_type, publisher_location, publisher_region, retained_edition_flags, publisher):
            media_rows[norm] += 1
            media_locations.setdefault(norm, Counter())[location] += 1
            media_regions.setdefault(norm, Counter())[region] += 1
            media_edition_flags.setdefault(norm, Counter())[flag] += 1
            media_publishers.setdefault(norm, Counter())[pub_name] += 1

        for norm, dt in zip(normalized_pub, dates):
            bounds = media_dates.setdefault(norm, [None, None])
            if pd.notna(dt):
                bounds[0] = dt if bounds[0] is None else min(bounds[0], dt)
                bounds[1] = dt if bounds[1] is None else max(bounds[1], dt)

        goid_counts.update(chunk["GOID"].map(compact_key))
        title_key = chunk["Title"].map(compact_key)
        date_key = dates.dt.strftime("%Y-%m-%d").fillna("")
        duplicate_mask = title_key.ne("") & date_key.ne("")
        same_media_keys = title_key + "||" + date_key + "||" + normalized_pub.map(compact_key)
        title_date_keys = title_key + "||" + date_key
        same_media_key_counts.update(same_media_keys[duplicate_mask])
        title_date_key_counts.update(title_date_keys[duplicate_mask])
        for key, title, date, media in zip(same_media_keys[duplicate_mask], chunk.loc[duplicate_mask, "Title"], date_key[duplicate_mask], normalized_pub[duplicate_mask]):
            same_media_key_examples.setdefault(key, (clean_text(title), date, media))
        for key, title, date in zip(title_date_keys[duplicate_mask], chunk.loc[duplicate_mask, "Title"], date_key[duplicate_mask]):
            title_date_key_examples.setdefault(key, (clean_text(title), date))

        subject_counts.update(split_terms(chunk["Subject Terms"]))
        class_counts.update(split_terms(chunk["Class Terms"]))
        author_counts.update(split_authors(chunk["Authors"]))
        for media, authors in zip(normalized_pub, chunk["Authors"]):
            for author in parse_term_value(authors):
                cleaned = canonical_term(author)
                if cleaned:
                    media_author_counts[(media, cleaned)] += 1
        for media, terms in zip(normalized_pub, chunk["Subject Terms"]):
            for term in parse_term_value(terms):
                cleaned = canonical_term(term)
                if cleaned:
                    media_terms[(media, cleaned)] += 1

        pages = pd.to_numeric(chunk["Pages"].replace("", np.nan), errors="coerce")
        page_values.append(pages.dropna())

        frac = min(1.0, topic_sample_target / max(row_count, 1) / 2)
        topic_cols = chunk[["GOID", "Title", "Date", "Source Type", "Publication Title", "Authors", "Object Type", "Subject Terms", "Class Terms", "Company Name"]].copy()
        topic_cols["Publication Title Normalized"] = normalized_pub
        topic_cols["Publisher Region"] = publisher_region
        topic_cols["Edition Flag"] = retained_edition_flags
        topic_cols["Year"] = years
        topic_cols["Decade"] = decades
        topic_cols["Topic Text"] = build_topic_text(chunk)
        sampled = sample_rows(topic_cols[topic_cols["Topic Text"].str.len() > 5], frac=frac, random_state=1000 + chunk_index)
        sample_frames.append(sampled)

    topic_sample = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    if len(topic_sample) > topic_sample_target:
        topic_sample = topic_sample.sample(n=topic_sample_target, random_state=42).reset_index(drop=True)

    duplicate_summary = pd.DataFrame(
        [
            {
                "metric": "retained_rows",
                "value": row_count,
            },
            {
                "metric": "raw_rows",
                "value": raw_row_count,
            },
            {
                "metric": "excluded_rows",
                "value": raw_row_count - row_count,
            },
            {
                "metric": "duplicate_goid_rows",
                "value": sum(c - 1 for k, c in goid_counts.items() if k and c > 1),
            },
            {
                "metric": "same_title_date_media_duplicate_rows",
                "value": sum(c - 1 for k, c in same_media_key_counts.items() if k and c > 1),
            },
            {
                "metric": "same_title_date_duplicate_rows",
                "value": sum(c - 1 for k, c in title_date_key_counts.items() if k and c > 1),
            },
            {
                "metric": "invalid_or_missing_date_rows",
                "value": invalid_dates,
            },
        ]
    )

    profile = {
        "raw_row_count": raw_row_count,
        "excluded_row_count": raw_row_count - row_count,
        "row_count": row_count,
        "min_date": str(min_date.date()) if min_date is not None else "",
        "max_date": str(max_date.date()) if max_date is not None else "",
        "invalid_dates": invalid_dates,
        "missing": dict(missing),
        "unique_counts": {k: len(v) for k, v in unique_values.items()},
        "topic_sample_size": int(len(topic_sample)),
    }

    write_counter_csv(value_counts["source_type"], "source_type_counts.csv", ["Source Type", "Documents"])
    write_counter_csv(value_counts["object_type"], "object_type_counts.csv", ["Object Type", "Documents"])
    write_counter_csv(value_counts["language"], "language_counts.csv", ["Language", "Documents"])
    write_counter_csv(value_counts["publication_raw"], "top_publications_raw.csv", ["Publication Title", "Documents"])
    write_counter_csv(value_counts["publication_normalized"], "top_publications_normalized.csv", ["Normalized Publication Title", "Documents"])
    write_counter_csv(value_counts["publisher_name"], "publisher_name_counts.csv", ["Publisher Name", "Documents"])
    write_counter_csv(value_counts["publisher_city_raw"], "publisher_city_raw_counts.csv", ["Publisher City Raw", "Documents"])
    write_counter_csv(value_counts["publisher_province_raw"], "publisher_province_raw_counts.csv", ["Publisher Province Raw", "Documents"])
    write_counter_csv(value_counts["publisher_city"], "publisher_city_counts.csv", ["Publisher City", "Documents"])
    write_counter_csv(value_counts["publisher_province"], "publisher_province_counts.csv", ["Publisher Province", "Documents"])
    write_counter_csv(value_counts["publisher_region"], "publisher_region_counts.csv", ["Publisher Region", "Documents"])
    write_counter_csv(value_counts["edition_flag"], "edition_flag_counts.csv", ["Edition Flag", "Documents"])
    write_counter_csv(value_counts["year"], "year_counts.csv", ["Year", "Documents"], sort_key=True)
    write_counter_csv(value_counts["decade"], "decade_counts.csv", ["Decade", "Documents"], sort_key=True)
    write_counter_csv(subject_counts, "top_subject_terms.csv", ["Subject Term", "Documents"])
    write_counter_csv(class_counts, "top_class_terms.csv", ["Class Term", "Documents"])
    write_counter_csv(author_counts, "top_authors.csv", ["Author", "Documents"])

    source_decade_df = pd.DataFrame(
        [{"Source Type": k[0], "Decade": k[1], "Documents": v} for k, v in source_decade.items()]
    ).sort_values(["Decade", "Source Type"])
    source_decade_df.to_csv(OUTPUT_DIR / "source_type_by_decade.csv", index=False)

    media_decade_df = pd.DataFrame(
        [{"Normalized Publication Title": k[0], "Decade": k[1], "Documents": v} for k, v in media_decade.items()]
    ).sort_values(["Normalized Publication Title", "Decade"])
    media_decade_df.to_csv(OUTPUT_DIR / "media_by_decade.csv", index=False)

    media_source_df = pd.DataFrame(
        [{"Normalized Publication Title": k[0], "Source Type": k[1], "Documents": v} for k, v in media_source.items()]
    ).sort_values(["Normalized Publication Title", "Source Type"])
    media_source_df.to_csv(OUTPUT_DIR / "media_by_source_type.csv", index=False)

    excluded_df = pd.DataFrame(
        [
            {
                "Normalized Publication Title": key[0],
                "Publication Title": key[1],
                "Publisher City": key[2],
                "Publisher Province": key[3],
                "Publisher Region": key[4],
                "Publisher Name": key[5],
                "Edition Flag": key[6],
                "Exclusion Reason": key[7],
                "Documents": value,
            }
            for key, value in excluded_media.items()
        ]
    ).sort_values("Documents", ascending=False)
    excluded_df.to_csv(OUTPUT_DIR / "excluded_media_summary.csv", index=False)

    media_summary = build_media_summary(
        media_rows,
        media_dates,
        media_locations,
        media_regions,
        media_edition_flags,
        media_raw_titles,
        media_display_titles,
        media_publishers,
        media_author_counts,
        media_terms,
    )
    media_summary.to_csv(OUTPUT_DIR / "media_summary.csv", index=False)

    if page_values:
        all_pages = pd.concat(page_values, ignore_index=True)
        pages_summary = all_pages.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).reset_index()
        pages_summary.columns = ["Statistic", "Pages"]
        pages_summary.to_csv(OUTPUT_DIR / "pages_summary.csv", index=False)

    missing_df = pd.DataFrame(
        [
            {
                "Column": col,
                "Missing Rows": count,
                "Missing Share": count / row_count if row_count else np.nan,
                "Observed Rows": row_count - count,
            }
            for col, count in missing.items()
        ]
    ).sort_values("Missing Share", ascending=False)
    missing_df.to_csv(OUTPUT_DIR / "column_completeness.csv", index=False)

    unique_df = pd.DataFrame([{"Field": k, "Unique Values": v} for k, v in profile["unique_counts"].items()])
    unique_df.to_csv(OUTPUT_DIR / "unique_counts.csv", index=False)
    duplicate_summary.to_csv(OUTPUT_DIR / "duplicate_summary.csv", index=False)
    build_duplicate_examples(same_media_key_counts, same_media_key_examples, "same_media").to_csv(
        OUTPUT_DIR / "duplicate_examples_same_media.csv", index=False
    )
    build_duplicate_examples(title_date_key_counts, title_date_key_examples, "title_date").to_csv(
        OUTPUT_DIR / "duplicate_examples_title_date.csv", index=False
    )

    norm_examples = build_normalization_examples(pub_norm_raw)
    norm_examples.to_csv(OUTPUT_DIR / "media_normalization_examples.csv", index=False)
    topic_sample.to_parquet(OUTPUT_DIR / "topic_sample.parquet", index=False)

    write_core_summary(profile, duplicate_summary)
    write_data_dictionary()

    (OUTPUT_DIR / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def write_counter_csv(counter: Counter, filename: str, columns: list[str], sort_key: bool = False) -> pd.DataFrame:
    items = list(counter.items())
    if sort_key:
        items = sorted(items, key=lambda x: x[0])
    else:
        items = sorted(items, key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(items, columns=columns)
    df.to_csv(OUTPUT_DIR / filename, index=False)
    return df


def build_normalization_examples(pub_norm_raw: Counter) -> pd.DataFrame:
    rows = []
    grouped: dict[str, Counter] = {}
    for (norm, raw), count in pub_norm_raw.items():
        grouped.setdefault(norm, Counter())[raw] += count
    for norm, raws in grouped.items():
        if len(raws) > 1:
            rows.append(
                {
                    "Normalized Publication Title": norm,
                    "Raw Name Variants": len(raws),
                    "Documents": sum(raws.values()),
                    "Examples": "; ".join([name for name, _ in raws.most_common(8)]),
                }
            )
    return pd.DataFrame(rows).sort_values(["Raw Name Variants", "Documents"], ascending=False)


def build_duplicate_examples(counter: Counter, examples: dict, duplicate_type: str) -> pd.DataFrame:
    rows = []
    for key, count in counter.most_common(100):
        if count <= 1:
            continue
        example = examples.get(key, ())
        row = {
            "duplicate_type": duplicate_type,
            "duplicate_rows": count - 1,
            "total_rows": count,
        }
        if duplicate_type == "same_media":
            title, date, media = example
            row.update({"title": title, "date": date, "normalized_publication_title": media})
        else:
            title, date = example
            row.update({"title": title, "date": date})
        rows.append(row)
    return pd.DataFrame(rows)


def build_media_summary(
    media_rows: Counter,
    media_dates: dict[str, list[pd.Timestamp | None]],
    media_locations: dict[str, Counter],
    media_regions: dict[str, Counter],
    media_edition_flags: dict[str, Counter],
    media_raw_titles: dict[str, Counter],
    media_display_titles: dict[str, Counter],
    media_publishers: dict[str, Counter],
    media_author_counts: Counter,
    media_terms: Counter,
) -> pd.DataFrame:
    author_sets: dict[str, set[str]] = {}
    author_top: dict[str, Counter] = {}
    term_top: dict[str, Counter] = {}
    for (media, author), count in media_author_counts.items():
        author_sets.setdefault(media, set()).add(author)
        author_top.setdefault(media, Counter())[author] += count
    for (media, term), count in media_terms.items():
        term_top.setdefault(media, Counter())[term] += count

    rows = []
    for media, documents in media_rows.most_common():
        first, last = media_dates.get(media, [None, None])
        display_title = media_display_titles.get(media, Counter()).most_common(1)
        rows.append(
            {
                "Normalized Publication Title": media,
                "Display Publication Title": display_title[0][0] if display_title else media,
                "Documents": documents,
                "First Date": first.date().isoformat() if first is not None else "",
                "Last Date": last.date().isoformat() if last is not None else "",
                "Raw Name Variants": len(media_raw_titles.get(media, {})),
                "Top Raw Names": "; ".join([name for name, _ in media_raw_titles.get(media, Counter()).most_common(5)]),
                "Top Publisher Locations": "; ".join([name for name, _ in media_locations.get(media, Counter()).most_common(5)]),
                "Publisher Regions": "; ".join([name for name, _ in media_regions.get(media, Counter()).most_common(5)]),
                "Edition Flags": "; ".join([name for name, _ in media_edition_flags.get(media, Counter()).most_common(5)]),
                "Top Publisher Names": "; ".join([name for name, _ in media_publishers.get(media, Counter()).most_common(5)]),
                "Unique Authors": len(author_sets.get(media, set())),
                "Top Authors": "; ".join([name for name, _ in author_top.get(media, Counter()).most_common(8)]),
                "Top Subject Terms": "; ".join([name for name, _ in term_top.get(media, Counter()).most_common(8)]),
            }
        )
    return pd.DataFrame(rows)


def write_core_summary(profile: dict, duplicate_summary: pd.DataFrame) -> None:
    subjects = pd.read_csv(OUTPUT_DIR / "top_subject_terms.csv").head(10)
    source = pd.read_csv(OUTPUT_DIR / "source_type_counts.csv")
    media = pd.read_csv(OUTPUT_DIR / "media_summary.csv")
    regions = pd.read_csv(OUTPUT_DIR / "publisher_region_counts.csv")
    duplicate_map = dict(zip(duplicate_summary["metric"], duplicate_summary["value"]))
    core = pd.DataFrame(
        [
            {"metric": "raw_documents", "value": profile["raw_row_count"]},
            {"metric": "retained_documents", "value": profile["row_count"]},
            {"metric": "excluded_documents", "value": profile["excluded_row_count"]},
            {"metric": "time_range_start", "value": profile["min_date"]},
            {"metric": "time_range_end", "value": profile["max_date"]},
            {"metric": "normalized_media_titles", "value": int(len(media))},
            {"metric": "source_type_count", "value": int(len(source))},
            {"metric": "top_source_type", "value": source.iloc[0]["Source Type"] if len(source) else ""},
            {"metric": "top_source_type_documents", "value": int(source.iloc[0]["Documents"]) if len(source) else 0},
            {"metric": "top_issue_term", "value": subjects.iloc[0]["Subject Term"] if len(subjects) else ""},
            {"metric": "top_issue_documents", "value": int(subjects.iloc[0]["Documents"]) if len(subjects) else 0},
            {"metric": "same_media_duplicate_rows", "value": int(duplicate_map.get("same_title_date_media_duplicate_rows", 0))},
            {"metric": "title_date_duplicate_rows", "value": int(duplicate_map.get("same_title_date_duplicate_rows", 0))},
            {"metric": "publisher_regions", "value": int(len(regions))},
        ]
    )
    core.to_csv(OUTPUT_DIR / "core_summary.csv", index=False)
    subjects.to_csv(OUTPUT_DIR / "core_issue_coverage.csv", index=False)


def write_data_dictionary() -> None:
    rows = [
        ("core_summary.csv", "metric", "Compact metric name."),
        ("core_summary.csv", "value", "Metric value."),
        ("core_issue_coverage.csv", "Subject Term", "Parsed subject/issue term from metadata."),
        ("core_issue_coverage.csv", "Documents", "Rows containing the term."),
        ("media_summary.csv", "Normalized Publication Title", "Stable normalized media identifier used for grouping."),
        ("media_summary.csv", "Display Publication Title", "Human-readable title chosen from the most common raw title variant."),
        ("media_summary.csv", "Documents", "Rows/documents after media-scope filtering."),
        ("media_summary.csv", "First Date", "Earliest parsed article date for the media title."),
        ("media_summary.csv", "Last Date", "Latest parsed article date for the media title."),
        ("media_summary.csv", "Publisher Regions", "North-American geography bucket inferred from publisher city/state metadata."),
        ("media_summary.csv", "Edition Flags", "Edition-like title pattern, if present; retained for exploration rather than filtering."),
        ("duplicate_summary.csv", "metric", "Duplicate/data-quality metric."),
        ("duplicate_summary.csv", "value", "Metric value."),
        ("duplicate_examples_same_media.csv", "duplicate_rows", "Rows beyond the first sharing title, date, and normalized media."),
        ("duplicate_examples_title_date.csv", "duplicate_rows", "Rows beyond the first sharing title and date across any media."),
        ("publisher_city_counts.csv", "Publisher City", "Disambiguated publisher city."),
        ("publisher_city_raw_counts.csv", "Publisher City Raw", "Original publisher city string before disambiguation."),
        ("topic_terms.csv", "Term", "Top title term in an LDA topic."),
        ("topic_by_media.csv", "Sample Documents", "Topic-assigned sample rows for each media title."),
        ("xml_field_manifest.csv", "panel_field", "Recommended metadata panel field."),
        ("xml_field_inventory.csv", "xml_path", "Observed XML tag path from sampled XML exports, if XML files are available."),
        ("metadata_panel.parquet", "journal_range_start", "Earliest observed date for the normalized publication title."),
        ("metadata_panel.parquet", "journal_range_end", "Latest observed date for the normalized publication title."),
        ("metadata_panel.parquet", "metadata_enriched_topic_text", "Title plus subject/class metadata for topic modeling sensitivity checks."),
    ]
    pd.DataFrame(rows, columns=["file", "field", "definition"]).to_csv(OUTPUT_DIR / "data_dictionary.csv", index=False)


def build_cleaned_metadata_outputs() -> dict:
    ensure_dirs()
    selected_rows = 0
    dedup_rows = 0
    duplicate_seen: set[str] = set()
    parquet_parts = []
    dedup_parts = []

    for chunk_index, chunk in enumerate(read_extended_chunks(), start=1):
        chunk = chunk.reindex(columns=KEY_COLUMNS + ["Start Page"], fill_value="")
        locations = [
            normalize_city_state(city, province)
            for city, province in zip(chunk["Publisher City"], chunk["Publisher Province"])
        ]
        region = pd.Series([item[2] for item in locations], index=chunk.index)
        english = chunk["Language"].map(lambda x: clean_text(x).lower()).eq("english")
        keep = region.isin(["United States", "Canada"]) & english
        if not keep.any():
            continue

        kept = chunk.loc[keep].copy()
        kept_locations = [locations[i] for i, idx in enumerate(chunk.index) if keep.loc[idx]]
        dates = pd.to_datetime(kept["Date"], errors="coerce")
        normalized_pub = kept["Publication Title"].map(normalize_publication)
        title_key = kept["Title"].map(compact_key)
        date_key = dates.dt.strftime("%Y-%m-%d").fillna("")
        duplicate_key = title_key + "||" + date_key + "||" + normalized_pub.map(compact_key)

        clean = pd.DataFrame(
            {
                "goid": kept["GOID"].map(clean_text),
                "title": kept["Title"].map(clean_text),
                "date": date_key,
                "year": dates.dt.year.astype("Int64"),
                "decade": (dates.dt.year // 10 * 10).astype("Int64"),
                "source_type": kept["Source Type"].map(clean_text),
                "object_type": kept["Object Type"].map(clean_text),
                "language": kept["Language"].map(clean_text),
                "authors": kept["Authors"].map(clean_text),
                "publication_id": kept["Publication ID"].map(clean_text),
                "publication_title_raw": kept["Publication Title"].map(clean_text),
                "publication_title_normalized": normalized_pub,
                "publication_title_display": [
                    publication_display_name(raw, norm)
                    for raw, norm in zip(kept["Publication Title"], normalized_pub)
                ],
                "publisher_name": kept["Publisher Name"].map(clean_text),
                "publisher_city_raw": kept["Publisher City"].map(clean_text),
                "publisher_province_raw": kept["Publisher Province"].map(clean_text),
                "publisher_city_normalized": [item[0] for item in kept_locations],
                "publisher_province_normalized": [item[1] for item in kept_locations],
                "publisher_region": [item[2] for item in kept_locations],
                "publisher_location_normalized": [item[3] for item in kept_locations],
                "edition_flag": kept["Publication Title"].map(edition_flag).map(lambda x: clean_text(x) or "main_or_unspecified"),
                "pages": pd.to_numeric(kept["Pages"].replace("", np.nan), errors="coerce"),
                "start_page": kept["Start Page"].map(clean_text),
                "company_name": kept["Company Name"].map(clean_text),
                "class_terms": kept["Class Terms"].map(clean_text),
                "subject_terms": kept["Subject Terms"].map(clean_text),
                "has_subject_terms": kept["Subject Terms"].map(clean_text).ne(""),
                "has_class_terms": kept["Class Terms"].map(clean_text).ne(""),
                "has_authors": kept["Authors"].map(clean_text).ne(""),
                "has_pages": kept["Pages"].map(clean_text).ne(""),
                "dedup_key_title_date_media": duplicate_key,
            }
        )
        clean["is_duplicate_title_date_media"] = clean["dedup_key_title_date_media"].duplicated(keep="first")
        clean["is_duplicate_title_date_media"] = [
            key in duplicate_seen or duplicate_flag
            for key, duplicate_flag in zip(clean["dedup_key_title_date_media"], clean["is_duplicate_title_date_media"])
        ]
        duplicate_seen.update(clean["dedup_key_title_date_media"].tolist())

        selected_rows += len(clean)
        dedup = clean.loc[~clean["is_duplicate_title_date_media"]].copy()
        dedup_rows += len(dedup)
        parquet_part = FINAL_DIR / f"cleaned_metadata_part_{chunk_index:02d}.parquet"
        dedup_part = FINAL_DIR / f"cleaned_metadata_dedup_part_{chunk_index:02d}.parquet"
        clean.to_parquet(parquet_part, index=False)
        dedup.to_parquet(dedup_part, index=False)
        parquet_parts.append(parquet_part)
        dedup_parts.append(dedup_part)

    cleaned = pd.concat([pd.read_parquet(path) for path in parquet_parts], ignore_index=True)
    deduped = pd.concat([pd.read_parquet(path) for path in dedup_parts], ignore_index=True)
    cleaned_parquet = FINAL_DIR / "cleaned_metadata_english_north_america.parquet"
    dedup_parquet = FINAL_DIR / "cleaned_metadata_english_north_america_dedup.parquet"
    cleaned_csv_zip = FINAL_DIR / "cleaned_metadata_english_north_america.csv.zip"
    dedup_csv_zip = FINAL_DIR / "cleaned_metadata_english_north_america_dedup.csv.zip"
    cleaned.to_parquet(cleaned_parquet, index=False)
    deduped.to_parquet(dedup_parquet, index=False)
    cleaned.to_csv(cleaned_csv_zip, index=False, compression={"method": "zip", "archive_name": "cleaned_metadata_english_north_america.csv"})
    deduped.to_csv(dedup_csv_zip, index=False, compression={"method": "zip", "archive_name": "cleaned_metadata_english_north_america_dedup.csv"})

    for path in parquet_parts + dedup_parts:
        path.unlink(missing_ok=True)

    summary = {
        "cleaned_rows": int(selected_rows),
        "deduplicated_rows": int(dedup_rows),
        "duplicate_rows_removed": int(selected_rows - dedup_rows),
        "cleaned_parquet": cleaned_parquet.name,
        "cleaned_csv_zip": cleaned_csv_zip.name,
        "deduplicated_parquet": dedup_parquet.name,
        "deduplicated_csv_zip": dedup_csv_zip.name,
    }
    (FINAL_DIR / "cleaned_metadata_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_cleaned_metadata_dictionary()
    return summary


def write_cleaned_metadata_dictionary() -> None:
    rows = [
        ("goid", "Original document identifier."),
        ("title", "Article/document title."),
        ("date", "Parsed document date in YYYY-MM-DD format."),
        ("year", "Parsed publication year."),
        ("decade", "Year rounded down to decade."),
        ("source_type", "Original source type."),
        ("object_type", "Original object type."),
        ("language", "Original language value; final cleaned set keeps English only."),
        ("authors", "Original authors metadata string."),
        ("publication_title_raw", "Original publication title."),
        ("publication_title_normalized", "Normalized publication grouping key."),
        ("publication_title_display", "Human-readable publication title."),
        ("publisher_city_normalized", "Disambiguated publisher city."),
        ("publisher_province_normalized", "Disambiguated state/province code."),
        ("publisher_region", "United States or Canada."),
        ("publisher_location_normalized", "City/state display string."),
        ("edition_flag", "Edition-like title pattern; retained for analysis."),
        ("has_subject_terms", "Whether Subject Terms is nonblank."),
        ("has_class_terms", "Whether Class Terms is nonblank."),
        ("has_authors", "Whether Authors is nonblank."),
        ("has_pages", "Whether Pages is nonblank."),
        ("dedup_key_title_date_media", "Conservative duplicate key: normalized title + date + normalized publication."),
        ("is_duplicate_title_date_media", "True for rows beyond the first sharing the conservative duplicate key."),
    ]
    pd.DataFrame(rows, columns=["field", "definition"]).to_csv(FINAL_DIR / "cleaned_metadata_data_dictionary.csv", index=False)


def write_xml_field_manifest() -> pd.DataFrame:
    rows = []
    observed_columns = set()
    with zipfile.ZipFile(EXTENDED_ZIP) as zf:
        with zf.open(EXTENDED_CSV) as f:
            observed_columns = set(pd.read_csv(f, nrows=0).columns)

    for item in XML_FIELD_MAP:
        csv_columns = item["csv_columns"]
        available_csv_columns = [col for col in csv_columns if col in observed_columns]
        rows.append(
            {
                "panel_field": item["panel_field"],
                "role": item["role"],
                "csv_columns": "; ".join(csv_columns),
                "available_csv_columns": "; ".join(available_csv_columns),
                "csv_available": bool(available_csv_columns),
                "xml_paths_to_check": "; ".join(item["xml_paths"]),
                "notes": item["notes"],
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUTPUT_DIR / "xml_field_manifest.csv", index=False)
    return manifest


def _local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _iter_xml_sources():
    for base in [DATA_DIR, ROOT / "raw"]:
        if not base.exists():
            continue
        for path in base.rglob("*.xml"):
            yield path.as_posix(), path.open("rb")
        for path in base.rglob("*.zip"):
            try:
                with zipfile.ZipFile(path) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(".xml"):
                            yield f"{path.as_posix()}::{name}", zf.open(name)
            except zipfile.BadZipFile:
                continue


def scan_xml_field_inventory(max_files: int = 100) -> pd.DataFrame:
    ensure_dirs()
    rows = []
    parsed_files = 0
    path_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    for source_name, handle in _iter_xml_sources():
        if parsed_files >= max_files:
            break
        parsed_files += 1
        stack: list[str] = []
        try:
            with handle:
                for event, elem in ET.iterparse(handle, events=("start", "end")):
                    if event == "start":
                        stack.append(_local_xml_name(elem.tag))
                        continue

                    xml_path = "/".join(stack)
                    text = clean_text(elem.text)
                    if text:
                        path_counts[xml_path] += 1
                        sample_values = examples.setdefault(xml_path, [])
                        if len(sample_values) < 3 and text not in sample_values:
                            sample_values.append(text[:160])
                    elem.clear()
                    if stack:
                        stack.pop()
        except ET.ParseError as exc:
            rows.append(
                {
                    "xml_path": "",
                    "observed_values": 0,
                    "example_values": "",
                    "source_note": f"Could not parse {source_name}: {exc}",
                }
            )

    if path_counts:
        rows.extend(
            {
                "xml_path": path,
                "observed_values": count,
                "example_values": " | ".join(examples.get(path, [])),
                "source_note": f"sampled {parsed_files} XML file(s)",
            }
            for path, count in path_counts.most_common()
        )
    elif not rows:
        rows.append(
            {
                "xml_path": "",
                "observed_values": 0,
                "example_values": "",
                "source_note": "No XML files were found in data/raw_extracted or raw; current export contains CSV metadata only.",
            }
        )

    inventory = pd.DataFrame(rows)
    inventory.to_csv(OUTPUT_DIR / "xml_field_inventory.csv", index=False)
    write_xml_field_manifest()
    return inventory


def _join_terms_for_topic(value: object, limit: int = 12) -> str:
    terms = [canonical_term(term) for term in parse_term_value(value)]
    terms = [term for term in terms if term]
    return " ".join(terms[:limit])


def _primary_author(value: object) -> str:
    authors = [canonical_term(author) for author in parse_term_value(value)]
    return authors[0] if authors else ""


def build_metadata_panel_outputs() -> dict:
    ensure_dirs()
    cleaned_path = FINAL_DIR / "cleaned_metadata_english_north_america.parquet"
    if not cleaned_path.exists():
        build_cleaned_metadata_outputs()

    df = pd.read_parquet(cleaned_path)
    dates = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = dates.dt.strftime("%Y-%m-%d")
    df["date"] = df["date"].fillna("")

    journal = (
        df.assign(_date=dates)
        .groupby("publication_title_normalized", dropna=False)
        .agg(
            journal_range_start=("_date", "min"),
            journal_range_end=("_date", "max"),
            journal_document_count=("goid", "size"),
            journal_observed_years=("year", "nunique"),
        )
        .reset_index()
    )
    journal["journal_range_start"] = journal["journal_range_start"].dt.strftime("%Y-%m-%d").fillna("")
    journal["journal_range_end"] = journal["journal_range_end"].dt.strftime("%Y-%m-%d").fillna("")
    start_year = pd.to_datetime(journal["journal_range_start"], errors="coerce").dt.year
    end_year = pd.to_datetime(journal["journal_range_end"], errors="coerce").dt.year
    journal["journal_range_years"] = (end_year - start_year + 1).clip(lower=0).astype("Int64")
    journal["journal_observed_year_share"] = (
        journal["journal_observed_years"] / journal["journal_range_years"].replace(0, np.nan)
    ).round(4)

    panel = df.merge(journal, on="publication_title_normalized", how="left")
    raw_title = panel["publication_title_raw"].fillna("")
    object_type = panel["object_type"].fillna("")
    index_only = raw_title.str.contains("index-only", case=False, na=False)
    undefined_object = object_type.str.lower().eq("undefined")
    panel["full_text_status"] = np.select(
        [index_only, undefined_object],
        ["publication_marked_index_only", "object_type_undefined"],
        default="not_observed_in_metadata_export",
    )
    panel["metadata_suggests_full_text_record"] = ~(index_only | undefined_object)
    panel["primary_author"] = panel["authors"].map(_primary_author)
    panel["author_count"] = panel["authors"].map(lambda value: len(parse_term_value(value)))
    panel["title_topic_text"] = panel["title"].fillna("").map(clean_text)
    panel["metadata_enriched_topic_text"] = (
        panel["title_topic_text"]
        + " "
        + panel["subject_terms"].map(_join_terms_for_topic)
        + " "
        + panel["class_terms"].map(_join_terms_for_topic)
    ).str.strip()

    keep_cols = [
        "goid",
        "title",
        "date",
        "year",
        "decade",
        "source_type",
        "object_type",
        "language",
        "authors",
        "primary_author",
        "author_count",
        "publication_id",
        "publication_title_raw",
        "publication_title_normalized",
        "publication_title_display",
        "journal_range_start",
        "journal_range_end",
        "journal_range_years",
        "journal_observed_years",
        "journal_observed_year_share",
        "journal_document_count",
        "publisher_name",
        "publisher_location_normalized",
        "publisher_region",
        "edition_flag",
        "pages",
        "start_page",
        "company_name",
        "class_terms",
        "subject_terms",
        "has_subject_terms",
        "has_class_terms",
        "has_authors",
        "has_pages",
        "full_text_status",
        "metadata_suggests_full_text_record",
        "title_topic_text",
        "metadata_enriched_topic_text",
        "dedup_key_title_date_media",
        "is_duplicate_title_date_media",
    ]
    panel = panel[keep_cols]

    panel_path = FINAL_DIR / "metadata_panel.parquet"
    panel_csv_zip = FINAL_DIR / "metadata_panel.csv.zip"
    journal_path = OUTPUT_DIR / "journal_coverage_summary.csv"
    panel.to_parquet(panel_path, index=False)
    panel.to_csv(panel_csv_zip, index=False, compression={"method": "zip", "archive_name": "metadata_panel.csv"})
    journal.sort_values("journal_document_count", ascending=False).to_csv(journal_path, index=False)
    write_metadata_panel_dictionary()
    write_metadata_export_manifest()

    summary = {
        "panel_rows": int(len(panel)),
        "panel_columns": int(len(panel.columns)),
        "panel_parquet": panel_path.name,
        "panel_csv_zip": panel_csv_zip.name,
        "journal_coverage_summary": journal_path.relative_to(ROOT).as_posix(),
        "full_text_status_counts": panel["full_text_status"].value_counts().to_dict(),
    }
    (FINAL_DIR / "metadata_panel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_metadata_panel_dictionary() -> None:
    rows = [
        ("goid", "Document-level identifier; one row in the panel is one metadata record."),
        ("title", "Document title."),
        ("source_type", "Broad source type, used as genre/source-family metadata."),
        ("object_type", "Document object type."),
        ("primary_author", "First parsed author where available."),
        ("author_count", "Number of parsed author names in the Authors field."),
        ("publication_title_normalized", "Stable normalized journal/publication/media key."),
        ("journal_range_start", "Earliest observed document date for the normalized publication."),
        ("journal_range_end", "Latest observed document date for the normalized publication."),
        ("journal_observed_year_share", "Observed publication years divided by inclusive first-to-last year range."),
        ("full_text_status", "Full-text availability evidence from metadata; current CSV has no explicit full-text field."),
        ("metadata_suggests_full_text_record", "False for index-only or undefined-object records; exploratory proxy, not proof of article body availability."),
        ("title_topic_text", "Title-only text for conservative topic modeling."),
        ("metadata_enriched_topic_text", "Title plus subject/class terms for a metadata-enriched topic modeling sensitivity run."),
    ]
    pd.DataFrame(rows, columns=["field", "definition"]).to_csv(FINAL_DIR / "metadata_panel_data_dictionary.csv", index=False)


def write_metadata_export_manifest() -> pd.DataFrame:
    paths = [
        FINAL_DIR / "metadata_panel.parquet",
        FINAL_DIR / "metadata_panel.csv.zip",
        FINAL_DIR / "metadata_panel_summary.json",
        FINAL_DIR / "metadata_panel_data_dictionary.csv",
        OUTPUT_DIR / "journal_coverage_summary.csv",
        OUTPUT_DIR / "xml_field_manifest.csv",
        OUTPUT_DIR / "xml_field_inventory.csv",
    ]
    rows = []
    for path in paths:
        rows.append(
            {
                "file": path.relative_to(ROOT).as_posix(),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "purpose": {
                    "metadata_panel.parquet": "Primary efficient panel for exploration and modeling.",
                    "metadata_panel.csv.zip": "Portable compressed CSV version of the metadata panel.",
                    "metadata_panel_summary.json": "Compact row/column and full-text-status summary.",
                    "metadata_panel_data_dictionary.csv": "Field definitions for the panel.",
                    "journal_coverage_summary.csv": "Publication-level range, coverage, and volume summary.",
                    "xml_field_manifest.csv": "Planned XML/CSV field map for metadata extraction.",
                    "xml_field_inventory.csv": "Observed XML field inventory from sampled XML files, when XML exists.",
                }.get(path.name, ""),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(FINAL_DIR / "metadata_export_manifest.csv", index=False)
    return manifest


def write_metadata_field_audit() -> pd.DataFrame:
    scan_xml_field_inventory()
    xml_manifest = pd.read_csv(OUTPUT_DIR / "xml_field_manifest.csv")
    full_text_csv_available = bool(
        xml_manifest.loc[xml_manifest["panel_field"].eq("full_text_status"), "csv_available"].fillna(False).any()
    )
    rows = [
        {
            "question": "issue_or_volume_metadata_present",
            "answer": "no",
            "evidence": "The current CSV export has Date, Pages, and Start Page, but no issue/volume enumeration field. XML inventory should be re-run if XML files are added.",
        },
        {
            "question": "issue_topic_metadata_present",
            "answer": "yes",
            "evidence": "Subject Terms and Class Terms provide topical/issue categories, but not serial issue completeness.",
        },
        {
            "question": "can_assess_complete_issue_coverage",
            "answer": "no",
            "evidence": "Date, publication title, pages, and start page can show temporal/article density, but cannot prove all issues of a serial publication are present.",
        },
        {
            "question": "full_text_online_field_present",
            "answer": "yes" if full_text_csv_available else "no",
            "evidence": "No explicit full-text availability column is present in the current CSV export; the XML manifest lists candidate tags to scan if XML files are available.",
        },
        {
            "question": "full_text_available_in_current_export",
            "answer": "no",
            "evidence": "The current local raw files contain citation.csv and extended.csv metadata exports only. The panel includes an exploratory index-only/undefined-object proxy, not a proof of full text.",
        },
        {
            "question": "metadata_panel_ready_for_topic_modeling",
            "answer": "yes",
            "evidence": "metadata_panel.parquet contains title_topic_text and metadata_enriched_topic_text, plus genre, journal, author, date, and coverage fields.",
        },
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(FINAL_DIR / "metadata_field_audit.csv", index=False)
    return audit


def write_final_summary_workbook() -> Path:
    write_metadata_field_audit()
    write_metadata_export_manifest()
    final_xlsx = FINAL_DIR / "summary_statistics_final.xlsx"
    sheets = {
        "Core Summary": pd.read_csv(OUTPUT_DIR / "core_summary.csv"),
        "Media Summary": pd.read_csv(OUTPUT_DIR / "media_summary.csv").head(40),
        "Issue Coverage": pd.read_csv(OUTPUT_DIR / "core_issue_coverage.csv"),
        "Data Quality": pd.read_csv(OUTPUT_DIR / "duplicate_summary.csv"),
        "Completeness": pd.read_csv(OUTPUT_DIR / "column_completeness.csv").head(12),
        "Publisher Cities": pd.read_csv(OUTPUT_DIR / "publisher_city_counts.csv"),
        "Type By Year": pd.read_csv(OUTPUT_DIR / "source_type_by_year_english_na.csv"),
        "Edition Flags": pd.read_csv(OUTPUT_DIR / "edition_flag_counts.csv"),
        "Topic Counts": pd.read_csv(OUTPUT_DIR / "topic_counts.csv"),
        "Field Audit": pd.read_csv(FINAL_DIR / "metadata_field_audit.csv"),
        "Dictionary": pd.read_csv(FINAL_DIR / "cleaned_metadata_data_dictionary.csv"),
        "XML Field Map": pd.read_csv(OUTPUT_DIR / "xml_field_manifest.csv"),
        "XML Inventory": pd.read_csv(OUTPUT_DIR / "xml_field_inventory.csv").head(100),
        "Export Manifest": pd.read_csv(FINAL_DIR / "metadata_export_manifest.csv"),
    }
    panel_dictionary = FINAL_DIR / "metadata_panel_data_dictionary.csv"
    journal_summary = OUTPUT_DIR / "journal_coverage_summary.csv"
    if panel_dictionary.exists():
        sheets["Panel Dictionary"] = pd.read_csv(panel_dictionary)
    if journal_summary.exists():
        sheets["Journal Coverage"] = pd.read_csv(journal_summary).head(75)
    with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return final_xlsx


def run_topic_modeling(n_topics: int = 12, max_features: int = 6_000) -> dict:
    cleaned_path = FINAL_DIR / "cleaned_metadata_english_north_america.parquet"
    if cleaned_path.exists():
        cols = [
            "goid",
            "title",
            "date",
            "source_type",
            "publication_title_normalized",
            "publisher_region",
            "edition_flag",
            "decade",
        ]
        base = pd.read_parquet(cleaned_path, columns=cols)
        base = base[base["title"].fillna("").str.len() > 10].copy()
        sample_n = min(100_000, len(base))
        base = base.sample(n=sample_n, random_state=42).reset_index(drop=True)
        sample = pd.DataFrame(
            {
                "GOID": base["goid"],
                "Title": base["title"],
                "Date": base["date"],
                "Source Type": base["source_type"],
                "Publication Title Normalized": base["publication_title_normalized"],
                "Publisher Region": base["publisher_region"],
                "Edition Flag": base["edition_flag"],
                "Decade": base["decade"],
                "Topic Text": base["title"].map(clean_text),
            }
        )
        source_scope = "cleaned English North-American metadata"
    else:
        sample_path = OUTPUT_DIR / "topic_sample.parquet"
        sample = pd.read_parquet(sample_path)
        sample = sample[sample["Topic Text"].fillna("").str.len() > 10].copy()
        source_scope = "legacy topic sample"

    vectorizer = CountVectorizer(
        stop_words=list(STOPWORDS),
        min_df=10,
        max_df=0.55,
        max_features=max_features,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",
    )
    X = vectorizer.fit_transform(sample["Topic Text"])
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
        max_iter=10,
        n_jobs=1,
    )
    doc_topic = lda.fit_transform(X)
    sample["Topic"] = doc_topic.argmax(axis=1)
    sample["Topic Confidence"] = doc_topic.max(axis=1)

    feature_names = np.array(vectorizer.get_feature_names_out())
    topic_rows = []
    topic_labels = {}
    for topic_idx, weights in enumerate(lda.components_):
        top_idx = weights.argsort()[-15:][::-1]
        terms = feature_names[top_idx].tolist()
        topic_labels[topic_idx] = ", ".join(terms[:5])
        for rank, term_idx in enumerate(top_idx, start=1):
            topic_rows.append(
                {
                    "Topic": topic_idx,
                    "Rank": rank,
                    "Term": feature_names[term_idx],
                    "Weight": float(weights[term_idx]),
                    "Topic Label": topic_labels[topic_idx],
                }
            )

    topic_terms = pd.DataFrame(topic_rows)
    topic_terms.to_csv(OUTPUT_DIR / "topic_terms.csv", index=False)

    topic_counts = (
        sample.groupby("Topic", dropna=False)
        .size()
        .reset_index(name="Sample Documents")
        .assign(**{"Topic Label": lambda d: d["Topic"].map(topic_labels)})
        .sort_values("Sample Documents", ascending=False)
    )
    topic_counts.to_csv(OUTPUT_DIR / "topic_counts.csv", index=False)

    topic_by_decade = (
        sample.dropna(subset=["Decade"])
        .groupby(["Decade", "Topic"])
        .size()
        .reset_index(name="Sample Documents")
        .assign(**{"Topic Label": lambda d: d["Topic"].map(topic_labels)})
    )
    topic_by_decade.to_csv(OUTPUT_DIR / "topic_by_decade.csv", index=False)

    topic_by_source = (
        sample.groupby(["Source Type", "Topic"])
        .size()
        .reset_index(name="Sample Documents")
        .assign(**{"Topic Label": lambda d: d["Topic"].map(topic_labels)})
    )
    topic_by_source.to_csv(OUTPUT_DIR / "topic_by_source_type.csv", index=False)

    topic_by_media = (
        sample.groupby(["Publication Title Normalized", "Topic"])
        .size()
        .reset_index(name="Sample Documents")
        .assign(**{"Topic Label": lambda d: d["Topic"].map(topic_labels)})
        .sort_values(["Publication Title Normalized", "Sample Documents"], ascending=[True, False])
    )
    topic_by_media.to_csv(OUTPUT_DIR / "topic_by_media.csv", index=False)

    sample[["GOID", "Title", "Date", "Source Type", "Publication Title Normalized", "Publisher Region", "Edition Flag", "Decade", "Topic", "Topic Confidence"]].to_csv(
        OUTPUT_DIR / "topic_sample_assignments.csv", index=False
    )

    make_wordclouds(sample, vectorizer, lda, feature_names)
    info = {
        "topics": n_topics,
        "documents": int(len(sample)),
        "features": int(len(feature_names)),
        "source_scope": source_scope,
        "language_filter": "English" if cleaned_path.exists() else "not guaranteed",
    }
    (OUTPUT_DIR / "topic_model_profile.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def make_wordcloud(text: str, path: Path, title: str | None = None) -> None:
    if not text.strip():
        return
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        stopwords=STOPWORDS,
        collocations=False,
        max_words=180,
    ).generate(text)
    plt.figure(figsize=(12, 6.75))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def make_wordclouds(sample: pd.DataFrame, vectorizer: CountVectorizer, lda: LatentDirichletAllocation, feature_names: np.ndarray) -> None:
    make_wordcloud(" ".join(sample["Topic Text"].dropna().astype(str).tolist()), FIGURE_DIR / "wordcloud_overall.png", "Overall Title Terms")

    for source in sample["Source Type"].value_counts().head(4).index:
        text = " ".join(sample.loc[sample["Source Type"] == source, "Topic Text"].dropna().astype(str).tolist())
        safe = re.sub(r"[^A-Za-z0-9]+", "_", str(source)).strip("_").lower()
        make_wordcloud(text, FIGURE_DIR / f"wordcloud_source_{safe}.png", f"Title terms: {source}")

    for decade in sample["Decade"].dropna().astype(int).value_counts().sort_index().tail(6).index:
        text = " ".join(sample.loc[sample["Decade"].astype("Int64") == decade, "Topic Text"].dropna().astype(str).tolist())
        make_wordcloud(text, FIGURE_DIR / f"wordcloud_decade_{decade}s.png", f"Title terms: {decade}s")

    for topic_idx, weights in enumerate(lda.components_):
        freqs = {feature_names[i]: float(weights[i]) for i in weights.argsort()[-80:]}
        wc = WordCloud(width=1200, height=800, background_color="white", stopwords=STOPWORDS, collocations=False).generate_from_frequencies(freqs)
        plt.figure(figsize=(9, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Topic {topic_idx}")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"wordcloud_topic_{topic_idx:02d}.png", dpi=160, bbox_inches="tight")
        plt.close()


def make_plots() -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    make_core_snapshot_plot()
    write_cleaned_timeseries_tables()

    year = pd.read_csv(OUTPUT_DIR / "year_counts.csv")
    plt.figure(figsize=(14, 5))
    sns.lineplot(data=year, x="Year", y="Documents", linewidth=1.8)
    plt.title("Document Count by Year")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "documents_by_year.png", dpi=160)
    plt.close()

    source = pd.read_csv(OUTPUT_DIR / "source_type_counts.csv").head(12)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=source, y="Source Type", x="Documents", color="#4C78A8")
    plt.title("Top Source Types")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "source_type_counts.png", dpi=160)
    plt.close()

    sy = pd.read_csv(OUTPUT_DIR / "source_type_by_year_english_na.csv")
    plt.figure(figsize=(13, 5))
    sns.lineplot(data=sy, x="year", y="documents", hue="source_type", linewidth=2)
    plt.title("English North-American Documents by Source Type and Year")
    plt.xlabel("")
    plt.ylabel("Documents")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "source_type_by_year_lines.png", dpi=170)
    plt.close()

    recent = sy[sy["year"] >= 1980].copy()
    plt.figure(figsize=(13, 5))
    sns.lineplot(data=recent, x="year", y="documents", hue="source_type", linewidth=2)
    plt.title("English North-American Documents by Source Type Since 1980")
    plt.xlabel("")
    plt.ylabel("Documents")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "source_type_by_year_since_1980.png", dpi=170)
    plt.close()

    pubs = pd.read_csv(OUTPUT_DIR / "top_publications_normalized.csv").head(20)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=pubs, y="Normalized Publication Title", x="Documents", color="#59A14F")
    plt.title("Top Normalized Publications")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "top_normalized_publications.png", dpi=160)
    plt.close()

    media = pd.read_csv(OUTPUT_DIR / "media_summary.csv").head(25)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=media, y="Display Publication Title", x="Documents", color="#76B7B2")
    plt.title("Documents by Retained North-American Publisher Media")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "media_summary_counts.png", dpi=160)
    plt.close()

    city = pd.read_csv(OUTPUT_DIR / "publisher_city_counts.csv").head(15)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=city, y="Publisher City", x="Documents", color="#9C755F")
    plt.title("Publisher City Counts After Disambiguation")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "publisher_city_counts.png", dpi=160)
    plt.close()

    comp = pd.read_csv(OUTPUT_DIR / "column_completeness.csv").sort_values("Missing Share", ascending=True)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=comp, y="Column", x="Missing Share", color="#E15759")
    plt.title("Missing Share by Column")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "column_missingness.png", dpi=160)
    plt.close()

    sd = pd.read_csv(OUTPUT_DIR / "source_type_by_decade.csv")
    if not sd.empty:
        pivot = sd.pivot_table(index="Source Type", columns="Decade", values="Documents", aggfunc="sum", fill_value=0)
        top_sources = pivot.sum(axis=1).sort_values(ascending=False).head(8).index
        pivot = pivot.loc[top_sources]
        plt.figure(figsize=(13, 5.5))
        sns.heatmap(np.log1p(pivot), cmap="viridis", cbar_kws={"label": "log1p(Documents)"})
        plt.title("Source Type Coverage by Decade")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "source_type_by_decade_heatmap.png", dpi=160)
        plt.close()

    topic_counts_path = OUTPUT_DIR / "topic_counts.csv"
    if topic_counts_path.exists():
        topics = pd.read_csv(topic_counts_path)
        plt.figure(figsize=(10, 5))
        sns.barplot(data=topics, y="Topic Label", x="Sample Documents", color="#F28E2B")
        plt.title("Topic Model: Sample Documents by Topic")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "topic_counts.png", dpi=160)
        plt.close()

        tbd = pd.read_csv(OUTPUT_DIR / "topic_by_decade.csv")
        if not tbd.empty:
            pivot = tbd.pivot_table(index="Topic Label", columns="Decade", values="Sample Documents", aggfunc="sum", fill_value=0)
            plt.figure(figsize=(13, 6))
            sns.heatmap(pivot, cmap="mako", cbar_kws={"label": "Sample Documents"})
            plt.title("Topic Coverage by Decade")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / "topic_by_decade_heatmap.png", dpi=160)
            plt.close()

        tbm = pd.read_csv(OUTPUT_DIR / "topic_by_media.csv")
        if not tbm.empty:
            top_media = tbm.groupby("Publication Title Normalized")["Sample Documents"].sum().sort_values(ascending=False).head(15).index
            pivot = tbm[tbm["Publication Title Normalized"].isin(top_media)].pivot_table(
                index="Publication Title Normalized",
                columns="Topic Label",
                values="Sample Documents",
                aggfunc="sum",
                fill_value=0,
            )
            plt.figure(figsize=(14, 7))
            sns.heatmap(pivot, cmap="rocket_r", cbar_kws={"label": "Sample Documents"})
            plt.title("Title Topic Mix by Top Media")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / "topic_by_media_heatmap.png", dpi=160)
            plt.close()


def write_cleaned_timeseries_tables() -> None:
    cleaned_path = FINAL_DIR / "cleaned_metadata_english_north_america.parquet"
    if not cleaned_path.exists():
        return
    df = pd.read_parquet(cleaned_path, columns=["year", "decade", "source_type", "publication_title_normalized"])
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    by_year = (
        df.groupby(["year", "source_type"])
        .size()
        .reset_index(name="documents")
        .sort_values(["year", "source_type"])
    )
    by_decade = (
        df.groupby(["decade", "source_type"])
        .size()
        .reset_index(name="documents")
        .sort_values(["decade", "source_type"])
    )
    by_year.to_csv(OUTPUT_DIR / "source_type_by_year_english_na.csv", index=False)
    by_decade.to_csv(OUTPUT_DIR / "source_type_by_decade_english_na.csv", index=False)


def make_core_snapshot_plot() -> None:
    profile = json.loads((OUTPUT_DIR / "profile.json").read_text(encoding="utf-8"))
    source = pd.read_csv(OUTPUT_DIR / "source_type_counts.csv")
    issues = pd.read_csv(OUTPUT_DIR / "core_issue_coverage.csv")
    decades = pd.read_csv(OUTPUT_DIR / "decade_counts.csv")

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.1])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1:])

    for ax in [ax0, ax1, ax2]:
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, linewidth=1.2, edgecolor="#cccccc"))

    ax0.text(0.06, 0.68, f"{int(profile['row_count']):,}", fontsize=24, fontweight="bold")
    ax0.text(0.06, 0.42, "retained documents", fontsize=12)
    ax0.text(0.06, 0.22, f"{int(profile['raw_row_count']):,} raw rows", fontsize=10, color="#555555")

    ax1.text(0.06, 0.64, f"{profile['min_date']} ->", fontsize=16, fontweight="bold")
    ax1.text(0.06, 0.42, f"{profile['max_date']}", fontsize=16, fontweight="bold")
    ax1.text(0.06, 0.22, "media time range", fontsize=10, color="#555555")

    top_issue = issues.iloc[0]
    ax2.text(0.06, 0.64, str(top_issue["Subject Term"])[:26], fontsize=15, fontweight="bold")
    ax2.text(0.06, 0.42, f"{int(top_issue['Documents']):,} documents", fontsize=14)
    ax2.text(0.06, 0.22, "top issue term", fontsize=10, color="#555555")

    sns.barplot(data=source, x="Documents", y="Source Type", ax=ax3, color="#4C78A8")
    ax3.set_title("Documents by Source Type")
    ax3.set_xlabel("")
    ax3.set_ylabel("")

    sns.lineplot(data=decades, x="Decade", y="Documents", marker="o", ax=ax4, color="#F28E2B")
    ax4.set_title("Documents by Decade")
    ax4.set_xlabel("")
    ax4.set_ylabel("")
    fig.suptitle("Core Coverage Snapshot: Time Range, Document Volume, and Issue Coverage", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURE_DIR / "core_coverage_snapshot.png", dpi=170)
    plt.close(fig)

def write_excel_summary() -> None:
    sheets = {
        "Core Summary": pd.read_csv(OUTPUT_DIR / "core_summary.csv"),
        "Core Issue Coverage": pd.read_csv(OUTPUT_DIR / "core_issue_coverage.csv"),
        "Duplicate Summary": pd.read_csv(OUTPUT_DIR / "duplicate_summary.csv"),
        "Completeness": pd.read_csv(OUTPUT_DIR / "column_completeness.csv").head(12),
        "Source Types": pd.read_csv(OUTPUT_DIR / "source_type_counts.csv"),
        "Publisher Cities": pd.read_csv(OUTPUT_DIR / "publisher_city_counts.csv"),
        "Decades": pd.read_csv(OUTPUT_DIR / "decade_counts.csv"),
        "Media Summary": pd.read_csv(OUTPUT_DIR / "media_summary.csv").head(75),
        "Excluded Media": pd.read_csv(OUTPUT_DIR / "excluded_media_summary.csv"),
        "Authors": pd.read_csv(OUTPUT_DIR / "top_authors.csv").head(50),
        "Subject Terms": pd.read_csv(OUTPUT_DIR / "top_subject_terms.csv").head(50),
        "Class Terms": pd.read_csv(OUTPUT_DIR / "top_class_terms.csv").head(50),
        "Normalization Examples": pd.read_csv(OUTPUT_DIR / "media_normalization_examples.csv").head(200),
        "Data Dictionary": pd.read_csv(OUTPUT_DIR / "data_dictionary.csv"),
    }
    topic_path = OUTPUT_DIR / "topic_terms.csv"
    if topic_path.exists():
        sheets["Topic Terms"] = pd.read_csv(topic_path)
        sheets["Topic Counts"] = pd.read_csv(OUTPUT_DIR / "topic_counts.csv")
        sheets["Topic By Media"] = pd.read_csv(OUTPUT_DIR / "topic_by_media.csv").head(500)
    target = SUMMARY_XLSX
    try:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
    except PermissionError as exc:
        target = SUMMARY_XLSX_FALLBACK
        record_processing_note(
            f"PermissionError while writing {SUMMARY_XLSX.name}; wrote fallback workbook {SUMMARY_XLSX_FALLBACK.name}. Original error: {exc}"
        )
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)


def pct(value: float) -> str:
    return f"{value:.2%}"


def write_report() -> None:
    profile = json.loads((OUTPUT_DIR / "profile.json").read_text(encoding="utf-8"))
    core = pd.read_csv(OUTPUT_DIR / "core_summary.csv")
    duplicates = pd.read_csv(OUTPUT_DIR / "duplicate_summary.csv")
    source = pd.read_csv(OUTPUT_DIR / "source_type_counts.csv")
    regions = pd.read_csv(OUTPUT_DIR / "publisher_region_counts.csv")
    edition_flags = pd.read_csv(OUTPUT_DIR / "edition_flag_counts.csv")
    media_summary = pd.read_csv(OUTPUT_DIR / "media_summary.csv").fillna("")
    excluded_media = pd.read_csv(OUTPUT_DIR / "excluded_media_summary.csv").fillna("")
    completeness = pd.read_csv(OUTPUT_DIR / "column_completeness.csv")
    subjects = pd.read_csv(OUTPUT_DIR / "top_subject_terms.csv")
    classes = pd.read_csv(OUTPUT_DIR / "top_class_terms.csv")
    authors = pd.read_csv(OUTPUT_DIR / "top_authors.csv")
    cities = pd.read_csv(OUTPUT_DIR / "publisher_city_counts.csv")
    raw_cities = pd.read_csv(OUTPUT_DIR / "publisher_city_raw_counts.csv")
    decades = pd.read_csv(OUTPUT_DIR / "decade_counts.csv")
    topics = pd.read_csv(OUTPUT_DIR / "topic_terms.csv")
    topic_counts = pd.read_csv(OUTPUT_DIR / "topic_counts.csv")
    topic_profile = json.loads((OUTPUT_DIR / "topic_model_profile.json").read_text(encoding="utf-8")) if (OUTPUT_DIR / "topic_model_profile.json").exists() else {}
    norm_examples = pd.read_csv(OUTPUT_DIR / "media_normalization_examples.csv")
    field_audit = write_metadata_field_audit()
    cleaned_summary_path = FINAL_DIR / "cleaned_metadata_summary.json"
    cleaned_summary = json.loads(cleaned_summary_path.read_text(encoding="utf-8")) if cleaned_summary_path.exists() else {}
    panel_summary_path = FINAL_DIR / "metadata_panel_summary.json"
    panel_summary = json.loads(panel_summary_path.read_text(encoding="utf-8")) if panel_summary_path.exists() else {}
    processing_notes = json.loads(PROCESSING_NOTES_PATH.read_text(encoding="utf-8")) if PROCESSING_NOTES_PATH.exists() else []
    workbook_path = SUMMARY_XLSX_FALLBACK if SUMMARY_XLSX_FALLBACK.exists() and processing_notes else SUMMARY_XLSX

    rows = int(profile["row_count"])
    raw_rows = int(profile["raw_row_count"])
    excluded_rows = int(profile["excluded_row_count"])
    dup_map = dict(zip(duplicates["metric"], duplicates["value"]))
    same_media_dup = int(dup_map.get("same_title_date_media_duplicate_rows", 0))
    broad_dup = int(dup_map.get("same_title_date_duplicate_rows", 0))
    author_missing = int(completeness.loc[completeness["Column"].eq("Authors"), "Missing Rows"].iloc[0])

    topic_summary = topic_counts.merge(
        topics[topics["Rank"] <= 8].groupby("Topic")["Term"].apply(lambda x: ", ".join(x)).reset_index(name="Top Terms"),
        on="Topic",
        how="left",
    )

    lines = [
        "# Metadata Quality and EDA Report",
        "",
        "## Core Coverage Snapshot",
        "This is the single summary point for the supervisor-facing questions: time range, number of news/documents, and issue coverage.",
        "",
        f"- **Time range of media:** {profile['min_date']} to {profile['max_date']}.",
        f"- **Number of news/documents:** {rows:,} retained rows from {raw_rows:,} raw rows.",
        f"- **Coverage of issues:** top parsed issue terms are {', '.join(subjects.head(5)['Subject Term'].astype(str).tolist())}.",
        f"- **Media coverage:** {len(media_summary):,} normalized media titles across {len(source):,} source types.",
        "",
        "Core summary table:",
        core.to_markdown(index=False),
        "",
        "## Scope and Filtering",
        f"- Retained scope: North-American publishers, including North-American publishers with international or regional edition titles.",
        f"- Scope rule: {MEDIA_FILTER_RULE}",
        f"- Excluded rows: {excluded_rows:,} ({pct(excluded_rows / raw_rows)} of raw rows), only when publisher geography appears outside North America.",
        "- Edition-like titles are not removed when publisher geography is North American; they are flagged separately.",
        "",
        "Publisher region counts:",
        regions.to_markdown(index=False),
        "",
        "Edition/title flags:",
        edition_flags.to_markdown(index=False),
        "",
        "Excluded publisher audit:",
        excluded_media.head(20).to_markdown(index=False) if len(excluded_media) else "No publisher rows were excluded.",
        "",
        "## Media, Time, and Volume",
        "Media titles use a normalized key for grouping and a display title for presentation.",
        "",
        media_summary.head(25).to_markdown(index=False),
        "",
        "Coverage by decade:",
        decades.to_markdown(index=False),
        "",
        "Source type counts:",
        source.to_markdown(index=False),
        "",
        "## Issue Coverage",
        "Issue coverage is inferred from parsed Subject Terms and Class Terms. These are metadata fields, not full-text keywords.",
        "The export does not include serial issue/volume enumeration, so it cannot prove whether every issue of a publication is present.",
        "",
        "Top Subject Terms:",
        subjects.head(30).to_markdown(index=False),
        "",
        "Top Class Terms:",
        classes.head(20).to_markdown(index=False),
        "",
        "## Data Quality and Deduplication",
        f"- Conservative same-title/date/media duplicate estimate: {same_media_dup:,} extra rows ({pct(same_media_dup / rows)} of retained rows).",
        f"- Broader same-title/date estimate: {broad_dup:,} extra rows ({pct(broad_dup / rows)} of retained rows).",
        f"- Author field is present for {rows - author_missing:,} retained rows and blank for {author_missing:,}.",
        "- GOID is unique in the retained data.",
        "",
        "Duplicate summary:",
        duplicates.to_markdown(index=False),
        "",
        "Column completeness:",
        completeness.to_markdown(index=False),
        "",
        "Metadata field audit:",
        field_audit.to_markdown(index=False),
        "",
        "Media normalization examples:",
        norm_examples.head(20).to_markdown(index=False) if len(norm_examples) else "No multi-variant normalized titles were found.",
        "",
        "## Publisher City Disambiguation",
        "City strings were normalized for obvious variants such as `NEW YORK` / `New York, N. Y.` and `Washington` / `Washington, D. C.`.",
        "",
        "Normalized publisher cities:",
        cities.head(20).to_markdown(index=False),
        "",
        "Raw publisher cities:",
        raw_cities.head(20).to_markdown(index=False),
        "",
        "## Authors",
        "Authors are parsed from list-like metadata strings; blank author cells remain a major limitation.",
        "",
        authors.head(30).to_markdown(index=False),
        "",
        "## Title Topic Modeling",
        "The topic model uses article titles only. Genre/form words such as magazine, journal, newspaper, trade, online, edition, report, and reports are removed; place/proper terms such as Wall Street and New York are retained.",
        f"Topic sample size: {int(topic_profile.get('documents', profile['topic_sample_size'])):,} title-bearing rows.",
        f"Topic source scope: {topic_profile.get('source_scope', 'not recorded')}.",
        f"Topic language filter: {topic_profile.get('language_filter', 'not recorded')}.",
        "",
        topic_summary.to_markdown(index=False),
        "",
        "## Metadata Panel for Exploration",
        "The upgraded panel is the recommended export for metadata exploration and topic-modeling sensitivity checks. It keeps one row per document metadata record, carries genre/source fields, author diagnostics, normalized journal/publication identifiers, observed journal date ranges, and both title-only and metadata-enriched topic text.",
        f"- Panel rows: {panel_summary.get('panel_rows', 'not generated')}.",
        f"- Panel columns: {panel_summary.get('panel_columns', 'not generated')}.",
        f"- Full-text status counts: {panel_summary.get('full_text_status_counts', 'not generated')}.",
        "- Current full-text flags are metadata evidence only; no article body text is present in the local CSV export.",
        "",
        "## Output Inventory",
        "- Notebook: `analysis/metadata_panel_topic_modeling_workflow.ipynb`.",
        "- Markdown report: `analysis/metadata_quality_report.md`.",
        "- PDF report: `analysis/metadata_quality_report.pdf`.",
        f"- Compact workbook: `{workbook_path.relative_to(ROOT).as_posix()}`.",
        "- Metadata panel: `analysis/final/metadata_panel.parquet` and `analysis/final/metadata_panel.csv.zip`.",
        "- Export manifest: `analysis/final/metadata_export_manifest.csv`.",
        "- XML/CSV field map: `analysis/outputs/xml_field_manifest.csv`.",
        "- XML field inventory: `analysis/outputs/xml_field_inventory.csv`.",
        "- Journal coverage summary: `analysis/outputs/journal_coverage_summary.csv`.",
        "- Data dictionary: `analysis/outputs/data_dictionary.csv`.",
        "- Clean core tables: `core_summary.csv`, `core_issue_coverage.csv`, `media_summary.csv`, `duplicate_summary.csv`.",
        "- Cleaned metadata for submission: `analysis/final/cleaned_metadata_english_north_america.parquet` and `.csv.zip`.",
        "- Conservative deduplicated cleaned metadata: `analysis/final/cleaned_metadata_english_north_america_dedup.parquet` and `.csv.zip`.",
        f"- Cleaned metadata rows: {cleaned_summary.get('cleaned_rows', 'not generated')}; deduplicated rows: {cleaned_summary.get('deduplicated_rows', 'not generated')}.",
        "- Audit tables: `excluded_media_summary.csv`, `media_normalization_examples.csv`, `duplicate_examples_same_media.csv`, `duplicate_examples_title_date.csv`.",
        "",
        "## Limitations",
        f"- {TEXT_LIMITATION}",
        "- No full-text online availability field is present in the current metadata export.",
        "- No serial issue/volume field is present, so issue completeness cannot be verified directly.",
        f"- {PIP_NOTE}",
        "- This export contains metadata, not full article text.",
        "- Topic modeling should be interpreted as title-theme modeling, not article-body topic modeling.",
        "- The North-American publisher scope is conservative and based on publisher city/state metadata.",
        "- Some publisher geography and author metadata remain missing or inconsistent in source data.",
    ]
    if processing_notes:
        lines.extend(["", "## Processing Notes"])
        lines.extend([f"- {note}" for note in processing_notes])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    write_pdf_report()


def add_text_page(pdf: PdfPages, title: str, lines: list[str], fontsize: int = 10) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    fig.text(0.08, 0.94, title, fontsize=18, fontweight="bold", va="top")
    y = 0.89
    for line in lines:
        fig.text(0.08, y, line, fontsize=fontsize, va="top", wrap=True)
        y -= 0.034 if line else 0.02
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            plt.axis("off")
            y = 0.94
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    if not image_path.exists():
        return
    img = plt.imread(image_path)
    fig = plt.figure(figsize=(11, 8.5))
    plt.axis("off")
    fig.text(0.04, 0.96, title, fontsize=16, fontweight="bold", va="top")
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.86])
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf_report() -> None:
    profile = json.loads((OUTPUT_DIR / "profile.json").read_text(encoding="utf-8"))
    subjects = pd.read_csv(OUTPUT_DIR / "top_subject_terms.csv").head(8)
    media = pd.read_csv(OUTPUT_DIR / "media_summary.csv").head(10).fillna("")
    duplicates = pd.read_csv(OUTPUT_DIR / "duplicate_summary.csv")
    dup_map = dict(zip(duplicates["metric"], duplicates["value"]))
    processing_notes = json.loads(PROCESSING_NOTES_PATH.read_text(encoding="utf-8")) if PROCESSING_NOTES_PATH.exists() else []
    summary_lines = [
        f"Dataset: ProQuest/TDM Studio metadata export.",
        f"Retained documents: {int(profile['row_count']):,} from {int(profile['raw_row_count']):,} raw rows.",
        f"Time range: {profile['min_date']} to {profile['max_date']}.",
        f"Normalized media titles: {len(pd.read_csv(OUTPUT_DIR / 'media_summary.csv')):,}.",
        f"Top issue terms: {', '.join(subjects['Subject Term'].astype(str).tolist())}.",
        f"Same-title/date/media duplicate rows: {int(dup_map.get('same_title_date_media_duplicate_rows', 0)):,}.",
        "",
        "Top retained media:",
    ]
    summary_lines.extend([f"- {row['Display Publication Title']}: {int(row['Documents']):,}" for _, row in media.iterrows()])
    if processing_notes:
        summary_lines.extend(["", "Processing notes:"])
        summary_lines.extend([f"- {note}" for note in processing_notes])

    with PdfPages(REPORT_PDF_PATH) as pdf:
        add_text_page(pdf, "Metadata EDA Report", summary_lines, fontsize=10)
        add_image_page(pdf, "Core Coverage Snapshot", FIGURE_DIR / "core_coverage_snapshot.png")
        add_image_page(pdf, "Documents by Year", FIGURE_DIR / "documents_by_year.png")
        add_image_page(pdf, "Documents by Source Type and Year", FIGURE_DIR / "source_type_by_year_lines.png")
        add_image_page(pdf, "Top Media", FIGURE_DIR / "media_summary_counts.png")
        add_image_page(pdf, "Issue/Title Topic Coverage by Decade", FIGURE_DIR / "topic_by_decade_heatmap.png")
        add_image_page(pdf, "Title Topic Mix by Media", FIGURE_DIR / "topic_by_media_heatmap.png")
        add_image_page(pdf, "Overall Title Word Cloud", FIGURE_DIR / "wordcloud_overall.png")


def run_full_eda() -> dict:
    profile = profile_metadata()
    panel_summary = build_metadata_panel_outputs()
    topic_info = run_topic_modeling()
    make_plots()
    write_excel_summary()
    write_report()
    profile["topic_info"] = topic_info
    profile["metadata_panel"] = panel_summary
    return profile
