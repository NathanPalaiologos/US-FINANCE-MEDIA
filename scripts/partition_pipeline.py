from __future__ import annotations

import ast
import csv
import io
import json
import math
import re
import sys
import warnings
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
ANALYSIS_DIR = ROOT / "analysis"
OUTPUT_DIR = ANALYSIS_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FINAL_DIR = ANALYSIS_DIR / "final"
EXTERNAL_DIR = ROOT / "data" / "external"
MANIFEST_PATH = ANALYSIS_DIR / "tdm_dataset_partitions_manifest.csv"
CURATED_MANIFEST_PATH = ANALYSIS_DIR / "deliverable" / "memory" / "tdm_dataset_partitions_manifest.csv"
REPORT_PATH = ANALYSIS_DIR / "metadata_quality_report.md"
REPORT_PDF_PATH = ANALYSIS_DIR / "metadata_quality_report.pdf"

MASTER_COLUMNS = [
    "GOID",
    "Title",
    "Date",
    "Source Type",
    "Authors",
    "Publication ID",
    "Publication Title",
    "Publisher City",
    "Publisher Province",
    "ISBN",
    "Publisher Name",
    "Publisher ZipCode",
    "Object Type",
    "Language",
    "Pages",
    "Degree Date",
    "Degree",
    "Degree Type",
    "School Name",
    "School Code",
    "School Location",
    "Department",
    "Advisors",
    "Committee Members",
    "Class Terms",
    "Subject Terms",
    "Paper Categories",
    "Paper Keywords",
    "Start Page",
    "Company Name",
    "Company NAIC",
]

COLUMN_RENAMES = {
    "GOID": "goid",
    "Title": "title",
    "Date": "date",
    "Source Type": "source_type",
    "Authors": "authors",
    "Publication ID": "publication_id",
    "Publication Title": "publication_title",
    "Publisher City": "publisher_city",
    "Publisher Province": "publisher_province",
    "ISBN": "isbn",
    "Publisher Name": "publisher_name",
    "Publisher ZipCode": "publisher_zipcode",
    "Object Type": "object_type",
    "Language": "language",
    "Pages": "pages",
    "Degree Date": "degree_date",
    "Degree": "degree",
    "Degree Type": "degree_type",
    "School Name": "school_name",
    "School Code": "school_code",
    "School Location": "school_location",
    "Department": "department",
    "Advisors": "advisors",
    "Committee Members": "committee_members",
    "Class Terms": "class_terms",
    "Subject Terms": "subject_terms",
    "Paper Categories": "paper_categories",
    "Paper Keywords": "paper_keywords",
    "Start Page": "start_page",
    "Company Name": "company_name",
    "Company NAIC": "company_naic",
}

PARTITION_DATASET_MAP = {
    "USBizNews18571913_metadata": "us_business_media__newspapers__1857_1913__fulltext_dedup",
    "USBizNews19141970_metadata": "us_business_media__newspapers__1914_1970__fulltext_dedup",
    "USBizNews19711998_metadata": "us_business_media__newspapers__1971_1998__fulltext_dedup",
    "USBizNews19992012_metadata": "us_business_media__newspapers__1999_2012__fulltext_dedup",
    "USBizNews20132026_metadata": "us_business_media__newspapers__2013_2026__fulltext_dedup",
    "USBizTrade19201970_metadata": "us_business_media__trade_journals__1920_1970__fulltext_dedup",
    "USBizTrade19711998_metadata": "us_business_media__trade_journals__1971_1998__fulltext_dedup",
    "USBizTrade19992012_metadata": "us_business_media__trade_journals__1999_2012__fulltext_dedup",
    "USBizTrade20132026_metadata": "us_business_media__trade_journals__2013_2026__fulltext_dedup",
    "USBusinessMediaMagazines1971-2026_metadata": "us_business_media__magazines__1971_2026__fulltext_dedup",
}

MISSING_TOKENS = {"", "nan", "NaN", "N/A", "n/a", "NA", "None", "none", "null", "NULL"}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
LM_URL = "https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"
LM_SOURCE_URL = "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"
LM_FILE = EXTERNAL_DIR / "Loughran-McDonald_MasterDictionary_1993-2025.csv"

CORPUS_STOPWORDS = set(ENGLISH_STOP_WORDS).union(
    {
        "article",
        "articles",
        "associated",
        "business",
        "copyright",
        "daily",
        "edition",
        "journal",
        "journals",
        "magazine",
        "magazines",
        "media",
        "new",
        "news",
        "newspaper",
        "newspapers",
        "online",
        "page",
        "pages",
        "press",
        "proquest",
        "publication",
        "publications",
        "publisher",
        "report",
        "reports",
        "source",
        "sources",
        "tdm",
        "times",
        "trade",
        "type",
        "types",
        "world",
    }
)
WORDCLOUD_STOPWORD_TOKENS = {token.upper().replace("-", "") for token in CORPUS_STOPWORDS}

PUBLICATION_GENERIC_TOKENS = {
    "bulletin",
    "chronicle",
    "daily",
    "digest",
    "early",
    "edition",
    "evening",
    "final",
    "gazette",
    "herald",
    "international",
    "journal",
    "journals",
    "late",
    "magazine",
    "magazines",
    "morning",
    "national",
    "news",
    "newspaper",
    "newspapers",
    "post",
    "press",
    "record",
    "reporter",
    "review",
    "sun",
    "times",
    "tribune",
    "weekly",
}

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
    "AB",
    "BC",
    "MB",
    "NB",
    "NL",
    "NS",
    "NT",
    "NU",
    "ON",
    "PE",
    "QC",
    "SK",
    "YT",
    "ALBERTA",
    "BRITISH COLUMBIA",
    "MANITOBA",
    "NEW BRUNSWICK",
    "NEWFOUNDLAND AND LABRADOR",
    "NOVA SCOTIA",
    "NORTHWEST TERRITORIES",
    "NUNAVUT",
    "ONTARIO",
    "PRINCE EDWARD ISLAND",
    "QUEBEC",
    "SASKATCHEWAN",
    "YUKON",
}

CITY_STATE_HINTS = {
    "new york n y": ("New York", "NY"),
    "new york": ("New York", None),
    "washington d c": ("Washington", "DC"),
    "washington": ("Washington", None),
}

NON_US_CITY_HINTS = {
    "abingdon": "United Kingdom",
    "london": "United Kingdom",
    "new delhi": "India",
    "mumbai": "India",
    "hong kong": "Hong Kong",
    "toronto": "Canada",
    "montreal": "Canada",
    "ottawa": "Canada",
    "vancouver": "Canada",
}


@dataclass(frozen=True)
class PartitionInfo:
    name: str
    path: Path
    master_path: Path
    expected_manifest_rows: int | None


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)


def clear_generated_outputs() -> None:
    ensure_dirs()
    keep_names = {"full_pipeline.log", "full_pipeline.err.log", "full_pipeline.pid"}
    for directory in [OUTPUT_DIR, FINAL_DIR]:
        for path in directory.rglob("*"):
            if path.is_file() and path.name not in keep_names:
                path.unlink()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text in MISSING_TOKENS:
        return ""
    return re.sub(r"\s+", " ", text)


def compact_key(value: object) -> str:
    text = clean_text(value).lower().replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_missing_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(list(MISSING_TOKENS), pd.NA)


def normalize_publication(value: object) -> str:
    text = clean_text(value)
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"\b(pre[- ]?\d{4}|online|fulltext|full text)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"['’]s\b", "s", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*the\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(inc|llc|ltd|corp|corporation|company|co)\b\.?", " ", text, flags=re.IGNORECASE)
    text = compact_key(text)
    text = re.sub(r"\bu s\b", "us", text)
    return text.title().replace("Us ", "US ") if text else "Unknown"


def publication_token_profile(value: object) -> dict[str, object]:
    normalized = compact_key(value)
    tokens = tuple(token for token in normalized.split() if token)
    anchor_tokens = tuple(token for token in tokens if token not in PUBLICATION_GENERIC_TOKENS)
    if not anchor_tokens:
        anchor_tokens = tokens
    return {
        "normalized": normalized,
        "tokens": tokens,
        "token_set": set(tokens),
        "anchor_tokens": anchor_tokens,
        "anchor_set": set(anchor_tokens),
    }


def compare_publication_titles(
    left: str,
    right: str,
    *,
    shared_publication_id: bool = False,
    shared_top_publisher: bool = False,
) -> dict[str, object]:
    from difflib import SequenceMatcher

    left_profile = publication_token_profile(left)
    right_profile = publication_token_profile(right)
    left_anchors = left_profile["anchor_set"]
    right_anchors = right_profile["anchor_set"]
    shared_anchors = left_anchors & right_anchors
    left_only = left_anchors - right_anchors
    right_only = right_anchors - left_anchors
    union_size = len(left_anchors | right_anchors)
    token_overlap = len(shared_anchors) / union_size if union_size else 1.0
    char_similarity = SequenceMatcher(None, left_profile["normalized"], right_profile["normalized"]).ratio()
    metadata_support = shared_publication_id or shared_top_publisher
    exact_normalized_match = left_profile["normalized"] == right_profile["normalized"]
    same_anchor_tokens = left_anchors == right_anchors
    subset_anchor_tokens = bool(left_anchors) and bool(right_anchors) and (left_anchors <= right_anchors or right_anchors <= left_anchors)
    anchor_conflict = bool(shared_anchors and left_only and right_only)
    low_information_anchor_match = len(shared_anchors) <= 1 and not exact_normalized_match

    auto_merge = False
    recommendation = "keep separate; low evidence"
    if exact_normalized_match:
        auto_merge = True
        recommendation = "auto-merge; exact normalized publication"
    elif anchor_conflict:
        recommendation = "keep separate; conflicting anchor tokens"
    elif same_anchor_tokens and low_information_anchor_match:
        recommendation = "keep separate; low-information anchor tokens"
    elif same_anchor_tokens and (char_similarity >= 0.6 or metadata_support):
        auto_merge = True
        recommendation = "auto-merge; same anchor tokens"
    elif subset_anchor_tokens and low_information_anchor_match:
        recommendation = "keep separate; low-information subset anchors"
    elif subset_anchor_tokens and shared_publication_id and char_similarity >= 0.72:
        auto_merge = True
        recommendation = "auto-merge; subset anchor tokens with shared publication id"
    elif metadata_support and token_overlap >= 0.6 and char_similarity >= 0.82:
        recommendation = "review; metadata-backed near match"
    elif token_overlap >= 0.75 and char_similarity >= 0.9:
        recommendation = "review; strong title overlap"

    needs_review = recommendation.startswith("review")
    return {
        "char_similarity": char_similarity,
        "token_overlap": token_overlap,
        "shared_anchor_tokens": "; ".join(sorted(shared_anchors)),
        "left_only_anchor_tokens": "; ".join(sorted(left_only)),
        "right_only_anchor_tokens": "; ".join(sorted(right_only)),
        "anchor_conflict": anchor_conflict,
        "low_information_anchor_match": low_information_anchor_match,
        "same_anchor_tokens": same_anchor_tokens,
        "subset_anchor_tokens": subset_anchor_tokens,
        "shared_publication_id": shared_publication_id,
        "shared_top_publisher": shared_top_publisher,
        "metadata_support": metadata_support,
        "auto_merge": auto_merge,
        "needs_review": needs_review,
        "recommendation": recommendation,
    }


def publication_display_name(raw_value: object, normalized_value: str) -> str:
    text = clean_text(raw_value)
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"^\s*the\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text or normalized_value


def normalize_title(value: object) -> str:
    return compact_key(value)


def parse_list_like(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [clean_text(item) for item in parsed if clean_text(item)]
        except (SyntaxError, ValueError):
            pass
    return [clean_text(part) for part in re.split(r"\s*[;|]\s*|\s{2,}", text) if clean_text(part)]


def join_terms(value: object, limit: int = 12) -> str:
    return " ".join(parse_list_like(value)[:limit])


def canonical_term(term: str) -> str:
    term = clean_text(term).strip("'\"")
    return term if not term.isupper() else term.title()


def partition_family(partition_name: str) -> str:
    if "Trade" in partition_name:
        return "trade_journals"
    if "Magazines" in partition_name:
        return "magazines"
    return "newspapers"


def normalize_city_state(city: object, province: object) -> tuple[str, str, str, str]:
    city_text = clean_text(city)
    province_text = clean_text(province)
    city_key = compact_key(city_text.replace(".", " "))
    province_key = compact_key(province_text).upper()

    hinted_city, hinted_state = CITY_STATE_HINTS.get(city_key, (None, None))
    normalized_city = hinted_city or city_text.title()
    normalized_state = province_key or hinted_state or ""

    if normalized_state in US_STATE_CODES:
        region = "United States"
    elif normalized_state in CANADA_PROVINCES or city_key in {"toronto", "montreal", "ottawa", "vancouver"}:
        region = "Canada"
    elif city_key in NON_US_CITY_HINTS:
        region = f"Non-US/Canada: {NON_US_CITY_HINTS[city_key]}"
    elif normalized_city:
        region = "Unknown/Needs Review"
    else:
        region = "Missing publisher location"

    location = f"{normalized_city}, {normalized_state}" if normalized_state else normalized_city
    return normalized_city or "", normalized_state, region, location or ""


def metadata_text(title: object, subject_terms: object, class_terms: object) -> str:
    parts = [clean_text(title), join_terms(subject_terms), join_terms(class_terms)]
    return " ".join(part for part in parts if part)


def tokenize(text: str) -> list[str]:
    return [token.upper().replace("-", "") for token in TOKEN_RE.findall(text)]


def csv_row_count(path: Path) -> int:
    with path.open("rb") as f:
        return max(sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b"")) - 1, 0)


def zip_csv_row_count(path: Path) -> int:
    with zipfile.ZipFile(path) as zf:
        member = [name for name in zf.namelist() if name.endswith(".csv")][0]
        with zf.open(member) as f:
            return max(sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b"")) - 1, 0)


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return next(csv.reader(f))


def read_zip_header(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        member = [name for name in zf.namelist() if name.endswith(".csv")][0]
        with zf.open(member) as f:
            return next(csv.reader(io.TextIOWrapper(f, encoding="utf-8", errors="replace")))


def discover_partitions() -> list[PartitionInfo]:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {RAW_DIR}")
    manifest_path = MANIFEST_PATH if MANIFEST_PATH.exists() else CURATED_MANIFEST_PATH
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    manifest_counts = dict(zip(manifest.get("dataset_name", []), manifest.get("document_count", [])))
    partitions = []
    for path in sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")):
        master_path = path / "master.csv"
        if not master_path.exists():
            continue
        dataset_name = PARTITION_DATASET_MAP.get(path.name)
        expected_rows = int(manifest_counts[dataset_name]) if dataset_name in manifest_counts else None
        partitions.append(PartitionInfo(path.name, path, master_path, expected_rows))
    if not partitions:
        raise FileNotFoundError(f"No partition master.csv files found under {RAW_DIR}")
    return partitions


def iter_master_chunks(partitions: Iterable[PartitionInfo], chunksize: int = 150_000) -> Iterable[tuple[PartitionInfo, pd.DataFrame]]:
    for partition in partitions:
        for chunk in pd.read_csv(
            partition.master_path,
            dtype=str,
            chunksize=chunksize,
            keep_default_na=False,
            low_memory=False,
        ):
            yield partition, chunk


def hash_frame(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df.fillna(""), index=False).to_numpy(dtype=np.uint64)


def duplicate_info(hashes: list[np.ndarray]) -> tuple[set[int], int]:
    if not hashes:
        return set(), 0
    values = np.concatenate(hashes)
    counts = pd.Series(values).value_counts(sort=False)
    duplicate_hashes = set(counts[counts > 1].index.astype(np.uint64).tolist())
    duplicate_extra_rows = int((counts[counts > 1] - 1).sum())
    return duplicate_hashes, duplicate_extra_rows


def load_lm_dictionary() -> dict[str, set[str]]:
    ensure_dirs()
    if not LM_FILE.exists():
        urllib.request.urlretrieve(LM_URL, LM_FILE)
    dictionary = pd.read_csv(LM_FILE)
    dictionary.columns = [col.strip().upper() for col in dictionary.columns]
    word_col = "WORD"
    categories = {
        "positive": "POSITIVE",
        "negative": "NEGATIVE",
        "uncertainty": "UNCERTAINTY",
        "litigious": "LITIGIOUS",
        "constraining": "CONSTRAINING",
        "strong_modal": "STRONG_MODAL",
        "weak_modal": "WEAK_MODAL",
    }
    lexicon: dict[str, set[str]] = {}
    for output_name, col in categories.items():
        if col in dictionary.columns:
            lexicon[output_name] = set(dictionary.loc[dictionary[col].fillna(0).astype(float) != 0, word_col].astype(str).str.upper())
        else:
            lexicon[output_name] = set()
    return lexicon


def sentiment_scores(text: str, lexicon: dict[str, set[str]]) -> dict[str, float | int]:
    tokens = tokenize(text)
    token_count = len(tokens)
    counts = {name: sum(token in words for token in tokens) for name, words in lexicon.items()}
    denom = token_count if token_count else 1
    scores: dict[str, float | int] = {"lm_token_count": token_count}
    for name, count in counts.items():
        scores[f"lm_{name}"] = int(count)
        scores[f"lm_{name}_rate"] = count / denom
    scores["lm_net_tone"] = (counts["positive"] - counts["negative"]) / denom
    return scores


def score_texts(texts: pd.Series, lexicon: dict[str, set[str]]) -> pd.DataFrame:
    rows = [sentiment_scores(text, lexicon) for text in texts.fillna("").astype(str)]
    return pd.DataFrame(rows, index=texts.index)


def build_sentiment_lookup(lexicon: dict[str, set[str]]) -> dict[str, tuple[str, ...]]:
    lookup: defaultdict[str, list[str]] = defaultdict(list)
    for category, words in lexicon.items():
        for word in words:
            lookup[word].append(category)
    return {word: tuple(categories) for word, categories in lookup.items()}


def wordcloud_tokens(text: str, stop_words: set[str] | None = None) -> list[str]:
    blocked = stop_words or WORDCLOUD_STOPWORD_TOKENS
    return [
        token.lower()
        for token in tokenize(text)
        if len(token) > 2 and token not in blocked and not token.isdigit()
    ]


def update_group_term_counters(
    texts: pd.Series,
    groups: pd.Series,
    counters: dict[object, Counter],
    *,
    stop_words: set[str] | None = None,
) -> None:
    blocked = stop_words or WORDCLOUD_STOPWORD_TOKENS
    frame = pd.DataFrame({"group": groups, "text": texts}).dropna(subset=["group"])
    if frame.empty:
        return
    combined_text = frame.groupby("group", dropna=False)["text"].agg(" ".join)
    for group_value, group_text in combined_text.items():
        counters[group_value].update(wordcloud_tokens(group_text, stop_words=blocked))


def update_group_sentiment_totals(
    texts: pd.Series,
    groups: pd.Series,
    totals: dict[object, Counter],
    *,
    sentiment_lookup: dict[str, tuple[str, ...]],
) -> None:
    frame = pd.DataFrame({"group": groups, "text": texts}).dropna(subset=["group"])
    if frame.empty:
        return
    documents_by_group = frame.groupby("group", dropna=False).size()
    combined_text = frame.groupby("group", dropna=False)["text"].agg(" ".join)
    for group_value, group_text in combined_text.items():
        token_counts = Counter(tokenize(group_text))
        group_totals = totals[group_value]
        group_totals["documents"] += int(documents_by_group.loc[group_value])
        group_totals["lm_token_count"] += int(sum(token_counts.values()))
        for token, token_count in token_counts.items():
            for category in sentiment_lookup.get(token, ()): 
                group_totals[f"lm_{category}"] += int(token_count)


def top_terms_by_group(group_counters: dict[object, Counter], limit: int = 120) -> dict[object, dict[str, int]]:
    return {
        group_value: {term: int(count) for term, count in counter.most_common(limit)}
        for group_value, counter in group_counters.items()
        if counter
    }


def sample_rows_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    per_group_limit: int,
    *,
    random_state: int,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sampled_index: list[object] = []
    for _, group_positions in df.groupby(group_cols, dropna=False).indices.items():
        position_values = list(group_positions)
        sample_size = min(len(position_values), per_group_limit)
        if sample_size == 0:
            continue
        sampled_index.extend(df.iloc[position_values].sample(sample_size, random_state=random_state).index.tolist())
    if not sampled_index:
        return df.iloc[0:0].copy()
    return df.loc[sampled_index].reset_index(drop=True)


def build_notebook_profile(
    partitions: list[PartitionInfo],
    *,
    chunksize: int = 200_000,
    topic_sample_per_group_chunk: int = 40,
    topic_sample_limit: int = 15_000,
    wordcloud_term_limit: int = 120,
) -> dict[str, object]:
    scan_cols = [
        "GOID",
        "Title",
        "Date",
        "Source Type",
        "Language",
        "Publication ID",
        "Publication Title",
        "Publisher City",
        "Publisher Province",
        "Publisher Name",
        "Object Type",
        "Pages",
        "Authors",
        "Subject Terms",
        "Class Terms",
        "Company Name",
    ]

    source_counts = Counter()
    language_counts = Counter()
    object_counts = Counter()
    year_counts = Counter()
    missing_counts = Counter()
    publisher_region_counts = Counter()
    publication_counts = Counter()
    publication_examples: defaultdict[str, Counter] = defaultdict(Counter)
    publication_id_examples: defaultdict[str, Counter] = defaultdict(Counter)
    publication_publisher_examples: defaultdict[str, Counter] = defaultdict(Counter)
    publication_source_type_examples: defaultdict[str, Counter] = defaultdict(Counter)
    subject_counts = Counter()
    class_counts = Counter()
    goid_hashes: list[np.ndarray] = []
    normalized_hashes: list[np.ndarray] = []
    title_date_hashes: list[np.ndarray] = []
    topic_sample_frames: list[pd.DataFrame] = []
    raw_rows = 0
    english_rows = 0
    invalid_dates = 0
    min_date = None
    max_date = None
    sentiment_lookup = build_sentiment_lookup(load_lm_dictionary())
    wordcloud_group_counters = {
        "decade": defaultdict(Counter),
        "source_type": defaultdict(Counter),
        "partition_family": defaultdict(Counter),
    }
    sentiment_group_totals = {
        "decade": defaultdict(Counter),
        "source_type": defaultdict(Counter),
        "partition_family": defaultdict(Counter),
    }
    wordcloud_group_sizes = {
        "decade": Counter(),
        "source_type": Counter(),
        "partition_family": Counter(),
    }

    for part in partitions:
        family = partition_family(part.name)
        for chunk in pd.read_csv(part.master_path, usecols=scan_cols, dtype=str, keep_default_na=False, chunksize=chunksize):
            raw_rows += len(chunk)
            chunk = normalize_missing_frame(chunk)
            for col in scan_cols:
                missing_counts[col] += int(chunk[col].isna().sum())

            dates = pd.to_datetime(chunk["Date"], errors="coerce")
            invalid_dates += int(dates.isna().sum())
            if dates.notna().any():
                min_date = dates.min() if min_date is None else min(min_date, dates.min())
                max_date = dates.max() if max_date is None else max(max_date, dates.max())

            source_counts.update(chunk["Source Type"].fillna("<missing>").value_counts().to_dict())
            language_counts.update(chunk["Language"].fillna("<missing>").value_counts().to_dict())
            object_counts.update(chunk["Object Type"].fillna("<missing>").value_counts().to_dict())
            locations = [normalize_city_state(c, s)[2] for c, s in zip(chunk["Publisher City"], chunk["Publisher Province"])]
            publisher_region_counts.update(pd.Series(locations).value_counts().to_dict())

            english = chunk[chunk["Language"].eq("English")].copy()
            english_rows += len(english)
            if english.empty:
                continue

            years = pd.to_datetime(english["Date"], errors="coerce").dt.year
            decades = ((years // 10) * 10).astype("Int64")
            year_counts.update(years.dropna().astype(int).value_counts().to_dict())

            normalized_publication = english["Publication Title"].map(normalize_publication)
            normalized_title = english["Title"].map(normalize_title)
            publication_counts.update(normalized_publication.value_counts().head(1000).to_dict())

            publication_detail = pd.DataFrame(
                {
                    "normalized_publication": normalized_publication,
                    "raw_publication": english["Publication Title"].fillna("Unknown"),
                    "publication_id": english["Publication ID"].fillna(""),
                    "publisher_name": english["Publisher Name"].fillna(""),
                    "source_type": english["Source Type"].fillna(""),
                }
            )
            for publication_name, group in publication_detail.groupby("normalized_publication", dropna=False):
                publication_examples[publication_name].update(group["raw_publication"].value_counts().head(5).to_dict())
                publication_id_examples[publication_name].update(group.loc[group["publication_id"].ne(""), "publication_id"].value_counts().head(5).to_dict())
                publication_publisher_examples[publication_name].update(group.loc[group["publisher_name"].ne(""), "publisher_name"].value_counts().head(5).to_dict())
                publication_source_type_examples[publication_name].update(group.loc[group["source_type"].ne(""), "source_type"].value_counts().head(5).to_dict())

            goid_hashes.append(hash_frame(english[["GOID"]].rename(columns={"GOID": "goid"})))
            normalized_hashes.append(
                hash_frame(
                    pd.DataFrame(
                        {
                            "title": normalized_title,
                            "date": english["Date"],
                            "publication": normalized_publication,
                        }
                    )
                )
            )
            title_date_hashes.append(hash_frame(pd.DataFrame({"title": normalized_title, "date": english["Date"]})))

            for value in english["Subject Terms"].dropna().head(25_000):
                subject_counts.update(canonical_term(term) for term in parse_list_like(value) if canonical_term(term))
            for value in english["Class Terms"].dropna().head(25_000):
                class_counts.update(canonical_term(term) for term in parse_list_like(value) if canonical_term(term))

            topic_text = pd.Series(
                [metadata_text(title, subject, cls) for title, subject, cls in zip(english["Title"], english["Subject Terms"], english["Class Terms"])],
                index=english.index,
            )
            valid_topic = topic_text.str.len() > 10
            topic_base = english.loc[valid_topic, ["Date", "Title", "Subject Terms", "Class Terms", "Source Type"]].copy()
            if topic_base.empty:
                continue
            topic_base["topic_text"] = topic_text.loc[valid_topic]
            topic_base["year"] = years.loc[valid_topic].astype("Int64")
            topic_base["decade"] = decades.loc[valid_topic].astype("Int64")
            topic_base["partition_family"] = family

            wordcloud_group_sizes["decade"].update(topic_base["decade"].dropna().astype(int).value_counts().to_dict())
            wordcloud_group_sizes["source_type"].update(topic_base["Source Type"].fillna("<missing>").value_counts().to_dict())
            wordcloud_group_sizes["partition_family"].update(topic_base["partition_family"].value_counts().to_dict())
            update_group_term_counters(topic_base["topic_text"], topic_base["decade"].astype(object), wordcloud_group_counters["decade"])
            update_group_term_counters(topic_base["topic_text"], topic_base["Source Type"], wordcloud_group_counters["source_type"])
            update_group_term_counters(topic_base["topic_text"], topic_base["partition_family"], wordcloud_group_counters["partition_family"])
            update_group_sentiment_totals(
                topic_base["topic_text"],
                topic_base["decade"].astype(object),
                sentiment_group_totals["decade"],
                sentiment_lookup=sentiment_lookup,
            )
            update_group_sentiment_totals(
                topic_base["topic_text"],
                topic_base["Source Type"],
                sentiment_group_totals["source_type"],
                sentiment_lookup=sentiment_lookup,
            )
            update_group_sentiment_totals(
                topic_base["topic_text"],
                topic_base["partition_family"],
                sentiment_group_totals["partition_family"],
                sentiment_lookup=sentiment_lookup,
            )

            sampled = sample_rows_by_group(
                topic_base.dropna(subset=["decade", "Source Type"]),
                ["decade", "Source Type"],
                topic_sample_per_group_chunk,
                random_state=42,
            )
            if not sampled.empty:
                topic_sample_frames.append(sampled)

    topic_sample = pd.concat(topic_sample_frames, ignore_index=True) if topic_sample_frames else pd.DataFrame()
    if not topic_sample.empty and len(topic_sample) > topic_sample_limit:
        group_cols = ["decade", "Source Type"]
        group_count = topic_sample.groupby(group_cols, dropna=False).ngroups
        per_group_limit = max(topic_sample_limit // max(group_count, 1), 1)
        topic_sample = sample_rows_by_group(
            topic_sample,
            group_cols,
            per_group_limit,
            random_state=123,
        )
        if len(topic_sample) > topic_sample_limit:
            topic_sample = topic_sample.sample(topic_sample_limit, random_state=123).reset_index(drop=True)

    wordcloud_term_frequencies = {
        group_name: top_terms_by_group(group_counters, limit=wordcloud_term_limit)
        for group_name, group_counters in wordcloud_group_counters.items()
    }
    serialized_sentiment_group_totals = {
        group_name: {
            group_value: {metric: int(value) for metric, value in totals.items()}
            for group_value, totals in group_totals.items()
            if totals
        }
        for group_name, group_totals in sentiment_group_totals.items()
    }

    return {
        "raw_rows": raw_rows,
        "english_rows": english_rows,
        "invalid_dates": invalid_dates,
        "min_date": min_date,
        "max_date": max_date,
        "source_counts": source_counts,
        "language_counts": language_counts,
        "object_counts": object_counts,
        "year_counts": year_counts,
        "missing_counts": missing_counts,
        "publisher_region_counts": publisher_region_counts,
        "publication_counts": publication_counts,
        "publication_examples": publication_examples,
        "publication_id_examples": publication_id_examples,
        "publication_publisher_examples": publication_publisher_examples,
        "publication_source_type_examples": publication_source_type_examples,
        "subject_counts": subject_counts,
        "class_counts": class_counts,
        "goid_hashes": goid_hashes,
        "normalized_hashes": normalized_hashes,
        "title_date_hashes": title_date_hashes,
        "topic_sample": topic_sample,
        "topic_model_sample_rows": len(topic_sample),
        "wordcloud_document_rows": int(sum(wordcloud_group_sizes["partition_family"].values())),
        "wordcloud_group_sizes": wordcloud_group_sizes,
        "wordcloud_term_frequencies": wordcloud_term_frequencies,
        "sentiment_group_totals": serialized_sentiment_group_totals,
    }


def group_term_frequencies(
    df: pd.DataFrame,
    group_col: str,
    text_col: str = "topic_text",
    *,
    stop_words: Iterable[str] | None = None,
    max_features: int = 120,
) -> dict[object, dict[str, int]]:
    frequencies: dict[object, dict[str, int]] = {}
    stop_words = stop_words or CORPUS_STOPWORDS
    for group_value, group in df.groupby(group_col, dropna=False):
        texts = group[text_col].fillna("").astype(str)
        texts = texts[texts.str.len() > 0]
        if texts.empty:
            continue
        vectorizer = CountVectorizer(
            stop_words=list(stop_words),
            max_features=max_features,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            continue
        terms = vectorizer.get_feature_names_out()
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        frequencies[group_value] = {
            term: int(count)
            for term, count in sorted(zip(terms, counts), key=lambda item: item[1], reverse=True)
            if count > 0
        }
    return frequencies


def inspect_partition_files(partitions: list[PartitionInfo]) -> pd.DataFrame:
    rows = []
    for partition in partitions:
        master_header = read_header(partition.master_path)
        master_rows = csv_row_count(partition.master_path)
        extended_path = partition.path / "extended.csv.zip"
        citation_path = partition.path / "citation.csv.zip"
        extended_rows = zip_csv_row_count(extended_path) if extended_path.exists() else None
        citation_rows = zip_csv_row_count(citation_path) if citation_path.exists() else None
        rows.append(
            {
                "source_partition": partition.name,
                "master_rows": master_rows,
                "extended_rows": extended_rows,
                "citation_rows": citation_rows,
                "manifest_rows": partition.expected_manifest_rows,
                "master_matches_manifest": master_rows == partition.expected_manifest_rows,
                "master_extended_rows_match": master_rows == extended_rows,
                "master_citation_rows_match": master_rows == citation_rows,
                "master_schema_ok": master_header == MASTER_COLUMNS,
                "master_columns": len(master_header),
                "extended_columns": len(read_zip_header(extended_path)) if extended_path.exists() else None,
                "citation_columns": len(read_zip_header(citation_path)) if citation_path.exists() else None,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "partition_file_validation.csv", index=False)
    return df


def prepare_chunk(chunk: pd.DataFrame, partition: PartitionInfo) -> pd.DataFrame:
    missing_cols = [col for col in MASTER_COLUMNS if col not in chunk.columns]
    if missing_cols:
        raise ValueError(f"{partition.name} is missing master columns: {missing_cols}")
    df = chunk.loc[:, MASTER_COLUMNS].rename(columns=COLUMN_RENAMES)
    df = normalize_missing_frame(df)
    df["source_partition"] = partition.name
    df["partition_family"] = partition_family(partition.name)

    dates = pd.to_datetime(df["date"], errors="coerce")
    df["publication_date"] = dates.dt.date.astype("string")
    df["year"] = dates.dt.year.astype("Int64")
    df["decade"] = ((dates.dt.year // 10) * 10).astype("Int64")

    df["title_normalized"] = df["title"].map(normalize_title)
    df["publication_title_normalized"] = df["publication_title"].map(normalize_publication)
    df["publication_title_canonical"] = df["publication_title_normalized"]
    df["publication_title_display"] = [
        publication_display_name(raw, norm)
        for raw, norm in zip(df["publication_title"], df["publication_title_normalized"])
    ]
    locations = [normalize_city_state(city, province) for city, province in zip(df["publisher_city"], df["publisher_province"])]
    df["publisher_city_normalized"] = [item[0] for item in locations]
    df["publisher_province_normalized"] = [item[1] for item in locations]
    df["publisher_region"] = [item[2] for item in locations]
    df["publisher_location_normalized"] = [item[3] for item in locations]

    authors = df["authors"].map(parse_list_like)
    subjects = df["subject_terms"].map(parse_list_like)
    classes = df["class_terms"].map(parse_list_like)
    companies = df["company_name"].map(parse_list_like)
    df["primary_author"] = authors.map(lambda items: items[0] if items else pd.NA)
    df["author_count"] = authors.map(len).astype("int16")
    df["subject_term_count"] = subjects.map(len).astype("int16")
    df["class_term_count"] = classes.map(len).astype("int16")
    df["company_count"] = companies.map(len).astype("int16")
    df["pages_numeric"] = pd.to_numeric(df["pages"], errors="coerce")
    return df


def first_pass_profile(partitions: list[PartitionInfo], chunksize: int, sample_per_group_chunk: int) -> dict:
    counters = {
        "source_type": Counter(),
        "language": Counter(),
        "object_type": Counter(),
        "publication": Counter(),
        "publication_normalized": Counter(),
        "publisher_region": Counter(),
        "publisher_location": Counter(),
        "year": Counter(),
        "decade": Counter(),
        "partition": Counter(),
        "subject_terms": Counter(),
        "class_terms": Counter(),
        "authors": Counter(),
    }
    missing = Counter()
    partition_rows = Counter()
    partition_english_rows = Counter()
    partition_min_date: dict[str, pd.Timestamp] = {}
    partition_max_date: dict[str, pd.Timestamp] = {}
    partition_invalid_dates = Counter()
    min_date = None
    max_date = None
    invalid_dates = 0
    all_rows = 0
    english_rows = 0
    page_count = 0
    page_sum = 0.0
    page_sum_sq = 0.0
    page_min = None
    page_max = None
    topic_samples: list[pd.DataFrame] = []
    hashes = {"goid": [], "exact": [], "normalized": [], "title_date": []}

    for partition, raw_chunk in iter_master_chunks(partitions, chunksize=chunksize):
        df = prepare_chunk(raw_chunk, partition)
        n = len(df)
        all_rows += n
        partition_rows[partition.name] += n
        for col in COLUMN_RENAMES.values():
            missing[col] += int(df[col].isna().sum())
        dates = pd.to_datetime(df["publication_date"], errors="coerce")
        invalid_dates += int(dates.isna().sum())
        if dates.notna().any():
            cmin = dates.min()
            cmax = dates.max()
            min_date = cmin if min_date is None or cmin < min_date else min_date
            max_date = cmax if max_date is None or cmax > max_date else max_date
            current_min = partition_min_date.get(partition.name)
            current_max = partition_max_date.get(partition.name)
            partition_min_date[partition.name] = cmin if current_min is None or cmin < current_min else current_min
            partition_max_date[partition.name] = cmax if current_max is None or cmax > current_max else current_max
        partition_invalid_dates[partition.name] += int(dates.isna().sum())

        counters["source_type"].update(df["source_type"].fillna("<missing>").value_counts().to_dict())
        counters["language"].update(df["language"].fillna("<missing>").value_counts().to_dict())
        counters["object_type"].update(df["object_type"].fillna("<missing>").value_counts().to_dict())
        counters["publisher_region"].update(df["publisher_region"].fillna("<missing>").value_counts().to_dict())
        counters["publisher_location"].update(df["publisher_location_normalized"].fillna("<missing>").value_counts().to_dict())
        counters["partition"].update(df["source_partition"].value_counts().to_dict())

        english = df[df["language"].eq("English")].copy()
        english_rows += len(english)
        partition_english_rows[partition.name] += len(english)
        counters["publication"].update(english["publication_title"].fillna("<missing>").value_counts().head(2000).to_dict())
        counters["publication_normalized"].update(english["publication_title_normalized"].fillna("<missing>").value_counts().head(2000).to_dict())
        counters["year"].update(english["year"].dropna().astype(int).value_counts().to_dict())
        counters["decade"].update(english["decade"].dropna().astype(int).value_counts().to_dict())

        for series_name, counter_name in [("subject_terms", "subject_terms"), ("class_terms", "class_terms"), ("authors", "authors")]:
            for value in english[series_name].dropna():
                for term in parse_list_like(value):
                    clean = canonical_term(term)
                    if clean:
                        counters[counter_name][clean] += 1

        pages = pd.to_numeric(english["pages"], errors="coerce").dropna()
        if len(pages):
            page_count += len(pages)
            page_sum += float(pages.sum())
            page_sum_sq += float((pages**2).sum())
            page_min = float(pages.min()) if page_min is None else min(page_min, float(pages.min()))
            page_max = float(pages.max()) if page_max is None else max(page_max, float(pages.max()))

        if len(english):
            hashes["goid"].append(hash_frame(english[["goid"]]))
            exact = english[["title", "date", "publication_title"]].fillna("")
            hashes["exact"].append(hash_frame(exact))
            normalized = english[["title_normalized", "date", "publication_title_normalized"]].fillna("")
            hashes["normalized"].append(hash_frame(normalized))
            title_date = english[["title_normalized", "date"]].fillna("")
            hashes["title_date"].append(hash_frame(title_date))

            sample_base = english[
                ["goid", "title", "subject_terms", "class_terms", "source_type", "publication_title_normalized", "year", "decade"]
            ].copy()
            sample_base["topic_text"] = [
                metadata_text(title, subject, cls)
                for title, subject, cls in zip(sample_base["title"], sample_base["subject_terms"], sample_base["class_terms"])
            ]
            sample_base = sample_base[sample_base["topic_text"].str.len() > 5]
            if len(sample_base):
                sampled = (
                    sample_base.groupby(["decade", "source_type"], dropna=False, group_keys=False)
                    .apply(lambda group: group.sample(min(len(group), sample_per_group_chunk), random_state=42))
                    .reset_index(drop=True)
                )
                topic_samples.append(sampled)

        print(f"profiled {partition.name}: {partition_rows[partition.name]:,} rows", flush=True)

    duplicate_sets = {}
    duplicate_summary = []
    for name, parts in hashes.items():
        dup_set, extra_rows = duplicate_info(parts)
        duplicate_sets[name] = dup_set
        duplicate_summary.append({"duplicate_rule": name, "duplicate_extra_rows": extra_rows, "duplicate_groups": len(dup_set)})

    page_mean = page_sum / page_count if page_count else np.nan
    page_var = (page_sum_sq / page_count) - (page_mean**2) if page_count else np.nan
    page_std = math.sqrt(max(page_var, 0.0)) if page_count else np.nan

    topic_sample = pd.concat(topic_samples, ignore_index=True) if topic_samples else pd.DataFrame()
    if len(topic_sample) and {"decade", "source_type"}.issubset(topic_sample.columns):
        topic_sample = (
            topic_sample.groupby(["decade", "source_type"], dropna=False, group_keys=False)
            .apply(lambda group: group.sample(min(len(group), 1200), random_state=123))
            .reset_index(drop=True)
        )
    elif len(topic_sample) == 0:
        topic_sample = pd.DataFrame(columns=["goid", "title", "subject_terms", "class_terms", "source_type", "publication_title_normalized", "year", "decade", "topic_text"])
    topic_sample.to_parquet(OUTPUT_DIR / "topic_training_sample.parquet", index=False)

    profile = {
        "all_rows": all_rows,
        "english_rows": english_rows,
        "min_date": str(min_date.date()) if min_date is not None else None,
        "max_date": str(max_date.date()) if max_date is not None else None,
        "invalid_dates": invalid_dates,
        "partition_rows": dict(partition_rows),
        "partition_english_rows": dict(partition_english_rows),
        "partition_min_date": {name: str(value.date()) for name, value in partition_min_date.items()},
        "partition_max_date": {name: str(value.date()) for name, value in partition_max_date.items()},
        "partition_invalid_dates": dict(partition_invalid_dates),
        "missing": dict(missing),
        "counters": counters,
        "duplicate_sets": duplicate_sets,
        "duplicate_summary": duplicate_summary,
        "page_summary": {
            "observed_pages": int(page_count),
            "mean": page_mean,
            "std": page_std,
            "min": page_min,
            "max": page_max,
        },
        "topic_sample_rows": len(topic_sample),
    }
    return profile


def counter_to_csv(counter: Counter, path: Path, name_col: str, value_col: str = "documents", limit: int | None = None) -> pd.DataFrame:
    rows = counter.most_common(limit)
    df = pd.DataFrame(rows, columns=[name_col, value_col])
    df.to_csv(path, index=False)
    return df


def write_profile_tables(profile: dict, partition_validation: pd.DataFrame) -> None:
    counters: dict[str, Counter] = profile["counters"]
    counter_to_csv(counters["source_type"], OUTPUT_DIR / "source_type_counts.csv", "source_type")
    counter_to_csv(counters["language"], OUTPUT_DIR / "language_counts.csv", "language")
    counter_to_csv(counters["object_type"], OUTPUT_DIR / "object_type_counts.csv", "object_type")
    counter_to_csv(counters["publisher_region"], OUTPUT_DIR / "publisher_region_counts.csv", "publisher_region")
    counter_to_csv(counters["publisher_location"], OUTPUT_DIR / "publisher_location_counts.csv", "publisher_location", limit=200)
    counter_to_csv(counters["publication_normalized"], OUTPUT_DIR / "top_publications_normalized.csv", "publication_title_normalized", limit=250)
    counter_to_csv(counters["publication"], OUTPUT_DIR / "top_publications_raw.csv", "publication_title", limit=250)
    counter_to_csv(counters["subject_terms"], OUTPUT_DIR / "top_subject_terms.csv", "subject_term", limit=1000)
    counter_to_csv(counters["class_terms"], OUTPUT_DIR / "top_class_terms.csv", "class_term", limit=1000)
    counter_to_csv(counters["authors"], OUTPUT_DIR / "top_authors.csv", "author", limit=1000)
    counter_to_csv(counters["year"], OUTPUT_DIR / "year_counts.csv", "year")
    counter_to_csv(counters["decade"], OUTPUT_DIR / "decade_counts.csv", "decade")

    pd.DataFrame(profile["duplicate_summary"]).to_csv(OUTPUT_DIR / "duplicate_summary.csv", index=False)
    pd.DataFrame(
        [{"column": col, "missing_rows": rows, "missing_share": rows / profile["all_rows"]} for col, rows in profile["missing"].items()]
    ).sort_values("missing_share", ascending=False).to_csv(OUTPUT_DIR / "column_completeness_all_rows.csv", index=False)

    partition_rows = pd.DataFrame(
        [
            {
                "source_partition": name,
                "raw_rows": rows,
                "english_rows": profile["partition_english_rows"].get(name, 0),
                "english_share": profile["partition_english_rows"].get(name, 0) / rows if rows else np.nan,
                "first_date": profile["partition_min_date"].get(name),
                "last_date": profile["partition_max_date"].get(name),
                "invalid_or_missing_dates": profile["partition_invalid_dates"].get(name, 0),
            }
            for name, rows in profile["partition_rows"].items()
        ]
    )
    partition_rows = partition_rows.merge(partition_validation, on="source_partition", how="left")
    partition_rows.to_csv(OUTPUT_DIR / "partition_quality_summary.csv", index=False)

    core = pd.DataFrame(
        [
            ("raw_documents", profile["all_rows"]),
            ("english_documents", profile["english_rows"]),
            ("time_range_start", profile["min_date"]),
            ("time_range_end", profile["max_date"]),
            ("invalid_or_missing_dates", profile["invalid_dates"]),
            ("source_type_count", len(counters["source_type"])),
            ("normalized_publications_top_counted", len(counters["publication_normalized"])),
            ("topic_training_sample_rows", profile["topic_sample_rows"]),
            ("lm_dictionary_source", LM_SOURCE_URL),
        ],
        columns=["metric", "value"],
    )
    core.to_csv(OUTPUT_DIR / "core_summary.csv", index=False)


def train_topic_model(n_topics: int = 12, max_features: int = 8000) -> tuple[TfidfVectorizer, NMF, dict[int, str]]:
    sample_path = OUTPUT_DIR / "topic_training_sample.parquet"
    sample = pd.read_parquet(sample_path)
    if sample.empty:
        raise ValueError("Topic training sample is empty.")
    vectorizer = TfidfVectorizer(
        stop_words=list(CORPUS_STOPWORDS),
        max_features=max_features,
        min_df=10,
        max_df=0.65,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vectorizer.fit_transform(sample["topic_text"].fillna(""))
    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=300)
    W = nmf.fit_transform(X)
    sample["topic_id"] = W.argmax(axis=1)
    sample["topic_weight"] = W.max(axis=1)

    terms = np.array(vectorizer.get_feature_names_out())
    labels = {}
    term_rows = []
    for topic_id, weights in enumerate(nmf.components_):
        top_idx = weights.argsort()[::-1][:15]
        top_terms = terms[top_idx]
        labels[topic_id] = ", ".join(top_terms[:5])
        for rank, idx in enumerate(top_idx, start=1):
            term_rows.append({"topic_id": topic_id, "rank": rank, "term": terms[idx], "weight": weights[idx], "topic_label": labels[topic_id]})

    pd.DataFrame(term_rows).to_csv(OUTPUT_DIR / "topic_terms.csv", index=False)
    sample.to_parquet(OUTPUT_DIR / "topic_sample_assignments.parquet", index=False)
    return vectorizer, nmf, labels


def assign_topics(texts: pd.Series, vectorizer: TfidfVectorizer, model: NMF, labels: dict[int, str]) -> pd.DataFrame:
    valid = texts.fillna("").astype(str).str.len() > 0
    topic_id = np.full(len(texts), -1, dtype=np.int16)
    topic_weight = np.zeros(len(texts), dtype=np.float32)
    if valid.any():
        X = vectorizer.transform(texts[valid])
        W = model.transform(X)
        topic_id[valid.to_numpy()] = W.argmax(axis=1).astype(np.int16)
        topic_weight[valid.to_numpy()] = W.max(axis=1).astype(np.float32)
    return pd.DataFrame(
        {
            "topic_id": topic_id,
            "topic_label": [labels.get(int(idx), "") for idx in topic_id],
            "topic_weight": topic_weight,
        },
        index=texts.index,
    )


def write_complete_metadata(
    partitions: list[PartitionInfo],
    profile: dict,
    vectorizer: TfidfVectorizer,
    topic_model: NMF,
    topic_labels: dict[int, str],
    lexicon: dict[str, set[str]],
    chunksize: int,
) -> dict:
    parquet_path = FINAL_DIR / "complete_metadata.parquet"
    csv_zip_path = FINAL_DIR / "complete_metadata.csv.zip"
    writer: pq.ParquetWriter | None = None
    seen_norm_hashes: set[int] = set()
    rows_written = 0
    duplicate_hash_arrays = {
        name: np.fromiter(values, dtype=np.uint64) if values else np.array([], dtype=np.uint64)
        for name, values in profile["duplicate_sets"].items()
    }

    with zipfile.ZipFile(csv_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        with zf.open("complete_metadata.csv", "w", force_zip64=True) as raw_csv:
            text_csv = io.TextIOWrapper(raw_csv, encoding="utf-8", newline="")
            wrote_header = False
            for partition, raw_chunk in iter_master_chunks(partitions, chunksize=chunksize):
                df = prepare_chunk(raw_chunk, partition)
                df = df[df["language"].eq("English")].copy()
                if df.empty:
                    continue

                goid_hash = hash_frame(df[["goid"]])
                exact_hash = hash_frame(df[["title", "date", "publication_title"]])
                norm_hash = hash_frame(df[["title_normalized", "date", "publication_title_normalized"]])
                title_date_hash = hash_frame(df[["title_normalized", "date"]])

                df["duplicate_goid"] = np.isin(goid_hash, duplicate_hash_arrays["goid"])
                df["duplicate_exact_title_date_publication"] = np.isin(exact_hash, duplicate_hash_arrays["exact"])
                df["duplicate_normalized_title_date_publication"] = np.isin(norm_hash, duplicate_hash_arrays["normalized"])
                df["duplicate_title_date"] = np.isin(title_date_hash, duplicate_hash_arrays["title_date"])
                extras = []
                for value in norm_hash:
                    key = int(value)
                    is_extra = key in seen_norm_hashes
                    extras.append(is_extra)
                    seen_norm_hashes.add(key)
                df["is_conservative_duplicate_extra"] = extras

                nlp_text = pd.Series(
                    [metadata_text(title, subject, cls) for title, subject, cls in zip(df["title"], df["subject_terms"], df["class_terms"])],
                    index=df.index,
                )
                title_text = df["title"].fillna("").astype(str)
                sentiment = score_texts(nlp_text, lexicon)
                topics = assign_topics(nlp_text, vectorizer, topic_model, topic_labels)
                title_topics = assign_topics(title_text, vectorizer, topic_model, topic_labels).rename(
                    columns={
                        "topic_id": "title_only_topic_id",
                        "topic_label": "title_only_topic_label",
                        "topic_weight": "title_only_topic_weight",
                    }
                )
                df = pd.concat([df, sentiment, topics, title_topics], axis=1)
                for text_col in df.select_dtypes(include=["object", "string"]).columns:
                    df[text_col] = df[text_col].astype("string")
                df["pages_numeric"] = pd.to_numeric(df["pages_numeric"], errors="coerce").astype("float64")

                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
                writer.write_table(table)
                df.to_csv(text_csv, index=False, header=not wrote_header)
                wrote_header = True
                rows_written += len(df)
                print(f"wrote complete metadata through {partition.name}: {rows_written:,} English rows", flush=True)
            text_csv.flush()
    if writer is not None:
        writer.close()
    return {"complete_rows": rows_written, "complete_parquet": str(parquet_path), "complete_csv_zip": str(csv_zip_path)}


def write_deduplicated_metadata() -> dict:
    source_path = FINAL_DIR / "complete_metadata.parquet"
    target_path = FINAL_DIR / "complete_metadata_dedup.parquet"
    csv_zip_path = FINAL_DIR / "complete_metadata_dedup.csv.zip"
    writer: pq.ParquetWriter | None = None
    rows = 0
    with zipfile.ZipFile(csv_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        with zf.open("complete_metadata_dedup.csv", "w", force_zip64=True) as raw_csv:
            text_csv = io.TextIOWrapper(raw_csv, encoding="utf-8", newline="")
            wrote_header = False
            parquet = pq.ParquetFile(source_path)
            for batch in parquet.iter_batches(batch_size=150_000):
                df = batch.to_pandas()
                df = df[~df["is_conservative_duplicate_extra"]].copy()
                if df.empty:
                    continue
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(target_path, table.schema, compression="zstd")
                writer.write_table(table)
                df.to_csv(text_csv, index=False, header=not wrote_header)
                wrote_header = True
                rows += len(df)
            text_csv.flush()
    if writer is not None:
        writer.close()
    return {"deduplicated_rows": rows, "dedup_parquet": str(target_path), "dedup_csv_zip": str(csv_zip_path)}


def write_csv_zip_from_parquet(parquet_path: Path, zip_path: Path, member_name: str, batch_size: int = 150_000) -> int:
    if zip_path.exists():
        zip_path.unlink()
    rows = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        with zf.open(member_name, "w", force_zip64=True) as raw_csv:
            text_csv = io.TextIOWrapper(raw_csv, encoding="utf-8", newline="")
            wrote_header = False
            parquet = pq.ParquetFile(parquet_path)
            for batch in parquet.iter_batches(batch_size=batch_size):
                df = batch.to_pandas()
                df.to_csv(text_csv, index=False, header=not wrote_header)
                wrote_header = True
                rows += len(df)
            text_csv.flush()
    return rows


def write_media_summary(group_col: str = "publication_title_canonical") -> pd.DataFrame:
    cols = [
        group_col,
        "publication_title_normalized",
        "publication_title_canonical",
        "publication_title_display",
        "publication_id",
        "publisher_name",
        "publisher_region",
        "source_type",
        "year",
        "goid",
        "primary_author",
        "subject_term_count",
    ]
    df = pd.read_parquet(FINAL_DIR / "complete_metadata.parquet", columns=list(dict.fromkeys(cols)))
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            display_publication_title=("publication_title_display", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
            documents=("goid", "size"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            normalized_variants=("publication_title_normalized", lambda x: "; ".join(sorted(set(x.dropna().astype(str)))[:8])),
            publication_ids=("publication_id", lambda x: "; ".join(sorted(set(x.dropna().astype(str)))[:5])),
            top_source_type=("source_type", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
            top_publisher=("publisher_name", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
            top_publisher_region=("publisher_region", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
            unique_authors=("primary_author", lambda x: x.dropna().nunique()),
            avg_subject_terms=("subject_term_count", "mean"),
        )
        .reset_index()
        .sort_values("documents", ascending=False)
    )
    grouped = grouped.rename(columns={group_col: "publication_title_canonical"})
    grouped.to_csv(OUTPUT_DIR / "media_summary.csv", index=False)
    return grouped


def build_publication_alias_audit(media: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    records = media.to_dict("records")
    for i, left in enumerate(records):
        for right in records[i + 1 :]:
            left_title = left["publication_title_canonical"]
            right_title = right["publication_title_canonical"]
            if abs(len(left_title) - len(right_title)) > 18:
                continue
            shared_id = bool(set(str(left["publication_ids"]).split("; ")) & set(str(right["publication_ids"]).split("; ")))
            shared_publisher = clean_text(left["top_publisher"]) and clean_text(left["top_publisher"]) == clean_text(right["top_publisher"])
            match = compare_publication_titles(
                left_title,
                right_title,
                shared_publication_id=shared_id,
                shared_top_publisher=bool(shared_publisher),
            )
            if match["auto_merge"] or match["needs_review"]:
                rows.append(
                    {
                        "publication_left": left_title,
                        "publication_right": right_title,
                        "char_similarity": match["char_similarity"],
                        "token_overlap": match["token_overlap"],
                        "shared_anchor_tokens": match["shared_anchor_tokens"],
                        "left_only_anchor_tokens": match["left_only_anchor_tokens"],
                        "right_only_anchor_tokens": match["right_only_anchor_tokens"],
                        "anchor_conflict": match["anchor_conflict"],
                        "shared_publication_id": shared_id,
                        "shared_top_publisher": bool(shared_publisher),
                        "auto_merge": match["auto_merge"],
                        "recommendation": match["recommendation"],
                    }
                )
    audit = pd.DataFrame(rows).sort_values(["auto_merge", "char_similarity", "token_overlap"], ascending=[False, False, False]) if rows else pd.DataFrame(
        columns=[
            "publication_left",
            "publication_right",
            "char_similarity",
            "token_overlap",
            "shared_anchor_tokens",
            "left_only_anchor_tokens",
            "right_only_anchor_tokens",
            "anchor_conflict",
            "shared_publication_id",
            "shared_top_publisher",
            "auto_merge",
            "recommendation",
        ]
    )
    canonical_map = build_publication_canonical_map(media, audit)
    return audit, canonical_map


def write_alias_audit(media: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    audit, canonical_map = build_publication_alias_audit(media)
    audit.to_csv(OUTPUT_DIR / "publication_alias_fuzzy_audit.csv", index=False)
    return audit, canonical_map


def build_publication_canonical_map(media: pd.DataFrame, audit: pd.DataFrame) -> dict[str, str]:
    titles = media["publication_title_canonical"].dropna().astype(str).tolist()
    if not titles:
        return {}

    parent = {title: title for title in titles}
    documents = dict(zip(media["publication_title_canonical"].astype(str), media["documents"]))

    def find(title: str) -> str:
        root = parent[title]
        while root != parent[root]:
            root = parent[root]
        while title != root:
            parent[title], title = root, parent[title]
        return root

    def choose_canonical(left_title: str, right_title: str) -> str:
        candidates = [left_title, right_title]
        return min(candidates, key=lambda name: (-documents.get(name, 0), len(name), name))

    def union(left_title: str, right_title: str) -> None:
        left_root = find(left_title)
        right_root = find(right_title)
        if left_root == right_root:
            return
        canonical = choose_canonical(left_root, right_root)
        other = right_root if canonical == left_root else left_root
        parent[other] = canonical

    if not audit.empty:
        for row in audit.loc[audit["auto_merge"].fillna(False)].itertuples(index=False):
            union(str(row.publication_left), str(row.publication_right))

    return {title: find(title) for title in titles}


def apply_publication_canonicalization(canonical_map: dict[str, str]) -> None:
    source_path = FINAL_DIR / "complete_metadata.parquet"
    temp_path = FINAL_DIR / "complete_metadata.canonicalizing.parquet"
    writer: pq.ParquetWriter | None = None
    parquet = pq.ParquetFile(source_path)
    for batch in parquet.iter_batches(batch_size=150_000):
        df = batch.to_pandas()
        canonical = df["publication_title_normalized"].map(canonical_map)
        existing = df["publication_title_canonical"] if "publication_title_canonical" in df.columns else df["publication_title_normalized"]
        df["publication_title_canonical"] = canonical.fillna(existing).fillna(df["publication_title_normalized"])
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(temp_path, table.schema, compression="zstd")
        writer.write_table(table)
    if writer is not None:
        writer.close()
    temp_path.replace(source_path)
    write_csv_zip_from_parquet(source_path, FINAL_DIR / "complete_metadata.csv.zip", "complete_metadata.csv")


def write_topic_and_sentiment_summaries() -> None:
    cols = [
        "goid",
        "year",
        "decade",
        "source_type",
        "publication_title_normalized",
        "topic_id",
        "topic_label",
        "topic_weight",
        "title_only_topic_id",
        "lm_token_count",
        "lm_positive",
        "lm_negative",
        "lm_uncertainty",
        "lm_litigious",
        "lm_constraining",
        "lm_net_tone",
    ]
    df = pd.read_parquet(FINAL_DIR / "complete_metadata.parquet", columns=cols)
    topic_counts = (
        df.groupby(["topic_id", "topic_label"], dropna=False)
        .size()
        .reset_index(name="documents")
        .sort_values("documents", ascending=False)
    )
    topic_counts.to_csv(OUTPUT_DIR / "topic_counts.csv", index=False)
    df.groupby(["decade", "topic_id", "topic_label"], dropna=False).size().reset_index(name="documents").to_csv(
        OUTPUT_DIR / "topic_by_decade.csv", index=False
    )
    df.groupby(["source_type", "topic_id", "topic_label"], dropna=False).size().reset_index(name="documents").to_csv(
        OUTPUT_DIR / "topic_by_source_type.csv", index=False
    )
    (
        df.groupby(["decade", "source_type"], dropna=False)
        .agg(
            documents=("goid", "size"),
            mean_net_tone=("lm_net_tone", "mean"),
            mean_positive_rate=("lm_positive", lambda x: x.sum() / max(df.loc[x.index, "lm_token_count"].sum(), 1)),
            mean_negative_rate=("lm_negative", lambda x: x.sum() / max(df.loc[x.index, "lm_token_count"].sum(), 1)),
            mean_uncertainty_rate=("lm_uncertainty", lambda x: x.sum() / max(df.loc[x.index, "lm_token_count"].sum(), 1)),
        )
        .reset_index()
        .to_csv(OUTPUT_DIR / "sentiment_by_decade_source_type.csv", index=False)
    )
    (
        df.groupby("decade", dropna=False)
        .agg(
            documents=("goid", "size"),
            mean_net_tone=("lm_net_tone", "mean"),
            positive_words=("lm_positive", "sum"),
            negative_words=("lm_negative", "sum"),
            uncertainty_words=("lm_uncertainty", "sum"),
            tokens=("lm_token_count", "sum"),
        )
        .reset_index()
        .to_csv(OUTPUT_DIR / "sentiment_by_decade.csv", index=False)
    )
    df[["goid", "topic_id", "topic_label", "topic_weight", "title_only_topic_id", "lm_net_tone"]].to_parquet(
        FINAL_DIR / "document_nlp_scores.parquet", index=False
    )


def write_final_sample_counts() -> None:
    df = pd.read_parquet(
        FINAL_DIR / "complete_metadata.parquet",
        columns=["source_type", "object_type", "publisher_region", "year", "decade"],
    )
    df["source_type"].value_counts(dropna=False).rename_axis("source_type").reset_index(name="documents").to_csv(
        OUTPUT_DIR / "source_type_counts.csv", index=False
    )
    df["object_type"].value_counts(dropna=False).rename_axis("object_type").reset_index(name="documents").to_csv(
        OUTPUT_DIR / "object_type_counts.csv", index=False
    )
    df["publisher_region"].value_counts(dropna=False).rename_axis("publisher_region").reset_index(name="documents").to_csv(
        OUTPUT_DIR / "publisher_region_counts.csv", index=False
    )
    df["year"].dropna().astype(int).value_counts().sort_index().rename_axis("year").reset_index(name="documents").to_csv(
        OUTPUT_DIR / "year_counts.csv", index=False
    )
    df["decade"].dropna().astype(int).value_counts().sort_index().rename_axis("decade").reset_index(name="documents").to_csv(
        OUTPUT_DIR / "decade_counts.csv", index=False
    )


def write_data_dictionaries() -> None:
    rows = [
        ("source_partition", "Raw TDM metadata partition directory."),
        ("partition_family", "Partition source family inferred from the export name."),
        ("goid", "TDM document identifier; primary document key."),
        ("publication_date", "Parsed publication date from Date."),
        ("year", "Publication year."),
        ("decade", "Publication decade."),
        ("publication_title_canonical", "Conservative canonical publication title built from token-aware alias rules."),
        ("publication_title_normalized", "Normalized publication title for grouping aliases."),
        ("publisher_region", "Publisher geography diagnostic; not used as the primary sample filter."),
        ("duplicate_normalized_title_date_publication", "True for rows in a duplicate group by normalized title, date, and publication."),
        ("is_conservative_duplicate_extra", "True for rows dropped from the conservative deduplicated export."),
        ("topic_id", "NMF topic assigned from Title + Subject Terms + Class Terms."),
        ("title_only_topic_id", "Sensitivity topic assignment using title text only."),
        ("lm_net_tone", "Loughran-McDonald positive minus negative word count divided by LM token count."),
    ]
    pd.DataFrame(rows, columns=["field", "definition"]).to_csv(FINAL_DIR / "complete_metadata_data_dictionary.csv", index=False)
    pd.DataFrame(
        [
            ("complete_metadata.parquet", "Primary English tidy metadata export with duplicate, sentiment, and topic diagnostics."),
            ("complete_metadata.csv.zip", "Compressed CSV companion for access convenience."),
            ("complete_metadata_dedup.parquet", "Conservative deduplicated version of complete_metadata."),
            ("document_nlp_scores.parquet", "Compact document-level topic and LM sentiment scores."),
            ("summary_statistics_final.xlsx", "Professor-facing summary statistics workbook."),
        ],
        columns=["file", "description"],
    ).to_csv(FINAL_DIR / "metadata_export_manifest.csv", index=False)


def write_summary_workbook() -> None:
    sheets = {
        "Core Summary": pd.read_csv(OUTPUT_DIR / "core_summary.csv"),
        "Partitions": pd.read_csv(OUTPUT_DIR / "partition_quality_summary.csv"),
        "Source Types": pd.read_csv(OUTPUT_DIR / "source_type_counts.csv"),
        "Languages": pd.read_csv(OUTPUT_DIR / "language_counts.csv"),
        "Object Types": pd.read_csv(OUTPUT_DIR / "object_type_counts.csv"),
        "Missingness": pd.read_csv(OUTPUT_DIR / "column_completeness_all_rows.csv"),
        "Duplicates": pd.read_csv(OUTPUT_DIR / "duplicate_summary.csv"),
        "Top Publications": pd.read_csv(OUTPUT_DIR / "media_summary.csv").head(100),
        "Publisher Regions": pd.read_csv(OUTPUT_DIR / "publisher_region_counts.csv"),
        "Subject Terms": pd.read_csv(OUTPUT_DIR / "top_subject_terms.csv").head(100),
        "Class Terms": pd.read_csv(OUTPUT_DIR / "top_class_terms.csv").head(100),
        "Topic Counts": pd.read_csv(OUTPUT_DIR / "topic_counts.csv"),
        "Topic Terms": pd.read_csv(OUTPUT_DIR / "topic_terms.csv"),
        "Sentiment Decade": pd.read_csv(OUTPUT_DIR / "sentiment_by_decade.csv"),
        "Alias Audit": pd.read_csv(OUTPUT_DIR / "publication_alias_fuzzy_audit.csv").head(200),
        "Data Dictionary": pd.read_csv(FINAL_DIR / "complete_metadata_data_dictionary.csv"),
    }
    with pd.ExcelWriter(FINAL_DIR / "summary_statistics_final.xlsx", engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def make_plots() -> None:
    sns.set_theme(style="white", context="notebook")
    year = pd.read_csv(OUTPUT_DIR / "year_counts.csv").sort_values("year")
    source = pd.read_csv(OUTPUT_DIR / "source_type_counts.csv")
    media = pd.read_csv(OUTPUT_DIR / "media_summary.csv").head(25)
    topic_decade = pd.read_csv(OUTPUT_DIR / "topic_by_decade.csv")
    sentiment = pd.read_csv(OUTPUT_DIR / "sentiment_by_decade.csv").sort_values("decade")

    plt.figure(figsize=(13, 5))
    sns.lineplot(data=year, x="year", y="documents", color="#2F6F73")
    plt.title("English U.S. Query Corpus Documents by Year")
    plt.xlabel("Publication year")
    plt.ylabel("Documents")
    sns.despine()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "documents_by_year.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=source, y="source_type", x="documents", color="#7A5C58")
    plt.title("Documents by Source Type")
    plt.xlabel("Documents")
    plt.ylabel("")
    sns.despine()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "source_type_counts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.barplot(data=media, y="display_publication_title", x="documents", color="#4E79A7")
    plt.title("Top Normalized Publications")
    plt.xlabel("Documents")
    plt.ylabel("")
    sns.despine()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "top_publications.png", dpi=160)
    plt.close()

    pivot = topic_decade.pivot_table(index="topic_label", columns="decade", values="documents", aggfunc="sum", fill_value=0)
    if not pivot.empty:
        pivot = pivot.div(pivot.sum(axis=0), axis=1)
        plt.figure(figsize=(13, 7))
        sns.heatmap(pivot, cmap="YlGnBu", cbar_kws={"label": "Share within decade"})
        plt.title("Topic Mix by Decade")
        plt.xlabel("Decade")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "topic_by_decade_heatmap.png", dpi=160)
        plt.close()

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=sentiment, x="decade", y="mean_net_tone", marker="o", color="#B55D60")
    plt.axhline(0, color="#999999", linewidth=1)
    plt.title("Loughran-McDonald Net Tone by Decade")
    plt.xlabel("Decade")
    plt.ylabel("Mean net tone")
    sns.despine()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "lm_net_tone_by_decade.png", dpi=160)
    plt.close()


def write_report(summary: dict) -> None:
    core = pd.read_csv(OUTPUT_DIR / "core_summary.csv")
    partitions = pd.read_csv(OUTPUT_DIR / "partition_quality_summary.csv")
    duplicates = pd.read_csv(OUTPUT_DIR / "duplicate_summary.csv")
    media = pd.read_csv(OUTPUT_DIR / "media_summary.csv")
    source = pd.read_csv(OUTPUT_DIR / "source_type_counts.csv")
    language = pd.read_csv(OUTPUT_DIR / "language_counts.csv")
    topics = pd.read_csv(OUTPUT_DIR / "topic_counts.csv")
    sentiment = pd.read_csv(OUTPUT_DIR / "sentiment_by_decade.csv")
    mismatch_rows = partitions[~partitions["master_matches_manifest"].fillna(False)]
    lines = [
        "# Partition-Aware Metadata Quality and EDA Report",
        "",
        "## Corpus and Scope",
        "The unit of observation is one TDM metadata document row. The primary tidy export keeps English records from the U.S. query corpus and does not use publisher geography as a sample restriction.",
        "",
        core.to_markdown(index=False),
        "",
        "## Partition Validation",
        "All local partitions contain `master.csv`, `extended.csv.zip`, and `citation.csv.zip`; `master.csv` is the canonical input because it carries the complete 31-column schema.",
        "",
        partitions.to_markdown(index=False),
        "",
        "Manifest mismatches:",
        mismatch_rows.to_markdown(index=False) if len(mismatch_rows) else "No row-count mismatches against the manifest.",
        "",
        "## Data Quality",
        "Duplicate flags are retained on the full tidy export. The deduplicated export drops only extra rows from the conservative normalized title/date/publication rule.",
        "",
        duplicates.to_markdown(index=False),
        "",
        "## Coverage",
        "Source types:",
        source.to_markdown(index=False),
        "",
        "Languages:",
        language.head(12).to_markdown(index=False),
        "",
        "Top normalized publications:",
        media.head(25).to_markdown(index=False),
        "",
        "## NLP and Sentiment",
        "The main NLP text is `Title + Subject Terms + Class Terms`; title-only topic assignments are included as a sensitivity check. The local export contains metadata, not full article body text.",
        f"Loughran-McDonald dictionary source: {LM_SOURCE_URL}",
        "",
        "Topic counts:",
        topics.to_markdown(index=False),
        "",
        "Sentiment by decade:",
        sentiment.to_markdown(index=False),
        "",
        "## Deliverables",
        f"- Complete metadata rows: {summary['complete_rows']:,}.",
        f"- Deduplicated metadata rows: {summary['deduplicated_rows']:,}.",
        "- Main export: `analysis/deliverable/data/complete_metadata.parquet`.",
        "- Access copy: `analysis/deliverable/data/complete_metadata.csv.zip`.",
        "- Summary workbook: `analysis/deliverable/reports/summary_statistics_final.xlsx`.",
        "- Notebook: `analysis/deliverable/notebooks/tdm_corpus_raw_eda_workbook.ipynb`.",
        "",
        "## Limitations",
        "- No full article text is present locally; NLP and sentiment are metadata/title based.",
        "- Subject and class terms are provider metadata and may encode ProQuest classification choices.",
        "- Fuzzy publication alias matches are audit suggestions only unless supported by shared publication ID or publisher metadata.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_pdf_report(lines: list[str]) -> None:
    with PdfPages(REPORT_PDF_PATH) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        y = 0.96
        for line in lines[:80]:
            if line.startswith("# "):
                fig.text(0.06, y, line.replace("# ", ""), fontsize=17, weight="bold", va="top")
                y -= 0.045
            elif line.startswith("## "):
                y -= 0.012
                fig.text(0.06, y, line.replace("## ", ""), fontsize=13, weight="bold", va="top")
                y -= 0.032
            elif line.startswith("|") or line.startswith(":") or not line.strip():
                continue
            else:
                fig.text(0.06, y, line[:115], fontsize=9, va="top")
                y -= 0.024
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11))
                plt.axis("off")
                y = 0.96
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for image in ["documents_by_year.png", "source_type_counts.png", "top_publications.png", "topic_by_decade_heatmap.png", "lm_net_tone_by_decade.png"]:
            path = FIGURE_DIR / image
            if path.exists():
                fig = plt.figure(figsize=(11, 8.5))
                plt.axis("off")
                img = plt.imread(path)
                ax = fig.add_axes([0.04, 0.04, 0.92, 0.9])
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def write_processing_notes(summary: dict) -> None:
    notes = [
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "note": "Refreshed the partition-aware outputs for the 10-partition raw corpus; old single-export artifacts are superseded.",
        },
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "note": "Primary sample is English records from the U.S. query corpus; publisher geography is reported as a diagnostic, not used as a filter.",
        },
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "note": f"Loughran-McDonald dictionary source: {LM_SOURCE_URL}.",
        },
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "note": "During the first full run, complete_metadata.parquet finished successfully but CSV ZIP writing hit the standard ZIP size limit; outputs were resumed from the validated parquet with force_zip64=True.",
        },
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "note": "Removed stale notebook artifacts from the old single-export workflow; the current workbook is analysis/deliverable/notebooks/tdm_corpus_raw_eda_workbook.ipynb.",
        },
    ]
    (OUTPUT_DIR / "processing_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    artifact_rows = []
    for directory in [OUTPUT_DIR, FINAL_DIR, ANALYSIS_DIR]:
        for path in sorted(directory.glob("*")):
            if path.is_file() and path.name not in {"artifact_manifest.csv"}:
                artifact_rows.append(
                    {
                        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                        "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                    }
                )
    pd.DataFrame(artifact_rows).to_csv(OUTPUT_DIR / "artifact_manifest.csv", index=False)
    (FINAL_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def validate_outputs(expected_complete_rows: int) -> dict:
    complete = FINAL_DIR / "complete_metadata.parquet"
    dedup = FINAL_DIR / "complete_metadata_dedup.parquet"
    if not complete.exists() or not dedup.exists():
        raise FileNotFoundError("Expected final parquet outputs are missing.")
    complete_meta = pq.ParquetFile(complete).metadata
    dedup_meta = pq.ParquetFile(dedup).metadata
    if complete_meta.num_rows != expected_complete_rows:
        raise ValueError(f"Complete parquet row count {complete_meta.num_rows:,} != expected {expected_complete_rows:,}.")
    with zipfile.ZipFile(FINAL_DIR / "complete_metadata.csv.zip") as zf:
        csv_member = zf.namelist()[0]
        with zf.open(csv_member) as f:
            header = f.readline().decode("utf-8").strip().split(",")
    required = {"goid", "source_partition", "publication_title_normalized", "topic_id", "lm_net_tone"}
    if not required.issubset(set(header)):
        raise ValueError(f"CSV header is missing required fields: {sorted(required - set(header))}")
    validation = {
        "complete_rows": complete_meta.num_rows,
        "deduplicated_rows": dedup_meta.num_rows,
        "complete_columns": complete_meta.num_columns,
        "csv_zip_readable": True,
        "required_columns_present": True,
    }
    (OUTPUT_DIR / "validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def run_partition_pipeline(chunksize: int = 150_000, sample_per_group_chunk: int = 80, clear_outputs: bool = True) -> dict:
    ensure_dirs()
    if clear_outputs:
        clear_generated_outputs()
    partitions = discover_partitions()
    partition_validation = inspect_partition_files(partitions)
    lexicon = load_lm_dictionary()
    profile = first_pass_profile(partitions, chunksize=chunksize, sample_per_group_chunk=sample_per_group_chunk)
    write_profile_tables(profile, partition_validation)
    vectorizer, topic_model, topic_labels = train_topic_model()
    complete_summary = write_complete_metadata(partitions, profile, vectorizer, topic_model, topic_labels, lexicon, chunksize=chunksize)
    pre_canonical_media = write_media_summary(group_col="publication_title_normalized")
    _, canonical_map = write_alias_audit(pre_canonical_media)
    apply_publication_canonicalization(canonical_map)
    dedup_summary = write_deduplicated_metadata()
    write_final_sample_counts()
    write_media_summary()
    write_topic_and_sentiment_summaries()
    write_data_dictionaries()
    write_summary_workbook()
    make_plots()
    summary = {**complete_summary, **dedup_summary, "raw_rows": profile["all_rows"], "english_rows": profile["english_rows"]}
    validation = validate_outputs(profile["english_rows"])
    summary.update(validation)
    write_report(summary)
    write_processing_notes(summary)
    return summary


def resume_from_complete_metadata() -> dict:
    complete_path = FINAL_DIR / "complete_metadata.parquet"
    if not complete_path.exists():
        raise FileNotFoundError(f"Missing complete metadata parquet: {complete_path}")
    complete_rows = pq.ParquetFile(complete_path).metadata.num_rows
    write_csv_zip_from_parquet(complete_path, FINAL_DIR / "complete_metadata.csv.zip", "complete_metadata.csv")
    pre_canonical_media = write_media_summary(group_col="publication_title_normalized")
    _, canonical_map = write_alias_audit(pre_canonical_media)
    apply_publication_canonicalization(canonical_map)
    dedup_summary = write_deduplicated_metadata()
    write_final_sample_counts()
    write_media_summary()
    write_topic_and_sentiment_summaries()
    write_data_dictionaries()
    write_summary_workbook()
    make_plots()
    core = pd.read_csv(OUTPUT_DIR / "core_summary.csv")
    core_map = dict(zip(core["metric"], core["value"]))
    summary = {
        "complete_rows": complete_rows,
        "complete_parquet": str(complete_path),
        "complete_csv_zip": str(FINAL_DIR / "complete_metadata.csv.zip"),
        **dedup_summary,
        "raw_rows": int(core_map.get("raw_documents", 0)),
        "english_rows": int(core_map.get("english_documents", complete_rows)),
    }
    validation = validate_outputs(summary["english_rows"])
    summary.update(validation)
    write_report(summary)
    write_processing_notes(summary)
    return summary


def main() -> None:
    summary = resume_from_complete_metadata() if "--resume-from-complete" in sys.argv else run_partition_pipeline()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
