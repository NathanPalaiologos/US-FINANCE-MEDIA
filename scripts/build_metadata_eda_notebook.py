from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "analysis" / "deliverable" / "notebooks" / "tdm_corpus_raw_eda_workbook.ipynb"
KERNEL_NAME = os.environ.get("TDM_KERNEL_NAME", "tdm-studio-eda")
KERNEL_DISPLAY_NAME = os.environ.get("TDM_KERNEL_DISPLAY", "TDM Studio (.venv)")


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": KERNEL_DISPLAY_NAME,
        "language": "python",
        "name": KERNEL_NAME,
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    nb.cells = [
        md(
            "# U.S. Business Media Metadata: Raw EDA Workbook\n\n"
            "This notebook starts from the ten raw TDM Studio metadata partitions. It is not a report loader: the checks below re-read `raw/`, use the helper functions in `scripts/partition_pipeline.py`, and build the working summaries in place so the cleaning choices are visible."
        ),
        md(
            "## What I am trying to learn\n\n"
            "The immediate question is whether this corpus is usable for a careful empirical-finance style descriptive study of U.S. business and financial media. I need to know what one row represents, how complete the partitions are, where coverage is thin, how much duplicate pressure exists, and whether the metadata text is rich enough for conservative topic and sentiment work."
        ),
        code(
            "from pathlib import Path\n"
            "from collections import Counter, defaultdict\n"
            "import json\n"
            "import sys\n"
            "import warnings\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "from sklearn.decomposition import LatentDirichletAllocation, NMF\n"
            "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n"
            "from wordcloud import WordCloud\n\n"
            "# Keep notebook imports relative to this file so execution works the same in VS Code and nbclient.\n"
            "sys.path.insert(0, str(Path('../../../scripts').resolve()))\n\n"
            "import partition_pipeline as hp\n\n"
            "RAW = '../../../raw'\n"
            "MANIFEST = '../../tdm_dataset_partitions_manifest.csv'\n"
            "if not Path(MANIFEST).exists():\n"
            "    MANIFEST = '../memory/tdm_dataset_partitions_manifest.csv'\n"
            "pd.set_option('display.max_colwidth', 120)\n"
            "pd.set_option('display.float_format', lambda x: f'{x:,.4f}')\n"
            "sns.set_theme(style='white', context='notebook')"
        ),
        md(
            "## Raw partition inventory\n\n"
            "Before looking at topics or sentiment, I want to make sure the files are what I think they are. The `master.csv` files are the source of truth because they include the full metadata schema; `extended` and `citation` are useful row-count cross-checks."
        ),
        code(
            "manifest = pd.read_csv(MANIFEST)\n"
            "manifest_counts = dict(zip(manifest['dataset_name'], manifest['document_count']))\n"
            "parts = hp.discover_partitions()\n\n"
            "inventory_rows = []\n"
            "for part in parts:\n"
            "    dataset_name = hp.PARTITION_DATASET_MAP.get(part.name)\n"
            "    master_rows = hp.csv_row_count(part.master_path)\n"
            "    extended_rows = hp.zip_csv_row_count(part.path / 'extended.csv.zip')\n"
            "    citation_rows = hp.zip_csv_row_count(part.path / 'citation.csv.zip')\n"
            "    inventory_rows.append({\n"
            "        'partition': part.name,\n"
            "        'family': hp.partition_family(part.name),\n"
            "        'master_rows': master_rows,\n"
            "        'extended_rows': extended_rows,\n"
            "        'citation_rows': citation_rows,\n"
            "        'manifest_rows': manifest_counts.get(dataset_name),\n"
            "        'master_schema_columns': len(hp.read_header(part.master_path)),\n"
            "        'master_schema_ok': hp.read_header(part.master_path) == hp.MASTER_COLUMNS,\n"
            "    })\n\n"
            "inventory = pd.DataFrame(inventory_rows)\n"
            "inventory['matches_manifest'] = inventory['master_rows'].eq(inventory['manifest_rows'])\n"
            "inventory['master_extended_match'] = inventory['master_rows'].eq(inventory['extended_rows'])\n"
            "inventory['master_citation_match'] = inventory['master_rows'].eq(inventory['citation_rows'])\n"
            "inventory"
        ),
        md(
            "The local files are internally consistent when `master`, `extended`, and `citation` row counts agree. The separate manifest comparison catches export/provenance mismatches."
        ),
        code("inventory.loc[~inventory['matches_manifest']]"),
        md(
            "## One row, one document?\n\n"
            "The claimed unit of observation is a document metadata record. I check dates, languages, source types, missingness, and several duplicate keys before treating the corpus as a row-level analytical dataset."
        ),
        code(
            "# Stream through the raw partitions once so the notebook stays laptop-friendly.\n"
            "profile = hp.build_notebook_profile(\n"
            "    parts,\n"
            "    chunksize=200_000,\n"
            "    topic_sample_per_group_chunk=36,\n"
            "    topic_sample_limit=15_000,\n"
            "    wordcloud_term_limit=120,\n"
            ")\n\n"
            "raw_rows = profile['raw_rows']\n"
            "english_rows = profile['english_rows']\n"
            "invalid_dates = profile['invalid_dates']\n"
            "min_date = profile['min_date']\n"
            "max_date = profile['max_date']\n"
            "source_counts = profile['source_counts']\n"
            "language_counts = profile['language_counts']\n"
            "object_counts = profile['object_counts']\n"
            "year_counts = profile['year_counts']\n"
            "missing_counts = profile['missing_counts']\n"
            "publisher_region_counts = profile['publisher_region_counts']\n"
            "publication_counts = profile['publication_counts']\n"
            "publication_examples = profile['publication_examples']\n"
            "publication_id_examples = profile['publication_id_examples']\n"
            "publication_publisher_examples = profile['publication_publisher_examples']\n"
            "publication_source_type_examples = profile['publication_source_type_examples']\n"
            "subject_counts = profile['subject_counts']\n"
            "class_counts = profile['class_counts']\n"
            "author_coverage_by_decade = profile['author_coverage_by_decade']\n"
            "author_coverage_by_decade_source = profile['author_coverage_by_decade_source']\n"
            "goid_hashes = profile['goid_hashes']\n"
            "normalized_hashes = profile['normalized_hashes']\n"
            "title_date_hashes = profile['title_date_hashes']\n"
            "topic_sample = profile['topic_sample'].copy()\n"
            "full_wordcloud_group_sizes = profile['wordcloud_group_sizes']\n"
            "full_wordcloud_term_frequencies = profile['wordcloud_term_frequencies']\n"
            "topic_model_sample_rows = profile['topic_model_sample_rows']\n"
            "wordcloud_document_rows = profile['wordcloud_document_rows']\n\n"
            "print(f'Raw rows: {raw_rows:,}')\n"
            "print(f'English rows: {english_rows:,}')\n"
            "print(f'Date range: {min_date.date()} to {max_date.date()}')\n"
            "print(f'Invalid/missing dates: {invalid_dates:,}')\n"
            "print(f'Topic-model sample rows: {topic_model_sample_rows:,}')\n"
            "print(f'Full-corpus word-cloud rows: {wordcloud_document_rows:,}')"
        ),
        code(
            "coverage = pd.DataFrame([\n"
            "    {'metric': 'raw rows', 'value': raw_rows},\n"
            "    {'metric': 'English rows', 'value': english_rows},\n"
            "    {'metric': 'English share', 'value': english_rows / raw_rows},\n"
            "    {'metric': 'first observed date', 'value': str(min_date.date())},\n"
            "    {'metric': 'last observed date', 'value': str(max_date.date())},\n"
            "    {'metric': 'invalid/missing dates', 'value': invalid_dates},\n"
            "])\n"
            "coverage"
        ),
        code(
            "missing = (\n"
            "    pd.DataFrame([{'column': k, 'missing_rows': v, 'missing_share': v / raw_rows} for k, v in missing_counts.items()])\n"
            "    .sort_values('missing_share', ascending=False)\n"
            ")\n"
            "missing.head(20)"
        ),
        md(
            "The high-missing fields are not all equally concerning. Missing `Company Name` or `Authors` limits some designs, but it does not invalidate document-level coverage. Missing dates or publication titles would be much more damaging; those are the fields I care about first."
        ),
        md("## Coverage over time and source type"),
        code(
            "source = pd.DataFrame(source_counts.most_common(), columns=['source_type', 'raw_documents'])\n"
            "language = pd.DataFrame(language_counts.most_common(), columns=['language', 'documents'])\n"
            "years = pd.DataFrame(sorted(year_counts.items()), columns=['year', 'english_documents'])\n"
            "regions = pd.DataFrame(publisher_region_counts.most_common(), columns=['publisher_region', 'raw_documents'])\n"
            "display(source)\n"
            "display(language.head(12))\n"
            "display(regions.head(12))"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(12, 4))\n"
            "sns.lineplot(data=years, x='year', y='english_documents', ax=ax, color='#2F6F73')\n"
            "ax.set_title('English metadata records by publication year')\n"
            "ax.set_xlabel('Year')\n"
            "ax.set_ylabel('Documents')\n"
            "sns.despine()\n"
            "plt.show()"
        ),
        md(
            "The early years are real coverage, but they are not balanced coverage. For any professor-facing writeup, I would describe this as observed database coverage over time rather than complete U.S. media history."
        ),
        md("## Publication titles and alias pressure"),
        code(
            "publication_profiles = pd.DataFrame([\n"
            "    {\n"
            "        'publication_title_normalized': name,\n"
            "        'publication_title_canonical': name,\n"
            "        'documents': count,\n"
            "        'sample_raw_titles': '; '.join(title for title, _ in publication_examples[name].most_common(3)),\n"
            "        'publication_ids': '; '.join(pub_id for pub_id, _ in publication_id_examples[name].most_common(3)),\n"
            "        'top_publisher': next((publisher for publisher, _ in publication_publisher_examples[name].most_common(1)), ''),\n"
            "        'top_source_type': next((source_type for source_type, _ in publication_source_type_examples[name].most_common(1)), ''),\n"
            "    }\n"
            "    for name, count in publication_counts.most_common(300)\n"
            "])\n"
            "alias_candidates, canonical_map = hp.build_publication_alias_audit(publication_profiles)\n"
            "publication_profiles['publication_title_canonical'] = publication_profiles['publication_title_normalized'].map(canonical_map).fillna(publication_profiles['publication_title_normalized'])\n"
            "top_publications = (\n"
            "    publication_profiles.groupby('publication_title_canonical', as_index=False)\n"
            "    .agg(\n"
            "        english_documents=('documents', 'sum'),\n"
            "        sample_raw_titles=('sample_raw_titles', 'first'),\n"
            "        normalized_variants=('publication_title_normalized', lambda x: '; '.join(sorted(set(x))[:5])),\n"
            "    )\n"
            "    .sort_values('english_documents', ascending=False)\n"
            ")\n"
            "top_publications.head(30)"
        ),
        code(
            "alias_candidates[['publication_left', 'publication_right', 'char_similarity', 'token_overlap', 'auto_merge', 'recommendation']].head(25)"
        ),
        md(
            "The alias review now uses token overlap and metadata evidence instead of raw string similarity alone. That keeps obvious variants together while separating location-shifted titles like city-specific business journals unless stronger evidence says otherwise."
        ),
        md(
            "## Most visible publications: corpus evidence plus real-world reputation\n\n"
            "Corpus frequency and real-world reputation answer different questions. The table below combines the most common publication titles in the data with a short analyst-defined reference list of major U.S. finance, business, and general-news outlets with strong business coverage. This is not a formal media ranking; it is a sanity check for whether the corpus contains the outlets a reader would expect to see."
        ),
        code(
            "media_summary_path = Path('../csv/media_summary.csv')\n"
            "if media_summary_path.exists():\n"
            "    observed_media = pd.read_csv(media_summary_path)\n"
            "    observed_media = observed_media.rename(columns={'documents': 'english_documents'})\n"
            "else:\n"
            "    observed_media = top_publications.rename(columns={'sample_raw_titles': 'display_publication_title'})\n"
            "    observed_media['first_year'] = pd.NA\n"
            "    observed_media['last_year'] = pd.NA\n"
            "    observed_media['top_source_type'] = pd.NA\n"
            "\n"
            "observed_media = observed_media.copy()\n"
            "if 'publication_title_canonical' not in observed_media.columns:\n"
            "    observed_media['publication_title_canonical'] = observed_media.get('publication_title_normalized', observed_media.get('display_publication_title'))\n"
            "if 'display_publication_title' not in observed_media.columns:\n"
            "    observed_media['display_publication_title'] = observed_media['publication_title_canonical']\n"
            "observed_media['publication_key'] = observed_media['publication_title_canonical'].map(hp.normalize_publication)\n"
            "observed_media['corpus_rank'] = observed_media['english_documents'].rank(method='first', ascending=False).astype(int)\n"
            "\n"
            "reference_media = pd.DataFrame([\n"
            "    {'reference_publication': 'Wall Street Journal', 'reference_role': 'national financial newspaper', 'expected_in_scope': True},\n"
            "    {'reference_publication': \"Barron's\", 'reference_role': 'investment weekly', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'Bloomberg Businessweek', 'reference_role': 'business magazine', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'Forbes', 'reference_role': 'business magazine', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'Fortune', 'reference_role': 'business magazine', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'Inc.', 'reference_role': 'entrepreneurship magazine', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'Fast Company', 'reference_role': 'business and technology magazine', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'MarketWatch', 'reference_role': 'markets and finance news', 'expected_in_scope': True},\n"
            "    {'reference_publication': \"Investor's Business Daily\", 'reference_role': 'markets newspaper/news site', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'American Banker', 'reference_role': 'banking trade publication', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'Business Insider', 'reference_role': 'business news site', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'CNBC', 'reference_role': 'business television/news site', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'New York Times', 'reference_role': 'national newspaper with business coverage', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'Washington Post', 'reference_role': 'national newspaper with business coverage', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'Reuters', 'reference_role': 'newswire with business coverage', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'Associated Press', 'reference_role': 'newswire with business coverage', 'expected_in_scope': True},\n"
            "    {'reference_publication': 'Financial Times', 'reference_role': 'global business newspaper, not U.S.-based', 'expected_in_scope': False},\n"
            "    {'reference_publication': 'The Economist', 'reference_role': 'global business/economics magazine, not U.S.-based', 'expected_in_scope': False},\n"
            "])\n"
            "reference_media['publication_key'] = reference_media['reference_publication'].map(hp.normalize_publication)\n"
            "\n"
            "reference_matches = reference_media.merge(\n"
            "    observed_media,\n"
            "    on='publication_key',\n"
            "    how='left',\n"
            "    suffixes=('', '_observed'),\n"
            ")\n"
            "reference_matches['real_world_reference'] = True\n"
            "reference_matches['why_included'] = 'reference outlet'\n"
            "\n"
            "observed_top = observed_media.head(35).copy()\n"
            "observed_top['reference_publication'] = observed_top['publication_title_canonical']\n"
            "reference_role_map = dict(zip(reference_media['publication_key'], reference_media['reference_role']))\n"
            "observed_top['reference_role'] = observed_top['publication_key'].map(reference_role_map).fillna('high-frequency title in corpus')\n"
            "observed_top['expected_in_scope'] = True\n"
            "observed_top['real_world_reference'] = observed_top['publication_key'].isin(reference_media['publication_key'])\n"
            "observed_top['why_included'] = 'top corpus frequency'\n"
            "\n"
            "combined_publications = pd.concat([observed_top, reference_matches], ignore_index=True, sort=False)\n"
            "combined_publications['observed_in_corpus'] = combined_publications['english_documents'].notna()\n"
            "combined_publications['publication'] = combined_publications['display_publication_title'].fillna(combined_publications['reference_publication'])\n"
            "popular_publications = (\n"
            "    combined_publications.sort_values(['observed_in_corpus', 'english_documents', 'real_world_reference'], ascending=[False, False, False])\n"
            "    .drop_duplicates('publication_key')\n"
            "    [[\n"
            "        'publication', 'english_documents', 'corpus_rank', 'first_year', 'last_year',\n"
            "        'top_source_type', 'real_world_reference', 'reference_role', 'why_included', 'observed_in_corpus'\n"
            "    ]]\n"
            "    .reset_index(drop=True)\n"
            ")\n"
            "popular_publications.head(60)"
        ),
        code(
            "missing_reference_publications = (\n"
            "    reference_matches.loc[\n"
            "        reference_matches['expected_in_scope'].eq(True) & reference_matches['english_documents'].isna(),\n"
            "        ['reference_publication', 'reference_role', 'expected_in_scope']\n"
            "    ]\n"
            "    .sort_values('reference_publication')\n"
            "    .reset_index(drop=True)\n"
            ")\n"
            "missing_reference_publications"
        ),
        md(
            "If the missing-reference table is non-empty, I would not immediately treat those outlets as impossible absences. Some sources may enter under wire-service, web-edition, or archive-specific titles, and some may be excluded by ProQuest product coverage. Still, a famous U.S. finance outlet missing from this watchlist is a real coverage warning to mention in the writeup."
        ),
        md("## Duplicate pressure"),
        code(
            "duplicate_rows = []\n"
            "for name, hashes in [('GOID', goid_hashes), ('normalized title/date/publication', normalized_hashes), ('title/date only', title_date_hashes)]:\n"
            "    duplicate_set, duplicate_extra_rows = hp.duplicate_info(hashes)\n"
            "    duplicate_rows.append({\n"
            "        'rule': name,\n"
            "        'duplicate_extra_rows': duplicate_extra_rows,\n"
            "        'duplicate_groups': len(duplicate_set),\n"
            "        'share_of_english_rows': duplicate_extra_rows / english_rows,\n"
            "    })\n"
            "duplicates = pd.DataFrame(duplicate_rows)\n"
            "duplicates"
        ),
        md(
            "The absence of duplicate `GOID` rows is reassuring. The title/date based duplicates are a different issue: they reflect reprints, repeated metadata records, or highly similar publication records, so I keep flags in the tidy data rather than pretending there is one universally correct deduplication rule."
        ),
        md("## What the subject metadata says"),
        code(
            "subjects = pd.DataFrame(subject_counts.most_common(30), columns=['subject_term', 'sampled_mentions'])\n"
            "classes = pd.DataFrame(class_counts.most_common(30), columns=['class_term', 'sampled_mentions'])\n"
            "display(subjects)\n"
            "display(classes)"
        ),
        md(
            "These are provider metadata terms, not article-body words. They are still useful because they show what ProQuest indexed this corpus as being about. I treat them as coverage metadata, not as direct language from journalists."
        ),
        md(
            "## Author information coverage by decade\n\n"
            "Author names are useful for byline, journalist-network, and repeated-contributor questions, but only if coverage is not too sparse. I summarize author coverage by decade and then zoom in on the post-2000 period, where born-digital and modern indexing practices should make bylines more consistently available."
        ),
        code(
            "author_decade = pd.DataFrame([\n"
            "    {'decade': decade, **totals}\n"
            "    for decade, totals in sorted(author_coverage_by_decade.items())\n"
            "])\n"
            "author_decade['author_coverage_share'] = author_decade['documents_with_author'] / author_decade['documents']\n"
            "author_decade['mean_author_mentions_per_document'] = author_decade['author_mentions'] / author_decade['documents']\n"
            "author_decade.tail(12)"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(10, 4))\n"
            "plot_author = author_decade.dropna(subset=['decade']).copy()\n"
            "sns.lineplot(data=plot_author, x='decade', y='author_coverage_share', marker='o', ax=ax, color='#4E79A7')\n"
            "ax.axvline(2000, color='#999999', linewidth=1)\n"
            "ax.set_title('Share of English records with author information by decade')\n"
            "ax.set_xlabel('Decade')\n"
            "ax.set_ylabel('Share with author/byline metadata')\n"
            "ax.set_ylim(0, min(1, max(plot_author['author_coverage_share'].max() * 1.15, 0.1)))\n"
            "sns.despine()\n"
            "plt.show()"
        ),
        code(
            "author_decade_source = pd.DataFrame([\n"
            "    {'decade': int(key.split('|', 1)[0]), 'source_type': key.split('|', 1)[1], **totals}\n"
            "    for key, totals in author_coverage_by_decade_source.items()\n"
            "])\n"
            "author_decade_source['author_coverage_share'] = author_decade_source['documents_with_author'] / author_decade_source['documents']\n"
            "post_2000_author = (\n"
            "    author_decade_source.loc[author_decade_source['decade'].ge(2000)]\n"
            "    .groupby('source_type', as_index=False)\n"
            "    .agg(\n"
            "        documents=('documents', 'sum'),\n"
            "        documents_with_author=('documents_with_author', 'sum'),\n"
            "        author_mentions=('author_mentions', 'sum'),\n"
            "    )\n"
            ")\n"
            "post_2000_author['author_coverage_share'] = post_2000_author['documents_with_author'] / post_2000_author['documents']\n"
            "post_2000_author['mean_author_mentions_per_document'] = post_2000_author['author_mentions'] / post_2000_author['documents']\n"
            "post_2000_author.sort_values('documents', ascending=False)"
        ),
        md(
            "The post-2000 table is the one I would use for any author-level design decision. If source types differ sharply in byline coverage, author analyses should either stratify by source type or report coverage as a sample restriction."
        ),
        md("## A first topic model on metadata text"),
        code(
            "topic_sample = topic_sample.dropna(subset=['decade', 'Source Type', 'partition_family']).copy()\n"
            "assert len(topic_sample), 'Topic-model sample is empty.'\n\n"
            "# Keep topic models on the bounded sample; the descriptive word clouds below use the full metadata corpus.\n"
            "vectorizer = TfidfVectorizer(\n"
            "    stop_words=list(hp.CORPUS_STOPWORDS),\n"
            "    max_features=5000,\n"
            "    min_df=8,\n"
            "    max_df=0.65,\n"
            "    ngram_range=(1, 2),\n"
            "    token_pattern=r'(?u)\\b[a-zA-Z][a-zA-Z]+\\b',\n"
            ")\n"
            "X = vectorizer.fit_transform(topic_sample['topic_text'])\n"
            "nmf = NMF(n_components=10, random_state=42, init='nndsvda', max_iter=300)\n"
            "W = nmf.fit_transform(X)\n"
            "terms = np.array(vectorizer.get_feature_names_out())\n"
            "topic_sample['topic'] = W.argmax(axis=1)\n\n"
            "topic_rows = []\n"
            "for topic_id, weights in enumerate(nmf.components_):\n"
            "    top_terms = terms[weights.argsort()[::-1][:10]]\n"
            "    topic_rows.append({'topic': topic_id, 'top_terms': ', '.join(top_terms), 'sample_documents': int((topic_sample['topic'] == topic_id).sum())})\n"
            "pd.DataFrame(topic_rows).sort_values('sample_documents', ascending=False)"
        ),
        md(
            "This NMF model is fit on a bounded stratified metadata sample so the notebook stays runnable on a laptop. I use the full corpus for grouped word clouds below, but I keep probabilistic topic models on a smaller sample because fitting them on all 6.2M rows would be disproportionately expensive in this notebook setting."
        ),
        md("## LDA topic model on the same metadata sample"),
        code(
            "lda_vectorizer = CountVectorizer(\n"
            "    stop_words=list(hp.CORPUS_STOPWORDS),\n"
            "    max_features=5000,\n"
            "    min_df=8,\n"
            "    max_df=0.65,\n"
            "    token_pattern=r'(?u)\\b[a-zA-Z][a-zA-Z]+\\b',\n"
            ")\n"
            "X_lda = lda_vectorizer.fit_transform(topic_sample['topic_text'])\n"
            "lda = LatentDirichletAllocation(n_components=10, random_state=42, learning_method='batch', max_iter=20)\n"
            "lda_weights = lda.fit_transform(X_lda)\n"
            "lda_terms = np.array(lda_vectorizer.get_feature_names_out())\n"
            "topic_sample['lda_topic'] = lda_weights.argmax(axis=1)\n\n"
            "lda_rows = []\n"
            "for topic_id, weights in enumerate(lda.components_):\n"
            "    top_terms = lda_terms[weights.argsort()[::-1][:10]]\n"
            "    lda_rows.append({'topic': topic_id, 'top_terms': ', '.join(top_terms), 'sample_documents': int((topic_sample['lda_topic'] == topic_id).sum())})\n"
            "pd.DataFrame(lda_rows).sort_values('sample_documents', ascending=False)"
        ),
        md(
            "The LDA view is complementary rather than authoritative. I read it as a second clustering lens on the same metadata sample, not as a claim that the notebook has recovered stable latent themes from full article text."
        ),
        md(
            "## Metadata word clouds by decade and media facet\n\n"
            "These clouds now use the full English corpus with usable metadata text, not the bounded topic-model sample. They summarize titles plus provider subject/class terms, not article bodies, so I treat them as a faceted metadata lens rather than a recovered topical truth."
        ),
        code(
            "analysis_scope = pd.DataFrame([\n"
            "    {'artifact': 'topic-model sample', 'documents': profile['topic_model_sample_rows']},\n"
            "    {'artifact': 'full-corpus word clouds', 'documents': profile['wordcloud_document_rows']},\n"
            "])\n"
            "decade_topic_counts = pd.DataFrame(sorted(full_wordcloud_group_sizes['decade'].items()), columns=['decade', 'documents_with_metadata_text'])\n"
            "source_type_topic_counts = pd.DataFrame(full_wordcloud_group_sizes['source_type'].most_common(), columns=['source_type', 'documents_with_metadata_text'])\n"
            "family_topic_counts = pd.DataFrame(full_wordcloud_group_sizes['partition_family'].most_common(), columns=['partition_family', 'documents_with_metadata_text'])\n"
            "display(analysis_scope)\n"
            "display(decade_topic_counts.tail(12))\n"
            "display(source_type_topic_counts)\n"
            "display(family_topic_counts)"
        ),
        code(
            "def render_group_wordclouds(frequency_map, group_sizes, ncols=3, colormap='viridis'):\n"
            "    # Each panel is one facet so the comparison stays readable even when group sizes differ a lot.\n"
            "    ordered = [(group, freq) for group, freq in frequency_map.items() if freq]\n"
            "    nrows = max((len(ordered) + ncols - 1) // ncols, 1)\n"
            "    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))\n"
            "    axes = np.atleast_1d(axes).ravel()\n"
            "    for ax in axes:\n"
            "        ax.axis('off')\n"
            "    for ax, (group_value, frequencies) in zip(axes, ordered):\n"
            "        cloud = WordCloud(width=900, height=500, background_color='white', colormap=colormap).generate_from_frequencies(frequencies)\n"
            "        ax.imshow(cloud, interpolation='bilinear')\n"
            "        label = int(group_value) if isinstance(group_value, (np.integer, int, float)) and pd.notna(group_value) else group_value\n"
            "        ax.set_title(f'{label} ({int(group_sizes[group_value]):,} docs)')\n"
            "        ax.axis('off')\n"
            "    plt.tight_layout()\n"
            "    plt.show()"
        ),
        code(
            "decade_wordclouds = dict(sorted(full_wordcloud_term_frequencies['decade'].items()))\n"
            "render_group_wordclouds(decade_wordclouds, full_wordcloud_group_sizes['decade'], ncols=4, colormap='viridis')"
        ),
        code(
            "source_type_wordclouds = full_wordcloud_term_frequencies['source_type']\n"
            "render_group_wordclouds(source_type_wordclouds, full_wordcloud_group_sizes['source_type'], ncols=3, colormap='magma')"
        ),
        code(
            "family_wordclouds = full_wordcloud_term_frequencies['partition_family']\n"
            "render_group_wordclouds(family_wordclouds, full_wordcloud_group_sizes['partition_family'], ncols=3, colormap='cividis')"
        ),
        md("## Loughran-McDonald tone on the same sample"),
        code(
            "lm = hp.load_lm_dictionary()\n"
            "tone_rows = []\n"
            "# Score the same sampled metadata text used for the topic models so these views stay comparable.\n"
            "for _, row in topic_sample.iterrows():\n"
            "    scores = hp.sentiment_scores(row['topic_text'], lm)\n"
            "    tone_rows.append({\n"
            "        'year': pd.to_datetime(row['Date'], errors='coerce').year,\n"
            "        'source_type': row['Source Type'],\n"
            "        **scores,\n"
            "    })\n"
            "tone = pd.DataFrame(tone_rows).dropna(subset=['year'])\n"
            "tone['decade'] = (tone['year'].astype(int) // 10) * 10\n"
            "tone_by_decade = tone.groupby('decade').agg(\n"
            "    sampled_documents=('lm_token_count', 'size'),\n"
            "    mean_net_tone=('lm_net_tone', 'mean'),\n"
            "    positive_words=('lm_positive', 'sum'),\n"
            "    negative_words=('lm_negative', 'sum'),\n"
            "    uncertainty_words=('lm_uncertainty', 'sum'),\n"
            ").reset_index()\n"
            "tone_by_decade"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "sns.lineplot(data=tone_by_decade, x='decade', y='mean_net_tone', marker='o', ax=ax, color='#B55D60')\n"
            "ax.axhline(0, color='#999999', linewidth=1)\n"
            "ax.set_title('Sampled Loughran-McDonald net tone by decade')\n"
            "ax.set_xlabel('Decade')\n"
            "ax.set_ylabel('Mean net tone')\n"
            "sns.despine()\n"
            "plt.show()"
        ),
        md(
            "## ANOVA: post-2000 tone differences by source type\n\n"
            "A naive ANOVA over millions of document rows would mostly tell us that the sample is huge. The more useful question here is narrower: in the bounded metadata sample used for topics and sentiment, do post-2000 source types have meaningfully different average Loughran-McDonald net tone? I treat this as a descriptive diagnostic, not a causal test."
        ),
        code(
            "anova_sample = tone.loc[\n"
            "    tone['year'].ge(2000) & tone['lm_token_count'].gt(0) & tone['source_type'].notna(),\n"
            "    ['source_type', 'lm_net_tone']\n"
            "].copy()\n"
            "group_sizes = anova_sample['source_type'].value_counts()\n"
            "kept_groups = group_sizes[group_sizes >= 30].index\n"
            "anova_sample = anova_sample[anova_sample['source_type'].isin(kept_groups)].copy()\n"
            "\n"
            "group_stats = (\n"
            "    anova_sample.groupby('source_type', as_index=False)\n"
            "    .agg(\n"
            "        documents=('lm_net_tone', 'size'),\n"
            "        mean_net_tone=('lm_net_tone', 'mean'),\n"
            "        sd_net_tone=('lm_net_tone', 'std'),\n"
            "    )\n"
            "    .sort_values('documents', ascending=False)\n"
            ")\n"
            "\n"
            "grand_mean = anova_sample['lm_net_tone'].mean()\n"
            "ss_between = sum(\n"
            "    len(group) * (group['lm_net_tone'].mean() - grand_mean) ** 2\n"
            "    for _, group in anova_sample.groupby('source_type')\n"
            ")\n"
            "ss_within = sum(\n"
            "    ((group['lm_net_tone'] - group['lm_net_tone'].mean()) ** 2).sum()\n"
            "    for _, group in anova_sample.groupby('source_type')\n"
            ")\n"
            "df_between = anova_sample['source_type'].nunique() - 1\n"
            "df_within = len(anova_sample) - anova_sample['source_type'].nunique()\n"
            "ms_between = ss_between / df_between if df_between else np.nan\n"
            "ms_within = ss_within / df_within if df_within else np.nan\n"
            "f_stat = ms_between / ms_within if ms_within else np.nan\n"
            "eta_squared = ss_between / (ss_between + ss_within) if (ss_between + ss_within) else np.nan\n"
            "\n"
            "anova_table = pd.DataFrame([\n"
            "    {'source': 'source_type', 'sum_sq': ss_between, 'df': df_between, 'mean_sq': ms_between, 'F': f_stat, 'eta_squared': eta_squared},\n"
            "    {'source': 'residual', 'sum_sq': ss_within, 'df': df_within, 'mean_sq': ms_within, 'F': np.nan, 'eta_squared': np.nan},\n"
            "])\n"
            "display(group_stats)\n"
            "anova_table"
        ),
        md(
            "I focus on effect size (`eta_squared`) and the group means rather than p-values. With large metadata samples, tiny differences can become statistically significant while remaining substantively unimportant. If `eta_squared` is small, source-type tone differences exist but explain little of the variation in document-level tone."
        ),
        md(
            "## Usable fields and changing metadata depth\n\n"
            "The corpus is not just titles and dates. The next checks summarize which fields are consistently populated enough to support analysis, then show whether field availability changes over time."
        ),
        code(
            "usable_field_names = [\n"
            "    'Title', 'Date', 'Source Type', 'Authors', 'Publication Title', 'Publisher Name',\n"
            "    'Publisher City', 'Publisher Province', 'Object Type', 'Pages', 'Subject Terms',\n"
            "    'Class Terms', 'Company Name'\n"
            "]\n"
            "usable_fields = (\n"
            "    missing.loc[missing['column'].isin(usable_field_names)]\n"
            "    .assign(nonmissing_share=lambda d: 1 - d['missing_share'])\n"
            "    [['column', 'nonmissing_share', 'missing_rows', 'missing_share']]\n"
            "    .sort_values('nonmissing_share', ascending=False)\n"
            ")\n"
            "usable_fields"
        ),
        code(
            "field_trend_sample = topic_sample.copy()\n"
            "field_trend_sample['has_subject_terms'] = field_trend_sample['Subject Terms'].notna() & field_trend_sample['Subject Terms'].astype(str).str.len().gt(0)\n"
            "field_trend_sample['has_class_terms'] = field_trend_sample['Class Terms'].notna() & field_trend_sample['Class Terms'].astype(str).str.len().gt(0)\n"
            "metadata_depth = (\n"
            "    field_trend_sample.groupby('decade', as_index=False)\n"
            "    .agg(\n"
            "        sampled_documents=('topic_text', 'size'),\n"
            "        subject_term_coverage=('has_subject_terms', 'mean'),\n"
            "        class_term_coverage=('has_class_terms', 'mean'),\n"
            "        mean_topic_text_length=('topic_text', lambda x: x.astype(str).str.len().mean()),\n"
            "    )\n"
            ")\n"
            "metadata_depth.tail(12)"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(10, 4))\n"
            "sns.lineplot(data=metadata_depth, x='decade', y='subject_term_coverage', marker='o', ax=ax, label='Subject terms')\n"
            "sns.lineplot(data=metadata_depth, x='decade', y='class_term_coverage', marker='o', ax=ax, label='Class terms')\n"
            "ax.set_title('Metadata term coverage in the topic-model sample')\n"
            "ax.set_xlabel('Decade')\n"
            "ax.set_ylabel('Share with field present')\n"
            "ax.set_ylim(0, 1.05)\n"
            "sns.despine()\n"
            "plt.show()"
        ),
        md(
            "These field trends matter for interpretation: a change in topic labels over time can reflect real media change, but it can also reflect richer or poorer provider metadata in particular periods."
        ),
        md(
            "## What I would tell a professor from this EDA\n\n"
            "- The corpus is large enough for serious descriptive work, but historical coverage is uneven and should be described carefully.\n"
            "- `GOID` works as the document key in the local files; title/date duplicates are real enough to keep explicit duplicate flags.\n"
            "- The English sample is the right primary scope for NLP, while non-English records should remain part of the raw coverage audit.\n"
            "- The full corpus is appropriate for descriptive word-frequency views, but NMF/LDA topic models should still be framed as metadata/title-based evidence fit on a bounded sample unless full article text is exported later.\n"
            "- Author coverage and famous-outlet coverage should be reported as diagnostics before making claims about journalist behavior or representativeness of U.S. finance media."
        ),
    ]
    return nb


def main(execute: bool = False) -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nbf.write(nb, NOTEBOOK_PATH)
    if execute:
        client = NotebookClient(
            nb,
            timeout=3600,
            kernel_name=KERNEL_NAME,
            resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
        )
        client.execute()
        nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main(execute="--execute" in sys.argv)
