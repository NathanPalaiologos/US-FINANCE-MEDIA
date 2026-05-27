from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "analysis" / "metadata_panel_topic_modeling_workflow.ipynb"
CURRENT_KERNEL_NAME = "sample-2025.12.578"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": CURRENT_KERNEL_NAME,
        "language": "python",
        "name": CURRENT_KERNEL_NAME,
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    nb.cells = [
        md(
            "# Metadata Panel and Topic Modeling Workflow\n\n"
            "This notebook is the working view for the metadata export. It audits the available XML/CSV fields, builds the document-level metadata panel, and then reads the prepared outputs in a report-friendly order."
        ),
        md("## Setup\n\nThe notebook keeps the current TDM Studio Python kernel and uses repository-relative paths so it can be rerun from the project root or from the `analysis/` folder."),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display, Markdown\n\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'scripts').exists() and (ROOT.parent / 'scripts').exists():\n"
            "    ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT / 'scripts'))\n\n"
            "from eda_helpers import (\n"
            "    scan_xml_field_inventory,\n"
            "    write_xml_field_manifest,\n"
            "    build_metadata_panel_outputs,\n"
            "    write_metadata_export_manifest,\n"
            ")\n\n"
            "OUT = ROOT / 'analysis' / 'outputs'\n"
            "FINAL = ROOT / 'analysis' / 'final'\n"
            "FIG = OUT / 'figures'\n"
            "pd.set_option('display.max_colwidth', 120)\n"
            "ROOT"
        ),
        md(
            "## XML and CSV Field Availability\n\n"
            "The TDM metadata README says columns are omitted when no document has a given tag. This audit therefore separates confirmed CSV fields from XML candidate paths that should be scanned when XML exports are available."
        ),
        code("write_xml_field_manifest()"),
        code("scan_xml_field_inventory(max_files=100)"),
        md(
            "## Build the Metadata Panel\n\n"
            "This is the efficient export step. It reuses the cleaned parquet when available, adds journal/publication ranges and coverage density, and creates both title-only and metadata-enriched text fields for topic-modeling experiments."
        ),
        code("build_metadata_panel_outputs()"),
        code("write_metadata_export_manifest()"),
        code("pd.read_json(FINAL / 'metadata_panel_summary.json', typ='series')"),
        code("pd.read_csv(FINAL / 'metadata_export_manifest.csv')"),
        md("## Panel Field Definitions"),
        code("pd.read_csv(FINAL / 'metadata_panel_data_dictionary.csv')"),
        md("## Journal Range and Coverage"),
        code("pd.read_csv(OUT / 'journal_coverage_summary.csv').head(25)"),
        md("## Metadata Panel Preview"),
        code(
            "panel_preview_cols = [\n"
            "    'goid', 'title', 'date', 'source_type', 'object_type',\n"
            "    'publication_title_display', 'journal_range_start', 'journal_range_end',\n"
            "    'full_text_status', 'primary_author', 'metadata_suggests_full_text_record'\n"
            "]\n"
            "pd.read_parquet(FINAL / 'metadata_panel.parquet', columns=panel_preview_cols).head(20)"
        ),
        md("## Core Summary"),
        code("pd.read_csv(OUT / 'core_summary.csv')"),
        code("display(Image(filename=str(FIG / 'core_coverage_snapshot.png')))"),
        md("## Metadata Field Audit"),
        code("pd.read_csv(FINAL / 'metadata_field_audit.csv')"),
        md("## Final Cleaned Dataset"),
        code("pd.read_json(FINAL / 'cleaned_metadata_summary.json', typ='series')"),
        code("pd.read_csv(FINAL / 'cleaned_metadata_data_dictionary.csv')"),
        md("## Scope"),
        code("pd.read_csv(OUT / 'publisher_region_counts.csv')"),
        code("pd.read_csv(OUT / 'edition_flag_counts.csv')"),
        code("pd.read_csv(OUT / 'excluded_media_summary.csv')"),
        md("## Media Coverage"),
        code("pd.read_csv(OUT / 'media_summary.csv').head(20)"),
        code("display(Image(filename=str(FIG / 'media_summary_counts.png')))"),
        md("## Time Coverage"),
        code("pd.read_csv(OUT / 'decade_counts.csv')"),
        code("display(Image(filename=str(FIG / 'documents_by_year.png')))"),
        md("## Source Type Over Time"),
        code("pd.read_csv(OUT / 'source_type_by_year_english_na.csv').head(20)"),
        code("display(Image(filename=str(FIG / 'source_type_by_year_lines.png')))"),
        code("display(Image(filename=str(FIG / 'source_type_by_year_since_1980.png')))"),
        md("## Issue Coverage"),
        code("pd.read_csv(OUT / 'core_issue_coverage.csv')"),
        code("pd.read_csv(OUT / 'top_subject_terms.csv').head(20)"),
        code("pd.read_csv(OUT / 'top_class_terms.csv').head(15)"),
        md("## Data Quality"),
        code("pd.read_csv(OUT / 'duplicate_summary.csv')"),
        code("pd.read_csv(OUT / 'column_completeness.csv')"),
        code("pd.read_csv(OUT / 'publisher_city_counts.csv').head(20)"),
        code("pd.read_csv(OUT / 'publisher_city_raw_counts.csv').head(20)"),
        md(
            "## Topic Modeling Inputs\n\n"
            "The existing model remains title-only for conservative interpretation. The panel also exposes `metadata_enriched_topic_text` so subject/class-term sensitivity runs can be compared without changing the unit of observation."
        ),
        code("pd.read_json(OUT / 'topic_model_profile.json', typ='series')"),
        code("pd.read_csv(OUT / 'topic_counts.csv')"),
        code("pd.read_csv(OUT / 'topic_terms.csv').query('Rank <= 8')"),
        code("display(Image(filename=str(FIG / 'topic_counts.png')))"),
        code("display(Image(filename=str(FIG / 'topic_by_decade_heatmap.png')))"),
        code("display(Image(filename=str(FIG / 'topic_by_media_heatmap.png')))"),
        md("## Final Files"),
        code("pd.DataFrame({'file': [p.name for p in sorted(FINAL.glob('*'))], 'bytes': [p.stat().st_size for p in sorted(FINAL.glob('*'))]})"),
    ]
    return nb


def main() -> None:
    nb = build_notebook()
    nbf.write(nb, NOTEBOOK_PATH)
    try:
        client = NotebookClient(nb, timeout=900, kernel_name=CURRENT_KERNEL_NAME)
        client.execute()
        nbf.write(nb, NOTEBOOK_PATH)
        print(f"Wrote and executed {NOTEBOOK_PATH}")
    except Exception as exc:
        print(f"Wrote {NOTEBOOK_PATH} but did not execute it with kernel {CURRENT_KERNEL_NAME}: {exc}")


if __name__ == "__main__":
    main()
