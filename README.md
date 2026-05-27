# TDM Studio Metadata EDA Pipeline

This project analyzes a ProQuest/TDM Studio metadata export for finance, economics, and business media.

## What the Pipeline Does

1. Reads the nested TDM Studio ZIP export from `raw/`.
2. Audits the available CSV columns and candidate XML paths for title, genre/source type, full-text availability, journal/publication, date range, coverage proxies, authors, and topic metadata.
3. Uses `extended.csv` as the main metadata table when XML files are not present locally.
4. Normalizes publication titles and publisher locations.
5. Retains North-American publishers for the main media scope.
6. Builds a submission-ready cleaned dataset containing English documents only.
7. Builds a document-level metadata panel for exploration and topic-modeling sensitivity checks.
8. Marks conservative duplicate candidates using normalized title + date + normalized publication.
9. Produces summary tables, issue/topic coverage tables, visualizations, a notebook, and reports.
10. Runs title-based topic modeling on the cleaned English North-American metadata only.

## Important Limits of the Source Metadata

- The export does not contain full article text.
- The current local CSV export does not contain a full-text-online availability field. The upgraded XML manifest records candidate XML tags to scan if XML exports are added.
- The export does not contain serial issue or volume fields.
- `Subject Terms` and `Class Terms` support topical issue coverage analysis, but they cannot prove whether every issue of a serial publication is present.

## Main Commands

Install dependencies:

```powershell
python -m pip install -r requirements_eda.txt
python -m ipykernel install --user --name tdm-studio-python --display-name "TDM Studio Python"
```

Run the full heavy pipeline:

```powershell
$env:PYTHONIOENCODING='utf-8'
@'
from pathlib import Path
import sys
sys.path.insert(0, str(Path('scripts').resolve()))
from eda_helpers import profile_metadata, build_cleaned_metadata_outputs, run_topic_modeling, make_plots, write_final_summary_workbook, write_report
from eda_helpers import scan_xml_field_inventory, build_metadata_panel_outputs

scan_xml_field_inventory()
profile_metadata()
build_cleaned_metadata_outputs()
build_metadata_panel_outputs()
run_topic_modeling()
make_plots()
write_final_summary_workbook()
write_report()
'@ | python -
```

Refresh the reading notebook after outputs already exist:

```powershell
python scripts\build_metadata_eda_notebook.py
```

## Final Deliverables

Use `analysis/final/` for submission-ready files:

- `summary_statistics_final.xlsx`
- `cleaned_metadata_english_north_america.parquet`
- `cleaned_metadata_english_north_america.csv.zip`
- `cleaned_metadata_english_north_america_dedup.parquet`
- `cleaned_metadata_english_north_america_dedup.csv.zip`
- `cleaned_metadata_data_dictionary.csv`
- `metadata_field_audit.csv`
- `metadata_panel.parquet`
- `metadata_panel.csv.zip`
- `metadata_panel_data_dictionary.csv`
- `metadata_export_manifest.csv`
- `mentor_email_draft.md`

Reports and notebook:

- `analysis/metadata_panel_topic_modeling_workflow.ipynb`
- `analysis/metadata_quality_report.md`
- `analysis/metadata_quality_report.pdf`

## Current Cleaned Dataset Scope

- Language: English only.
- Publisher region: United States or Canada.
- Publication/location fields: normalized and disambiguated.
- Deduplication: conservative title/date/media key is included; a deduplicated export is provided separately.

## Notes for Maintenance

- `analysis/outputs/` contains detailed intermediate outputs.
- `analysis/final/` contains the files intended for sharing/submission.
- If `summary_statistics.xlsx` is open in Excel, Windows may lock it. The pipeline records this as a processing note rather than hiding the issue.
