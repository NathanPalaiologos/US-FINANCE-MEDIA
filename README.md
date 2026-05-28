# TDM Studio Partition-Aware Metadata EDA

This project analyzes ten TDM Studio metadata partitions for U.S. business, finance, economics, and commercial media. The current pipeline treats `raw/*/master.csv` as the canonical input because it contains the full 31-column metadata schema.

## Scope

- Unit of observation: one TDM metadata document row.
- Primary key: `GOID`.
- Primary tidy sample: English records from the U.S. query corpus.
- Publisher geography is a diagnostic, not a sample filter.
- The local export contains metadata only, not full article body text.

## Commands

Create and activate the recommended Windows virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_eda.txt
```

Register the notebook kernel used by the raw-first EDA workbook:

```powershell
python -m ipykernel install --user --name tdm-studio-eda --display-name "TDM Studio (.venv)"
```

Install dependencies into an existing environment:

```powershell
python -m pip install -r requirements_eda.txt
```

The notebook builder writes `analysis/deliverable/notebooks/tdm_corpus_raw_eda_workbook.ipynb` with the `tdm-studio-eda` kernel by default. If you need a different kernel identity, set `TDM_KERNEL_NAME` and `TDM_KERNEL_DISPLAY` before running `scripts\build_metadata_eda_notebook.py`.

Run focused tests:

```powershell
python -m unittest discover -s tests
```

Run the full pipeline:

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts\partition_pipeline.py
```

Create the raw-first EDA workbook:

```powershell
python scripts\build_metadata_eda_notebook.py
```

Curate the output tree and remove intermediate clutter:

```powershell
python scripts\curate_deliverables.py
```

## Curated Output

The useful deliverables live under `analysis/deliverable/`:

- `data/`: full row-level parquet/CSV exports and compact document NLP scores.
- `csv/`: compact summary and audit tables.
- `reports/`: summary workbook, markdown report, and figures. No PDF report is kept.
- `memory/`: run summary, validation, data dictionary, manifest, and processing notes.
- `notebooks/`: standalone raw-first EDA workbook.

## Important Limits

- Topic and sentiment analysis are metadata/title based.
- The raw-first notebook now builds grouped word clouds from the full English corpus with usable metadata text, while NMF and LDA topic models still fit on a bounded metadata sample for runtime reasons.
- The main NLP text is `Title + Subject Terms + Class Terms`; title-only sensitivity scores are stored in the full tidy export.
- Loughran-McDonald sentiment uses the official SRAF/Notre Dame dictionary source.
- Fuzzy publication alias matches are audit suggestions only.
- The saved manifest records 955,690 rows for the 2013-2026 trade-journal partition, while the local export contains 955,594 rows.
