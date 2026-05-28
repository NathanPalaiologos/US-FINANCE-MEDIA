from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
DELIVERABLE = ANALYSIS / "deliverable"


DIRS = {
    "data": DELIVERABLE / "data",
    "csv": DELIVERABLE / "csv",
    "reports": DELIVERABLE / "reports",
    "memory": DELIVERABLE / "memory",
    "notebooks": DELIVERABLE / "notebooks",
}

FINAL_FILES = {
    "data": [
        "complete_metadata.parquet",
        "complete_metadata.csv.zip",
        "complete_metadata_dedup.parquet",
        "document_nlp_scores.parquet",
    ],
    "reports": [
        "summary_statistics_final.xlsx",
    ],
    "memory": [
        "complete_metadata_data_dictionary.csv",
        "metadata_export_manifest.csv",
        "run_summary.json",
    ],
}

OUTPUT_CSV_FILES = [
    "core_summary.csv",
    "partition_quality_summary.csv",
    "column_completeness_all_rows.csv",
    "duplicate_summary.csv",
    "source_type_counts.csv",
    "language_counts.csv",
    "media_summary.csv",
    "publication_alias_fuzzy_audit.csv",
    "top_subject_terms.csv",
    "top_class_terms.csv",
    "topic_counts.csv",
    "topic_terms.csv",
    "sentiment_by_decade.csv",
    "validation_summary.json",
    "processing_notes.json",
]

ROOT_ANALYSIS_FILES = {
    "reports": ["metadata_quality_report.md"],
    "memory": ["tdm_dataset_partitions_manifest.csv", "tdm_dataset_partition_notes.md"],
}


def ensure_clean_deliverable() -> None:
    if DELIVERABLE.exists():
        shutil.rmtree(DELIVERABLE)
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def move_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    shutil.move(str(source), str(target))


def copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def curate() -> None:
    ensure_clean_deliverable()

    final_dir = ANALYSIS / "final"
    output_dir = ANALYSIS / "outputs"

    for folder, names in FINAL_FILES.items():
        for name in names:
            move_if_exists(final_dir / name, DIRS[folder] / name)

    for name in OUTPUT_CSV_FILES:
        target_folder = "memory" if name.endswith(".json") else "csv"
        move_if_exists(output_dir / name, DIRS[target_folder] / name)

    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        target_figures = DIRS["reports"] / "figures"
        target_figures.mkdir(parents=True, exist_ok=True)
        for figure in figures_dir.glob("*.png"):
            move_if_exists(figure, target_figures / figure.name)

    for folder, names in ROOT_ANALYSIS_FILES.items():
        for name in names:
            move_if_exists(ANALYSIS / name, DIRS[folder] / name)

    pdf_report = ANALYSIS / "metadata_quality_report.pdf"
    if pdf_report.exists():
        pdf_report.unlink()

    stale_notebook = ANALYSIS / "metadata_panel_topic_modeling_workflow.ipynb"
    if stale_notebook.exists():
        stale_notebook.unlink()

    for stale_dir in [final_dir, output_dir]:
        if stale_dir.exists():
            shutil.rmtree(stale_dir)

    artifact_rows = []
    for path in sorted(DELIVERABLE.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv":
            artifact_rows.append(
                {
                    "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )

    manifest_path = DIRS["memory"] / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(artifact_rows, indent=2), encoding="utf-8")

    readme = DELIVERABLE / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Curated Deliverables",
                "",
                "This directory is the submission-facing output tree for the partition-aware TDM metadata analysis.",
                "",
                "- `data/`: full row-level parquet/CSV exports and compact document NLP scores.",
                "- `csv/`: compact audit, summary, topic, and sentiment tables used by the notebook/report.",
                "- `reports/`: professor-facing workbook, markdown report, and figures. No PDF report is kept.",
                "- `memory/`: run summary, validation, data dictionary, partition manifest, and processing notes.",
                "- `notebooks/`: standalone raw-first EDA workbook.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    curate()
