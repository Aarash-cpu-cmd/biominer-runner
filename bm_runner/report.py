"""Deterministic Markdown report for a DE run.

The report is a pure function of the table, the design and the
provenance fields it quotes — timestamps and wall-clock timings stay in
``provenance.json`` so this file hashes identically across reruns of the
same inputs.
"""

from __future__ import annotations

import pandas as pd

TOP_GENES_IN_REPORT = 20


def _sig(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "NA"
    if v != v:  # NaN
        return "NA"
    return f"{v:.{digits}g}"


def render_markdown(*, contrast: dict, design: dict, input_info: dict,
                    output_info: dict, params: dict, effective: dict,
                    gene_filter: dict, classification: dict,
                    deps: dict, table: pd.DataFrame,
                    figures: dict[str, str], runner_version: str,
                    norm_report: dict) -> str:
    numerator = contrast["numerator"]
    denominator = contrast["denominator"]
    lines: list[str] = []
    a = lines.append
    a(f"# Differential expression report — {numerator} vs {denominator}")
    a("")
    a("Generated locally by the frozen BioMiner runner. Every input and "
      "output below is pinned by its SHA-256; the full provenance record "
      "is in `provenance.json`.")
    a("")
    a("## Summary")
    a("")
    a(f"- Contrast: **{numerator}** (tested) vs **{denominator}** "
      f"(reference), two unpaired groups")
    a(f"- Genes tested: {output_info.get('n_rows')} "
      f"(filter: {gene_filter.get('n_genes_before')} -> "
      f"{gene_filter.get('n_genes_after')}, "
      f">= {gene_filter.get('min_counts')} counts in "
      f">= {gene_filter.get('min_samples')} samples)")
    a(f"- Significant at padj < {classification.get('alpha')}: "
      f"**{classification.get('n_significant')}** "
      f"({classification.get('n_up')} up, "
      f"{classification.get('n_down')} down)")
    a(f"- With |log2 fold change| >= {classification.get('lfc_threshold')}: "
      f"{classification.get('n_effect_candidates')} "
      f"({classification.get('n_effect_up')} up, "
      f"{classification.get('n_effect_down')} down)")
    a("")
    a("## Engine")
    a("")
    a(f"- Engine: {params.get('engine')} {params.get('engine_version')} "
      f"(frozen profile `{params.get('profile_id')}` "
      f"v{params.get('profile_version')})")
    a(f"- Runner: {runner_version}; python "
      f"{deps.get('python', 'see provenance')}")
    a(f"- Dispersion fit requested: "
      f"`{effective.get('fit_type_requested')}`; effective: "
      f"`{effective.get('fit_type_effective')}`")
    if effective.get("dispersion_trend_fallback"):
        a("")
        a("> **Dispersion trend fallback**: the parametric dispersion "
          "curve did not fit, so the engine used the alternative "
          f"`{effective.get('fit_type_effective')}` trend. Results are "
          "valid but this deviation from the frozen request is recorded "
          "in provenance and should be mentioned in downstream claims.")
    a("")
    a(f"- Normalization for figures: `{norm_report.get('method')}` "
      f"({norm_report.get('n_reference_genes')} reference genes)")
    a("")
    a("## Design")
    a("")
    sizes = design.get("sizes") or {}
    for level in sorted(sizes):
        a(f"- {level}: {sizes[level]} sample(s)")
    a("")
    a("## Inputs")
    a("")
    a(f"- counts: `{input_info.get('counts_filename')}` — sha256 "
      f"`{input_info.get('counts_sha256')}`")
    a(f"- sheet: `{input_info.get('sheet_filename')}` — sha256 "
      f"`{input_info.get('sheet_sha256')}`")
    transform = input_info.get("input_transform") or {}
    a(f"- input transform: `{transform.get('type', 'unknown')}`")
    a(f"- samples: {', '.join(str(s) for s in input_info.get('sample_ids', []))}")
    a("")
    a("## Figures")
    a("")
    for name in sorted(figures):
        a(f"- `{name}` — sha256 `{figures[name]}`")
    a("")
    a("## Top genes")
    a("")
    shown = table.head(TOP_GENES_IN_REPORT)
    a("| gene | baseMean | log2FoldChange | padj |")
    a("|---|---|---|---|")
    for row in shown.itertuples(index=False):
        a(f"| {row.gene} | {_sig(row.baseMean)} | "
          f"{_sig(row.log2FoldChange)} | {_sig(row.padj)} |")
    if len(table) > len(shown):
        a(f"\n_…{len(table) - len(shown)} further rows in `de_table.tsv`_")
    a("")
    a("## Outputs")
    a("")
    a(f"- `de_table.tsv` — sha256 `{output_info.get('de_table_sha256')}`")
    if output_info.get("report_sha256"):
        a(f"- `report.md` — sha256 `{output_info['report_sha256']}`")
    a("")
    a("## Scope and limitations")
    a("")
    a("- Two-group unpaired design only; no batch or pairing adjustment "
      "in this profile.")
    a("- This is a local computation (`execution_origin=local`, "
      "`validation_status=schema_validated`); it is not a server "
      "attempt, not a grant, and not a scientific qualification.")
    a("- Reproduce with the same runner version and the input digests "
      "above; the parameter digest is "
      f"`{params.get('params_sha256', 'see provenance')}`.")
    a("")
    return "\n".join(lines)
