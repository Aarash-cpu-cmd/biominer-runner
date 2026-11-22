"""Deterministic pure-SVG figures for the DE report (publication-grade).

The frozen runner may not take a plotting dependency, so the three report
figures are rendered by hand. Every artifact is a pure function of its
inputs: the same table and counts produce the same bytes on every run,
which is what lets provenance carry their digests.

Publication-grade conventions applied here:

* Okabe-Ito colorblind-safe categorical palette; ColorBrewer RdBu
  (blue-low / white / red-high) for the diverging heatmap, with a labeled
  colorbar so the scale is never implicit.
* Journal-style typography: Helvetica/Arial stack, italic gene symbols,
  open (L-shaped) axes, light grid, tick labels without trailing zeros.
* Volcano points are classified by the *effect* rule the report uses
  (padj < alpha AND |log2FC| >= lfc_threshold), so figure and table never
  disagree; the top genes of each direction are labeled with a white halo.
* Every figure carries a subtitle (what was plotted) and a caption
  (classification rule / selection rule), so a reviewer can read the
  figure without the surrounding report.

``normalized_log`` is shared by the PCA and heatmap paths so both figures
project the same transformed matrix.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

FIGURE_WIDTH = 800
FIGURE_HEIGHT = 560
TOP_VARIANCE_GENES = 500
HEATMAP_TOP_GENES = 50

# Okabe-Ito colorblind-safe palette (https://jfly.uni-koeln.de/color/).
_UP = "#D55E00"      # vermilion
_DOWN = "#0072B2"    # blue
_NS = "#999999"      # neutral gray
_GRID = "#E8E8E8"
_AXIS = "#444444"
_INK = "#1B2631"
_MUTED = "#566573"
_FONT = "Helvetica, Arial, sans-serif"

_GROUP_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")

# ColorBrewer RdBu anchors, blue = low, white = 0, red = high (z-score).
_HEAT_STOPS = (
    (-2.0, (33, 102, 172)),
    (-1.0, (103, 169, 207)),
    (0.0, (247, 247, 247)),
    (1.0, (239, 138, 98)),
    (2.0, (178, 24, 43)),
)


def _fmt(value: float) -> str:
    text = f"{value:.2f}"
    return "0" if text in ("-0", "-0.00") else text


def _tick_label(value: float) -> str:
    """Tick text without trailing zeros; never '-0'."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _ticks(vmin: float, vmax: float, n: int = 6) -> list[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return [vmin]
    span = vmax - vmin
    step = 10 ** math.floor(math.log10(span / n))
    for mult in (1, 2, 2.5, 5, 10):
        if step * mult * n >= span:
            step *= mult
            break
    start = math.floor(vmin / step) * step
    ticks = []
    v = start
    while v <= vmax + step * 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _scale(vmin: float, vmax: float, lo: float, hi: float):
    if vmax <= vmin:
        vmax = vmin + 1e-9

    def f(value: float) -> float:
        return lo + (value - vmin) * (hi - lo) / (vmax - vmin)
    return f


def normalized_log(counts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """log2(count / size_factor + 1) with DESeq-style median-of-ratios.

    ``counts`` is samples x genes. Genes present in every sample set the
    size factors; if no gene qualifies, library-size scaling is used and
    the method is recorded so the report can say what happened.
    """
    positive = counts.loc[:, (counts > 0).all(axis=0)]
    if positive.shape[1] > 0:
        geo = np.exp(np.log(positive).mean(axis=0))
        ratios = positive.div(geo, axis=1)
        factors = ratios.median(axis=1)
        factors = factors.replace(0, np.nan).fillna(1.0)
        method = "median_of_ratios"
        n_ref = int(positive.shape[1])
    else:
        totals = counts.sum(axis=1).replace(0, np.nan)
        factors = totals / totals.mean()
        factors = factors.fillna(1.0)
        method = "library_size"
        n_ref = 0
    normed = counts.div(factors, axis=0)
    return np.log2(normed + 1.0), {"method": method,
                                   "n_reference_genes": n_ref}


def _halo_text(x: float, y: float, text: str, *, size: int = 10,
               anchor: str = "start", italic: bool = False,
               fill: str = _INK) -> str:
    """Text with a white halo so labels stay legible over dense marks."""
    style = ' font-style="italic"' if italic else ""
    return (f'<text x="{_fmt(x)}" y="{_fmt(y)}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{fill}"{style} '
            f'stroke="white" stroke-width="3" paint-order="stroke">'
            f'{html.escape(text)}</text>')


def _frame(width: int, height: int, body: list[str], *, title: str,
           subtitle: str = "", caption: str = "") -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="{_FONT}" font-size="13">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="26" text-anchor="middle" '
        f'font-size="18" font-weight="bold" fill="{_INK}">'
        f"{html.escape(title)}</text>",
    ]
    if subtitle:
        parts.append(
            f'<text x="{width / 2:.0f}" y="44" text-anchor="middle" '
            f'font-size="11" fill="{_MUTED}">{html.escape(subtitle)}</text>')
    parts.extend(body)
    if caption:
        parts.append(
            f'<text x="{width / 2:.0f}" y="{height - 10}" '
            f'text-anchor="middle" font-size="10" fill="{_MUTED}">'
            f'{html.escape(caption)}</text>')
    parts.append("</svg>\n")
    return "".join(parts)


def _axes(body: list[str], *, left: int, top: int, right: int, bottom: int,
          x_ticks: list[float], y_ticks: list[float],
          xlabel: str, ylabel: str,
          x_range: tuple | None = None, y_range: tuple | None = None) -> tuple:
    """Open (L-shaped) axes: left + bottom spines, light grid, tick marks.

    The scale is anchored to the *data* range (``x_range``/``y_range``),
    never to the tick range: nice ticks may stop short of the data, and
    anchoring to them would push points outside the plot frame.
    """
    xmin, xmax = x_range if x_range else (min(x_ticks), max(x_ticks))
    ymin, ymax = y_range if y_range else (min(y_ticks), max(y_ticks))
    xs = _scale(xmin, xmax, left, right)
    ys = _scale(ymin, ymax, bottom, top)
    for t in x_ticks:
        x = xs(t)
        body.append(f'<line x1="{_fmt(x)}" y1="{top}" x2="{_fmt(x)}" '
                    f'y2="{bottom}" stroke="{_GRID}"/>')
        body.append(f'<line x1="{_fmt(x)}" y1="{bottom}" x2="{_fmt(x)}" '
                    f'y2="{bottom + 5}" stroke="{_AXIS}"/>')
        body.append(f'<text x="{_fmt(x)}" y="{bottom + 20}" '
                    f'text-anchor="middle" font-size="12" fill="{_INK}">'
                    f'{_tick_label(t)}</text>')
    for t in y_ticks:
        y = ys(t)
        body.append(f'<line x1="{left}" y1="{_fmt(y)}" x2="{right}" '
                    f'y2="{_fmt(y)}" stroke="{_GRID}"/>')
        body.append(f'<line x1="{left - 5}" y1="{_fmt(y)}" x2="{left}" '
                    f'y2="{_fmt(y)}" stroke="{_AXIS}"/>')
        body.append(f'<text x="{left - 9}" y="{_fmt(y + 4)}" '
                    f'text-anchor="end" font-size="12" fill="{_INK}">'
                    f'{_tick_label(t)}</text>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
                f'stroke="{_AXIS}" stroke-width="1.2"/>')
    body.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" '
                f'y2="{bottom}" stroke="{_AXIS}" stroke-width="1.2"/>')
    body.append(f'<text x="{(left + right) / 2:.0f}" y="{bottom + 42}" '
                f'text-anchor="middle" font-size="13" fill="{_INK}">'
                f'{html.escape(xlabel)}</text>')
    body.append(f'<text x="22" y="{(top + bottom) / 2:.0f}" '
                f'text-anchor="middle" font-size="13" fill="{_INK}" '
                f'transform="rotate(-90 22 {(top + bottom) / 2:.0f})">'
                f'{html.escape(ylabel)}</text>')
    return xs, ys


def _legend(body: list[str], entries: list[tuple[str, str]], *,
            x: int, y: int, width: int = 200) -> None:
    """White legend box; entries are (label, color) pairs, drawn as dots."""
    height = 14 + 16 * len(entries)
    body.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
                f'fill="white" fill-opacity="0.92" stroke="{_GRID}"/>')
    for i, (label, color) in enumerate(entries):
        ty = y + 20 + i * 16
        body.append(f'<circle cx="{x + 14}" cy="{ty - 4}" r="4.5" '
                    f'fill="{color}"/>')
        body.append(f'<text x="{x + 26}" y="{ty}" font-size="12" '
                    f'fill="{_INK}">{html.escape(label)}</text>')


def volcano_svg(table: pd.DataFrame, *, alpha: float, lfc_threshold: float,
                numerator: str, denominator: str) -> str:
    """Volcano of log2FoldChange vs -log10(padj), publication-grade.

    Points are colored by the report's *effect* rule (padj < alpha AND
    |lfc| >= lfc_threshold) so figure and classification table agree.
    Rows with non-finite coordinates are dropped; the point order is the
    gene sort order so the bytes do not depend on frame ordering. The top
    genes of each direction (by padj, ties by gene id) are labeled.
    """
    frame = table[["gene", "log2FoldChange", "padj"]].copy()
    frame["lfc"] = pd.to_numeric(frame["log2FoldChange"], errors="coerce")
    frame["padj"] = pd.to_numeric(frame["padj"], errors="coerce")
    frame = frame.dropna(subset=["lfc", "padj"])
    pos = frame.loc[frame["padj"] > 0, "padj"]
    floor = float(pos.min() / 10.0) if len(pos) else 1e-300
    n_zero = int((frame["padj"] == 0).sum())
    frame["neglog"] = -np.log10(frame["padj"].clip(lower=floor))
    effect = (frame["padj"] < alpha) & (frame["lfc"].abs() >= lfc_threshold)
    frame["effect_up"] = effect & (frame["lfc"] > 0)
    frame["effect_down"] = effect & (frame["lfc"] < 0)
    frame = frame.sort_values("gene")
    n_up = int(frame["effect_up"].sum())
    n_down = int(frame["effect_down"].sum())
    n_ns = int(len(frame) - n_up - n_down)

    left, top, right, bottom = 76, 64, FIGURE_WIDTH - 30, FIGURE_HEIGHT - 96
    xmax = float(max(1.0, frame["lfc"].abs().max() * 1.05)) if len(frame) else 1.0
    xmax = math.ceil(xmax * 2) / 2
    ymax = float(max(1.0, frame["neglog"].max() * 1.05)) if len(frame) else 1.0
    ymax = math.ceil(ymax)
    body: list[str] = []
    xs, ys = _axes(body, left=left, top=top, right=right, bottom=bottom,
                   x_ticks=_ticks(-xmax, xmax), y_ticks=_ticks(0, ymax),
                   x_range=(-xmax, xmax), y_range=(0.0, ymax),
                   xlabel="log2 fold change "
                          f"({numerator} vs {denominator})",
                   ylabel="-log10(adjusted p)")
    for thr in (-lfc_threshold, lfc_threshold):
        body.append(f'<line x1="{_fmt(xs(thr))}" y1="{top}" '
                    f'x2="{_fmt(xs(thr))}" y2="{bottom}" '
                    f'stroke="{_AXIS}" stroke-dasharray="4 3" '
                    f'opacity="0.55"/>')
    body.append(f'<line x1="{left}" y1="{_fmt(ys(-math.log10(alpha)))}" '
                f'x2="{right}" y2="{_fmt(ys(-math.log10(alpha)))}" '
                f'stroke="{_AXIS}" stroke-dasharray="4 3" opacity="0.55"/>')
    for row in frame.itertuples(index=False):
        color = (_UP if row.effect_up else
                 _DOWN if row.effect_down else _NS)
        body.append(f'<circle cx="{_fmt(xs(row.lfc))}" '
                    f'cy="{_fmt(ys(row.neglog))}" r="2.4" fill="{color}" '
                    f'fill-opacity="0.75"/>')
    # Label the top genes of each direction; deterministic order and a
    # simple vertical nudge keep labels from stacking on each other.
    labeled: list[tuple[float, float, str]] = []
    for direction in ("effect_up", "effect_down"):
        hits = frame.loc[frame[direction]].sort_values(["padj", "gene"])
        for row in hits.head(5).itertuples(index=False):
            labeled.append((float(xs(row.lfc)), float(ys(row.neglog)),
                            str(row.gene)))
    placed: list[tuple[float, float]] = []
    for px, py, gene in sorted(labeled, key=lambda t: (t[1], t[0], t[2])):
        ly = py - 7
        while any(abs(ly - qy) < 11 and abs(px - qx) < 90
                  for qx, qy in placed):
            ly -= 12
        if ly < top + 10:  # never label outside the plot frame
            continue
        placed.append((px, ly))
        anchor = "start" if px < (left + right) / 2 else "end"
        dx = 7 if anchor == "start" else -7
        body.append(_halo_text(px + dx, ly, gene, size=10, anchor=anchor,
                               italic=True))
    _legend(body, [
        (f"up in {numerator} (n={n_up})", _UP),
        (f"down in {numerator} (n={n_down})", _DOWN),
        (f"not significant (n={n_ns})", _NS),
    ], x=right - 210, y=top + 6, width=204)
    caption = (f"{len(frame)} genes tested · colored: padj < {alpha:g} and "
               f"|log2FC| ≥ {lfc_threshold:g} · dashed lines: thresholds")
    if n_zero:
        caption += f" · {n_zero} padj=0 plotted at floor"
    return _frame(FIGURE_WIDTH, FIGURE_HEIGHT, body,
                  title="Volcano — differential expression",
                  subtitle=f"{numerator} vs {denominator} · "
                           "adjusted p: Benjamini-Hochberg",
                  caption=caption)


def pca_svg(log_counts: pd.DataFrame, conditions: pd.Series,
            norm_report: dict, *, numerator: str, denominator: str) -> str:
    """PC1/PC2 scatter of samples over the top-variance genes.

    Gene selection sorts by descending variance then gene id, the SVD is
    deterministic for a fixed input order, and sign flips are resolved by
    fixing the sign of the largest-magnitude loading.
    """
    variance = log_counts.var(axis=0)
    order = sorted(variance.index, key=lambda g: (-variance[g], g))
    genes_sel = order[:TOP_VARIANCE_GENES]
    matrix = log_counts[genes_sel].to_numpy(dtype="float64")
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    scores = _u[:, :2] * s[:2]
    for comp in range(2):
        col = scores[:, comp]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0:
            scores[:, comp] = -col
    total = float((s ** 2).sum())
    var = (s ** 2) / total if total > 0 else np.zeros_like(s)
    levels = sorted(conditions.unique())
    wanted = [denominator, numerator]
    levels = [lv for lv in wanted if lv in levels] + \
        [lv for lv in levels if lv not in wanted]

    left, top, right, bottom = 76, 64, FIGURE_WIDTH - 30, FIGURE_HEIGHT - 96
    xs_all, ys_all = scores[:, 0], scores[:, 1]
    pad = 0.08
    xr = (float(xs_all.min()), float(xs_all.max()))
    yr = (float(ys_all.min()), float(ys_all.max()))
    xr = (xr[0] - (xr[1] - xr[0] or 1) * pad, xr[1] + (xr[1] - xr[0] or 1) * pad)
    yr = (yr[0] - (yr[1] - yr[0] or 1) * pad, yr[1] + (yr[1] - yr[0] or 1) * pad)
    body: list[str] = []
    xs, ys = _axes(body, left=left, top=top, right=right, bottom=bottom,
                   x_ticks=_ticks(*xr), y_ticks=_ticks(*yr),
                   x_range=xr, y_range=yr,
                   xlabel=f"PC1 ({var[0] * 100:.1f}% variance)",
                   ylabel=f"PC2 ({var[1] * 100:.1f}% variance)")
    color_of = {lv: _GROUP_COLORS[i % len(_GROUP_COLORS)]
                for i, lv in enumerate(levels)}
    n_of = {lv: int((conditions == lv).sum()) for lv in levels}
    for sample in sorted(log_counts.index):
        i = list(log_counts.index).index(sample)
        cond = str(conditions[sample])
        body.append(f'<circle cx="{_fmt(xs(scores[i, 0]))}" '
                    f'cy="{_fmt(ys(scores[i, 1]))}" r="5.5" '
                    f'fill="{color_of[cond]}" fill-opacity="0.9" '
                    f'stroke="white" stroke-width="1"/>')
    for sample in sorted(log_counts.index):
        i = list(log_counts.index).index(sample)
        body.append(_halo_text(xs(scores[i, 0]) + 9, ys(scores[i, 1]) + 4,
                               str(sample), size=10))
    _legend(body, [(f"{lv} (n={n_of[lv]})", color_of[lv]) for lv in levels],
            x=right - 210, y=top + 6, width=204)
    return _frame(FIGURE_WIDTH, FIGURE_HEIGHT, body,
                  title="PCA — samples",
                  subtitle=f"{len(log_counts)} samples · top {len(genes_sel)} "
                           f"variable genes · {norm_report['method']} "
                           "normalization",
                  caption="log2(normalized counts + 1), gene-centered; "
                          "sign fixed by largest-magnitude loading")


def _heat_color(z: float) -> str:
    """ColorBrewer RdBu (blue low / white 0 / red high), clamped to ±2."""
    z = max(-2.0, min(2.0, z))
    for (z0, c0), (z1, c1) in zip(_HEAT_STOPS, _HEAT_STOPS[1:]):
        if z <= z1:
            t = (z - z0) / (z1 - z0)
            r, g, b = (int(round(a + (b2 - a) * t))
                       for a, b2 in zip(c0, c1))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#b2182b"


def heatmap_svg(log_counts: pd.DataFrame, table: pd.DataFrame,
                conditions: pd.Series, *, alpha: float) -> str:
    """Z-scored heatmap of the top genes by adjusted p, with colorbar.

    Genes are the up-to-50 lowest padj rows (ties broken by gene id);
    samples keep sheet order and are grouped by condition. The diverging
    scale is explicit: a labeled colorbar spans -2..+2 row z-score.
    """
    ranked = table[["gene", "padj"]].copy()
    ranked["padj"] = pd.to_numeric(ranked["padj"], errors="coerce")
    ranked = ranked.dropna().sort_values(["padj", "gene"])
    genes = [g for g in ranked["gene"].head(HEATMAP_TOP_GENES)
             if g in log_counts.columns]
    if not genes:
        genes = list(log_counts.columns[:HEATMAP_TOP_GENES])
    matrix = log_counts[genes].T  # genes x samples
    z = matrix.sub(matrix.mean(axis=1), axis=0)
    scale = matrix.std(axis=1, ddof=0).replace(0, np.nan)
    z = z.div(scale, axis=0).fillna(0.0).to_numpy(dtype="float64")

    samples = list(log_counts.index)
    cond_of = {s: str(conditions[s]) for s in samples}
    cell_w = 20
    cell_h = 13
    left = 200
    top = 76
    width = max(FIGURE_WIDTH, left + cell_w * len(samples) + 60)
    height = top + cell_h * len(genes) + 170
    body: list[str] = []
    for gi, _gene in enumerate(genes):
        y = top + gi * cell_h
        body.append(f'<text x="{left - 8}" y="{y + cell_h - 3}" '
                    f'text-anchor="end" font-size="10" font-style="italic" '
                    f'fill="{_INK}">{html.escape(str(genes[gi]))}</text>')
        for si, _sample in enumerate(samples):
            x = left + si * cell_w
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w}" '
                        f'height="{cell_h}" fill="{_heat_color(z[gi, si])}"/>')
    body.append(f'<rect x="{left}" y="{top}" '
                f'width="{cell_w * len(samples)}" '
                f'height="{cell_h * len(genes)}" fill="none" '
                f'stroke="{_AXIS}" stroke-width="0.8"/>')
    cond_row = top - 4
    for si, sample in enumerate(samples):
        x = left + si * cell_w
        cond = cond_of[sample]
        idx = sorted(set(cond_of.values())).index(cond)
        color = _GROUP_COLORS[idx % len(_GROUP_COLORS)]
        body.append(f'<rect x="{x}" y="{cond_row - 10}" width="{cell_w}" '
                    f'height="8" fill="{color}"/>')
        body.append(f'<text x="{x + cell_w / 2:.0f}" y="{cond_row + 18}" '
                    f'font-size="10" fill="{_INK}" text-anchor="end" '
                    f'transform="rotate(-45 {x + cell_w / 2:.0f} '
                    f'{cond_row + 18})">{html.escape(sample)}</text>')
    # Colorbar: 64 deterministic swatches, ticks at -2..2, explicit label.
    bar_y = top + cell_h * len(genes) + 26
    bar_w, bar_h, n_swatch = 220, 10, 64
    for i in range(n_swatch):
        zv = -2.0 + 4.0 * (i + 0.5) / n_swatch
        body.append(f'<rect x="{left + bar_w * i / n_swatch:.2f}" '
                    f'y="{bar_y}" width="{bar_w / n_swatch + 0.5:.2f}" '
                    f'height="{bar_h}" fill="{_heat_color(zv)}"/>')
    body.append(f'<rect x="{left}" y="{bar_y}" width="{bar_w}" '
                f'height="{bar_h}" fill="none" stroke="{_AXIS}" '
                f'stroke-width="0.8"/>')
    for t in (-2, -1, 0, 1, 2):
        tx = left + bar_w * (t + 2) / 4
        body.append(f'<line x1="{tx:.2f}" y1="{bar_y + bar_h}" '
                    f'x2="{tx:.2f}" y2="{bar_y + bar_h + 4}" '
                    f'stroke="{_AXIS}"/>')
        body.append(f'<text x="{tx:.2f}" y="{bar_y + bar_h + 16}" '
                    f'text-anchor="middle" font-size="11" fill="{_INK}">'
                    f'{t}</text>')
    body.append(f'<text x="{left + bar_w + 12}" y="{bar_y + bar_h - 1}" '
                f'font-size="11" fill="{_INK}">row z-score</text>')
    legend_y = bar_y + 40
    for i, lv in enumerate(sorted(set(cond_of.values()))):
        x = left + i * 170
        color = _GROUP_COLORS[i % len(_GROUP_COLORS)]
        body.append(f'<rect x="{x}" y="{legend_y}" width="12" height="12" '
                    f'fill="{color}"/>')
        body.append(f'<text x="{x + 18}" y="{legend_y + 11}" font-size="12" '
                    f'fill="{_INK}">{html.escape(lv)}</text>')
    return _frame(width, height, body,
                  title="Heatmap — top differential genes",
                  subtitle=f"top {len(genes)} genes by adjusted p "
                           f"(padj < {alpha:g} first) · row z-score of "
                           "log2 normalized counts",
                  caption="blue = below gene mean · red = above · "
                          "samples grouped by condition (bar above columns)")


def write_figures(out_dir: Path, table: pd.DataFrame,
                  log_counts: pd.DataFrame, conditions: pd.Series,
                  norm_report: dict, *, alpha: float, lfc_threshold: float,
                  numerator: str, denominator: str) -> dict[str, str]:
    """Render the three report figures and return name -> sha256."""
    import hashlib

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "volcano.svg": volcano_svg(
            table, alpha=alpha, lfc_threshold=lfc_threshold,
            numerator=numerator, denominator=denominator),
        "pca.svg": pca_svg(
            log_counts, conditions, norm_report,
            numerator=numerator, denominator=denominator),
        "heatmap.svg": heatmap_svg(log_counts, table, conditions,
                                   alpha=alpha),
    }
    digests: dict[str, str] = {}
    for name in sorted(payloads):
        data = payloads[name].encode("utf-8")
        (fig_dir / name).write_bytes(data)
        digests[f"figures/{name}"] = hashlib.sha256(data).hexdigest()
    return digests