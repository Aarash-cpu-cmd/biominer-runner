"""Deterministic pure-SVG figures for the DE report.

The frozen runner may not take a plotting dependency, so the three report
figures are rendered by hand. Every artifact is a pure function of its
inputs: the same table and counts produce the same bytes on every run,
which is what lets provenance carry their digests.

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
_UP = "#c0392b"
_DOWN = "#2471a3"
_NS = "#b3b6b7"
_GRID = "#e5e7e9"
_INK = "#1b2631"

_GROUP_COLORS = ("#1f77b4", "#d35400", "#117864", "#6c3483")


def _fmt(value: float) -> str:
    text = f"{value:.2f}"
    return "0" if text in ("-0", "-0.00") else text


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


def _frame(width: int, height: int, body: list[str], *, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="sans-serif" font-size="12">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<text x="{width / 2:.0f}" y="24" text-anchor="middle" '
        f'font-size="16" font-weight="bold" fill="{_INK}">'
        f"{html.escape(title)}</text>"
        + "".join(body) + "</svg>\n")


def _axes(body: list[str], *, left: int, top: int, right: int, bottom: int,
          x_ticks: list[float], y_ticks: list[float],
          xlabel: str, ylabel: str) -> tuple:
    xmin, xmax = min(x_ticks), max(x_ticks)
    ymin, ymax = min(y_ticks), max(y_ticks)
    xs = _scale(xmin, xmax, left, right)
    ys = _scale(ymin, ymax, bottom, top)
    for t in x_ticks:
        x = xs(t)
        body.append(f'<line x1="{_fmt(x)}" y1="{top}" x2="{_fmt(x)}" '
                    f'y2="{bottom}" stroke="{_GRID}"/>')
        body.append(f'<text x="{_fmt(x)}" y="{bottom + 16}" '
                    f'text-anchor="middle" fill="{_INK}">{_fmt(t)}</text>')
    for t in y_ticks:
        y = ys(t)
        body.append(f'<line x1="{left}" y1="{_fmt(y)}" x2="{right}" '
                    f'y2="{_fmt(y)}" stroke="{_GRID}"/>')
        body.append(f'<text x="{left - 8}" y="{_fmt(y + 4)}" '
                    f'text-anchor="end" fill="{_INK}">{_fmt(t)}</text>')
    body.append(f'<rect x="{left}" y="{top}" width="{right - left}" '
                f'height="{bottom - top}" fill="none" stroke="{_INK}"/>')
    body.append(f'<text x="{(left + right) / 2:.0f}" y="{bottom + 40}" '
                f'text-anchor="middle" fill="{_INK}">'
                f'{html.escape(xlabel)}</text>')
    body.append(f'<text x="20" y="{(top + bottom) / 2:.0f}" '
                f'text-anchor="middle" fill="{_INK}" transform="rotate(-90 '
                f'20 {(top + bottom) / 2:.0f})">{html.escape(ylabel)}</text>')
    return xs, ys


def volcano_svg(table: pd.DataFrame, *, alpha: float, lfc_threshold: float,
                numerator: str, denominator: str) -> str:
    """Volcano of log2FoldChange vs -log10(padj).

    Rows with non-finite coordinates are dropped; the point order is the
    gene sort order so the bytes do not depend on frame ordering.
    """
    frame = table[["gene", "log2FoldChange", "padj"]].copy()
    frame["lfc"] = pd.to_numeric(frame["log2FoldChange"], errors="coerce")
    frame["padj"] = pd.to_numeric(frame["padj"], errors="coerce")
    frame = frame.dropna(subset=["lfc", "padj"])
    pos = frame.loc[frame["padj"] > 0, "padj"]
    floor = float(pos.min() / 10.0) if len(pos) else 1e-300
    frame["neglog"] = -np.log10(frame["padj"].clip(lower=floor))
    significant = frame["padj"] < alpha
    frame = frame.sort_values("gene")

    left, top, right, bottom = 70, 50, FIGURE_WIDTH - 30, FIGURE_HEIGHT - 70
    xmax = float(max(1.0, frame["lfc"].abs().max() * 1.05)) if len(frame) else 1.0
    xmax = math.ceil(xmax * 2) / 2
    ymax = float(max(1.0, frame["neglog"].max() * 1.05)) if len(frame) else 1.0
    ymax = math.ceil(ymax)
    body: list[str] = []
    xs, ys = _axes(body, left=left, top=top, right=right, bottom=bottom,
                   x_ticks=_ticks(-xmax, xmax), y_ticks=_ticks(0, ymax),
                   xlabel="log2 fold change "
                          f"({numerator} vs {denominator})",
                   ylabel="-log10(adjusted p)")
    body.append(f'<line x1="{_fmt(xs(-lfc_threshold))}" y1="{top}" '
                f'x2="{_fmt(xs(-lfc_threshold))}" y2="{bottom}" '
                f'stroke="{_INK}" stroke-dasharray="4 3" opacity="0.5"/>')
    body.append(f'<line x1="{_fmt(xs(lfc_threshold))}" y1="{top}" '
                f'x2="{_fmt(xs(lfc_threshold))}" y2="{bottom}" '
                f'stroke="{_INK}" stroke-dasharray="4 3" opacity="0.5"/>')
    body.append(f'<line x1="{left}" y1="{_fmt(ys(-math.log10(alpha)))}" '
                f'x2="{right}" y2="{_fmt(ys(-math.log10(alpha)))}" '
                f'stroke="{_INK}" stroke-dasharray="4 3" opacity="0.5"/>')
    for row in frame.itertuples(index=False):
        sig = bool(row.padj < alpha)
        color = (_UP if row.lfc > 0 else _DOWN) if sig else _NS
        body.append(f'<circle cx="{_fmt(xs(row.lfc))}" '
                    f'cy="{_fmt(ys(row.neglog))}" r="2.4" fill="{color}" '
                    f'fill-opacity="0.75"/>')
    body.append(f'<rect x="{right - 190}" y="{top + 6}" width="180" '
                f'height="52" fill="white" stroke="{_GRID}"/>')
    for i, (label, color) in enumerate((
            (f"up ({numerator})", _UP), (f"down ({numerator})", _DOWN),
            ("not significant", _NS))):
        y = top + 20 + i * 14
        body.append(f'<circle cx="{right - 176}" cy="{y - 4}" r="4" '
                    f'fill="{color}"/>')
        body.append(f'<text x="{right - 166}" y="{y}" fill="{_INK}">'
                    f'{html.escape(label)}</text>')
    return _frame(FIGURE_WIDTH, FIGURE_HEIGHT, body,
                  title="Volcano — differential expression")


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

    left, top, right, bottom = 70, 50, FIGURE_WIDTH - 30, FIGURE_HEIGHT - 70
    xs_all, ys_all = scores[:, 0], scores[:, 1]
    pad = 0.08
    xr = (float(xs_all.min()), float(xs_all.max()))
    yr = (float(ys_all.min()), float(ys_all.max()))
    xr = (xr[0] - (xr[1] - xr[0] or 1) * pad, xr[1] + (xr[1] - xr[0] or 1) * pad)
    yr = (yr[0] - (yr[1] - yr[0] or 1) * pad, yr[1] + (yr[1] - yr[0] or 1) * pad)
    body: list[str] = []
    xs, ys = _axes(body, left=left, top=top, right=right, bottom=bottom,
                   x_ticks=_ticks(*xr), y_ticks=_ticks(*yr),
                   xlabel=f"PC1 ({var[0] * 100:.1f}% variance)",
                   ylabel=f"PC2 ({var[1] * 100:.1f}% variance)")
    color_of = {lv: _GROUP_COLORS[i % len(_GROUP_COLORS)]
                for i, lv in enumerate(levels)}
    for sample in sorted(log_counts.index):
        i = list(log_counts.index).index(sample)
        cond = str(conditions[sample])
        body.append(f'<circle cx="{_fmt(xs(scores[i, 0]))}" '
                    f'cy="{_fmt(ys(scores[i, 1]))}" r="5" '
                    f'fill="{color_of[cond]}" fill-opacity="0.85"/>')
        body.append(f'<text x="{_fmt(xs(scores[i, 0]) + 8)}" '
                    f'y="{_fmt(ys(scores[i, 1]) + 4)}" font-size="10" '
                    f'fill="{_INK}">{html.escape(sample)}</text>')
    body.append(f'<rect x="{right - 190}" y="{top + 6}" width="180" '
                f'height="{18 + 14 * len(levels)}" fill="white" '
                f'stroke="{_GRID}"/>')
    for i, lv in enumerate(levels):
        y = top + 20 + i * 14
        body.append(f'<circle cx="{right - 176}" cy="{y - 4}" r="4" '
                    f'fill="{color_of[lv]}"/>')
        body.append(f'<text x="{right - 166}" y="{y}" fill="{_INK}">'
                    f'{html.escape(lv)}</text>')
    body.append(f'<text x="{left}" y="{top - 8}" font-size="10" '
                f'fill="{_INK}">top {len(genes_sel)} variable genes, '
                f'{html.escape(norm_report["method"])} normalization</text>')
    return _frame(FIGURE_WIDTH, FIGURE_HEIGHT, body,
                  title="PCA — samples")


def _heat_color(z: float) -> str:
    z = max(-2.0, min(2.0, z)) / 2.0
    if z >= 0:
        r, g, b = 255, int(255 * (1 - z)), int(255 * (1 - z))
    else:
        r, g, b = int(255 * (1 + z)), int(255 * (1 + z)), 255
    return f"#{r:02x}{g:02x}{b:02x}"


def heatmap_svg(log_counts: pd.DataFrame, table: pd.DataFrame,
                conditions: pd.Series, *, alpha: float) -> str:
    """Z-scored heatmap of the top genes by adjusted p.

    Genes are the up-to-50 lowest padj rows (ties broken by gene id);
    samples keep sheet order and are grouped by condition.
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
    cell_w = 18
    cell_h = 12
    left = 190
    top = 60
    width = left + cell_w * len(samples) + 40
    height = top + cell_h * len(genes) + 110
    body: list[str] = []
    body.append(f'<text x="{left}" y="{top - 12}" font-size="10" '
                f'fill="{_INK}">top {len(genes)} genes by adjusted p '
                f'(padj &lt; {_fmt(alpha)} first), row z-score of '
                f'log2 normalized counts</text>')
    for gi, _gene in enumerate(genes):
        y = top + gi * cell_h
        body.append(f'<text x="{left - 6}" y="{y + cell_h - 3}" '
                    f'text-anchor="end" font-size="9" fill="{_INK}">'
                    f'{html.escape(str(genes[gi]))}</text>')
        for si, _sample in enumerate(samples):
            x = left + si * cell_w
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w}" '
                        f'height="{cell_h}" fill="{_heat_color(z[gi, si])}"/>')
    cond_row = top - 4
    for si, sample in enumerate(samples):
        x = left + si * cell_w
        cond = cond_of[sample]
        idx = sorted(set(cond_of.values())).index(cond)
        color = _GROUP_COLORS[idx % len(_GROUP_COLORS)]
        body.append(f'<rect x="{x}" y="{cond_row - 10}" width="{cell_w}" '
                    f'height="8" fill="{color}"/>')
        body.append(f'<text x="{x + cell_w / 2:.0f}" y="{cond_row + 18}" '
                    f'font-size="9" fill="{_INK}" text-anchor="end" '
                    f'transform="rotate(-45 {x + cell_w / 2:.0f} '
                    f'{cond_row + 18})">{html.escape(sample)}</text>')
    legend_y = top + cell_h * len(genes) + 20
    for i, lv in enumerate(sorted(set(cond_of.values()))):
        x = left + i * 150
        color = _GROUP_COLORS[i % len(_GROUP_COLORS)]
        body.append(f'<rect x="{x}" y="{legend_y}" width="12" height="12" '
                    f'fill="{color}"/>')
        body.append(f'<text x="{x + 18}" y="{legend_y + 10}" '
                    f'fill="{_INK}">{html.escape(lv)}</text>')
    return _frame(width, height, body,
                  title="Heatmap — top differential genes")


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
