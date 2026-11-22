"""Publication-grade figure contracts: determinism, classification parity,
colorbar/scale explicitness, and well-formed SVG."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from bm_runner import figures as F


def _table() -> pd.DataFrame:
    # up-effect, down-effect, padj-significant but below lfc threshold,
    # and a plain NS gene.
    return pd.DataFrame({
        "gene": ["GENE_UP", "GENE_DOWN", "GENE_SMALL", "GENE_NS"],
        "log2FoldChange": [2.5, -2.5, 0.3, 0.1],
        "padj": [1e-6, 1e-8, 1e-4, 0.9],
    })


def _counts() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    genes = [f"G{i:03d}" for i in range(40)]
    samples = [f"S{i}" for i in range(6)]
    base = rng.poisson(200, size=(6, 40)).astype(float)
    base[:3, :8] *= 4.0  # group effect on the first genes
    counts = pd.DataFrame(base, index=samples, columns=genes)
    conditions = pd.Series(["control"] * 3 + ["treated"] * 3, index=samples)
    return counts, conditions


def test_volcano_effect_rule_matches_report_classification():
    svg = F.volcano_svg(_table(), alpha=0.05, lfc_threshold=1.0,
                        numerator="treated", denominator="control")
    # GENE_SMALL is padj-significant but |lfc| < threshold -> NS gray.
    assert "up in treated (n=1)" in svg
    assert "down in treated (n=1)" in svg
    assert "not significant (n=2)" in svg
    assert F._NS in svg
    # top genes of each direction are labeled, italic
    assert "GENE_UP" in svg and "GENE_DOWN" in svg
    assert 'font-style="italic"' in svg
    # classification rule is stated in the caption
    assert "padj &lt; 0.05" in svg and "|log2FC| ≥ 1" in svg


def test_volcano_zero_padj_floor_note():
    table = _table()
    table.loc[0, "padj"] = 0.0
    svg = F.volcano_svg(table, alpha=0.05, lfc_threshold=1.0,
                        numerator="treated", denominator="control")
    assert "padj=0 plotted at floor" in svg


def test_volcano_all_ns_does_not_crash():
    table = pd.DataFrame({"gene": ["A", "B"], "log2FoldChange": [0.1, -0.2],
                          "padj": [0.9, 0.8]})
    svg = F.volcano_svg(table, alpha=0.05, lfc_threshold=1.0,
                        numerator="t", denominator="c")
    assert "not significant (n=2)" in svg
    ET.fromstring(svg)


def test_heatmap_colorbar_and_palette():
    counts, conditions = _counts()
    log_counts, _ = F.normalized_log(counts)
    table = pd.DataFrame({"gene": list(log_counts.columns[:10]),
                          "padj": np.linspace(1e-5, 0.2, 10)})
    svg = F.heatmap_svg(log_counts, table, conditions, alpha=0.05)
    assert "row z-score" in svg
    assert ">-2<" in svg and ">2<" in svg
    assert "blue = below gene mean" in svg
    # ColorBrewer RdBu endpoints and midpoint, clamped
    assert F._heat_color(-2.0) == "#2166ac"
    assert F._heat_color(0.0) == "#f7f7f7"
    assert F._heat_color(2.0) == "#b2182b"
    assert F._heat_color(-9.0) == "#2166ac"
    assert F._heat_color(9.0) == "#b2182b"
    ET.fromstring(svg)


def test_pca_variance_labels_and_group_counts():
    counts, conditions = _counts()
    log_counts, norm = F.normalized_log(counts)
    svg = F.pca_svg(log_counts, conditions, norm,
                    numerator="treated", denominator="control")
    assert "% variance)" in svg
    assert "control (n=3)" in svg and "treated (n=3)" in svg
    assert norm["method"] in svg
    ET.fromstring(svg)


@pytest.mark.parametrize("name", ["volcano", "pca", "heatmap"])
def test_figures_byte_deterministic(name):
    counts, conditions = _counts()
    log_counts, norm = F.normalized_log(counts)
    table = pd.DataFrame({"gene": list(log_counts.columns[:10]),
                          "log2FoldChange": np.linspace(-2, 2, 10),
                          "padj": np.linspace(1e-5, 0.2, 10)})
    if name == "volcano":
        args = (table,)
        kwargs = dict(alpha=0.05, lfc_threshold=1.0,
                      numerator="treated", denominator="control")
        fn = F.volcano_svg
    elif name == "pca":
        args = (log_counts, conditions, norm)
        kwargs = dict(numerator="treated", denominator="control")
        fn = F.pca_svg
    else:
        args = (log_counts, table, conditions)
        kwargs = dict(alpha=0.05)
        fn = F.heatmap_svg
    assert fn(*args, **kwargs) == fn(*args, **kwargs)


def test_tick_labels_clean():
    assert F._tick_label(0.0) == "0"
    assert F._tick_label(-0.0) == "0"
    assert F._tick_label(2.0) == "2"
    assert F._tick_label(-2.5) == "-2.5"
    assert F._tick_label(0.10) == "0.1"


def test_pca_points_stay_inside_frame():
    """Regression: anchoring the scale to nice ticks (not the data range)
    pushed samples whose score exceeded the last tick outside the frame."""
    rng = np.random.default_rng(42)
    genes = [f"GENE{i:03d}" for i in range(300)]
    samples = [f"CTRL_{i+1}" for i in range(4)] + [f"TRT_{i+1}" for i in range(4)]
    base = rng.poisson(300, size=(8, 300)).astype(float)
    base[4:, :25] *= 6.0
    base[4:, 25:45] *= 0.15
    counts = pd.DataFrame(base, index=samples, columns=genes)
    conditions = pd.Series(["control"] * 4 + ["treated"] * 4, index=samples)
    log_counts, norm = F.normalized_log(counts)
    svg = F.pca_svg(log_counts, conditions, norm,
                    numerator="treated", denominator="control")
    root = ET.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    circles = [c for c in root.findall(".//s:circle", ns)
               if c.get("r") == "5.5"]
    assert len(circles) == 8
    top, bottom = 64, F.FIGURE_HEIGHT - 96
    left, right = 76, F.FIGURE_WIDTH - 30
    for c in circles:
        assert top <= float(c.get("cy")) <= bottom, c.get("cy")
        assert left <= float(c.get("cx")) <= right, c.get("cx")


def test_write_figures_digests_match_bytes(tmp_path):
    counts, conditions = _counts()
    log_counts, norm = F.normalized_log(counts)
    table = pd.DataFrame({"gene": list(log_counts.columns[:10]),
                          "log2FoldChange": np.linspace(-2, 2, 10),
                          "padj": np.linspace(1e-5, 0.2, 10)})
    import hashlib
    digests = F.write_figures(tmp_path, table, log_counts, conditions, norm,
                              alpha=0.05, lfc_threshold=1.0,
                              numerator="treated", denominator="control")
    assert set(digests) == {"figures/volcano.svg", "figures/pca.svg",
                            "figures/heatmap.svg"}
    for name, digest in digests.items():
        data = (tmp_path / name).read_bytes()
        assert digest == hashlib.sha256(data).hexdigest()
        ET.fromstring(data.decode("utf-8"))