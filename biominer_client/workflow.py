"""End-to-end local workflow: download from BioMiner, analyse locally, save back.

The pipeline is deliberately split into small, testable steps so an agent
(or a human) can run them one at a time::

    export_dataset   -> raw integer counts + sample sheet (byte tickets)
    build_contrast   -> declare a two-group contrast, subset counts to match
    run_de           -> bm_runner differential table + provenance (local)
    fetch_gene_sets  -> frozen GMT snapshots + manifests (byte tickets)
    run_enrichment   -> bm_runner GSEA / ORA (local)
    save_result      -> upload provenance + de_table, register the result

``local_analysis`` chains all of them. Nothing here computes science on the
server: the server stores bytes and registers a receipt; ``bm_runner`` does
the statistics on the user's machine.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .client import BioMinerClient, BioMinerError

_SYMBOL_COLUMNS = ("gene_symbol", "gene_name", "symbol")


def _stable_key(prefix: str, *parts: Any) -> str:
    """Deterministic idempotency key derived from content.

    Re-running the same analysis (same study intent, same result bytes)
    replays the existing study/result instead of creating a duplicate. A
    random UUID per call would silently accumulate duplicates on retry.
    """
    digest = hashlib.sha256(
        "\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:32]}"


def _file_sha(path: Any) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# Provenance fields that legitimately vary between two runs of the SAME
# analysis (wall clock, timing/RSS sampling). They must not take part in
# the idempotency key, otherwise every re-run registers a duplicate result
# instead of replaying (bug #12: re-run produced replayed=false even though
# the de_table was byte-identical).
_PROVENANCE_VOLATILE_KEYS = ("created_at", "metrics")


def _provenance_semantic_sha(path: Any) -> str:
    """SHA-256 of the provenance with volatile fields removed.

    Everything else (params, engine, input/output digests, dependencies,
    design) is deterministic for identical inputs, so two runs of the same
    contrast hash equal and the server replays the stored result.
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in _PROVENANCE_VOLATILE_KEYS:
        doc.pop(key, None)
    canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_characteristics(text: str) -> dict[str, str]:
    """Parse a ``key: value,key: value`` characteristics string."""
    out: dict[str, str] = {}
    for part in str(text or "").split(","):
        key, sep, value = part.partition(":")
        if sep:
            out[key.strip().lower()] = value.strip()
    return out


def summarize_conditions(export: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the export sheet's condition fields with sample counts.

    Zero-knowledge contrast discovery: lists every ``characteristics`` key
    (plus a populated ``condition`` column) and its value→count distribution,
    so a caller can pick ``--condition-key`` / ``--numerator`` /
    ``--denominator`` from real labels instead of guessing ``treatment``.
    """
    sheet = pd.read_csv(export["sheet_path"], sep="\t", dtype=str,
                        keep_default_na=False)
    fields: dict[str, dict[str, int]] = {}

    def _bump(key: str, value: str) -> None:
        value = value.strip()
        if not value:
            return
        bucket = fields.setdefault(key, {})
        bucket[value] = bucket.get(value, 0) + 1

    for _, row in sheet.iterrows():
        _bump("condition", str(row.get("condition", "")))
        for k, v in _parse_characteristics(row.get("characteristics", "")).items():
            _bump(k, v)
    ordered = {
        k: dict(sorted(v.items(), key=lambda kv: (-kv[1], kv[0])))
        for k, v in sorted(fields.items())
    }
    return {"n_samples": int(len(sheet)), "condition_fields": ordered}


def export_dataset(client: BioMinerClient, gse_id: str,
                   out_dir: Path) -> dict[str, Any]:
    """Download the ARCHS4 raw-count package for one series.

    Returns the export metadata plus local paths to ``counts.tsv.gz`` and
    ``sample_sheet.tsv``. Every file's SHA-256 is verified on download.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = client.call_tool("expression_export_prepare", {"gse_id": gse_id})
    export = data["export"]
    paths: dict[str, Path] = {}
    for ticket in data.get("downloads", []):
        dest = client.download_ticket(ticket, out_dir / ticket["filename"])
        paths[ticket["filename"]] = dest
    return {"export": export,
            "counts_path": paths.get("counts.tsv.gz"),
            "sheet_path": paths.get("sample_sheet.tsv"),
            "dir": out_dir}


def build_contrast(export: dict[str, Any], numerator: str, denominator: str,
                   out_dir: Path, *, condition_key: str = "treatment",
                   condition_map: dict[str, str] | None = None) -> dict[str, Any]:
    """Declare a two-group contrast and write runner-ready inputs.

    The exported sample sheet carries no condition label. This step derives
    one — from an explicit ``condition_map`` (sample_id -> label), from the
    sheet's ``condition`` column if already populated, or from a
    ``characteristics`` field named by ``condition_key`` (default
    ``treatment``) — then selects the samples belonging to the two groups
    and subsets the counts so columns match the sheet exactly and in order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = pd.read_csv(export["sheet_path"], sep="\t", dtype=str,
                        keep_default_na=False)
    labels: dict[str, str] = {}
    for _, row in sheet.iterrows():
        sid = str(row["sample_id"]).strip()
        if condition_map and sid in condition_map:
            label = condition_map[sid]
        elif str(row.get("condition", "")).strip():
            label = str(row["condition"]).strip()
        else:
            label = _parse_characteristics(row.get("characteristics", "")).get(
                condition_key.lower(), "")
        labels[sid] = label

    chosen = {numerator: [], denominator: []}
    for sid, label in labels.items():
        if label in chosen:
            chosen[label].append(sid)
    for group in chosen.values():
        group.sort()
    if not chosen[numerator] or not chosen[denominator]:
        raise ValueError(
            f"contrast {numerator!r} vs {denominator!r} has an empty group "
            f"(sizes: {numerator}={len(chosen[numerator])}, "
            f"{denominator}={len(chosen[denominator])}); "
            f"available labels: {sorted(set(labels.values()))}")

    # Denominator first, then numerator; stable within group.
    order = chosen[denominator] + chosen[numerator]
    condition_by_id = {**{s: denominator for s in chosen[denominator]},
                       **{s: numerator for s in chosen[numerator]}}
    sheet_out = pd.DataFrame(
        {"sample_id": order, "condition": [condition_by_id[s] for s in order]})
    sheet_path = out_dir / "contrast_sheet.tsv"
    sheet_out.to_csv(sheet_path, sep="\t", index=False)

    counts = pd.read_csv(export["counts_path"], sep="\t", dtype=str)
    gene_col = counts.columns[0]
    symbol_col = next((c for c in _SYMBOL_COLUMNS if c in counts.columns), None)
    keep = [gene_col] + ([symbol_col] if symbol_col else []) + order
    missing = [s for s in order if s not in counts.columns]
    if missing:
        raise ValueError(f"counts missing selected samples: {missing}")
    subset = counts[keep]
    counts_path = out_dir / "contrast_counts.tsv"
    subset.to_csv(counts_path, sep="\t", index=False)

    return {"counts_path": counts_path, "sheet_path": sheet_path,
            "numerator": numerator, "denominator": denominator,
            "n_samples": len(order),
            "group_sizes": {denominator: len(chosen[denominator]),
                            numerator: len(chosen[numerator])},
            "sample_ids": order}


def run_de(contrast: dict[str, Any], out_dir: Path,
           params: dict | None = None) -> dict[str, Any]:
    """Run the local bm_runner differential analysis; return its provenance."""
    from bm_runner import runner

    out_dir = Path(out_dir)
    provenance = runner.run(
        Path(contrast["counts_path"]), Path(contrast["sheet_path"]),
        contrast["numerator"], contrast["denominator"], out_dir, params)
    return {"provenance": provenance,
            "provenance_path": out_dir / "provenance.json",
            "de_table_path": out_dir / "de_table.tsv",
            "gene_symbols_path": out_dir / "gene_symbols.tsv",
            "dir": out_dir}


def fetch_gene_sets(client: BioMinerClient, out_dir: Path,
                    libraries: tuple[str, ...] = ("go_bp",)) -> dict[str, Any]:
    """Download frozen GMT snapshots + write their runner manifests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    package = client.gene_set_package()["package"]
    result: dict[str, Any] = {}
    for lib in package.get("libraries", []):
        name = lib["library"]
        if name not in libraries or not lib.get("available", False):
            continue
        gmt_path = client.download_ticket(lib["download"],
                                          out_dir / lib["download"]["filename"])
        manifest_path = out_dir / f"{name}.manifest.json"
        manifest_path.write_text(json.dumps(lib["runner_manifest"], indent=2),
                                 encoding="utf-8")
        result[name] = {"gmt_path": gmt_path, "manifest_path": manifest_path,
                        "library": lib}
    if not result:
        raise ValueError(f"none of the requested libraries available: {libraries}")
    return result


def run_enrichment(de: dict[str, Any], gene_sets: dict[str, Any],
                   out_dir: Path, *, kind: str = "gsea",
                   library: str = "go_bp", species: str = "human") -> dict[str, Any]:
    """Run local GSEA or ORA against a downloaded gene-set library."""
    from bm_runner import pathways

    out_dir = Path(out_dir)
    gs = gene_sets[library]
    fn = pathways.run_gsea if kind == "gsea" else pathways.run_ora
    provenance = fn(
        Path(de["de_table_path"]), Path(gs["gmt_path"]), out_dir,
        gene_symbols_path=Path(de["gene_symbols_path"]),
        manifest_path=Path(gs["manifest_path"]), species=species)
    return {"provenance": provenance, "dir": out_dir, "kind": kind,
            "library": library}


def save_result(client: BioMinerClient, study_id: str,
                de: dict[str, Any], contrast: dict[str, Any],
                *, idempotency_key: str | None = None,
                result_kind: str = "de") -> dict[str, Any]:
    """Upload provenance + de_table and register the local result.

    The server validates the provenance schema and stores the bytes; it
    does not recompute or endorse the statistics. Configure the client
    with a ``receipt_cache`` (the CLI does this by default) so re-runs
    reuse the original receipts and the server replays instead of
    refusing the re-bound idempotency key with CONFLICT.
    """
    prov_semantic = _provenance_semantic_sha(de["provenance_path"])
    prov_receipt = client.upload_file(Path(de["provenance_path"]),
                                      semantic_sha=prov_semantic)
    de_receipt = client.upload_file(Path(de["de_table_path"]))
    if idempotency_key is None:
        # Semantic provenance hash (volatile created_at/metrics excluded) +
        # de_table bytes: stable across re-runs of the same contrast, so the
        # server replays instead of accumulating duplicates (bug #12).
        idempotency_key = _stable_key(
            "lr", study_id, contrast["numerator"], contrast["denominator"],
            result_kind, prov_semantic, _file_sha(de["de_table_path"]))

    def _register(prov: dict, de_r: dict) -> dict:
        return client.call_tool("local_result_register", {
            "study_id": study_id,
            "contrast": {"numerator": contrast["numerator"],
                         "denominator": contrast["denominator"]},
            "provenance_receipt_id": prov["receipt_id"],
            "de_table_receipt_id": de_r["receipt_id"],
            "result_kind": result_kind,
            "idempotency_key": idempotency_key})

    try:
        data = _register(prov_receipt, de_receipt)
    except BioMinerError as exc:
        if exc.code != "CONFLICT":
            raise
        # The key is already bound to an earlier request body (e.g. a
        # pre-fix run uploaded byte-different provenance). Replay that
        # original registration by retrying with previously cached
        # receipts instead of failing the re-run (bug #12 migration).
        data = None
        for old_prov in client.cached_receipts(Path(de["provenance_path"]).name):
            if old_prov["receipt_id"] == prov_receipt["receipt_id"]:
                continue
            try:
                data = _register(old_prov, de_receipt)
                prov_receipt = old_prov
                client.rebind_semantic_receipt(
                    Path(de["provenance_path"]).name, prov_semantic, old_prov)
                break
            except BioMinerError as retry_exc:
                if retry_exc.code != "CONFLICT":
                    raise
        if data is None:
            raise
    return {"register": data,
            "provenance_receipt_id": prov_receipt["receipt_id"],
            "de_table_receipt_id": de_receipt["receipt_id"]}


def save_enrichment_result(client: BioMinerClient, study_id: str,
                           enrichment: dict[str, Any],
                           contrast: dict[str, Any], *,
                           parent_result_id: str,
                           idempotency_key: str | None = None) -> dict[str, Any]:
    """Upload pathway provenance + table and register a gsea/ora result.

    The server cross-checks the provenance against the parent DE digest
    and the uploaded table bytes, then stores them; it does not recompute
    or endorse the statistics. ``enrichment`` is one value of the dict
    returned by :func:`run_enrichment` (``kind``, ``dir``, ``library``).
    As with :func:`save_result`, configure a ``receipt_cache`` on the
    client so re-runs replay instead of hitting a CONFLICT.
    """
    kind = enrichment["kind"]
    if kind not in ("gsea", "ora"):
        raise ValueError(f"unsupported enrichment kind: {kind!r}")
    gdir = Path(enrichment["dir"])
    # bm_runner names the pathway provenance per kind: GSEA writes
    # provenance.json, ORA writes ora_provenance.json.
    prov_path = gdir / ("provenance.json" if kind == "gsea"
                        else "ora_provenance.json")
    table_path = gdir / ("gsea_results.tsv" if kind == "gsea"
                         else "ora_results.tsv")
    prov_semantic = _provenance_semantic_sha(prov_path)
    # Namespace the receipt cache by kind: DE and pathway provenance share
    # the on-disk filename "provenance.json"; a shared cache name would
    # cross-wire the bug-#12 CONFLICT recovery between result kinds.
    prov_name = f"{kind}_provenance.json"
    prov_receipt = client.upload_bytes(prov_name, prov_path.read_bytes(),
                                       semantic_sha=prov_semantic)
    tab_receipt = client.upload_file(table_path)
    if idempotency_key is None:
        # Semantic provenance hash (volatile created_at/metrics excluded) +
        # pathway table bytes: stable across re-runs of the same contrast
        # and library, so the server replays instead of duplicating.
        idempotency_key = _stable_key(
            "lr", study_id, contrast["numerator"], contrast["denominator"],
            kind, enrichment.get("library", ""), prov_semantic,
            _file_sha(table_path))

    def _register(prov: dict, tab: dict) -> dict:
        return client.call_tool("local_result_register", {
            "study_id": study_id,
            "contrast": {"numerator": contrast["numerator"],
                         "denominator": contrast["denominator"]},
            "result_kind": kind,
            "parent_result_id": parent_result_id,
            "pathway_table_receipt_id": tab["receipt_id"],
            "provenance_receipt_id": prov["receipt_id"],
            "idempotency_key": idempotency_key})

    try:
        data = _register(prov_receipt, tab_receipt)
    except BioMinerError as exc:
        if exc.code != "CONFLICT":
            raise
        # Same migration recovery as save_result (bug #12): the key is
        # bound to an earlier body; replay with the original receipts.
        data = None
        for old_prov in client.cached_receipts(prov_name):
            if old_prov["receipt_id"] == prov_receipt["receipt_id"]:
                continue
            try:
                data = _register(old_prov, tab_receipt)
                prov_receipt = old_prov
                client.rebind_semantic_receipt(
                    prov_name, prov_semantic, old_prov)
                break
            except BioMinerError as retry_exc:
                if retry_exc.code != "CONFLICT":
                    raise
        if data is None:
            raise
    return {"register": data,
            "provenance_receipt_id": prov_receipt["receipt_id"],
            "pathway_table_receipt_id": tab_receipt["receipt_id"]}


def local_analysis(client: BioMinerClient, gse_id: str, numerator: str,
                   denominator: str, workdir: Path, *,
                   condition_key: str = "treatment",
                   condition_map: dict[str, str] | None = None,
                   libraries: tuple[str, ...] = ("go_bp",),
                   enrichment: tuple[str, ...] = ("gsea", "ora"),
                   study_id: str | None = None,
                   organism: str = "Homo sapiens",
                   assay_type: str = "rna_seq",
                   register: bool = True,
                   params: dict | None = None,
                   species: str | None = None,
                   progress=None) -> dict[str, Any]:
    """Chain the full download -> local analysis -> save pipeline.

    ``species`` overrides the enrichment species; by default it follows the
    export manifest's ``organism`` (the runner normalizes e.g. "Homo sapiens"
    → human and honestly refuses unsupported species rather than
    mis-annotating). ``progress`` is an optional ``callable(str)`` invoked at
    each step so a CLI can show liveness during the multi-minute DE fit.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def _say(msg: str) -> None:
        if progress is not None:
            progress(msg)

    _say(f"[1/6] exporting {gse_id} from the server ...")
    exported = export_dataset(client, gse_id, workdir / "export")
    manifest_organism = str(exported["export"].get("organism") or organism)
    _say(f"[2/6] building contrast {numerator!r} vs {denominator!r} "
         f"(key={condition_key!r}) ...")
    contrast = build_contrast(exported, numerator, denominator,
                              workdir / "contrast", condition_key=condition_key,
                              condition_map=condition_map)
    _say(f"[3/6] local DE on {contrast['n_samples']} samples "
         "(pydeseq2 fit; this can take a few minutes) ...")
    de = run_de(contrast, workdir / "de", params)

    enrichments: dict[str, Any] = {}
    if enrichment:
        _say("[4/6] fetching the frozen gene-set package ...")
        gene_sets = fetch_gene_sets(client, workdir / "gene_sets", libraries)
        library = libraries[0]
        enrich_species = species or manifest_organism
        for kind in enrichment:
            _say(f"[5/6] local {kind.upper()} ({library}, "
                 f"species={enrich_species}) ...")
            enrichments[kind] = run_enrichment(
                de, gene_sets, workdir / kind, kind=kind, library=library,
                species=enrich_species)

    saved: dict[str, Any] | None = None
    enrichment_saved: dict[str, Any] = {}
    if register:
        _say("[6/6] registering study + uploading DE/provenance"
             + (f" + {'/'.join(enrichments)} results" if enrichments else "")
             + " ...")
        if study_id is None:
            study_id = client.call_tool("study_create", {
                "body": {"organism": manifest_organism, "assay_type": assay_type,
                         "title": f"{gse_id}: {numerator} vs {denominator}"},
                "idempotency_key": _stable_key(
                    "study", gse_id, numerator, denominator, condition_key,
                    manifest_organism, assay_type)})["study_id"]
        saved = save_result(client, study_id, de, contrast)
        # Persist the enrichment results too: a one-click run that computes
        # GSEA/ORA but leaves them only on disk loses the biology on refresh.
        parent = ((saved.get("register") or {}).get("result") or {}).get("result_id")
        if parent:
            for name, enr in enrichments.items():
                enrichment_saved[name] = save_enrichment_result(
                    client, study_id, enr, contrast, parent_result_id=parent)

    return {"gse_id": gse_id, "workdir": workdir, "export": exported["export"],
            "contrast": contrast, "de": de, "enrichments": enrichments,
            "study_id": study_id, "saved": saved,
            "enrichment_results": enrichment_saved}
