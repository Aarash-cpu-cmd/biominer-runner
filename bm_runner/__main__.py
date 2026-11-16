"""CLI for the fixed-version local runner.

Exit codes:
    0  success — provenance.json written
    2  refused input (structured JSON on stdout, nothing written)
    1  internal error (traceback on stderr)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .pathways import run_gsea, run_ora
from .runner import (RUNNER_VERSION, Refusal, check_engine, dependencies,
                     run, run_quant)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bm_runner",
                                     description="fixed-version bulk DE runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("self-check", help="print runner and engine versions")

    run_p = sub.add_parser("run", help="compute a two-group differential table")
    source = run_p.add_mutually_exclusive_group(required=True)
    source.add_argument("--counts", type=Path,
                        help="gene x sample integer count table")
    source.add_argument("--quant-dir", type=Path,
                        help="directory of salmon/kallisto sample folders")
    run_p.add_argument("--tx2gene", type=Path,
                       help="transcript-to-gene table (required with --quant-dir)")
    run_p.add_argument("--sheet", required=True, type=Path)
    run_p.add_argument("--numerator", required=True)
    run_p.add_argument("--denominator", required=True)
    run_p.add_argument("--out", required=True, type=Path)
    run_p.add_argument("--params", default="{}",
                       help="JSON object; only n_cpus (1..4) may be overridden")

    def _pathway_parser(name: str, help_text: str):
        parser = sub.add_parser(name, help=help_text)
        parser.add_argument("--de-table", required=True, type=Path)
        parser.add_argument("--gene-sets", required=True, type=Path,
                            help="frozen GMT file")
        parser.add_argument("--gene-symbols", type=Path,
                            help="gene_symbols.tsv from the DE run")
        parser.add_argument("--gene-set-manifest", type=Path,
                            help="frozen manifest with the expected sha256")
        parser.add_argument("--species", default="human")
        parser.add_argument("--out", required=True, type=Path)
        return parser

    _pathway_parser("gsea", "run GSEA prerank over a DE table")
    _pathway_parser("ora", "run hypergeometric ORA over a DE table")

    val_p = sub.add_parser(
        "validate",
        help="check inputs without fitting; writes nothing")
    vsource = val_p.add_mutually_exclusive_group(required=True)
    vsource.add_argument("--counts", type=Path,
                         help="gene x sample integer count table")
    vsource.add_argument("--quant-dir", type=Path,
                         help="directory of salmon/kallisto sample folders")
    val_p.add_argument("--tx2gene", type=Path,
                       help="transcript-to-gene table "
                            "(required with --quant-dir)")
    val_p.add_argument("--sheet", required=True, type=Path)
    val_p.add_argument("--numerator", required=True)
    val_p.add_argument("--denominator", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "self-check":
        print(json.dumps({"runner_version": RUNNER_VERSION,
                          "dependencies": dependencies()}, indent=2))
        return 0

    if args.cmd == "validate":
        from .runner import validate_inputs
        try:
            if args.quant_dir and not args.tx2gene:
                raise ValueError("--quant-dir requires --tx2gene")
            if args.counts and args.tx2gene:
                raise ValueError("--tx2gene is only valid with --quant-dir")
            summary = (validate_inputs(args.quant_dir, args.tx2gene,
                                       args.sheet, args.numerator,
                                       args.denominator)
                       if args.quant_dir else
                       validate_inputs(None, None, args.sheet,
                                       args.numerator, args.denominator,
                                       counts_path=args.counts))
        except Refusal as exc:
            print(json.dumps({"ok": False, "code": exc.code,
                              "message": exc.message,
                              "details": exc.details}))
            return 2
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "code": "INTERNAL_ERROR",
                              "message": f"{type(exc).__name__}: {exc}"}))
            return 1
        print(json.dumps({"ok": True, **summary}, indent=2))
        return 0

    try:
        params = json.loads(getattr(args, "params", "{}"))
        if not isinstance(params, dict):
            raise ValueError("params must be a JSON object")
    except ValueError as exc:
        print(json.dumps({"ok": False, "code": "INVALID_ARGS",
                          "message": str(exc)}))
        return 2

    if getattr(args, "counts", None) and getattr(args, "tx2gene", None):
        print(json.dumps({"ok": False, "code": "INVALID_ARGS",
                          "message": "--tx2gene is only valid with --quant-dir"}))
        return 2
    if getattr(args, "quant_dir", None) and not getattr(args, "tx2gene", None):
        print(json.dumps({"ok": False, "code": "INVALID_ARGS",
                          "message": "--quant-dir requires --tx2gene"}))
        return 2

    try:
        check_engine()
        if args.cmd == "gsea":
            provenance = run_gsea(args.de_table, args.gene_sets, args.out,
                                  gene_symbols_path=args.gene_symbols,
                                  manifest_path=args.gene_set_manifest,
                                  species=args.species)
        elif args.cmd == "ora":
            provenance = run_ora(args.de_table, args.gene_sets, args.out,
                                 gene_symbols_path=args.gene_symbols,
                                 manifest_path=args.gene_set_manifest,
                                 species=args.species)
        elif args.quant_dir:
            provenance = run_quant(args.quant_dir, args.tx2gene, args.sheet,
                                   args.numerator, args.denominator, args.out,
                                   params)
        else:
            provenance = run(args.counts, args.sheet, args.numerator,
                             args.denominator, args.out, params)
    except Refusal as exc:
        print(json.dumps({"ok": False, "code": exc.code,
                          "message": exc.message,
                          "details": exc.details}))
        return 2
    except Exception as exc:  # noqa: BLE001 - internal failure, keep the trace
        traceback.print_exc()
        print(json.dumps({"ok": False, "code": "INTERNAL_ERROR",
                          "message": f"{type(exc).__name__}: {exc}"}))
        return 1

    print(json.dumps({"ok": True, "out": str(args.out),
                      "provenance": str(args.out / "provenance.json"),
                      "de_table": str(args.out / "de_table.tsv"),
                      "runner_version": RUNNER_VERSION,
                      "n_significant": (provenance.get("output") or {}).get(
                          "n_significant")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
