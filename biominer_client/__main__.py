"""Command-line interface for the BioMiner local-runner connector.

Examples
--------
    biominer-client login --username alice --token-file .biominer_token
    biominer-client export --gse GSE52778 --out work/export
    biominer-client run --gse GSE52778 --numerator Dexamethasone \
        --denominator Untreated --out work/airway
    biominer-client save --study-id study_x --provenance work/airway/de/provenance.json \
        --de-table work/airway/de/de_table.tsv --numerator Dexamethasone --denominator Untreated

The token is read from ``--token-file`` or ``BIOMINER_TOKEN``; it is never
printed. ``register`` / ``login`` write it to ``--token-file`` (default
``.biominer_token``) with owner-only permissions where the platform supports it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .client import BioMinerClient, BioMinerError
from . import workflow


def _read_token(args) -> str | None:
    if args.token_file and Path(args.token_file).is_file():
        return Path(args.token_file).read_text(encoding="utf-8").strip()
    return os.environ.get("BIOMINER_TOKEN")


def _write_token(path: str, token: str) -> None:
    p = Path(path)
    p.write_text(token + "\n", encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass


def _client(args) -> BioMinerClient:
    return BioMinerClient(base_url=args.base_url, token=_read_token(args),
                          receipt_cache=args.token_file + ".receipts.json")


def _emit(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="biominer-client",
                                     description=__doc__)
    parser.add_argument("--base-url", default=None,
                        help="BioMiner root URL (default: BIOMINER_BASE_URL "
                             "or the public endpoint)")
    parser.add_argument("--token-file", default=".biominer_token",
                        help="file holding the bearer token")
    # Accept --base-url / --token-file both before AND after the subcommand
    # (users naturally write `login --username x --token-file f`). SUPPRESS
    # keeps an explicit global value from being clobbered by the subparser.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", default=argparse.SUPPRESS)
    common.add_argument("--token-file", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("register", "login"):
        p = sub.add_parser(name, help=f"{name} a BioMiner account",
                           parents=[common])
        p.add_argument("--username", required=True)
        p.add_argument("--password", required=True)

    p = sub.add_parser("export", help="download a series' counts + sample sheet",
                       parents=[common])
    p.add_argument("--gse", required=True)
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("run", help="full download -> local DE/enrichment -> save",
                       parents=[common])
    p.add_argument("--gse", required=True)
    p.add_argument("--numerator", required=True)
    p.add_argument("--denominator", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--condition-key", default="treatment",
                   help="characteristics field used to label groups")
    p.add_argument("--libraries", default="go_bp",
                   help="comma-separated gene-set libraries (go_bp,reactome,wikipathways)")
    p.add_argument("--enrichment", default="gsea,ora",
                   help="comma-separated enrichment kinds, or 'none'")
    p.add_argument("--study-id", default=None)
    p.add_argument("--organism", default="Homo sapiens")
    p.add_argument("--no-register", action="store_true",
                   help="analyse locally but do not save back to BioMiner")

    p = sub.add_parser("save", help="upload + register an existing local result",
                       parents=[common])
    p.add_argument("--study-id", required=True)
    p.add_argument("--provenance", required=True, type=Path)
    p.add_argument("--de-table", required=True, type=Path)
    p.add_argument("--numerator", required=True)
    p.add_argument("--denominator", required=True)

    args = parser.parse_args(argv)

    try:
        if args.cmd in ("register", "login"):
            with _client(args) as c:
                fn = c.register if args.cmd == "register" else c.login
                token = fn(args.username, args.password)
            _write_token(args.token_file, token)
            _emit({"ok": True, "cmd": args.cmd, "username": args.username,
                   "token_file": str(args.token_file)})
            return 0

        if args.cmd == "export":
            with _client(args) as c:
                result = workflow.export_dataset(c, args.gse, args.out)
            _emit({"ok": True, "gse_id": args.gse,
                   "n_samples": result["export"].get("n_samples"),
                   "counts_path": str(result["counts_path"]),
                   "sheet_path": str(result["sheet_path"])})
            return 0

        if args.cmd == "run":
            libraries = tuple(x for x in args.libraries.split(",") if x)
            enrichment = (() if args.enrichment.lower() == "none"
                          else tuple(x for x in args.enrichment.split(",") if x))
            with _client(args) as c:
                result = workflow.local_analysis(
                    c, args.gse, args.numerator, args.denominator, args.out,
                    condition_key=args.condition_key, libraries=libraries,
                    enrichment=enrichment, study_id=args.study_id,
                    organism=args.organism, register=not args.no_register)
            cls = (result["de"]["provenance"].get("classification") or {})
            _emit({"ok": True, "gse_id": args.gse,
                   "contrast": f'{args.numerator} vs {args.denominator}',
                   "n_samples": result["contrast"]["n_samples"],
                   "group_sizes": result["contrast"]["group_sizes"],
                   "classification": cls,
                   "enrichments": list(result["enrichments"]),
                   "study_id": result["study_id"],
                   "result": (result["saved"] or {}).get("register"),
                   "workdir": str(result["workdir"])})
            return 0

        if args.cmd == "save":
            de = {"provenance_path": args.provenance,
                  "de_table_path": args.de_table}
            contrast = {"numerator": args.numerator,
                        "denominator": args.denominator}
            with _client(args) as c:
                saved = workflow.save_result(c, args.study_id, de, contrast)
            _emit({"ok": True, **saved})
            return 0

    except BioMinerError as exc:
        _emit({"ok": False, "code": exc.code, "message": exc.message,
               "details": exc.details})
        return 2
    except (ValueError, FileNotFoundError) as exc:
        _emit({"ok": False, "code": "INVALID_INPUT", "message": str(exc)})
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
