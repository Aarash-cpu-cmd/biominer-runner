# bm_runner — fixed-version local differential-expression runner

The runner is the only component allowed to compute a differential table on the
client side. It refuses anything outside the frozen support matrix, fits the
D23 `bulk_wald_poscounts_raw_v1` profile, and writes a provenance record the
server can verify independently.

## Supported in V1

| Item | Frozen value |
|------|--------------|
| Data | bulk RNA-seq counts: raw integers, or salmon/kallisto estimated counts summed per gene and rounded |
| Design | exactly two groups, unpaired, no batch adjustment |
| Engine | pydeseq2 **0.5.4**, `~condition`, reference level first |
| Prefilter | count >= 10 in >= 2 samples (D23) |
| Parameters | alpha 0.05, parametric fit, **poscounts** size factors, `low_memory=true`, cooks + independent filtering on |
| Significance | finite `padj < 0.05`; effect candidates add finite raw `\|log2FC\| >= 1.0` |
| Shrinkage | none (D23) |
| Threads | 1 by default, `--params '{"n_cpus": N}'` accepts 1..4 |
| Gene cap | 70,000 (an unfiltered human ARCHS4 matrix has 67,186) |
| Sample cap | 200 |

Refused with a structured code: paired designs (`subject_id`/`pair_id`),
multi-level batch columns, more or fewer than two conditions, groups with fewer
than two samples, negative or non-integer counts (TPM/FPKM), duplicate gene
identifiers or sample ids, mismatched sample sets, counts whose sample columns
are not in the same order as the sheet `sample_id` column (`SHEET_ORDER_MISMATCH`
— a permutation of otherwise matching ids is refused, not silently reordered),
more than 70,000 genes or 200 samples, zero genes surviving the D23 prefilter
(`NO_TESTABLE_GENES`, no fit is attempted), unsupported parameter overrides, and
any environment whose pydeseq2 is not 0.5.4.

A run that finds zero significant genes is a **legal result**, not a failure.

## Usage

```bash
pip install -r requirements.lock.txt
python -m bm_runner self-check
python -m bm_runner run \
  --counts counts.tsv --sheet sheet.tsv \
  --numerator treated --denominator control \
  --out out/
```

Quantified input (salmon or kallisto):

```bash
python -m bm_runner run \
  --quant-dir quant/ --tx2gene tx2gene.tsv \
  --sheet sheet.tsv --numerator treated --denominator control \
  --out out/
```

`quant/` contains one subdirectory per sample (`quant/GSM1/quant.sf` or
`quant/GSM1/abundance.tsv`); sample identity comes from the directory name.
Transcript estimated counts are summed per gene and rounded half-up; the
transform and the unrounded digest are recorded in provenance. This is a
declared V1 limitation.

Pathway analysis (human only in V1):

```bash
python -m bm_runner gsea \
  --de-table out/de_table.tsv --gene-symbols out/gene_symbols.tsv \
  --gene-sets library.gmt --gene-set-manifest manifest.json \
  --out out/gsea

python -m bm_runner ora \
  --de-table out/de_table.tsv --gene-symbols out/gene_symbols.tsv \
  --gene-sets library.gmt --gene-set-manifest manifest.json \
  --out out/ora
```

GSEA freezes gseapy 1.3.0 prerank, Wald `stat` ranking, 1000 permutations,
seed 42, set size 15..500, one thread. ORA uses a one-sided hypergeometric
test over effect-filtered DEGs (padj < 0.05 and |log2FC| >= 1), up and down
separately, Benjamini-Hochberg within each direction, with the testable
post-filter genes as background. Both verify the GMT SHA-256 against the
manifest when one is supplied, and both write `bm_local_pathway_v1`
provenance. Mouse is refused with `SPECIES_UNSUPPORTED` until V2.

Inputs:

* `counts.tsv` — first column is a gene identifier (Ensembl or symbol), an
  optional `gene_symbol` column carries the display symbol used by pathway
  analysis, remaining columns are samples with non-negative integer values.
* `sheet.tsv` — at minimum `sample_id` and `condition`.

Outputs in `--out`:

| File | Contents |
|------|----------|
| `de_table.tsv` | gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj (NA for non-finite) |
| `gene_symbols.tsv` | `gene_id`, `gene_symbol` used by pathway analysis |
| `provenance.json` | schema `bm_local_result_v1`: input/params/output digests, D23 profile, effective fit type, filter counts, contrast, engine and dependency versions, runner version, exit code, elapsed seconds, peak RSS, group sizes |
| `run.log` | what was read and produced |

Pathway outputs add `gsea_results.tsv` + `ranking.tsv` + `run_config.json` +
`gene_set_provenance.json` + `mapping_report.json` + `provenance.json` (GSEA),
or `ora_results.tsv` + `ora_provenance.json` (ORA).

Exit codes: `0` success, `2` refused input (structured JSON on stdout, nothing
written), `1` internal error.

## What the provenance does and does not claim

It claims: these bytes, this contrast, this engine version, these parameters,
this filter, this machine, this duration. It does **not** claim scientific
validity, and it is not a server attempt or an execution grant. The server
stores it with `execution_origin=local` and `validation_status=schema_validated`.
