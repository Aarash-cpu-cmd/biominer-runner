# 本地工作流客户端 `biominer_client`

`bm_runner` 负责**本机计算**；`biominer_client` 负责把本机计算接回 BioMiner
托管服务：**从服务端下载数据 → 在本机跑 DE/GSEA/ORA → 把产物回传并登记**。
服务端只保存字节、颁发票据、登记回执，**不在服务端重算或背书统计结果**。

```
expression_export_prepare ──票据──> counts.tsv.gz + sample_sheet.tsv   (下载)
        build_contrast                  声明两组对比，按样本子集化 counts
        bm_runner.run                   本机 PyDESeq2 → de_table + provenance
gene_set_package_get ──票据──> 冻结 GMT + manifest                      (下载)
        bm_runner pathways              本机 GSEA / ORA
study_create + upload_prepare ──票据──> provenance.json / de_table.tsv  (上传)
        local_result_register           登记结果，返回 result_id
```

## 认证

客户端**不内置任何凭据**。两条路径：

**零账号（默认）**：没有令牌时，`export` / `conditions` / `run` / `save` 会自动
向服务端铸造一枚**短命匿名凭据**（无需注册，权限与注册用户一致，云计算除外）。
匿名凭据不落盘、到期/预算耗尽即自毁；服务端按稳定设备指纹 find-or-mint，
重复运行仍落在同一 study 命名空间，幂等回放不受影响。

**注册账号**：令牌通过 `BIOMINER_TOKEN` 环境变量或 `--token-file` 提供；
`register` / `login` 会把令牌写入 `--token-file`（默认 `.biominer_token`，
在支持的平台上设为仅属主可读），不会打印令牌。

```bash
biominer-client login --username alice --password '***' --token-file .biominer_token
# 或注册新账号
biominer-client register --username alice --password '***'
```

服务端地址默认指向公共端点，可用 `--base-url` 或 `BIOMINER_BASE_URL` 覆盖。

## 命令行

```bash
# 先看真实分组：列出 characteristics 各字段与取值样本数（零账号即可）
biominer-client conditions --gse GSE52778

# 仅下载某系列的原始整数 counts 与样本表（逐个校验 SHA-256）
biominer-client export --gse GSE52778 --out work/export

# 完整流程：下载 → 本机 DE + GSEA/ORA → 回传登记（进度打到 stderr）
biominer-client run --gse GSE52778 --condition-key treatment \
    --numerator Dexamethasone --denominator Untreated \
    --out work/airway --libraries go_bp --enrichment gsea,ora

# 只在本机分析、不回传
biominer-client run --gse GSE52778 --numerator Dexamethasone \
    --denominator Untreated --out work/airway --no-register

# 自定义分组：sample_id -> 标签 的 JSON 映射（覆盖表内标签）
biominer-client run --gse GSE52778 --numerator case --denominator ctrl \
    --condition-map my_groups.json --out work/custom

# 把已有的本机产物回传登记
biominer-client save --study-id study_xxx \
    --provenance work/airway/de/provenance.json \
    --de-table work/airway/de/de_table.tsv \
    --numerator Dexamethasone --denominator Untreated
```

`run` 的分组标签默认从样本表 `characteristics` 里的 `treatment` 字段推导
（可用 `--condition-key` 改字段名）；若样本表已带 `condition` 列则优先使用。
**不知道字段名/标签时，先跑 `conditions`**——它按样本数列出每个字段的真实
取值；选空组时错误信息会附带可用标签与 `conditions` 提示。
客户端会按所选两组**子集化 counts**，使列与样本表逐一同序——这是 runner 的
硬性输入约束（列错位会被结构化拒绝）。

富集物种默认跟随导出 manifest 的 `organism`（当前 ARCHS4 本地源为人源，
runner 将 "Homo sapiens" 归一为 human）；`--species` 可显式覆盖，不支持的
物种会被 runner 诚实拒绝（SPECIES_UNSUPPORTED）而非错误注释。

## Python API

```python
from biominer_client import BioMinerClient, workflow

with BioMinerClient(token="...") as c:
    result = workflow.local_analysis(
        c, "GSE52778", "Dexamethasone", "Untreated", "work/airway",
        libraries=("go_bp",), enrichment=("gsea", "ora"))
    print(result["de"]["provenance"]["classification"])
    print(result["saved"]["register"]["result"]["result_id"])
```

一键跑会把 DE 与 GSEA/ORA 结果都注册回服务端（富集结果挂在 DE 结果的
`parent_result_id` 下），返回值的 `enrichment_results` 按 kind 给出各自回执。

各步骤也可单独调用：`export_dataset`、`build_contrast`、`run_de`、
`fetch_gene_sets`、`run_enrichment`、`save_result`、`save_enrichment_result`。
需要显式指定分组时用
`build_contrast(..., condition_map={"GSM...": "treated", ...})`。
直接用 Python API 时请给客户端配置 `receipt_cache`（CLI 默认已配），
否则重跑会因票据变化触发服务端 CONFLICT 而非幂等回放。

## 真实示例（已在生产端跑通）

GSE52778（人气道平滑肌细胞，地塞米松处理）：4 个 Dexamethasone vs 4 个
Untreated，67,186 个基因输入、18,247 个可检验基因，**955 个显著（620 上调 /
335 下调，padj<0.05）**。上调头部为 FKBP5、ZBTB16、DUSP1、PER1、MAOA 等经典
糖皮质激素靶基因；GSEA 检验 2,172 个 GO BP 通路，ORA 检验 10,794 个通路。
产物登记为 `lres_…`，`validation_status=schema_validated`、
`execution_origin=local`，可用 `local_result_get` 复核。

## 契约要点

- **下载/上传走一次性票据**：`expression_export_prepare` / `gene_set_package_get`
  返回带 SHA-256 的下载票据；`upload_prepare` 返回上传票据，字节以
  `application/octet-stream` POST 到票据 URL，得到 `receipt`。票据短期有效、
  绑定属主，**不携带科学权威**；回执只证明服务端收到了恰好这些字节。
- **登记校验 provenance**：`local_result_register` 需要 `study_id`、
  `contrast={numerator,denominator}`、`provenance_receipt_id`、
  `de_table_receipt_id`、`idempotency_key`。服务端校验 `provenance.json`
  含 `execution_origin / engine / dependencies / params / contrast / input /
  output / metrics / created_at` 等字段——这些正是 `bm_runner.run()` 的产物。
- **幂等**：上传与登记都带幂等键，重放返回同一回执/结果，不重复登记。

## 安全

- 不内置凭据；令牌仅来自环境变量或本地令牌文件。
- 默认直连（`trust_env=False`），避免本地代理静默截获票据字节；TLS 用 certifi
  根证书校验，**不要对真实部署关闭校验**。
- 客户端不计算科学结果，也不把统计有效性写入回执——有效性由 `bm_runner`
  的冻结参数与逐级 SHA-256 溯源独立支撑。
