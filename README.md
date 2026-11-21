# bm_runner — BioMiner 本地计算 runner

**原始数据不出本机的转录组分析计算组件。** 这是 [BioMiner](https://biominer.hxzhi.cc.cd) 的开源客户端部分：差异表达（DE）、通路富集（GSEA/ORA）、投稿级 SVG 图表和可核验的溯源记录，全部在你的机器上完成。

```
你的 counts / salmon / kallisto 定量
        │
        ▼
   bm_runner（本仓库，本机运行）
        ├── PyDESeq2 0.5.4 差异表达（固定参数剖面，拒绝不成立的设计）
        ├── gseapy 1.3.0 GSEA / ORA
        ├── 纯 SVG 图：volcano / PCA / heatmap（零额外依赖，字节确定性）
        └── report.md + provenance.json（输入/参数/环境/输出逐级 SHA-256）
```

云端服务（可选）只做元数据、票据与结果登记，**从不接触你的原始表达矩阵**。本仓库不含服务端代码与签名库。

## 安装

需要 Python 3.11+。

**终端用户（一条命令装好 runner + 连接器）：**

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows（Linux/macOS: source .venv/bin/activate）
pip install "bm-runner @ git+https://github.com/Aarash-cpu-cmd/biominer-runner.git"
python -m bm_runner self-check
biominer-client --help
```

> **受限网络 / 中国大陆镜像**：若 pip 直连 pypi.org / files.pythonhosted.org 出现
> SSL 证书错误或下载超时（企业 TLS 审查、CDN 不可达），改用清华镜像并信任该主机：
>
> ```bash
> export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
> export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
> # Windows PowerShell: $env:PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
> #                     $env:PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
> ```
>
> 这只影响 pip 取包。连接器 `biominer-client` 用 certifi 校验 BioMiner 服务端证书、
> 默认不走系统代理（`trust_env=False`），因此机器上的代理 / 自定义 CA 环境变量
> 不会干扰它与托管服务的连接。

**开发者（从源码可编辑安装）：**

```bash
pip install -r requirements.lock.txt
pip install -e .
python -m bm_runner self-check
```

`requirements.lock.txt` 是冻结环境：引擎版本被锁定（pydeseq2 必须为 0.5.4），因为 DE/GSEA 数值是引擎特定的，换版本就换了结果。

## 使用

```bash
# 原始整数 counts
python -m bm_runner run \
  --counts counts.tsv --sheet sheet.tsv \
  --numerator treated --denominator control \
  --out out/

# salmon / kallisto 定量（按基因求和取整，变换记入溯源）
python -m bm_runner run \
  --quant-dir quant/ --tx2gene tx2gene.tsv \
  --sheet sheet.tsv --numerator treated --denominator control \
  --out out/

# 通路富集（本机 GMT，如 WikiPathways；V1 仅人类）
python -m bm_runner gsea \
  --de-table out/de_table.tsv --gene-symbols out/gene_symbols.tsv \
  --gene-sets library.gmt --out out/gsea

python -m bm_runner ora \
  --de-table out/de_table.tsv --gene-symbols out/gene_symbols.tsv \
  --gene-sets library.gmt --out out/ora

# 跑之前预检输入（不拟合、不写盘；不成立的设计在此就会被拒绝）
python -m bm_runner validate \
  --counts counts.tsv --sheet sheet.tsv \
  --numerator treated --denominator control
```

完整命令、输入格式与输出契约见 [docs/RUNNER_SPEC.md](docs/RUNNER_SPEC.md)。

## 设计原则：先判断，再计算

runner 只接受冻结支持矩阵内的分析（V1：两组、非配对、无批次；原始整数 counts 或按基因汇总的定量）。对不成立的输入——配对设计缺配对列、TPM/FPKM、样本表错位、组内样本不足等——**结构化拒绝并说明原因**，而不是硬跑出一张漂亮的火山图。找到 0 个显著基因是合法结果，不是失败。

每次运行写出 `provenance.json`：输入文件、参数、引擎版本、环境指纹与输出文件逐级 SHA-256。它声明"这些字节、这个对比、这个引擎、这些参数、这台机器"，不声明科学有效性——任何人都可以据哈希独立复核。

## 与 BioMiner 托管服务的关系

- 本仓库 = 本机计算层，**可独立离线使用**。
- 托管服务（MCP，Streamable-HTTP）提供 GEO 数据集目录、表达矩阵导出、设计评估、签名检索与结果登记；runner 的产物可以登记到服务端形成父子结果链（如 DE → GSEA）。
- MCP 端点：`https://biominer.hxzhi.cc.cd/mcp/workspace/mcp`（凭据在服务端签发；下载/上传走一次性票据，不携带凭据）。
- 服务端代码与公共签名库不在本仓库开源范围内。

## 本地工作流客户端 `biominer_client`

随仓库提供连接器 `biominer_client`（CLI：`biominer-client`），把本机计算接回托管服务：**下载数据 → 本机跑 DE/GSEA/ORA → 回传并登记结果**。服务端只存字节、颁发票据、登记回执，不重算统计。

```bash
# 零账号一键跑：无令牌时自动铸造短命匿名凭据（无需注册，权限与注册用户一致，云计算除外）
# 先看真实分组字段与取值样本数，再选对比
biominer-client conditions --gse GSE52778
# 下载 → 本机 DE/GSEA/ORA → 回传登记
# 重复运行同一对比会幂等回放（replayed=true），不产生重复 study/结果
biominer-client run --gse GSE52778 --condition-key treatment \
    --numerator Dexamethasone --denominator Untreated --out work/airway \
    --libraries go_bp,reactome,wikipathways

# 或使用注册账号（令牌写入 .biominer_token，绝不打印）
biominer-client register --username alice --password '***'
biominer-client login --username alice --password '***'
```

不内置任何凭据：令牌来自 `BIOMINER_TOKEN` / 本地令牌文件，缺省时自动匿名铸造（不落盘、到期自毁）。完整流程、Python API、契约与真实示例见 [docs/LOCAL_WORKFLOW.md](docs/LOCAL_WORKFLOW.md)。

## 测试

```bash
pip install pytest
python -m pytest tests -q
```

测试用真实 PyDESeq2 跑合成 counts（含拒绝路径、确定性、溯源校验）；个别与服务端旧引擎做奇偶校验的用例在本仓库中自动跳过。

## 许可证

[Apache-2.0](LICENSE)。依赖组件（PyDESeq2、gseapy 等）各自遵循其上游许可证。