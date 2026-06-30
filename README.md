# HMG-Benchmark

LoCoMo 长对话记忆问答（QA）基准的可复现评测工具集，用于衡量 [HMG](https://github.com/HMG-AI) 的端到端记忆召回与回答能力。

本仓库**只包含评测工具集（harness）**，不包含 HMG 源码、API key 或被测数据集本体。HMG 作为外部依赖通过 `hmg-server` 二进制调用。

---

## 基准简介

- **数据集**：[LoCoMo](https://snap.stanford.edu/locomo/)（Snap Research, ACL 2024），10 段长对话，共 1986 个问答；主分仅统计 category 1–4（multi-hop / temporal / open-domain / single-hop），共 **1540 题**，与 mem0 官方口径一致。
- **评测类型**：端到端（E2E）answerer + judge QA（J-score 二分类判分）。
- **指标**：accuracy = correct / 1540。

### 方法链（可复现的最高分配置）

```
HMG 真实语义召回（MultilingualE5Small，需 HMG ≥ 含 E5 prefix 修复的版本）
        │
        ├──► query-rewrite 多查询融合（LLM 为每题生成 N 个 facet 变体，多路 recall + RRF）
        │
        └──► 双源 RRF 融合（互补召回，突破单路 recall 上限）
                        │
                        ▼
            top-200 大上下文 + Answer LLM（GLM-5.2 思维链）
                        │
                        ▼
                  Judge LLM 判分 → accuracy
```

### 实测结果（本工具集产出，详见 `results/`）

| # | 配置 | accuracy | 说明 |
| --- | --- | ---: | --- |
| 1 | GLM-5.2 baseline（原始 top-20 检索，思维链关） | **79.68%** | 1227/1540 |
| 2 | 确定性 hash + 6-query fusion + 双源 RRF + 思维链 | **93.83%** | 1445/1540，超 mem0 top-200 SOTA(92.5%) |
| 3 | 真实语义（E5 prefix 修复）+ 6-query fusion + 双源 RRF + 思维链 | **95.26%** | 1467/1540，interim 峰值 ~97.7% |

> 关键前提：配置 3 依赖 HMG 修复了 E5 不对称模型未加 `passage:`/`query:` 前缀导致召回崩盘的 bug。该修复已合入 HMG 主分支，下一版本分发后即可复现高分。

---

## 前置条件

1. **HMG**（含 E5 prefix 修复的版本）：`hmg` / `hmg-server` 在 PATH，或通过 `HMG_SERVER_BIN` 指向二进制。
   - 首次运行前预下载嵌入模型：`hmg model embedding download`。
   - 自检：`hmg model embedding status` 应显示 fastembed 启用（非 deterministic fallback）。
2. **LoCoMo 数据集**：从 [Snap Research LoCoMo](https://snap.stanford.edu/locomo/) 获取 `locomo10.json`，放到仓库根目录（**不要提交到本仓库**，见数据许可）。
3. **Python ≥ 3.11** + 依赖：`pip install -r requirements.txt`。
4. **Answer/Judge LLM**：OpenAI 兼容端点。默认 GLM-5.2：
   ```sh
   export ZHIPU_API_KEY="<your-key>"
   # 可选：export OPENAI_BASE_URL=...  / 其他兼容端点
   ```

---

## 复现步骤

> 以下命令在仓库根目录执行。把 `locomo10.json` 放在根目录。

### 步骤 0：准备
```sh
pip install -r requirements.txt
export ZHIPU_API_KEY="<your-key>"
# 验证 HMG 语义嵌入就绪
hmg model embedding status
```

### 步骤 1：把对话 ingest 进 HMG stores（真实语义索引）
```sh
python -m harness.ingest benchmark_stores
```
产出：`benchmark_stores/conv-*`（每个对话一个 store）。

### 步骤 2：单查询自适应召回（baseline 检索 + hit 评测）
```sh
python -m harness.retrieval \
  --dataset locomo10.json \
  --stores-dir benchmark_stores \
  --output retrieval_top50.json
```
产出 `retrieval_top50.json`，报告 hit@20 / hit@50。

### 步骤 3：query-rewrite 多查询融合检索（核心提分）
```sh
python -m harness.fusion \
  --dataset locomo10.json \
  --stores-dir benchmark_stores \
  --output fused_top200.json \
  --top-n 200 --n-rewrites 5
```
对每题用 LLM 生成 5 个不同 facet（人名/事件/时间/属性/地点）的检索变体，多路 recall + RRF 融合，突破 HMG 单路 recall 的 50 atom 上限。

### 步骤 4（可选）：双源 RRF 融合
如果你同时有一份原始检索 JSON（如 `retrieval_top50.json`），可与 fusion 结果做双源 RRF 互补：
```sh
python -m harness.fuse_dual \
  --fusion fused_top200.json \
  --orig retrieval_top50.json \
  --top-n 200 --output fused_dual_top200.json
```

### 步骤 5：E2E answerer + judge（最终评分）
```sh
python -m harness.e2e_qa \
  --retrieval-json fused_dual_top200.json \
  --top-k 200 --enable-thinking \
  --run-mode full \
  --output-dir results_run \
  --max-workers 8 --rpm 150
```
产出 `results_run/hmg_locomo_e2e_full_top_200_results.json`，含 overall / by-category accuracy。

### 步骤 6（可选）：语义重排诊断
验证真实语义嵌入的排序价值（hit@k 提升对比）：
```sh
python -m harness.semantic_rerank --fusion-json fused_dual_top200.json
```

---

## harness 脚本说明

| 脚本 | 作用 |
| --- | --- |
| `harness/hmg_client.py` | HMG MCP 客户端（通过 `hmg-server` 二进制，`HMG_SERVER_BIN` 可覆盖） |
| `harness/ingest.py` | 把 LoCoMo 对话 ingest 进 HMG stores |
| `harness/retrieval.py` | 单查询自适应召回 + evidence-hit 评测 |
| `harness/fusion.py` | query-rewrite 多查询融合（RRF） |
| `harness/fuse_dual.py` | 双源 RRF 融合（互补召回） |
| `harness/e2e_qa.py` | GLM answerer + judge E2E QA（支持 `--top-k` / `--enable-thinking`） |
| `harness/semantic_rerank.py` | 语义重排诊断（本地 e5-small ONNX，验证嵌入质量） |
| `prompts/locomo_prompts.py` | LoCoMo answer / judge 提示词（基于 mem0 风格的宽松判分） |

关键算法 — RRF 融合（`harness/fusion.py`）：
```python
def rrf_fuse(lists, k=60):
    score = defaultdict(float)
    for src in lists:                    # 每路 recall
        for rank, atom in enumerate(src, 1):
            score[atom_key(atom)] += 1.0 / (k + rank)   # reciprocal rank
    return sorted by score desc
```

---

## 数据与许可

- **LoCoMo 数据集**不属于本仓库，需自行从 Snap Research 获取并遵守其许可。`locomo10.json` 与生成的 stores / 检索 JSON 均已在 `.gitignore` 中排除，**不要提交数据**。
- 本 harness 代码以仓库根 `LICENSE` 许可发布。
- 判分采用 LLM-as-judge，存在模型版本、provider 实现、采样与重试带来的波动；结果以本仓库 `results/` 的指标为准。

## 引用

- LoCoMo: *LoCoMo: Long Context Multi-Session Conversational Memory*, Snap Research, ACL 2024.
- mem0 baseline: [mem0ai/memory-benchmarks](https://github.com/mem0ai/memory-benchmarks).
