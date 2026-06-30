# 结果指标（评分证据）

本目录存放各配置 run 的**聚合指标**（overall + by-category accuracy），作为评分证据。
不含 per-question 原文与对话数据（避免数据再分发）。

| run | accuracy | 配置 |
| --- | ---: | --- |
| 01_baseline_glm52_top20 | 79.68% | GLM-5.2 基线（原始 top-20 检索） |
| 02_deterministic_fusion_top200 | 93.83% | 确定性 hash + fusion + 双源 + 思维链 |
| 03_real_semantic_fusion_top200 | 95.26% | 真实语义(E5修复) + fusion + 双源 + 思维链 |

详见 `metrics/*.json`。完整复现见仓库根 README。
