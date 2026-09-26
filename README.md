# 用户历史短时，商品文本能补上 ID 推荐漏掉的目标吗？

推荐系统要根据用户已有交互，预测下一次发生交互的商品。交互少时，ID 路线能利用的行为线索也少；商品文本看起来可能补位。但 Text 单路整体命中更少，不能只凭这一点判断它是否有用。

我先问一个更具体的问题：**当 ID 没把真实下一商品排进前十时，Text 能否找回它？这种补充在短历史用户中是否更多？** 事前预测是短历史中的 Text 补充更大；同时还要检查这些独有命中能否在融合后留下来。

## 结果怎样改变判断

在同一批 Beauty 测试事件、同一 12,101 件候选商品上，Text 找回 ID 漏例的比例在短历史组为 **296/6,776（4.37%）**，长历史组为 **370/14,444（2.56%）**。差值为 **+1.81 个百分点**，用户 bootstrap 95% 区间为 **+1.25 至 +2.35 个百分点**。这支持一个有限判断：在这次代理实验中，短历史组的 ID 漏例更常被 Text 命中。

但有互补命中，不代表简单融合能保住它。Text 单独命中而 ID 未命中的目标共有 **666 个**，等权融合只保留 **168 个（25.2%）**；它同时丢掉 515 个 ID 命中，全体净增仅 21 个。短组净增 57，长组净减 36。

验证集选择的全局 ID 权重为 **0.9**，在这次测试中前十命中相对 ID 净增 **169 次**，整体 NDCG@10 也从 **0.02869** 升到 **0.02995**。但首位命中从 **1.167%** 降到 **0.868%**，长历史组 NDCG@10 从 **0.02820** 降到 **0.02750**。看过首轮测试后追加的分组调权中，两组验证集也都选出 0.9，没有新收益。

为解释短长差异，目标商品的低、中、高训练频次组内，短组的 Text 救回率仍分别高 **2.61、2.38、1.21 个百分点**；粗分频次无法单独解释差异。频次分组只用于测试后诊断、没有用于路由或选参；ID 分数仍含训练流行度回退。这个测试的“短历史组”实际全是 **4 件可见历史**。另一项事后诊断发现，短组 Text 前十平均有 **3.672 个位置**被历史商品占据，长组为 **4.637 个**；测试目标在完整既往历史中均未出现。这可能挤占新目标的排名位置，但还不能说明它解释了全部差异。[补充诊断](docs/beauty-followup-diagnostics.md)与[历史商品占位诊断](docs/seen-item-diagnostic.md)列出细节；排除可见历史候选和平均并列排名的敏感性结果待核对后再决定下一步。

![Beauty CPU 代理实验汇总](results/beauty-cpu-proxy-summary.svg)

## 怎样得到这些结果

这是 Beauty 上的 CPU 代理实验：ID 分数来自训练序列中的商品转移和流行度，Text 分数来自商品目录描述的 TF-IDF。每个查询都在同一候选集排序；ID 与 Text 分数先转换为候选百分位，再比较单路和融合。原始序列商品 ID `i` 对应元数据 ID `i+1`。

初轮分析方法在查看测试结果前已有书面记录，但没有预注册。初轮测试结果查看后才增加分组权重探索；权重仍只由验证集选择，测试标签没有参与调参。这次代理不是 EnsRec 学习表征训练，也没有复现论文的 cosine 融合。完整设计、切分和结果见[轻量探测计划](docs/lightweight-probe-plan.md)与[研究方案](docs/protocol.md)。

当前百分位实现用候选商品 ID 拆分同分项；稀疏文本分数中许多商品相似度为零，因此融合结果可能受候选 ID 顺序影响。

## 获取数据并运行

按 [EnsRec 上游 README 的 Data Preparation 说明](https://github.com/snap-research/EnsRec#data-preparation)准备 Beauty 数据目录，其中应包含 `items/`、`training/`、`evaluation/` 和 `testing/` TFRecord 文件。原始数据、用户序列和商品文本不放入本仓库。

安装代理依赖后运行：

```bash
python -m pip install -r requirements-proxy.txt
python scripts/beauty_cpu_proxy.py \
  --data-dir /path/to/EnsRec/data/beauty \
  --output-dir outputs/beauty-cpu-proxy
```

需要在本地逐事件复核时，可加 `--write-event-diagnostics`；明细写入输出目录且默认关闭，公开结果只保留聚合数据。

## 相关材料

- [研究案例走读](docs/research-case-walkthrough.md)：从问题到下一步决策的完整推理链
- [轻量探测计划](docs/lightweight-probe-plan.md)：指标口径、代理方法、分组结果与待补诊断
- [原始方案与证据边界](docs/protocol.md)：真实 EnsRec 计划与当前代理结果的区别
- [聚合结果](results/beauty-cpu-proxy-results.json)与[汇总图](results/beauty-cpu-proxy-summary.svg)
- [代理脚本](scripts/beauty_cpu_proxy.py)
- 实现前的[导出探针](docs/id-text-export-probe.md)、[集成审计](docs/integration-audit.md)与[配置审计](docs/config-audit.md)
