# Beauty 代理实验补充诊断

本文补充首轮 CPU proxy 结果，不改变模型、ID/Text 分数、候选库、19 项可见历史上限、validation cutoff 或 alpha 选择规则。它们是描述性分析；短/长分层与目标商品频次相关，不能作因果解释。

## 排名指标

在与原 Recall@10 相同的完整候选排序中保留 target rank。NDCG@10 对前 10 名按 `1/log2(rank+1)` 折损，Hit@1 为第一名命中比例。下表为 all / short / long 的 test 指标；`conditional` 与 `global` 相同，因为两组 validation 都选出 alpha=0.9。

| 方法 | 分层 | NDCG@10 | Hit@1 |
| --- | --- | ---: | ---: |
| ID | 全部 | 0.02869 | 1.167% (261/22,363) |
| ID | 短 | 0.02972 | 1.215% (87/7,162) |
| ID | 长 | 0.02820 | 1.145% (174/15,201) |
| Text | 全部 | 0.01540 | 0.022% (5/22,363) |
| Text | 短 | 0.02093 | 0.014% (1/7,162) |
| Text | 长 | 0.01279 | 0.026% (4/15,201) |
| Equal (alpha=0.5) | 全部 | 0.02428 | 0.470% (105/22,363) |
| Equal (alpha=0.5) | 短 | 0.03030 | 0.614% (44/7,162) |
| Equal (alpha=0.5) | 长 | 0.02144 | 0.401% (61/15,201) |
| Global (alpha=0.9) | 全部 | 0.02995 | 0.868% (194/22,363) |
| Global (alpha=0.9) | 短 | 0.03516 | 1.187% (85/7,162) |
| Global (alpha=0.9) | 长 | 0.02750 | 0.717% (109/15,201) |
| Conditional (alpha=0.9 / 0.9) | 全部 / 短 / 长 | 与 Global 完全相同 | 与 Global 完全相同 |

现有 Recall@10 字段及选择值未变。本次 test cutoff=4，短历史组实际全部为可见长度 4（7,162 人）；长历史组为 5–19 项（15,201 人），这里的 short 不是冷启动或 0–4 项的泛指。Validation 的完整 alpha × short/long Recall@10 邻域表保存在结果 JSON 的 `validation.recall_at_10_by_alpha_by_cohort`；所选 alpha 附近为：

| ID alpha | Validation short Recall@10 | Validation long Recall@10 |
| ---: | ---: | ---: |
| 0.8 | 8.258% | 6.257% |
| 0.9 | 8.557% | 6.721% |
| 1.0 | 7.002% | 6.056% |

两个 cohort 都在 alpha=0.9 达到本次 0.1 网格的最大 Recall@10。该网格诊断不涉及 test 选参。

排名质量指标也显示指标间的取舍：global alpha=0.9 的 overall Recall@10 高于 ID（5.87% vs 5.11%），但 Hit@1 较低（0.868% vs 1.167%）；long cohort 的 NDCG@10 也略低于 ID（0.02750 vs 0.02820）。因此只看 Recall@10 的净命中增加，不能概括首位排序或长历史排名质量。

## 目标商品热度与可见历史长度

目标商品频次按训练 split 中该商品出现次数计算；在全 catalog 频次上取 `1/3` 和 `2/3` 线性分位数，cutpoints 为 5 与 9 次。区间分别为 `<=5`、`(5,9]`、`>9` 次。频次分组仅用于 test 后验诊断，没有参与预测、路由或权重选择。Text rescue 表示 ID 未进 Top10、Text 进 Top10；分母是该格中 ID Top10 miss 数。

| 训练频次组 | 可见历史组 | Text rescue / ID miss | 命中率 |
| --- | --- | ---: | ---: |
| 低频 (0–5) | 短 | 119/1,910 | 6.23% |
| 低频 (0–5) | 长 | 150/4,145 | 3.62% |
| 中频 (6–9) | 短 | 58/1,164 | 4.98% |
| 中频 (6–9) | 长 | 59/2,269 | 2.60% |
| 高频 (10–369) | 短 | 119/3,702 | 3.21% |
| 高频 (10–369) | 长 | 161/8,030 | 2.00% |

各频次组的 short-minus-long rescue 差依次为 **+2.61、+2.38、+1.21 个百分点**。短历史差异在各频次组内仍存在，因此整体差异不能只由这三个粗粒度商品频次组的构成解释；这些分层没有控制其他混杂因素。

按 test 的可见历史精确长度分组：

| 可见长度 | 用户数 | Text rescue / ID miss | 命中率 |
| ---: | ---: | ---: | ---: |
| 4 | 7,162 | 296/6,776 | 4.37% |
| 5 | 4,221 | 127/3,993 | 3.18% |
| 6 | 2,680 | 86/2,551 | 3.37% |
| 7 | 1,811 | 50/1,721 | 2.91% |
| 8 | 1,366 | 29/1,295 | 2.24% |
| 9 | 881 | 19/849 | 2.24% |
| 10–16 | 2,735 | 36/2,617 | 1.38% |
| 17 | 141 | 0/132 | 0.00% |
| 18 | 118 | 3/114 | 2.63% |
| 19 (truncated bucket) | 1,248 | 20/1,172 | 1.71% |

长度 19 表示至少 19 个 target 前事件被截成最近 19 项，不代表用户原序列恰好长 19。长度 1–3 在 test 中没有用户，结果 JSON 将这些空桶保留为零用户、`hit_rate: null`；任何空 ID-miss 分母对应的差值同样为 `null`。

## 逐事件复核与运行范围

默认只输出聚合 JSON/SVG。需要逐事件复核时，可显式运行：

```bash
python scripts/beauty_cpu_proxy.py \
  --data-dir /path/to/EnsRec/data/beauty \
  --output-dir outputs/beauty-cpu-proxy \
  --write-event-diagnostics
```

该选项写入被 `.gitignore` 忽略的 `beauty-cpu-proxy.event-diagnostics.jsonl`，每行包含 test dataset user ID、target item ID、可见历史长度、ID/Text/equal/global/conditional 的 target top-10 rank（未进 Top10 写 `0`），以及 equal/global 相对 ID 的 added/lost flags；不包含原始文本或历史序列。逐事件文件只能用于本地审阅，不应提交或分享。

本次 Beauty 全量离线 CPU 运行读取 12,101 个 catalog item 与 validation/test 各 22,363 个事件，耗时 **154.68 秒**（Python 3.13.3、NumPy 2.4.4、SciPy 1.18.1、scikit-learn 1.9.1、tfrecord 1.14.5）。这是本机端到端离线耗时，不代表在线延迟。原 Recall@10/命中计数、alpha 选择和既有聚合字段均与 `8bb9ea4` 逐字段一致。
