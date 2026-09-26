# Beauty 排序与候选策略敏感性

这是首轮 Beauty CPU proxy 之后两项预先冻结规则的敏感性对照，不替换主结果，也不根据 test 挑胜者。两项保持训练序列、ID 转移表、catalog TF-IDF、候选 inventory（B 按策略过滤）、可见历史上限 19、validation median cutoff=4、alpha 网格不变；alpha 选择的同分规则与最终融合分数同分规则也不变。A 只改变 Text 百分位的同分处理。每项都只用 validation 选择全局和短/长 alpha，test 只用于报告。完整 all/short/long 指标分别存于独立 JSON。

## A. Text ties 改为 average rank

原策略对 Text 相似度并列项按商品 ID 排序；本对照只将 Text ties 改为 `scipy.stats.rankdata(method="average")` 转换 percentile。ID 排序和其他输入均不变。全局、validation short、validation long 都仍选出 alpha=0.9。

短历史 Text rescue 仍为 296/6,776（4.37%），长历史 370/14,444（2.56%），短减长 +1.81 pp（user bootstrap 95% CI +1.25 至 +2.35 pp）。Text-only Top10 命中不变是预期结果：average rank 是单调变换，而最终 Text-only 排序仍由商品 ID 解同分。因此 rescue 不变不能视为独立的 tie-policy 稳健性证据。此对照真正检验的是 average percentile 会否影响融合：本次 Recall@10/Hit@1 未变，Global NDCG@10 也相同；Equal NDCG@10 仅有极小变化：全部 0.0242814→0.0242798，长组 0.0214443→0.0214419，短组不变。该结论限于这项固定平均秩政策。

## B. 所有方法排除可见历史商品

对每个 query，ID、Text、Equal、Global 使用同一个 eligible mask：从候选排序中移除该 query 的可见历史商品，之后在剩余商品中各自计算 percentile。Text 历史 profile 仍使用原始可见历史。每个 evaluation event 都保留；若 target 本身在可见历史中就记为 miss。本次 validation/test target 均未在对应可见历史出现（各 0/22,363），所以没有标签被过滤。test 仍为 22,363 个事件。Catalog inventory 为 12,101 项；单条 test query 的 eligible 数为 12,082–12,097，均值 12,093.90。

Validation 的全局及 short/long alpha 仍为 0.9。各策略 Recall@10 如下，括号内为命中数：

| 分组 | ID | Text | Equal | Global |
| --- | ---: | ---: | ---: | ---: |
| 全部 (n=22,363) | 5.78% (1,293) | 5.71% (1,276) | 6.75% (1,509) | 7.28% (1,628) |
| 短 (n=7,162) | 5.65% (405) | 7.19% (515) | 6.97% (499) | 7.16% (513) |
| 长 (n=15,201) | 5.84% (888) | 5.01% (761) | 6.64% (1,010) | 7.33% (1,115) |

短/长 Text rescue 分别为 382/6,757（5.65%）和 536/14,313（3.74%），短减长 **+1.91 pp**（user bootstrap 95% CI +1.29 至 +2.52 pp）。Global 相对 ID 新增 653、丢失 318、净增 335 个命中；短组净增 108，长组净增 227。

| 方法 | 分组 | NDCG@10 | Hit@1 |
| --- | --- | ---: | ---: |
| ID | 全部 / 短 / 长 | 0.03499 / 0.03227 / 0.03627 | 1.780% / 1.452% / 1.934% |
| Text | 全部 / 短 / 长 | 0.03050 / 0.03941 / 0.02630 | 1.105% / 1.466% / 0.934% |
| Equal | 全部 / 短 / 长 | 0.03963 / 0.04276 / 0.03816 | 1.824% / 2.150% / 1.671% |
| Global | 全部 / 短 / 长 | 0.04333 / 0.04331 / 0.04334 | 2.070% / 2.039% / 2.085% |

排除可见历史后，短历史 rescue 差仍为正；但这是改变候选政策后的另一种评估任务，不能与原候选全集主结果混为同一指标，也不能据此宣称线上收益。

## 运行与产物

Average-rank 和 unseen-candidate 两次运行分别耗时 153.93 秒、146.33 秒。本机端到端离线 CPU 耗时不代表在线延迟或部署成本。主结果 JSON 保留原候选全集、item-ID tie policy 数值；两项策略的完整聚合指标分别保存在 [average-ties JSON](../results/beauty-cpu-proxy-average-ties-results.json) 和 [unseen-candidates JSON](../results/beauty-cpu-proxy-unseen-candidates-results.json)。中文主图仍展示基线结果，并标出“按商品 ID”与“不排除可见历史候选”的政策。
