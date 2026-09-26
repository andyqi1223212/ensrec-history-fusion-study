# Beauty 历史物品占位诊断

从独立仓库运行，指定原始 Beauty TFRecord 目录和聚合结果路径：

```bash
python scripts/beauty_seen_item_diagnostic.py --data-dir /path/to/EnsRec/data/beauty --output outputs/beauty-seen-item-diagnostic.json
```

历史长度 cutoff 只按 validation 中位数选择，本次为 4。Test 短历史组（7,162 人）平均有 3.672 个 top-10 槽位被可见历史物品占据，top-1 为历史物品的比例是 97.18%；长历史组（15,201 人）分别为 4.637 个和 91.16%。因此长历史平均多占 0.965 个槽位。短/长组目标在可见历史及完整 pre-target 历史中的重复数均为 0。

Text target hits 与原 proxy 一致：test all/short/long 为 924/404/520；本次验证集为 1,116。输出仅含聚合计数和比例，不含用户、物品明细或原始文本。槽位占用差异只是潜在机制的 post hoc 证据，不能说明它解释了全部 rescue 差异，也不能据此建立因果关系；诊断不改候选评分或过滤策略。
