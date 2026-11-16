# 2026-11-16 实验归档

[三步实验结果与结论](../../docs/三步实验结果与结论.md)汇总本轮结果、参数、置信区间和检验。

| 文件或目录 | 内容 |
|---|---|
| `01_texts_50/` | 最终50条原创、200条评论，生成调用与校验 |
| `00_texts_original/`、`01_texts/` | 原始10条与中间20条原创候选库 |
| `02_initialization_50/` | 最终候选库上的96组筛选、8组复测、真实LLM验证及邻域检查 |
| `02_initialization/` | 早期候选库上的预试记录 |
| `03_positions/` | 3张冻结网络、9组个人参数与位置、27条10轮轨迹、完整原始调用日志 |
| `03_positions/prefix8/` | 恢复运行前的8轮快照；主结果使用完整10轮 |
| `figures/` | 位置置信区间、热力图、10轮时间变化 |
| `parameters_final.csv` | 完整初始化与依据 |
| `run_summary.csv`、`position_comparisons.csv`、`time_comparisons.csv` | 每组结果及统计比较 |
| `followup_initialization.json`、`recommended_positions.json` | 后续TH初始化及9组具体节点名单 |
| `source_snapshots/`、`*.executed10.py`、执行清单 | 实际运行源码和哈希快照 |
| `archive_manifest.json` | 归档文件SHA-256清单 |

原始`.jsonl`使用gzip无损压缩为`.jsonl.gz`。原有压缩轨迹直接保存；调用记录包含真实响应、重试与缓存复用标记。归档源码和原始元数据保留运行时路径；可执行入口位于仓库`scripts/`，可移植配置位于`configs/initialized_20261116.json`。

在仓库根目录安装并验证：

```powershell
python -m pip install -e ".[experiments]"
python -m unittest discover -s tests -v
python -m scripts.verify_results
python -m scripts.summarize_study
```

汇总与校验无需API key。新实验先参照根目录`.env.example`在本地设置密钥，然后运行：

```powershell
python -m scripts.majority_study positions --output runs/positions_new
```

该命令使用冻结初始化和最终文本库，在新目录生成网络、配对位置与10轮真实LLM轨迹；仅观察节点调用感知。若要重做N=100初始化，在同一新目录依次运行`screen`、`confirm`、`controls`、`refine`、`freeze`阶段。`texts50`可基于中间20条原创重新扩充候选库。
