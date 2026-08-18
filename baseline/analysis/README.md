# Baseline 分析复现实验操作手册

本目录对已经完成的 baseline reproduction 做离线分析和固定对手评估。它不重新训练模型，也不改变训练协议。本文只固定分析协议、操作方法、输入输出和验收条件，不记录任何具体实验得分。

以下命令均假定当前工作目录为 `baseline/`：

```powershell
cd I:\dissertation\src\baseline
```

应使用运行 baseline reproduction 时相同的 Python 环境。JavaScript heuristic 对手还要求本机可执行 `node`。

## 分析范围

当前包含三类分析：

1. **训练效率与 replay 派生指标**：从逐轮训练指标计算生成效率、GPU 时间占比、buffer 流入/消耗、样本暴露、淘汰量、turnover 和样本年龄。
2. **最终 replay 快照分析**：分析第 210 轮保存的 150 轮 rolling window，包括 canonical state 多样性、重复率、state-effective count、每轮游戏数量和长度分布，以及按 `step` 重置恢复的游戏边界。
3. **固定篮子评估**：让指定 checkpoint 对阵相同的四类固定对手，使用固定 seed、固定颜色安排和固定 150 回合上限，生成逐局记录、checkpoint 汇总、对手分层汇总和 provisional Elo。

本目录目前**不生成** hold-out 评估、plateau 判定结果或论文图片。`baseline_gate2.yaml` 中这些路径是后续分析的预留接口，不能据此认为相应实验已经运行。

## 配置权威性

| 配置 | 负责内容 |
|---|---|
| `analysis/configs/baseline_gate2.yaml` | baseline run ID、replay/state hash/派生指标定义、输入来源和分析输出路径登记 |
| `analysis/configs/fixed_basket_v1.yaml` | 当前固定篮子的唯一权威协议：checkpoint、对手、局数、颜色、MCTS、温度和 seed |

固定篮子脚本不会读取 `baseline_gate2.yaml` 中的 `fixed_basket` 段。若两份配置中的固定篮子字段不同，以 `fixed_basket_v1.yaml` 为准。

state hash 固定定义为：

```text
SHA256(dtype + shape + contiguous canonical-board bytes)
```

不做额外 symmetry reduction，也不把原始状态或状态哈希写入 replay 汇总输出。

## 输入来源

默认分析对象是：

```text
outputs/baseline_reproduction_seed1001_4090/
```

| 输入 | 来源 | 使用者 |
|---|---|---|
| `metrics.jsonl` | baseline 每轮训练记录 | `derive_baseline_metrics.py`、`summarize_replay.py` |
| `resolved_config.yaml` | baseline 实际解析配置 | `derive_baseline_metrics.py`、`summarize_replay.py` |
| `run_metadata.json` | run ID、模型配置和 checkpoint 来源 | `evaluate_fixed_basket.py` |
| `checkpoints/latest.examples` | 第 210 轮 rolling replay 快照 | `summarize_replay.py` |
| 初始 checkpoint | `run_metadata.json` 记录的预训练/初始 checkpoint 路径 | `evaluate_fixed_basket.py` 的 checkpoint 0 |
| checkpoint 20–210 | baseline run 的 checkpoint manifest/目录 | `evaluate_fixed_basket.py` |

评估脚本在开始比赛前解析 checkpoint 路径并记录 SHA-256。checkpoint 0 不假定存在于 baseline checkpoint 目录中；其路径来自 `run_metadata.json`。

## 是否修改原始结果

这些脚本不会修改 `metrics.jsonl`、`resolved_config.yaml`、`run_metadata.json`、`latest.examples` 或 checkpoint 内容。

- `derive_baseline_metrics.py` 默认在原 run 下新增或原子更新 `gate2/` 中自己的三个派生产物，但不覆盖任何原始训练文件。
- replay 和 fixed-basket 产物写入独立的 `outputs/baseline_seed1001_4090_analysis/`。
- fixed-basket 每局完成后只向分析目录的 `games.jsonl` 追加并立即刷新。
- `--resume` 读取既有逐局记录并跳过已经完成的稳定比赛键，不会重复比赛。
- `--verify-source-integrity` 比较评估前后的源 run 文件清单；任何变化都会使评估失败。
- 既有 random + greedy sanity evaluation 不属于本协议，也不会被这些脚本覆盖。

重复运行汇总脚本会更新其自身的 CSV/manifest 汇总。正式逐局结果已经存在时，评估脚本要求显式传入 `--resume`。

## 1. 派生 baseline 指标

运行：

```powershell
python analysis/scripts/derive_baseline_metrics.py `
  --metrics outputs/baseline_reproduction_seed1001_4090/metrics.jsonl `
  --resolved-config outputs/baseline_reproduction_seed1001_4090/resolved_config.yaml `
  --output-dir outputs/baseline_reproduction_seed1001_4090/gate2
```

输出：

```text
outputs/baseline_reproduction_seed1001_4090/gate2/
|-- derived_metrics.csv
|-- baseline_resource_summary.json
`-- data_quality_report.json
```

`derived_metrics.csv` 每轮至少包含：

- `fresh_states_per_update`
- `states_per_gpu_hour`
- `games_per_gpu_hour`
- `self_play_fraction`
- `training_fraction`
- `buffer_inflow_fraction`
- `buffer_fraction_consumed`
- `mean_sample_exposure`
- `selected_sample_reuse`
- `evicted_states`
- `turnover_fraction`
- `mean_sample_age`
- `median_sample_age`
- `p90_sample_age`

成功条件：进程退出码为 0，`data_quality_report.json` 的 `status` 为 `passed`，210 轮全部通过，四项 `checks` 均为 `true`，且所有派生字段完整、有限。最近 150 轮 `positions_generated` 重建出的 replay buffer 必须与每轮记录的 `replay_buffer_size` 完全一致。

## 2. 汇总最终 replay

默认路径已经与正式 baseline run 对齐，因此可直接运行：

```powershell
python analysis/scripts/summarize_replay.py
```

等价的显式命令为：

```powershell
python analysis/scripts/summarize_replay.py `
  --replay outputs/baseline_reproduction_seed1001_4090/checkpoints/latest.examples `
  --metrics outputs/baseline_reproduction_seed1001_4090/metrics.jsonl `
  --resolved-config outputs/baseline_reproduction_seed1001_4090/resolved_config.yaml `
  --output-dir outputs/baseline_seed1001_4090_analysis/replay `
  --expected-iteration 210 `
  --expected-history-buckets 150 `
  --expected-total-states 284234
```

输出：

```text
outputs/baseline_seed1001_4090_analysis/replay/
|-- replay_iteration_stats.csv
|-- replay_final_summary.json
`-- trajectory_stats.csv
```

成功时终端必须显示：

```text
Replay iteration: 210
History buckets: 150
Recovered range: 61-210
Total states: 284234
Empty buckets: 0
Count matches metrics: yes
Output status: completed
```

此外，`replay_final_summary.json` 中的所有 `validations` 必须为 `true`。快照只保留第 61–210 轮，因此不能恢复第 1–60 轮的 buffer-level diversity，也不能恢复训练时每个状态实际被抽取的次数。

## 3. 固定篮子预检与 pilot

仅解析协议、checkpoint 和 JS determinism，不比赛：

```powershell
python analysis/scripts/evaluate_fixed_basket.py `
  --mode pilot `
  --checkpoints 0 210 `
  --games-per-opponent 2 `
  --prepare-only
```

运行 16 局 pilot：

```powershell
python analysis/scripts/evaluate_fixed_basket.py `
  --mode pilot `
  --checkpoints 0 210 `
  --games-per-opponent 2 `
  --verify-source-integrity

python analysis/scripts/summarize_fixed_basket.py --mode pilot
```

pilot 输出目录：

```text
outputs/baseline_seed1001_4090_analysis/fixed_basket_v1_pilot/
|-- protocol.resolved.yaml
|-- evaluation_manifest.json
|-- games.jsonl
|-- checkpoint_summary.csv
`-- evaluation.log
```

pilot 成功条件：checkpoint 0 和 210 均能加载；四类对手各有模型执白和执黑一局；JS 进程正常启动和退出；`games.jsonl` 恰好 16 行；`invalid_move=0`、`bot_error=0`；第 6 个模型动作开始温度为 0；源 run 不变；再次使用 `--resume` 时不重复已有比赛。

## 4. 运行完整固定篮子评估

首次运行：

```powershell
python analysis/scripts/evaluate_fixed_basket.py `
  --mode formal `
  --verify-source-integrity
```

中断后继续：

```powershell
python analysis/scripts/evaluate_fixed_basket.py `
  --mode formal `
  --resume `
  --verify-source-integrity
```

评估协议固定为 12 个 checkpoint、4 个对手、每个 checkpoint–opponent 50 局，共 2400 局。每组前 25 局模型执白、后 25 局模型执黑；`max_turns=150`。每局 seed 由 protocol ID、base seed、checkpoint、opponent ID 和 game index 稳定生成。

只在确认某条记录需要重跑时使用以下恢复选项：

```powershell
# 重跑所有指定错误终止类型
python analysis/scripts/evaluate_fixed_basket.py --mode formal --resume --retry-termination bot_error

# 重跑一个稳定比赛键
python analysis/scripts/evaluate_fixed_basket.py --mode formal --resume --retry-game 210:heuristic_200:2
```

正式输出目录：

```text
outputs/baseline_seed1001_4090_analysis/fixed_basket_v1/
|-- protocol.resolved.yaml
|-- evaluation_manifest.json
|-- games.jsonl
|-- checkpoint_summary.csv
|-- opponent_summary.csv
|-- elo_summary.csv
`-- evaluation.log
```

评估完成后运行汇总：

```powershell
python analysis/scripts/summarize_fixed_basket.py --mode formal
```

正式评估成功条件：

- `evaluation_manifest.json` 的 `status` 为 `completed`；
- 12 个 checkpoint 和 4 个对手全部覆盖；
- 48 个 checkpoint–opponent 分组各 50 局，且模型执白/执黑为 25/25；
- `games.jsonl` 有 2400 个唯一稳定比赛键；
- `invalid_move=0`、`bot_error=0`、`fault=0`；
- 达到回合上限的比赛允许存在，但必须记录为 `termination=max_turns` 并单独汇总；
- 每局温度历史符合前 5 个模型动作 0.18、从第 6 个模型动作起 0；
- 使用完整性检查时，`source_integrity.status=passed` 且 `changed_paths` 为空；
- 汇总终端显示 `Observed games: 2400`、`Expected games: 2400` 和 `Summary status: completed`；
- `checkpoint_summary.csv` 为 12 行，`opponent_summary.csv` 为 48 行；
- score rate 满足 `(wins + 0.5 * draws) / games`，固定篮子总分等于四种对手得分的等权平均；
- 95% 置信区间按“对手 × 模型执棋颜色”分层 bootstrap；
- 当前 Elo 必须标记为 `provisional` 并记录固定随机 seed。Adaptive 完成后，应将 Baseline checkpoint、Adaptive checkpoint 和固定对手放入同一个最终 Elo 拟合中。

## 测试

修改分析代码后运行：

```powershell
python -m pytest analysis/tests -q
```

测试失败时不要发布或引用新汇总；先修复对应脚本，再重新生成受影响的分析产物。

## 论文图表映射

仓库当前没有固定论文图号。论文排版时应按下表取数，不要从终端日志或手工复制的中间值作图。

| 论文内容 | 权威数据源 | 建议用途 |
|---|---|---|
| 训练资源与效率表 | `gate2/baseline_resource_summary.json` | 总 GPU hours、生成/训练占比、aggregate throughput |
| 逐轮效率曲线 | `gate2/derived_metrics.csv` | states/games per GPU hour、fresh states/update、buffer 消耗与样本暴露 |
| replay 动态图 | `gate2/derived_metrics.csv` | turnover、eviction、mean/median/p90 sample age 随 iteration 变化 |
| 最终 replay 多样性表或图 | `replay/replay_iteration_stats.csv`、`replay/replay_final_summary.json` | incoming unique ratio、duplicate rate、state-effective count、最终 unique ratio |
| 自博弈轨迹长度分布 | `replay/trajectory_stats.csv` | 每轮游戏数、长度分布和轨迹边界质量检查 |
| Baseline checkpoint 学习曲线 | `fixed_basket_v1/checkpoint_summary.csv` | 固定篮子 score rate 与分层 bootstrap 置信区间 |
| 分对手性能图或表 | `fixed_basket_v1/opponent_summary.csv` | checkpoint × opponent 的分层表现和颜色平衡 |
| Elo 附录或诊断图 | `fixed_basket_v1/elo_summary.csv` | 仅用于 provisional baseline Elo；最终论文 Elo 使用联合拟合结果 |
| 方法与可复现性附录 | `evaluation_manifest.json`、`protocol.resolved.yaml` | seed、checkpoint 来源、协议、完整性状态和数据质量说明 |
| 逐局审计 | `fixed_basket_v1/games.jsonl` | 对异常终止、回合上限、颜色和比赛 seed 做追溯，不直接作为论文主表 |

任何论文图表都应保留其源 CSV/JSON 路径、筛选规则和统计方法。图中使用的 fixed-basket score 必须来自 `checkpoint_summary.csv`；不能与既有 random + greedy sanity evaluation 混用。
