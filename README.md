当前模型按 PDF 前五页组织，参数与运行说明见 [MODEL_GUIDE.md](MODEL_GUIDE.md)，学习率默认 alpha=0.05。

# Who is in the majority?

## 2026-11-16 三步实验

已完成50条原创/200条评论、N=100初始化搜索，以及N=1000、50名观察者、27组10轮的连接位置实验。TH末3轮误判均值46.5%，选作后续位置初始化；TH−TL为+7.5个百分点，95% CI [+2.4,+13.2]，Holm p=0.1172。

- [三步实验结果与结论（Markdown）](docs/三步实验结果与结论.md)
- [完整实验归档与复现](experiments/2026-11-16/README.md)
- [选定初始化](configs/initialized_20261116.json)

## 2026-11-20 三项拓展实验

三项拓展已经在 `gxy` 分支完成并归档。反馈机制、网络环境和个体抵抗力分别使用 `scripts/ext_feedback.py`、`scripts/ext_network.py` 和 `scripts/ext_resistance.py`，每项都提供 `prepare`、`run`、`analyze` 阶段，个体抵抗力另有 `conversions` 阶段。

- 共 108 条真实 API 轨迹，全部 10 轮完成，无缺失或插补。
- 三项协议记录冻结输入和各自的运行器 SHA-256。准备阶段验证既有协议，运行阶段要求与该协议指定的代码版本一致。
- `python -m unittest discover -s tests -v` 验证 47 项测试。
- `python -m scripts.verify_results` 验证主实验一的 203 个归档文件。
- 三个拓展目录的 `archive_manifest.json` 均可独立校验。

完整结果、统计口径与限制见 [拓展实验方案与结果](docs/拓展实验方案与结果.md)，实施步骤见 [三项拓展实验计划](docs/superpowers/plans/2026-11-20-three-extension-experiments.md)。

安装实验依赖使用`python -m pip install -e ".[experiments]"`；核对保存结果使用`python -m scripts.verify_results`，重建图表与汇总使用`python -m scripts.summarize_study`。API配置见`.env.example`，实际密钥只在本地提供。

以下为基础框架使用说明。

多数幻觉与公开表达实验。数据结构、轮次语义和扩展边界见 [DESIGN.md](DESIGN.md)。

当前设定写在 `configs/study.json`：**2024 年美国总统大选投票前，少数倡议者反对特朗普当选，其余节点初始支持**；私人立场在主实验中固定。LLM 输出始终是“支持特朗普当选的人数比例”，多数幻觉统计检查是否误认为反对者占多数。

## 使用

Python 3.11+，在项目目录安装依赖：

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

本地 `.env` 保存 API 配置，已被 Git 忽略，模板见 `.env.example`。环境变量优先于文件。`DEEPSEEK_BASE_URL=http://127.0.0.1:3000/v1`，SDK 会追加 `/chat/completions`；请求模型为 `deepseek-v4-flash`。API key 不写入代码、运行清单或日志。

```powershell
# 生成 3 张正式网络：各 100 个节点、30 名固定观测者
python -m majority network --output runs/networks_new

# 双方原创各 5 条，每条双方评论各 2 条
python -m majority texts --per-side 5 --comments 2 --output runs/new_pool

# 用冻结候选池发出一次真实感知请求
python -m majority check-llm --bank runs/candidate_pool_final/bank.json --output runs/check_new

# 汇总成功、失败、无效响应、已知 token 用量和调用耗时
python -m majority summarize runs/candidate_pool_final/calls.jsonl
```

输出目录/文件必须不存在，防止覆盖已冻结材料。单次运行失败会保留调用日志，不静默重试或把异常改写成支持率 0.5。未返回 usage 时记录缺失；汇总 token 总数只包含已知用量。

本次生成文件：

- `runs/networks/network-0.json` 至 `network-2.json`：网络、生成配置、种子及结构统计。
- `runs/candidate_pool_final/bank.json`：10 条原创、40 条评论，含编号、引用关系和提示词哈希。
- `runs/candidate_pool_final/candidate_pool.md`：便于阅读的 50 条文本。
- `runs/candidate_pool_final/calls.jsonl` 和 `calls.summary.json`：生成过程原始响应与用量。
- `runs/candidate_pool_final/prompts.json`：生成时使用的提示词正文与哈希快照。
- `runs/api_check_final/`：使用最终候选池的真实感知检查，实际输入另存于调用日志。

网关在验证中接受了 `deepseek-v4-flash` 请求，响应 `model` 字段为 `deepseek-flash`。代码保留请求名与返回名，不推断底层版本。

## 运行单分支

先从网络文件选择未进入 `observers` 的少数节点作为倡议者，再显式提供系数：

```powershell
python -m majority run --network runs/networks/network-0.json --bank runs/candidate_pool_final/bank.json --coefficients configs/coefficients.demo.json --advocates "填入非观测节点编号，以逗号分隔" --rounds 20 --rho 3 --output runs/run_new
```

`coefficients.demo.json` 只用于验证流程，正式实验需替换为预试拟合结果。此命令每轮对实际读到消息的节点调用 LLM；空阅读沿用旧判断。它不自动完成种子配对或预试，手选种子不代表高/低幻觉配置。成组运行可使用 `plan_group`/`run_group`，为各分支提供独立的 `LLMPerception` 实例。

输出 `trajectory.jsonl` 包含冻结输入、消息/受众、阅读/引用、通知、状态、行为概率、感知原始输出、用量、研究指标及两套判断汇总（固定观测集合与全体节点）。`published` 属于当前轮，`next_decision` 属于下一轮，公开行为指标只使用已发布行为。

## 提示词与统计口径

`majority/prompts/` 下三个提示词分别负责原创、评论和感知。生成温度 0.7，感知温度 0。完整历史保留，不隐式摘要或截断；上下文超限时调用失败并留痕。感知输入不含分支名、真实支持率、倡议者名单、受众或全网结构，消息编号转成不含分支信息的别名。

候选生成校验数量、40–80 字长度、重复和结构；感知校验支持率范围、简短依据和已见证据编号。文本立场与历史场景须语义审阅后冻结。各次调用保存提示词哈希、请求条件及原始响应；温度 0 不保证远程服务每次返回相同结果。

| 字段 | 当前反对派倡议者设定下的含义 |
| --- | --- |
| `support` / `mean_support` | 对支持特朗普当选的人数比例的判断 |
| `true_support` | 实际私人支持率，固定为 1−k/N |
| `feed_majority` | 观测者中读到反对消息占多数的比例 J |
| `perceived_majority` | 观测者中判断支持率 < 0.5 的比例 Qill |
| `perception_bias` | 感知支持率减真实支持率 |
| `advocate_perception_bias` | 感知反对率减真实反对率，倡议立场的 B |
| `public_support` | 公开支持特朗普的观测者比例 |
| `public_advocate_stance` | 公开反对特朗普的观测者比例 |
| `opponent_silence` | 私人立场与倡议者相反的人中，沉默者比例 |
| `public_conformity` | 私人支持特朗普的人中，公开反对者比例 |

空阅读保留在观测分母中且 J 计为 0；沿用感知照常进入 Qill，另统计 `carried_forward`。恰好 0.5 不算任一方多数。描述性判断分布与误差不等于成组实验差异的置信区间。

## 当前仍待实现

三项拓展所需的种子配对、成组 bootstrap/Holm、绘图和私人态度更新已经由拓展脚本落地。基础包仍保留明确边界：`majority.preparation.calibrate()` 尚未实现，基础 CLI 也没有把预试拟合、正式预筛、k 网格补测和推荐曝光编排成一条通用流水线。拓展实验使用冻结协议和独立入口，不把这些未实现的基础接口宣称为已完成。

兼容接口参考：[OpenAI Chat Completions](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)。
