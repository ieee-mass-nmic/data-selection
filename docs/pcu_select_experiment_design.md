# PCU-Select 实验设计文档

> 配套文档：[研究方案](pcu_select_research_plan.md) · [实现细化版](pcu_select_design.md)
> 本文档只规约 **实验如何做**：实验矩阵、对照、协议、指标、统计口径与排期。不写具体实验代码。
> 方法本身（scorer、多保真效用、site mask、自适应聚类选择、OOD 校准）以实现版文档为准。

---

## 0. 实验总览

本课题需要回答的核心命题：**在固定 backbone family、稳定 PEFT 子空间下，一次离线训练得到的任务条件化 PEFT-aware 数据效用 scorer，能在多个 PEFT 配置上复用，并在"性能"与"多 PEFT 总成本"两个维度上同时占优。**

围绕用户要求的五个实验，映射到研究问题（RQ）如下：

| 实验 | 名称 | 回答的 RQ | 主张类型 | 主证据 |
|---|---|---|---|---|
| **E1** | 不同 PEFT 下，本方法 vs 基线 | 在每个 PEFT 上是否都选出更有效的数据？ | 性能优先 | 下游任务指标 + 排序指标 |
| **E2** | 跨 PEFT 迁移/摊薄成本对比 | 服务多个 PEFT 时，本方法总成本是否更低？ | 成本优先 | GPU-hours 曲线 + break-even |
| **E3** | 消融 | 每个关键模块是否真有贡献？ | 方法论 | 逐模块掉点幅度 |
| **E4** | 同一 PEFT family 下不同配置对比 | PEFT 条件化是否真的捕捉到"配置改变了样本价值"？ | 方法论核心 | 配置间选择差异 + 错配掉点 |
| **E5** | 未见 PEFT 的效果 | 对 unseen 配置 / unseen family 是否泛化（含校准）？ | 泛化边界 | ID/OOD 分层结果 |

**贯穿全文的两条铁律：**

1. **Compute-controlled（同算力对照）**：所有"性能"比较必须在**目标 PEFT 下的训练+评估协议完全一致**的前提下进行（同 backbone、同 PEFT 超参 recipe、同 epoch/step、同 budget、同 seed 集合）。唯一变量是**被选中的数据子集**。
2. **Cost-accounted（全程记账）**：每个阶段（特征、低保真、高保真、scorer 训练、打分、选择、目标训练、评估）都写 `cost/accounting.jsonl`，用于 E2 与所有"性能/成本"图。

---

## 1. 通用实验配置（所有实验共享）

### 1.1 模型（均为 7B 及以上）

主声明限定在**固定 backbone family、family 内不同尺寸**（与实现版 §6.2 一致）。

| 角色 | 主实验（Family A: Llama） | 鲁棒性复现（Family B: Qwen2.5） |
|---|---|---|
| Selector model（抽梯度签名 / 特征） | `Llama-2-7B` | `Qwen2.5-7B` |
| 高保真 anchor + 目标微调 backbone | `Llama-2-7B`、`Llama-2-13B` | `Qwen2.5-7B`、`Qwen2.5-14B` |

说明：
- **主结论在 Family A 上给出**；Family B 仅用于证明结论不是单一 family 的偶然（**不**做跨 family zero-shot 迁移，跨 family 属 future work）。
- Selector 取 family 内最小可用模型（7B），目标微调可放大到 13B/14B，验证"selector 用小模型、目标用大模型"的设计（实现版 §6.1）。这本身是 E3 的一个消融轴（selector 尺寸）。
- 选 Llama-2 而非更新模型，是为了与 LESS / RDS+ / IFD 等基线的公开实现对齐，降低复现争议；Qwen2.5 作为现代模型的鲁棒性补充。
- 若算力紧张，E2/E4/E5 可只在 Family A 上完成；E1/E3 至少各跑一个任务在 Family B 上复现。

### 1.2 候选数据池（meta-pool / candidate pool）

- 规模：**N ≈ 300k**，混合多领域，保证语义冗余与长尾并存（否则数据选择无意义）。
- 组成建议（比例可调，需在论文中固定并公开）：
  - 通用指令：Tulu-v2-mix / Open-Hermes 子集
  - 数学推理：MetaMathQA / GSM8K-train CoT
  - 代码：Magicoder-OSS-Instruct / Evol-Instruct-Code
  - 知识问答：FLAN-v2 子集
  - 多语言：Aya / 多语言指令子集
  - 安全对齐：safety preference / refusal 子集
- **去测试集泄漏**：候选池与所有任务的 test split 做 n-gram + embedding 近重去重。
- 离线元训练用的 **meta-pool** 与在线应用的 **candidate pool** 用同一池（主实验），但高保真标签只采样其中 `Q_H=10k` 个三元组。

### 1.3 任务集合与评估协议

每个任务都需 (a) `train/dev` 抽 validation sketch（默认 32 条，实现版 §9.1），(b) 独立 `test` 用于最终评估。Sketch 与 test 严格隔离。

| 任务 | 能力 | 数据 | 评估指标 | 解码 |
|---|---|---|---|---|
| **GSM8K** | 数学推理 | GSM8K | Exact Match (acc) | greedy, 8-shot→0-shot CoT |
| **MATH**（子集） | 难推理 | MATH | Accuracy | greedy CoT |
| **HumanEval + MBPP** | 代码 | - | Pass@1, Pass@10 | temp=0.2, n=20 |
| **MMLU** | 知识 | - | Accuracy | log-likelihood |
| **TyDiQA-GoldP**（子集） | 多语言 | - | F1 / EM | greedy |
| **AlpacaEval 2 / MT-Bench** | 指令遵循 | - | LC win-rate / GPT-judge | 官方协议 |
| **Safety**（held-out） | 安全保持 | refusal eval | refusal acc / over-refusal | - |

- **主任务**（贯穿 E1–E5）：**GSM8K、HumanEval、MMLU、TyDiQA** 四个，覆盖推理/代码/知识/多语言四类，避免单任务过拟合结论。
- AlpacaEval/MT-Bench、MATH、Safety 作为补充任务，至少在 E1 出现一次。
- GPT-judge 类评估固定评审模型与 prompt 版本，报告版本号。

### 1.4 PEFT 配置空间（全实验统一注册表）

把 PEFT 配置划分为三个集合，所有实验复用这一注册表。`★` 表示进入 scorer 离线训练支持分布（"seen"）。

#### A. 训练支持集（SEEN，scorer 在这些配置上训练）

| ID | family | modules | layers | rank/bottleneck | lr | 备注 |
|---|---|---|---|---|---|---|
| `L-r8-qv` ★ | lora | q,v | all | r=8, α=16 | 2e-4 | LoRA 基准 |
| `L-r16-qkvo` ★ | lora | q,k,v,o | all | r=16, α=32 | 2e-4 | 更宽 attn |
| `L-r8-mlp` ★ | lora | up,down | all | r=8, α=16 | 2e-4 | MLP-only |
| `IA3-attnmlp` ★ | ia3 | attn+ffn | all | - | 5e-4 | IA3 标准 |
| `AD-b64` ★ | adapter | bottleneck | all | b=64 | 3e-4 | Houlsby adapter |

#### B. 同 family 未见配置（UNSEEN-config，用于 E4 / E5a）

| ID | family | modules | layers | rank/bottleneck | lr | 与 SEEN 的差异 |
|---|---|---|---|---|---|---|
| `L-r4-qv` | lora | q,v | all | r=4 | 2e-4 | 极小容量 |
| `L-r32-qkvo` | lora | q,k,v,o | all | r=32 | 2e-4 | 极大容量 |
| `L-r64-all` | lora | all-linear | all | r=64 | 1e-4 | 容量+placement 双变 |
| `L-r8-lowlayers` | lora | q,v | low 1/3 | r=8 | 2e-4 | placement 偏移 |
| `L-r8-highlayers` | lora | q,v | high 1/3 | r=8 | 2e-4 | placement 偏移 |
| `L-r16-hlr` | lora | q,k,v,o | all | r=16 | 5e-4 | recipe（lr）偏移 |
| `AD-b16` / `AD-b256` | adapter | bottleneck | all | b=16 / 256 | 3e-4 | 容量极值 |
| `IA3-attnonly` | ia3 | attn | all | - | 5e-4 | placement 偏移 |

#### C. 未见 family（OOD-family，用于 E5b）

| ID | family | 说明 |
|---|---|---|
| `PRE-l16` | prefix | Prefix tuning, prefix_len=16 |
| `PT-l32` | ptuning | P-Tuning v2, len=32 |
| `BF` | bitfit | bias-only |

> 注：Prefix/P-Tuning/BitFit 不进入主结论，仅作为 OOD 泛化与失败案例（与研究方案 §12.3 一致）。

### 1.5 数据预算

- 默认三档：**budget B ∈ {5%, 10%, 30%}** of N。
- 主表用 **10%**；5%/30% 用于预算敏感性曲线。
- 额外参照点：**100%（full pool）** 上界、**random@budget** 下界。

### 1.6 评估指标三层

1. **下游性能**（最终主张）：§1.3 各任务指标。
2. **选择质量 / 排序指标**（机制证据，在 held-out `(x,p,t)` 三元组上对照高保真真值）：Spearman ρ、Kendall τ、NDCG@K、Top-K hit rate、pairwise ranking acc。
3. **成本**：离线 GPU-h、每个目标 PEFT 应用 GPU-h、总 GPU-h、峰值显存、持久化存储、目标训练节省、break-even T。

### 1.7 统计与可复现口径

- **种子**：每个 (方法 × PEFT × 任务 × budget) 配置跑 **3 个目标微调 seed**，报告 **mean ± std**；主表做配对显著性检验（Wilcoxon signed-rank 或 paired t-test，跨任务/配置配对）。
- **Sketch 方差**：对至少一个任务，用 3 个不同 sketch seed 重抽 sketch，报告 sketch 敏感性（呼应研究方案 §15.6 数据泄漏质疑）。
- **公平性 checklist**（每个 PEFT 固定并公开）：相同 backbone ckpt、相同 PEFT recipe（lr/warmup/scheduler/steps）、相同有效 batch、相同 eval 协议、相同 budget 下的相同训练 step 数（即子集不同但训练 compute 相同）。
- **算力对照变体**：基线分两类报告——(i) *unconstrained*（各基线用其推荐配置）与 (ii) *compute-matched*（限制各方法的"选择阶段" wall-time 到同一预算），后者用于回应"为何不直接用便宜方法"。

---

## 2. E1 — 不同 PEFT 下，本方法 vs 基线

### 2.1 目的

验证 **在每一个 PEFT 配置上**，本方法选出的子集训练后的下游性能都 ≥ 各类基线（不只是跨 PEFT 复用时才有优势）。这是"性能优先"主张的根基。

### 2.2 实验矩阵

```
方法 (≈12) × SEEN PEFT (5: L-r8-qv, L-r16-qkvo, L-r8-mlp, IA3-attnmlp, AD-b64)
          × 任务 (4 主任务) × budget (10% 主, 5%/30% 补) × seed (3)
          × backbone (Llama-2-7B 主; 13B 抽样复现)
```

主表固定 budget=10%、Llama-2-7B，得到 **方法 × PEFT × 任务** 的大表，并给出每个 PEFT 的跨任务均值。

### 2.3 对照基线（分组）

| 组 | 基线 | 备注 |
|---|---|---|
| 下界/上界 | Random、Balanced-Random、Full-pool(100%) | 锚定区间 |
| 启发式 | Length、Loss(high)、Perplexity、IFD | 无需训练动态 |
| 表示类 | Embedding-NN-to-sketch、RDS+、Diversity-only clustering | 只看语义相似/多样 |
| 训练动态类 | **LESS**、Influence/gradient-similarity、S2L | PEFT-specific，是最强对手 |
| **本方法** | PCU-Select（adaptive cluster + uncertainty，默认配置） | - |

要点：
- **LESS / influence 类基线是 per-PEFT 重算的**——对每个 PEFT 配置都要重新计算梯度特征。E1 里它们被允许"重算到最优"（unconstrained），以构成最强性能对手；它们的重算成本在 **E2** 里被清算。
- RDS+/Embedding-NN 的 query 用同一份 validation sketch，保证任务条件输入公平。

### 2.4 成功判据

- 主张 A（强）：本方法在 ≥ 4/5 个 PEFT × 多数任务上 ≥ 最强训练动态基线（LESS），且显著优于所有表示类/启发式基线。
- 主张 B（弱回退）：若个别 PEFT 上略逊于 per-PEFT LESS，则结论收敛为"**同等或接近性能、但显著更低的多 PEFT 总成本**"（交由 E2 支撑）。
- 机制佐证：在 held-out 三元组上，本方法的 NDCG@K / Spearman 高于表示类基线。

### 2.5 失败分析挂钩
- 若被纯 embedding 基线超越 → 检查 scorer 是否退化为语义相似（看 E3 去 site-mask / 去 activation 的消融）。
- 若高保真标签噪声大 → 看 E3 多保真消融与 §1.7 的 seed/sketch 方差。

---

## 3. E2 — 不同方法在多 PEFT 上的迁移/摊薄成本对比

### 3.1 目的

证明方法的核心经济性主张：**离线成本可被多个目标 PEFT 摊薄**，服务的 PEFT 数越多，本方法相对 per-PEFT 方法越省。这是与 LESS/influence 类方法区分的关键。

### 3.2 成本模型（与实现版 §16 对齐）

- 本方法总成本：`C_offline + T · C_apply`
  - `C_offline = C_feat + C_lo + C_hi + C_scorer`（一次性）
  - `C_apply = C_feat-new(可缓存→≈0) + C_score + C_select + C_target-train`
- Per-PEFT 基线总成本：`T · C_specific`
  - 对 LESS：`C_specific = C_grad-feature(per PEFT) + C_select + C_target-train`
  - 对 RDS+/PPL：`C_specific` 较小（只 forward），是"便宜但弱"的对手

### 3.3 实验设置

```
T ∈ {1, 3, 5, 10}  个目标 PEFT（从 SEEN ∪ UNSEEN-config 注册表抽取，覆盖 family/容量/placement）
对每个方法：实测各阶段 GPU-hours（不估算），代入成本模型，画两类曲线：
  (a) 总 GPU-h vs T            —— 找交点 = break-even T*
  (b) 性能(跨 PEFT 均值) vs 总成本  —— Pareto 前沿
```

### 3.4 对比方法

- PCU-Select（含离线一次性成本）
- LESS / influence（每 PEFT 重算梯度特征）
- RDS+ / PPL / IFD（每 PEFT forward 重算，便宜）
- Random（零选择成本基线）

### 3.5 必报产物

1. **成本分解堆叠柱状图**：每个方法的 `C_feat / C_lo / C_hi / C_scorer / C_apply` 拆分。
2. **break-even 曲线**：`T* = C_offline / (C_specific - C_apply)`，标出与 LESS、与 RDS+ 各自的交点。
3. **Pareto 图**：横轴总 GPU-h，纵轴跨 PEFT 平均下游性能；本方法应在右上（多 PEFT 区）占优。
4. 峰值显存 / 持久化存储 / target-train 节省 对照表。

### 3.6 成功判据
- 存在合理的 `T*`（例如 `T* ≤ 5`）使得 `T > T*` 时本方法总成本最低。
- 若 `T*` 过大（离线太贵）→ 回退手段：降 `Q_H`、降 anchor 数、降 horizon（在文中给出 `T*` 对这些超参的敏感性，呼应研究方案 §12.3）。

---

## 4. E3 — 消融实验

### 4.1 目的
逐一拆掉关键模块，证明每个组件都有独立贡献，回应"模块太多/只是 LESS+PEFT 向量"的质疑（研究方案 §15.1/§15.7）。

### 4.2 消融轴（每次只改一个，其余为默认配置）

| # | 消融轴 | 变体 | 验证什么 |
|---|---|---|---|
| A | **PEFT 条件** | 去掉 `z_p`（family one-hot / 全去） | PEFT-conditioning 必要性（最关键） |
| B | **任务条件** | 去掉 `z_t`；sketch size ∈ {0,8,16,32,64} | 任务草图必要性 + size 拐点 |
| C | **多保真** | lo-only / hi-only / lo+hi；hi 预算 ∈ {2k,5k,10k} | 多保真不是冗余复杂度 |
| D | **样本表示** | `e_x` only / +`d_x` / +`a_x`（激活签名） | site 交互需要 activation signature |
| E | **PEFT 表示粒度** | family one-hot → +site mask → +capacity → +recipe → +fingerprint | 结构化编码逐项收益 |
| F | **不确定性** | 去掉 `σ̂` 风险惩罚（λ_unc=0, λ=0） | uncertainty 的部署价值 |
| G | **选择策略** | global top-k / uniform-cluster / adaptive-cluster / (DPP 可选)；α ∈ {0,0.3,0.6,0.9,1.0} | 多样性约束 + α 拐点 |
| H | **训练目标** | 去 rank / 去 reg / 去 proxy 蒸馏（逐项 λ=0） | 各 loss 项贡献 |
| I | **selector 尺寸** | 7B selector → 13B selector（同 family） | "小 selector 够用"假设 |
| J | **时间池化 / horizon** | mean vs last-token；horizon {1} / {4} / {1,4} | 低保真/高保真细节 |

### 4.3 设置
- 主消融在 **GSM8K + HumanEval 两个任务、SEEN PEFT 取 2 个代表（`L-r16-qkvo`、`AD-b64`）、budget=10%** 上做，控制实验量。
- 每个变体报告：下游性能（主）+ held-out 排序指标（NDCG@K，机制）。
- **关键消融 A 与 E** 必须在全部 4 主任务上做（这是核心贡献的直接证据）。

### 4.4 成功判据（与研究方案 §20 的最小验证对齐）
- 去 `z_p`：NDCG@K 掉 ≥ 5%，下游性能可见下降。
- lo-only vs lo+hi：lo+hi 在排序与下游上优于任一单独 fidelity（否则把 hi 降级为可选并明说）。
- adaptive-cluster > global top-k（多样性收益），且存在最优 α。
- E 的每一项（site mask / capacity / recipe）有单调或近单调增益；若 fingerprint 无增益 → 降级为可选模块。

---

## 5. E4 — 同一 PEFT 下不同配置的效果对比

### 5.1 目的（本课题方法论核心）

证明命题"**同一样本在同一 family 的不同配置下价值不同，且本方法能捕捉这种差异**"（研究方案前提 1、2）。这是整套 PEFT-conditioning 立论的直接验证——如果不同配置下最优子集相同，那么 PEFT 条件就是多余的。

### 5.2 配置扫描（固定 LoRA family，逐轴变化）

| 轴 | 取值 | 来自注册表 |
|---|---|---|
| rank（容量） | 4 / 8 / 16 / 32 / 64 | `L-r4-qv` … `L-r64-all` |
| target modules（placement） | qv / qkvo / all-linear | `L-r8-qv` / `L-r16-qkvo` / `L-r64-all` |
| layer range（placement） | low / mid / high / all | `L-r8-lowlayers` / `-highlayers` / `-qv` |
| lr（recipe） | 1e-4 / 2e-4 / 5e-4 | `L-r16-qkvo` / `L-r16-hlr` |

> 这些配置部分在 SEEN、部分在 UNSEEN-config；E4 同时服务于"配置敏感性"与"对 unseen 配置的 ID 内插"两个目的。

### 5.3 三个子实验

**E4-a 选择差异度（机制）**：对每对配置 `(p_i, p_j)`，计算本方法在同一候选池上选出的子集的 **Jaccard 重叠**、Top-K Spearman。预期：配置差异越大（如 rank4 vs rank64、low vs high layers），重叠越低。给出"配置差异 → 选择差异"的相关图。作为对照，给出 RDS+/PPL（**与 PEFT 无关**，重叠恒为 1）以凸显本方法的条件敏感性。

**E4-b 每配置性能（性能）**：在每个配置上，本方法 vs Random vs RDS+ vs LESS 的下游性能。预期本方法在每个配置上都领先，且领先幅度随配置"偏离常规"程度增大。

**E4-c 错配/交叉迁移（决定性证据）**：构造 **错配矩阵**——用为配置 `p_i` 选的子集去训练配置 `p_j`（`i≠j`），与"为 `p_j` 正确条件化选的子集"对比。
- 期望：对角线（正确条件化）> 非对角线（错配），掉点幅度量化"PEFT-conditioning 的价值"。
- 同时给出 PEFT-agnostic 基线（RDS+）的同一矩阵作为对照——它对角线与非对角线无差异，进一步反衬。

### 5.4 成功判据
- E4-a：配置差异与选择差异显著正相关（Spearman > 0.5）。
- E4-c：错配掉点显著（配对检验 p<0.05），且 PCU 的"正确条件化增益"> PEFT-agnostic 基线的随机波动。
- 若配置间选择几乎不变（E4-a 重叠≈1）→ 说明 scorer 未真正用上 `z_p`，回到 E3-A/E3-E 诊断。

---

## 6. E5 — 未见 PEFT 的效果

### 6.1 目的
刻画泛化边界，并验证 OOD 校准模式的有效性（实现版 §13）。分两层，**绝不混为一谈**（研究方案 §15.5）。

### 6.2 分层设置

| 层级 | 测试配置 | scorer 见过？ | 模式 |
|---|---|---|---|
| **L0 ID 内插** | UNSEEN-config 中 rank/placement 落在 SEEN 凸包内（如 `L-r32-qkvo`、`AD-b16`） | family 见过、具体配置没见过 | zero-shot 直接打分 |
| **L1 ID 外推** | 极端配置（`L-r64-all`、`AD-b256`、`L-r8-highlayers`） | family 见过、配置超出训练范围 | zero-shot + 校准对照 |
| **L2 OOD family** | `PRE-l16`、`PT-l32`、`BF`（prefix/ptuning/bitfit） | family 完全没见过 | **必须**走校准模式 |

### 6.3 校准协议（L1/L2）
按实现版 §13.2：
1. Mahalanobis `d²(p*)` 判定是否 OOD（阈值 = SEEN 配置 `d²` 的 95 分位）；报告每个测试配置的 `d²` 与是否触发校准。
2. 触发后：抽 **200 / 500** 条样本，对 `p*` 算少量高保真（horizon=1、单 anchor，≈1/4 成本），冻结 scorer 主体只训 calibration head。
3. 对比三种模式：**zero-shot 直接打分** / **校准(200)** / **校准(500)** / per-PEFT LESS（上界参照）。

### 6.4 实验矩阵
```
测试配置 (L0:2, L1:3, L2:3) × 任务 (GSM8K, HumanEval, MMLU)
× 模式 (zero-shot / cal-200 / cal-500) × seed(3)
对照：Random(下界)、RDS+(PEFT-agnostic)、per-PEFT LESS(上界)
```

### 6.5 成功判据 / 预期形态
- **L0**：zero-shot 即接近 per-PEFT LESS，明显优于 Random/RDS+ → 证明 `z_p` 支持 ID 内插。
- **L1**：zero-shot 有衰减，cal-500 基本补回 → 证明校准模式有效且廉价。
- **L2**：zero-shot 可能失败（**允许并如实报告为失败案例**），cal-500 应显著优于 Random；若仍失败，明确写入"不适用边界"（研究方案 §16.2）。
- 必报 **`d²` vs 性能衰减 散点**：验证 Mahalanobis 距离能预测何时需要校准（OOD 判定器本身的有效性）。

---

## 7. 统一实现 / 公平性说明（写作时需固定并公开）

1. **目标微调协议**：所有方法用同一 PEFT recipe 与同一训练 step 数；budget 改变的是"数据子集"，不改变 compute（必要时按 step 截断，保证 compute-matched）。
2. **基线移植**：LESS / RDS+ / IFD / S2L 使用官方实现或经核对的复现；对每个 PEFT 重新计算其所需信号（这正是 E2 要清算的成本）。
3. **sketch 即 query**：表示类基线（RDS+/Embedding-NN）与本方法共用同一 validation sketch 作为任务 query，杜绝"任务信息不对等"。
4. **去泄漏**：sketch ⟂ test；候选池 ⟂ test。报告去重统计。
5. **随机性**：固定 global_seed；目标微调 3 seed；至少一个任务做 3 sketch-seed。
6. **记账**：所有阶段写 `cost/accounting.jsonl`（实现版 §16.1 字段），E2 直接读取，不允许事后估算。

---

## 8. 计算资源与排期估算（粗略，用于排产）

> 数量级估算，按 8×A100-80G 计；真实值以 `accounting.jsonl` 为准。

| 阶段 | 主要成本来源 | 量级 | 备注 |
|---|---|---|---|
| 特征提取（C_feat） | N=300k 一次 forward + 激活签名 | 低 | 一次性、可缓存 |
| 低保真（C_lo） | N×forward+backward 站点签名 | 中 | 一次性、跨 PEFT 复用 |
| 高保真（C_hi） | Q_H=10k × A=2 × horizon{1,4} 短更新 | **中高** | 离线主成本，E2 重点 |
| Scorer 训练 | 1.5M 参数小网络 | 极低 | - |
| 每目标 PEFT 应用 | 打分(≈0) + 选择 + 目标微调(10% 数据) | 由目标训练主导 | 随 T 线性 |
| 目标微调评估 | 各任务 inference | 中 | Pass@k/解码占大头 |

排产建议：
- **先跑最小闭环**（研究方案 §20 / 实现版 §20 三实验）确认 `u^lo↔u^hi` 相关、`z_p` 有效、同算力胜过基线，再铺开 E1–E5。
- 优先级：**E1 → E4 → E3 → E2 → E5**（先立性能与机制，再立成本，最后立泛化边界）。Family B 复现放最后。

---

## 9. 结果产物清单（论文图表模板）

| 编号 | 形式 | 内容 |
|---|---|---|
| T1 | 大表 | E1：方法 × PEFT × 任务（budget=10%, 7B），含均值与显著性 |
| F1 | 折线 | E1：budget(5/10/30%) 敏感性 |
| F2 | 折线+交点 | E2：总 GPU-h vs T，break-even |
| F3 | 散点 | E2：性能 vs 总成本 Pareto |
| T2 | 消融表 | E3：各轴逐项掉点（性能 + NDCG@K） |
| F4 | 折线 | E3-B/E3-G：sketch size / α 拐点 |
| F5 | 热图 | E4-c：错配矩阵（对角 vs 非对角） |
| F6 | 散点 | E4-a：配置差异 vs 选择差异 |
| F7 | 分组柱 | E5：L0/L1/L2 × (zero-shot/cal-200/cal-500) |
| F8 | 散点 | E5：Mahalanobis d² vs 性能衰减 |
| T3 | 表 | 成本/显存/存储 明细 + 公平性 checklist |

---

## 10. 风险与回退矩阵

| 风险 | 触发信号 | 回退 |
|---|---|---|
| 单 PEFT 上输给 embedding 基线 | E1 主表掉队 | 主张转为"成本优先"，靠 E2;同时查 E3-D/E |
| 离线成本摊不薄 | E2 的 T* 过大 | 降 Q_H/anchor/horizon，报敏感性 |
| 配置间选择无差异 | E4-a 重叠≈1 | 诊断 z_p 未生效（E3-A/E），可能需更强 site/activation 表示 |
| OOD family 失败 | E5-L2 zero-shot 崩 | 限定主张到 fixed-family，L2 列为失败案例 + 校准兜底 |
| 高保真标签噪声大 | seed/sketch 方差大、u_lo↔u_hi 相关弱 | 加 anchor/seed 于高不确定区，RankNorm 复核 |
| 评估泄漏质疑 | 审稿质疑 sketch | 公开去重与 sketch 协议，报告 sketch-seed 方差 |

---

## 附：与研究方案 §12 的对应关系

本文档是研究方案 §12「实验设计」的**收敛与重组**，对应关系：

- 研究方案 实验1（单 PEFT 选择）→ 本文 **E1**
- 研究方案 实验2（跨 PEFT 复用）→ 本文 **E5**（泛化）+ **E4**（配置敏感）
- 研究方案 实验3（多 PEFT 总成本）→ 本文 **E2**
- 研究方案 实验4/5/6/7（多保真/条件表示/任务草图/选择策略消融）→ 本文 **E3** 的 C/E、B、G 轴
- 新增贡献：**E4 错配矩阵**（PEFT-conditioning 的决定性证据）、**E5 的 ID/OOD 三层分解 + d² 有效性**、所有实验统一的 compute-matched 与记账口径。
</content>
</invoke>
