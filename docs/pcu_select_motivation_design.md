# PCU-Select 动机实验设计文档

> 配套文档：[研究方案](pcu_select_research_plan.md) · [实现细化版](pcu_select_design.md) · [主实验设计](pcu_select_experiment_design.md)
> 本文档只规约 **课题动机如何被实验证成**：在方法（scorer / 多保真 / 自适应选择）尚未介入之前，用便宜且**与本方法无关**的信号，证明「PEFT-conditioned 数据选择」这一立论的前提是成立的。
> 产出三件：**Table 1**（PEFT 改的位置与参数量不同）、**Figure 1**（数据价值排序的不一致）、**Figure 2**（跨 PEFT 迁移矩阵的对角占优）。

---

## 0. 动机实验要回答什么

研究方案 §1.2 把整个课题挂在四个前提上，其中**最根本、也最容易被审稿人一句话否掉**的是：

> **前提 1：同一样本在不同 PEFT 下的价值并不相同。**

如果这个前提不成立——即不同 PEFT 下「该选哪些数据」几乎一样——那么 PEFT-conditioning（`z_p`）、跨 PEFT 复用、错配矩阵（E4-c）这一整条主线全部失去意义，方法退化为一个普通的 task-only 数据选择器。因此动机实验的唯一任务是：

| 产出 | 论证的命题 | 反方假设（要被排除的） |
|---|---|---|
| **Table 1** | PEFT 在**结构上**就不同：改的层/模块/算子/容量不同 → `z_p` 有可编码的结构 | 「PEFT 之间只是名字不同」 |
| **Figure 1** | PEFT 在**数据价值上**不同：逐样本价值排序随 PEFT 显著分歧，且分歧**超过噪声地板** | 「排序看起来不同只是因为信号有噪声」 |
| **Figure 2** | 这种分歧**对下游有后果**：为 PEFT *i* 选的数据训练 PEFT *j* 会掉点（对角占优） | 「随便用哪个 PEFT 选的数据都一样好」 |

三件产出构成一条递进的逻辑链：**结构不同（T1）→ 价值排序不同（F1）→ 训练后果不同（F2）**。Table 1 是描述性的（不需训练），Figure 1/2 是实证性的。

### 0.1 一条贯穿全文的方法论铁律：不许自证循环

动机实验**严禁使用本课题自己的 scorer `s_φ`** 作为数据价值信号。否则「我们的方法认为不同 PEFT 价值不同」是同义反复，没有说服力。

因此本文档的数据价值一律来自**与 PCU 无关的、PEFT-specific 的真值或公认代理**：

- **主信号（真值）`u^hi`**：真实短程 PEFT 更新带来的验证损失下降 Δ（leave-one-in 影响），即 [short_update.py](../src/pcu_select/hi_fidelity/short_update.py) 的 `ShortUpdater.delta(...)`。这是「为这个 PEFT 训练这条样本，到底有没有用」的物理定义，独立于任何学习到的 scorer。
- **交叉验证信号 `u^grad`**：LESS 式 per-PEFT 梯度相似度（[selectors.py](../src/pcu_select/baselines/selectors.py) `_less`），用于证明 F1 的结论不依赖某一种价值定义。
- **PEFT-agnostic 对照 `u^rds`**：RDS+（语义相似，与 `z_p` 无关）。它对所有 PEFT 给出**同一个**排序，在 F1/F2 中作为「天花板般的一致性」反衬本应观察到的分歧。

> 一句话：动机实验用的是**别人也会承认的数据价值**，证明的是**这种价值本身就随 PEFT 改变**——这正是 PCU 想去学的目标量，而非 PCU 的输出。

---

## 1. 通用设置（动机实验共享，刻意做小）

动机实验的定位是「先导验证」（研究方案 §20 的精神），必须**比主实验便宜一个量级**，能在铺开 E1–E5 之前快速跑出。因此在主实验通用配置（[实验设计 §1](pcu_select_experiment_design.md)）基础上做如下收缩：

| 维度 | 主实验 | 动机实验 | 理由 |
|---|---|---|---|
| Backbone | Llama-2-7B/13B + Qwen 复现 | **Llama-2-7B 单一** | 动机不需鲁棒性复现 |
| 候选池 | N≈300k | **估值池 N_val≈2k**（F1）/ 选择池 N_sel≈20k（F2） | Δ 标注成本随样本数线性 |
| 任务 | 4 主任务 + 补充 | **GSM8K + HumanEval 两个**（一推理一代码，跨度足够） | 两个足以显示「任务 × PEFT」交互非平凡 |
| 数据价值 | 学习到的 `s_φ` | **真值 Δ + LESS 代理（均非 PCU）** | §0.1 铁律 |
| 预算 | 5/10/30% | **10% 单档**（F2 选择） | 单档即可显示对角占优 |
| seed | 3 目标训练 seed | **F1：2 anchor + 2 seed（用于噪声地板）；F2：3 目标训练 seed** | 噪声地板是 F1 的命门（§3.3） |

### 1.1 动机实验 PEFT 集合（统一注册表的子集）

从 [registry.py](../src/pcu_select/experiments/registry.py) `PEFT_REGISTRY` 中抽 **8 个配置**，刻意沿三条结构轴张开，使「结构距离 → 价值分歧」可被观察：

| ID | family | 改的位置 | 算子 | 容量 | 在本集合中代表的轴 |
|---|---|---|---|---|---|
| `L-r8-qv` | lora | attn: q,v / all layers | additive low-rank | r=8 | **基准** |
| `L-r8-mlp` | lora | mlp: up,down / all layers | additive low-rank | r=8 | placement：attn↔mlp |
| `L-r4-qv` | lora | attn: q,v / all layers | additive low-rank | r=4 | capacity：小 |
| `L-r32-qkvo` | lora | attn: q,k,v,o / all layers | additive low-rank | r=32 | capacity+placement：大 |
| `L-r8-lowlayers` | lora | attn: q,v / **low 1/3** | additive low-rank | r=8 | placement：层段 |
| `L-r8-highlayers` | lora | attn: q,v / **high 1/3** | additive low-rank | r=8 | placement：层段 |
| `IA3-attnmlp` | ia3 | k,v,down / all layers | **multiplicative** | — | family/算子：乘性 |
| `AD-b64` | adapter | block residual / all layers | **additive bottleneck** | b=64 | family/算子：瓶颈 |

> 这 8 个里 5 个属 `seen`/3 个属 `unseen_config`，但动机实验**不区分 seen/unseen**（那是泛化主张，属 E5）；这里只用它们的结构差异。

设计意图：集合内既有**跨 family**（lora/ia3/adapter，算子不同），又有**同 family 内**的容量轴（r4/r8/r32）和 placement 轴（qv/mlp/lowlayers/highlayers）。这样 Figure 1 不仅能显示「LoRA vs IA3 价值不同」（容易），还能显示**更强的命题**：「连同为 LoRA、只差 rank 或层段的两个配置，数据价值排序都已显著分歧」——这才真正堵死「PEFT 差异可被超参一笔带过」的质疑。

---

## 2. Table 1 — PEFT 配置与可训练参数

### 2.1 目的

以**纯描述性**方式确立「PEFT 不是一个名字，而是一组结构化的、互不相同的干预」（研究方案 §1.2 前提 2）。这张表为 `z_p`（site mask + capacity + recipe，[design §8](pcu_select_design.md)）提供「确有结构可编码」的直接依据，也是后续 Figure 1/2 中「结构距离」一词的量化锚点。

### 2.2 列定义

对 §1.1 的 8 个配置各一行，列：

| 列 | 含义 | 来源 |
|---|---|---|
| `PEFT ID` | 配置名 | registry |
| `Family` | lora / ia3 / adapter | registry |
| `Inserted into` | 改哪些 module（q/k/v/o/up/down/residual） | `target_modules` |
| `Layers` | all / low⅓ / high⅓ | `target_layers` |
| `Operator` | additive-lowrank / multiplicative / additive-bottleneck | [design §2.3](pcu_select_design.md) 映射 |
| `# Trainable` | 可训练参数量（绝对值） | 见 §2.3 公式 |
| `% of backbone` | 占 7B 全参比例 | `#Trainable / 6.7e9` |
| `Touched sites \|Ω_p\|` | 命中的干预站点数（24 站点中） | `site_mask_of(p)`（[site_mask.py](../src/pcu_select/peft_space/site_mask.py)） |

最后两列（站点数 / 参数量）是 Figure 1 解读的关键：它把「为什么这两个 PEFT 价值排序差得多」量化为「它们改的站点集合 / 容量差得多」。

### 2.3 参数量计算（`d=4096, L=32` for Llama-2-7B）

不手填、由脚本从 registry 算出（避免与配置漂移）：

- **LoRA**：`#trainable = 2 · r · d · n_layers · n_modules`
  - `L-r8-qv`：`2·8·4096·32·2 ≈ 4.19M`（≈0.063%）
  - `L-r4-qv`：≈2.10M；`L-r32-qkvo`：`2·32·4096·32·4 ≈ 33.6M`（≈0.50%）
  - `L-r8-lowlayers`/`highlayers`：层数取 ⌈32/3⌉=11 → `2·8·4096·11·2 ≈ 1.44M`
  - `L-r8-mlp`：模块换成 up/down，注意 MLP 维度 `d_ffn≈11008` → `2·8·(4096+11008)·32 ≈ 7.73M`
- **IA3**（`IA3-attnmlp`，缩放 k/v/down 的输出维向量）：`#trainable = (d_k + d_v + d_ffn)·L ≈ (4096+4096+11008)·32 ≈ 0.61M`（≈0.009%）
- **Adapter**（`AD-b64`，Houlsby 双瓶颈）：`#trainable ≈ 2 · (2·b·d) · L = 2·2·64·4096·32 ≈ 67.1M`（≈1.0%）

> 上述为**示意量级**，正式数字以脚本输出为准。要点不在精确值，而在**横跨两个数量级**（IA3≈0.6M ↔ Adapter≈67M）且**改的位置互不重叠**——这正是「同一样本价值会随 PEFT 改变」的结构性前提。

### 2.4 落地

- 新增 `scripts/experiments/build_table1.py`：遍历 `resolve_peft(name, "llama2-7b")`，调 `count_trainable_params(peft)` 与 `site_mask_of(peft)`，输出 CSV + LaTeX。
- 若 `count_trainable_params` 尚无，加到 [peft_space/encoder.py](../src/pcu_select/peft_space/encoder.py)（它已在算 `c_p` 的 `log(trainable_params)`，复用其内部计数即可）。

---

## 3. Figure 1 — 数据价值排序的不一致

### 3.1 目的与形态

证明**前提 1 的实证版**：在同一候选样本集合上，不同 PEFT 给出的逐样本数据价值排序显著不同，且**这种不同不是噪声**。

> 论文图：**左 panel = Spearman 相关热图（8×8）**，**右 panel = Top-5% 重叠热图（8×8）**。两张都是 PEFT × PEFT 矩阵，对角线=1，非对角线越低越说明分歧越大。

### 3.2 数据价值信号（每个样本一个分数，per PEFT）

对 §1.1 的每个 PEFT `p`，在估值池 `D_val`（N_val≈2k）上为每条样本算价值向量 `r_p ∈ R^{N_val}`：

1. **主信号 `u^hi`（真值，主图用这个）**：
   `u^hi(x, p, t) = E_a RankNorm_x[ Δ_{a,p,h=1}(x, t) ]`，
   其中 `Δ` 来自 [short_update.py](../src/pcu_select/hi_fidelity/short_update.py) `ShortUpdater.delta(peft=p, sample=x, sketch=V_t, horizon=1, seed=·)`，anchor 数 `A=2`（`θ_base, θ_warm`，[design §10.1](pcu_select_design.md)），组内 RankNorm 见 [labeler.py](../src/pcu_select/hi_fidelity/labeler.py) `_rank_norm_within_bucket`。
   - 直接复用高保真标注管线，只是把三元组限制在 `D_val × {8 个 PEFT} × {2 任务}`。
   - horizon=1、A=2 是为了便宜；这与主实验 `Q_H=10k` 无关，是独立的小批标注。
2. **交叉信号 `u^grad`（LESS 式，副图/附录用）**：[selectors.py](../src/pcu_select/baselines/selectors.py) `score_baseline("less", inp, peft=p)`。证明 F1 的分歧结论**不依赖**「真值」这一种定义。
3. **PEFT-agnostic 对照 `u^rds`**：`score_baseline("rds_plus", inp)`，对所有 `p` 相同。

### 3.3 命门：噪声地板对照（必须有，否则全图无意义）

「两个 PEFT 排序的 Spearman=0.4」本身**说明不了任何事**——可能只是 Δ 信号噪声大。必须建立**噪声地板**：同一个 PEFT 内、仅换 anchor/seed 时排序自己和自己的相关性。

对每个 PEFT `p`，用两个**独立**估值（不同 anchor 或不同 seed）得到 `r_p^{(1)}, r_p^{(2)}`，定义：

```
ρ_intra(p)  = Spearman(r_p^{(1)}, r_p^{(2)})           # 同一 PEFT 的自一致性（信号上限）
ρ_inter(p,q)= Spearman( mean_seed r_p, mean_seed r_q ) # 跨 PEFT 分歧（待测量）
```

**结论只有在 `ρ_inter(p,q) 显著 < ρ_intra` 时才成立**：即「换 PEFT 造成的排序变化」明显大于「换 seed 造成的排序变化」。把噪声地板 `ρ̄_intra`（8 个 PEFT 的均值）画成左图热图上的一条参考刻度/对角注记。

> 这是整张 Figure 1 能不能用的唯一判据。审稿人最可能的攻击就是「你的分歧是噪声」，噪声地板把它正面挡掉。

### 3.4 两个 panel 的精确定义

- **左（Spearman）**：`S[i][j] = ρ_inter(p_i, p_j)`，用 [metrics.py](../src/pcu_select/eval/metrics.py) `spearman`。对角=1。配色让接近 `ρ̄_intra` 的格子接近「白」（=与噪声地板无异），低于地板的越蓝（=真分歧）。
- **右（Top-5% overlap）**：`O[i][j] = jaccard(top5%(r_{p_i}), top5%(r_{p_j}))`，用 [metrics.py](../src/pcu_select/eval/metrics.py) `jaccard`。这是**直接对选择有意义的量**——预算 5% 时两个 PEFT 真正会选进去的样本集合有多大重叠。同样画出 `intra` 的 Top-5% overlap 作地板。

> Top-5% overlap 比 Spearman 更贴近课题：我们最终只关心 top-k 选谁。两个 PEFT 即使全局 Spearman 不低，top-5% 也可能差得很远（尾部敏感），这恰好是「选择层面」的分歧。

### 3.5 结构化解读（让图不只是"看起来不同"）

把矩阵的非对角值对 §1.1 的三条结构轴回归/分组，叠加一张小图或表：

- 同 family 同 placement、只差容量（`L-r4-qv` vs `L-r8-qv`）：分歧应**最小**但仍 > 地板；
- 同 family、placement 差（`L-r8-qv` vs `L-r8-mlp`，或 low vs high layers）：分歧**中等**；
- 跨 family/算子（`L-r8-qv` vs `IA3-attnmlp` vs `AD-b64`）：分歧**最大**。

即给出「**结构距离越大 → 价值排序分歧越大**」的单调关系（呼应 E4-a「配置差异 → 选择差异正相关」，但这里用的是与 PCU 无关的真值，因此是动机而非自证）。

### 3.6 成功判据

- **主判据**：存在大量配置对满足 `ρ_inter(p,q) < ρ̄_intra − margin`（建议 margin 取 `ρ_intra` 的 95% bootstrap 区间宽），且至少跨 family 的对的 Top-5% overlap ≤ 0.5。
- **稳健性**：把主信号从 `u^hi` 换成 `u^grad`，分歧的**定性结论（结构距离单调）不变**（数值可不同）。
- **对照**：`u^rds`（PEFT-agnostic）对所有 PEFT 给出 overlap=1 的「全黑」行/列，直观反衬。
- **失败兜底**：若 `ρ_inter ≈ ρ_intra`（分歧≈噪声）→ 立即检查 (a) Δ 噪声是否过大（加 anchor/seed，看 §3.3）、(b) 是否只在容量轴上塌缩（若仅 family 间有差异、family 内无差异，则主张收缩为「跨 family 才需 conditioning」，并据此调整 E4 配置扫描的卖点）。

---

## 4. Figure 2 — 跨 PEFT 迁移矩阵（对角占优）

### 4.1 目的与形态

Figure 1 证明「排序在分歧」，但审稿人会追问：**分歧有后果吗？** Figure 2 用真实下游训练回答：为 PEFT *i* 选的数据，去训练 PEFT *j*，性能不如「为 *j* 自己选的数据」。

> 论文图：**P×P 热图**，行 = 训练用的 PEFT（target，被微调的那个），列 = 选数据用的 PEFT（source，价值排序来自谁）。看点 = **对角占优（diagonal dominance）**。

### 4.2 协议（这是 E4-c 错配矩阵的"动机版"，但选择信号非 PCU）

为控成本与避免自证，Figure 2 用**精简 PEFT 子集**（建议 5 个，跨度足够即可）：`{L-r8-qv, L-r32-qkvo, L-r8-mlp, IA3-attnmlp, AD-b64}`。

```
选择池 D_sel（N_sel≈20k），预算 B=10%
for source p_j in 5 configs:
    用 §3.2 的 PEFT-specific 真值 u^hi(·, p_j, t) 取 top-10% → 子集 S_j   # 选择=按真值排序，非 PCU
for target p_i in 5 configs:
    for source p_j in 5 configs:
        train_and_eval(peft=p_i, samples=S_j, ...)  →  perf[i][j]        # eval.target_train
        (3 个目标训练 seed，报 mean±std)
```

- 选择信号：直接用**该 source PEFT 的真值排序** top-k（不经过任何学习的 scorer）。这把命题压到最干净的形式：「即使你拥有 source PEFT 的完美数据价值真值，拿去喂另一个 PEFT 也是次优的」。
- 训练与评估：[target_train.py](../src/pcu_select/eval/target_train.py) `train_and_eval`，同一 recipe / step / eval，唯一变量是子集（[实验设计 §1.7](pcu_select_experiment_design.md) compute-matched）。native 后端（lora/ia3/adapter）全部支持，5 个配置都能训。
- 复用 [run_e4.py](../scripts/experiments/run_e4.py) 的 E4-c 双层 `product(configs, configs)` 结构；差别仅是 `select(...)` 换成「按 source 的 `u^hi` 取 top-k」而非 PCU。

### 4.3 归一化（让对角占优在不同量纲下可见）

不同 target PEFT 的绝对性能不可比（IA3 容量小、Adapter 容量大）。因此**按行归一化**后再画热图：

```
norm[i][j] = ( perf[i][j] − perf_random[i] ) / ( perf[i][i] − perf_random[i] )
```

- `perf_random[i]`：用随机 10% 子集训练 `p_i` 的性能（每行一个下界锚点）。
- 这样**对角线 `norm[i][i]=1`**，非对角 `<1` 即对角占优；`<0` 表示比随机还差（强分歧信号）。
- 同时报一张**未归一化**的原始 `perf[i][j]`（mean±std）放附录，保证可追溯。

### 4.4 关键对照：PEFT-agnostic 选择的同一张矩阵

并排给出**用 `u^rds`（RDS+，PEFT-agnostic）选择**的迁移矩阵。它对所有 source 选**同一个**子集，因此**没有对角结构**（行内近似常数）。两张矩阵对比即为决定性证据：

- PEFT-specific 选择 → 明显对角占优；
- PEFT-agnostic 选择 → 平坦无对角。

→ 「对角占优」这一结构**只在你按 PEFT 条件化地选数据时才出现」，正是 PCU 想自动化的东西。

### 4.5 成功判据

- **主判据**：对每一行 `i`，`norm[i][i]=1 > mean_{j≠i} norm[i][j]`，且**逐行配对检验**（跨 3 seed，对角 vs 该行非对角均值）`p<0.05`（Wilcoxon signed-rank，[实验设计 §1.7](pcu_select_experiment_design.md)）。
- **量化**：报告平均「错配掉点」`Gap = mean_i (norm[i][i] − mean_{j≠i} norm[i][j])`，并按 source-target 的结构距离分层（跨 family 错配掉点应 > 同 family 错配）。
- **对照对比**：PEFT-specific 矩阵的 `Gap` 显著大于 PEFT-agnostic 矩阵的 `Gap`（后者应≈0）。
- **失败兜底**：若对角不占优（错配不掉点）→ 与 Figure 1 联合诊断：要么 F1 已显示分歧其实很小（主张需收缩到「仅成本动机」，靠 [E2](pcu_select_experiment_design.md)），要么 10% 预算太宽松导致子集差异被淹没（降到 5% 重测，呼应预算敏感性）。

---

## 5. 与主实验、与"自证循环"的边界

| | 动机实验（本文） | 主实验 E4（[实验设计 §5](pcu_select_experiment_design.md)） |
|---|---|---|
| 数据价值来自 | **真值 Δ / LESS（非 PCU）** | **PCU 的 `s_φ`** |
| 证明的事 | 「价值**确实**随 PEFT 变」（世界的性质） | 「PCU **能捕捉**这种变化」（方法的能力） |
| 角色 | 立论动机（intro / motivation 章） | 方法论核心证据（experiments 章） |
| 成本 | 小（2k 估值 + 25 训练格） | 大（全注册表 × 全任务 × 3 seed） |

二者**互补不重复**：动机实验证明"问题存在"，E4 证明"我们解决了它"。Figure 2 与 E4-c 形状一样（错配矩阵）是有意为之——读者在 intro 先看到「错配会掉点」（真值版），到 experiments 再看到「PCU 的条件化能复现并预测这种结构」（方法版），形成首尾呼应。

> **必须在论文里写清**：Figure 1/2 的数据价值不来自本方法，否则就是循环论证。这一行声明本身就是防御审稿质疑 §15.1（「是不是只是 LESS+PEFT 向量」）的弹药。

---

## 6. 落地清单与排期

### 6.1 新增/复用脚本

| 产出 | 脚本 | 复用 |
|---|---|---|
| Table 1 | `scripts/experiments/build_table1.py`（新） | `resolve_peft`, `site_mask_of`, 参数计数 |
| Figure 1 标注 | `scripts/experiments/build_motivation_values.py`（新） | `ShortUpdater.delta`, `HiFidelityLabeler`, `score_baseline("less"/"rds_plus")` |
| Figure 1 绘图 | `scripts/plots/plot_motivation_f1.py`（新） | `metrics.spearman`, `metrics.jaccard` |
| Figure 2 训练 | `scripts/experiments/run_motivation_transfer.py`（新） | `run_e4.py` 的 `product` 结构 + `train_and_eval` |
| Figure 2 绘图 | `scripts/plots/plot_motivation_f2.py`（新） | 行归一化热图 |

所有标注/结果写入 `runs/<exp>/motivation/`（values.parquet, transfer.jsonl），与主实验 `results/` 平行，记账照写 `cost/accounting.jsonl`（[design §16](pcu_select_design.md)）。

### 6.2 成本量级（8×A100 估算，以 accounting.jsonl 为准）

| 步骤 | 量级 | 说明 |
|---|---|---|
| Table 1 | ~0（CPU） | 纯计数 |
| F1 标注 | **中低**：`2k 样本 × 8 PEFT × 2 任务 × 2 anchor × 2 seed × h=1` 短更新 | 噪声地板的 2 seed 是主要乘子；可先 1 任务试跑 |
| F2 训练 | **中**：`5×5 矩阵 × 2 任务 × 3 seed` 次 10%-子集 PEFT 微调 + RDS+ 对照同规模 | native 后端、bf16、anchor 冻结，单格便宜 |

### 6.3 推荐执行顺序

1. **Table 1**（当天，零 GPU）——先把结构差异摆出来。
2. **Figure 1 单任务（GSM8K）小跑**：先验证 `ρ_inter < ρ_intra`（噪声地板成立）。**这是 go/no-go 关卡**——若噪声地板都过不了，先修 Δ 标注再谈其余。
3. Figure 1 补 HumanEval + LESS 交叉验证。
4. **Figure 2**：先跑 PEFT-specific 矩阵确认对角占优，再补 RDS+ 对照矩阵。
5. 写作：三图进 motivation 章，串成「结构不同 → 价值不同 → 后果不同」。

---

## 7. 风险与回退

| 风险 | 触发信号 | 回退 |
|---|---|---|
| 分歧≈噪声 | F1 `ρ_inter ≈ ρ_intra` | 加 anchor/seed 降 Δ 噪声；若确实无分歧 → family 内退为「仅跨 family 需 conditioning」，相应收缩 E4 卖点 |
| 仅容量轴有分歧 | F1 family 内 overlap≈1、仅 family 间低 | 主张精确化为「placement/family 驱动，容量次要」，并在 Table 1 强调站点不重叠 |
| 对角不占优 | F2 错配不掉点 | 降预算到 5% 重测；仍不占优则动机转向「成本优先」（[E2](pcu_select_experiment_design.md)），性能动机让位 |
| 被指自证循环 | 审稿质疑 | 强调 F1/F2 价值来自真值 Δ/LESS，非 `s_φ`（§5 声明） |
| Δ 标注太贵 | F1 标注 wall-time 超预算 | 降 N_val 到 1k、anchor 到 1（但保留 2 seed 以维持噪声地板）、horizon 固定 1 |
```
