# PCU-Select 完整设计方案（实现版）

> 本文档是 [pcu_select_research_plan.md](pcu_select_research_plan.md) 的实现细化版本，补强了原方案中未充分说明或不利于直接实现的细节，并对若干设计点做了收敛性修订。研究动机、整体策略与实验框架沿用原方案，本文档不再重复，仅对可执行层的接口与算法进行规约。

---

## 0. 与原方案的差异摘要

| 类别 | 原方案 | 本文档 |
|---|---|---|
| 干预站点 Ω | 未规定具体集合 | 明确为 8 层 × 3 模块 = 24 个站点（默认） |
| `α_p^ω` | 描述为"由 site mask、capacity、operator type 决定" | 给出可执行公式 |
| 样本梯度签名 | 文字描述 | 明确投影维度 `d_proj=256`、共享随机投影矩阵 |
| Anchor checkpoint | 数量为 2，未说明来源 | 规定为 `θ_base + θ_warm`，warm 由统一协议生成 |
| 短程更新 | h 步、optimizer 未定 | 规定 batch、优化器状态、单样本 h-step 协议 |
| 特征缓存 | 概念性提及 | 规定为 parquet/arrow，给出列 schema |
| OOD 判定 | 文字 | 用 z_p 上的 Mahalanobis 距离 + 95 分位阈值 |
| 任务草图 | 32–64 条 | 默认 32，标注 `train-split-only`、不接触测试集 |
| 高保真损失 | 高保真 + 低保真联合 | 先低保真预训练，再高保真微调；不一次性 mixing |
| 验证集 sketch 编码 | 提到 mean/attention pooling | 默认 `set-transformer` 风格 attention pool |
| 多保真采样 | 三阶段，未量化 | 给出每阶段比例与触发条件 |
| Scorer 部署延迟 | 文字 | 规定批前向 + bf16，保证 amortized cost ≪ forward |

---

## 1. 全局符号与数据契约

### 1.1 类型定义概览

```python
SampleID    = str            # 内容哈希
PEFTID      = str            # 内容哈希
TaskID      = str            # 任务名 + sketch 哈希
SiteID      = tuple[int, str]  # (layer_idx, module_name)
```

### 1.2 关键张量维度约定

| 名称 | 维度 | 说明 |
|---|---|---|
| `e_x.joint` | `d_sem`（默认 768） | 指令+回复联合 embedding |
| `d_x` | `d_diff`（默认 16） | 难度/质量手工统计 |
| `a_x` | `n_layers_signature * d_layer_stat`（默认 8×8=64） | 分层激活签名 |
| `z_x` | `d_sem + d_diff + d_act`（默认 848） | 拼接得到的样本表示 |
| `g_x^ω` | `d_proj`（默认 256） | 站点 ω 上的投影梯度签名 |
| `m_p` | `\|Ω\| * 4`（默认 96） | site mask + 3 种 operator 指示位 |
| `c_p` | `d_cap`（默认 16） | 容量向量 |
| `r_p` | `d_rec`（默认 16） | recipe 向量 |
| `f_p` | `d_fp`（可选，默认 64） | functional fingerprint |
| `z_p` | `m_p + c_p + r_p + f_p`（默认 192） | PEFT 表示 |
| `z_t` | 等同 `z_x` | 任务条件表示（草图样本表示池化得到） |

### 1.3 一切实体均通过内容哈希构造 ID

- `SampleID = sha256(instruction + "\n" + response + dataset_tag)[:16]`
- `PEFTID   = sha256(canonical_yaml(PEFTConfig))[:16]`
- `TaskID   = sha256(task_name + sketch_indices_csv)[:16]`

这保证 cache 不会随调度顺序变化，断点续算成立。

---

## 2. 干预站点 Ω 的精确定义

### 2.1 默认站点配置

对一个 L 层 decoder-only transformer，定义：

- 选取层索引集合 `L_sig = uniform_indices(L, k=8)`，例如 L=32 时为 `[3,7,11,15,19,23,27,31]`。
- 每层 3 个 hook 点：
  - `attn_out`：自注意力模块输出（在残差加法之前）
  - `mlp_out`：MLP 模块输出（在残差加法之前）
  - `block_residual`：本 block 残差合并后的输出（layer norm 之前）

得到默认站点集合：

```
Ω_default = { (l, m) : l ∈ L_sig, m ∈ {attn_out, mlp_out, block_residual} }
|Ω_default| = 24
```

### 2.2 为什么不用 `q_proj/k_proj/v_proj` 这种更细的位置？

- 不同 backbone 的 fused QKV vs split QKV 不一致，复用 hook 麻烦；
- 自注意力 OUT 一致覆盖 q/k/v/o 的训练效果；
- 24 个 site 已经在缓存大小（≈2.5GB for N=100k）和判别力之间取到合理平衡。

### 2.3 PEFT → 站点的映射

| PEFT | 影响的 site 类型 | operator type |
|---|---|---|
| LoRA on q_proj/v_proj | `attn_out`（间接） | additive_low_rank |
| LoRA on up_proj/down_proj | `mlp_out` | additive_low_rank |
| IA3 on attention | `attn_out` | multiplicative |
| IA3 on FFN | `mlp_out` | multiplicative |
| Bottleneck Adapter | `block_residual` | additive_bottleneck |
| Prefix tuning | 所有 `attn_out`（前缀注意力效果） | prefix |
| BitFit | 所有 site | bias_shift |

具体的 `site_mask_of(PEFTConfig) -> dict[SiteID, float]` 函数实现见 [`src/pcu_select/peft_space/site_mask.py`](../src/pcu_select/peft_space/site_mask.py)。

---

## 3. PEFT 条件权重 `α_p^ω` 的可执行公式

原方案描述为"由 PEFT site mask、capacity、operator type 决定"，本文给出精确形式。

设：
- `mask(p, ω) ∈ {0, 1}`：PEFT p 是否作用于站点 ω；
- `cap(p, ω)`：在该站点的可训练参数数量（如 LoRA 在该层的 `2 * rank * d_model`）；
- `op(p, ω) ∈ {additive_low_rank, multiplicative, additive_bottleneck, prefix, bias_shift}`；
- `η`、`ρ_op`：超参数，控制 capacity 缩放与 operator 类型权重。

则：

```
α_p^ω = mask(p, ω) · ρ_{op(p,ω)} · tanh( η · log(1 + cap(p, ω) / d_model) )
```

并对整个站点集合归一化：

```
α̃_p^ω = α_p^ω / max(ε, Σ_{ω'} α_p^{ω'})
```

使得最终低保真效用 `u^lo` 不被 site 数量主导，而是反映 site 上"分布权重"。

默认超参：

- `η = 1.0`
- `ρ = {additive_low_rank: 1.0, multiplicative: 0.6, additive_bottleneck: 0.8, prefix: 0.4, bias_shift: 0.3}`

`ρ_op` 的含义：从相同 site 处同等容量出发，不同 operator 对训练动态的有效扰动并不相同。这些值用作先验初值，可以在实验中调或并入 scorer 学习。

---

## 4. 样本损失 ℓ(x) 的定义

由 `TaskConfig.loss_type` 决定：

- **`response_lm`**（instruction tuning，默认）：在响应 token（即 prompt 之后的 token）上的交叉熵，instruction tokens mask 掉。
- **`full_lm`**（continued pretraining 风格）：所有 token 的交叉熵。
- **`preference`**：DPO 风格的偏好对样本损失。本课题不主推，但接口预留。

```
ℓ(x) = - Σ_{i ∈ response_mask} log p_θ(x_i | x_{<i})
```

任务草图损失 `L_V`：草图样本的 ℓ(v) 之和，对样本数做均值。

---

## 5. 站点梯度签名

### 5.1 计算流程

```
for each sample x:
  with hooks at each ω ∈ Ω registering retain_grad on h^ω:
    forward(x), compute ℓ(x)
    ℓ.backward()
    for each ω: g_raw_ω = h^ω.grad   # shape: (B, T_x, d_model)
                g_pool_ω = pool_over_T(g_raw_ω)  # (B, d_model)
  for each ω:
    g_x^ω = Φ_ω @ g_pool_ω           # (B, d_proj)
    g_x^ω = g_x^ω / ||g_x^ω||₂
```

### 5.2 时间维池化

- 默认 `mean_over_response_tokens`（与 loss mask 对齐）。
- 备选：`last_token` 或 `attention_weighted`（实验消融）。

### 5.3 随机投影矩阵

- `Φ_ω ∈ R^{d_proj × d_model}`，每个站点一份；
- `seed_ω = hash(site_id, global_seed)`；
- 元素 ~ `N(0, 1/d_proj)`（Johnson-Lindenstrauss 缩放）；
- 一次生成，永久持久化（避免重复时不一致）。

### 5.4 任务草图签名

对草图 V，先对每个样本计算 `g_v^ω`，然后池化：

```
g_t^ω = normalize( (1/|V|) · Σ_{v ∈ V} g_v^ω )
```

> 注意：草图签名计算的模型与样本签名是**同一个 selector model**（见 §6.1），而不是任意 backbone。

### 5.5 低保真效用

```
u^lo(x, p, t) = Σ_ω α̃_p^ω · cos(g_x^ω, g_t^ω)
```

---

## 6. Selector model 的选择

### 6.1 用什么模型抽取梯度签名？

**关键决策**：站点梯度签名是否需要与目标 backbone 同源？

本方案选择 **selector model = backbone family 中最小的可用模型**（例如 LLaMA 7B 而非 65B），原因：

1. 梯度签名是相对量，主要刻画"哪些站点关心这个样本"，不需要绝对精度；
2. 小模型省时间与显存；
3. 同 family 保证 layer 索引语义一致；
4. 大模型阶段的 PEFT 微调是另一回事，不依赖 selector model。

若实验显示性能不足，可以升级到与目标 backbone 同尺寸；这是消融选项。

### 6.2 与目标 backbone 不同的处理

- `L_sig` 是 selector model 上的层索引；不同 backbone 不直接共享 site 坐标。
- 因此一次离线训练只面向一个 backbone family（如 LLaMA-7B/13B）；跨 family 不在主声明范围。

---

## 7. 样本特征 z_x 的精确构造

### 7.1 语义表示 `e_x`

- `e_x.instr`：用 `sentence-transformers/all-mpnet-base-v2` 或同等模型对 instruction 编码（768 维）。
- `e_x.resp`：response 同上。
- `e_x.joint`：把 instruction + response 拼接后编码（截断 512 tokens），用于聚类与 task pool 默认通道。
- `e_x.source_id`：可选 `OneHot(source_dataset)`。

主输入到 scorer 的是 `e_x.joint`。其余字段保留供消融。

### 7.2 难度统计 `d_x`

固定 16 维：

```
[ log_len_instr, log_len_resp, log_total_len,
  resp/instr_ratio,
  loss_mean, loss_std, loss_max, perplexity,
  avg_logprob, entropy_mean, entropy_max,
  is_cot, is_code, is_qa, language_id_onehot(3 dims) ]
```

`loss/ppl/logprob/entropy` 在 selector model 上一次前向得到。

### 7.3 分层激活签名 `a_x`

对 `L_sig` 的每一层，记录：

```
[ ||h^attn_out||₂, ||h^mlp_out||₂, ||h^residual||₂,
  attn_entropy, attn_head_norm_var,
  mlp_activation_norm,
  hidden_token_var, last_token_dot_first ]
```

每层 8 维，共 8 层 × 8 = 64 维。

> 注意：这与站点梯度签名共享同一次前向，但梯度签名需要额外一次 backward，可以选择性跳过没有 loss mask 的情况。

---

## 8. PEFT 表示 `z_p` 的精确构造

### 8.1 Site mask `m_p`

形状：`|Ω| × 4` = 96 维，第二维是 `[is_active, op_additive, op_multiplicative, op_prefix]` 的 4 位指示。

> Bottleneck adapter 与 additive low-rank 在 operator 维度合并为 `op_additive`；prefix 单列；bias_shift 在 `is_active=1, op_additive=0/mul=0/prefix=0` 下识别（保留 1 个空位以便未来添加）。

### 8.2 容量向量 `c_p`

```
[ log(trainable_params), trainable_ratio,
  lora_rank_normalized, lora_alpha_normalized,
  adapter_bottleneck_normalized, prefix_len_normalized,
  log_extra_flops, log_extra_memory_mb,
  inference_latency_ratio,
  affects_kv_cache,
  per_op_capacity_share (5 dims),
  per_op_indicator_count ]
```

16 维。

### 8.3 Recipe 向量 `r_p`

```
[ log(lr), warmup_ratio, weight_decay,
  scheduler_onehot(4),
  optimizer_onehot(3),
  log(batch_size), dropout,
  log(max_steps), grad_clip,
  init_method_onehot(2) ]
```

16 维。

### 8.4 Functional fingerprint `f_p`（可选）

- 选定 fingerprint probe set：64 条 task-agnostic 样本（短问答 + 短指令）。
- 对 `θ_base`，应用一次 PEFT 的小 warmup（h_fp=2 步、固定数据 8 条），记录每个 site 的 `||Δh^ω||₂` 与下游 loss 变化，拼接成 64 维向量。

可选模块，由 `PEFTConfig.use_fingerprint` 开关控制；默认 `false`，在消融实验中打开。

---

## 9. 任务草图编码

### 9.1 草图构造协议

- 来源：**仅** 目标任务的 `train` 或 `dev` split，**绝不**取自最终 `test` split；
- 默认大小：`N_V = 32`；
- 抽样：固定 seed，按 length 分位数三层（短/中/长）等比例采样，避免长度偏倚；
- 持久化：`sketch_{task}_{seed}.json`。

### 9.2 池化

对草图样本集合 `V`，每个样本得到 `z_v`（与 `z_x` 同构）。然后：

```python
class TaskEncoder(nn.Module):
    # 设置一个 latent query 数组 Q ∈ R^{q × d}
    # MultiheadAttention(Q, K=V_set, V=V_set) → mean over q → z_t
```

默认 `q = 4`、`d = 256`。Set-transformer 风格 attention pool 比 mean pool 更稳健，对长尾草图样本也更鲁棒。

---

## 10. 高保真效用 `u^hi`

### 10.1 Anchor checkpoint 协议

`A = 2`：

- `θ_base`：原 backbone，未做任何微调；
- `θ_warm`：在 meta-pool 中随机抽 1k 样本、做 base-config LoRA（rank=8, attn-only）训练 200 步得到，作为"已轻量微调"代表。

两个 anchor 都被冻结，PEFT 参数从其上启动。

### 10.2 短程更新协议

对一个 `(x, p, a, h)`：

1. Clone PEFT params from fresh init (per `p.init_method`)；
2. Set optimizer = `p.recipe.optimizer`（默认 `adamw`），状态空白；
3. 重复 `h` 次：在 `x` 上做一次 forward + backward + step；
4. 在 `V` 上 forward-only 计算 `L_V`；
5. 返回 `Δ = L_V(θ_a) - L_V(θ_a + Adapt^h_p(x))`。

> 实现要点：所有 PEFT 微调使用 `torch.compile` 或 `fp16/bf16` 加速；anchor 模型主参数全程冻结，使得 backward 只对 PEFT 参数生效，省 backward 时间约 50–90%。

### 10.3 噪声抑制

- 组内 RankNorm：对同一 `(p, t)` bucket 内的所有 `x` 的 `Δ` 做 `rank/N` 标准化，转为 `[0,1]`。
- 不同 horizon 做加权：默认 `w_1 = 0.4, w_4 = 0.6`。
- 默认 seed 数 1；scorer 判定不确定区域时通过 active sampling 加 seed 而不是均匀重复。

### 10.4 高保真三元组采样

总预算 `Q_H = 10k`，分阶段：

| 阶段 | 数量 | 策略 |
|---|---|---|
| Phase 1（覆盖） | 50% | 分层采样：sample cluster × PEFT family × PEFT capacity bucket |
| Phase 2（不确定性） | 30% | `q_query = σ̂ · (1 + γ · ReLU(û^lo))`，`γ=0.5` |
| Phase 3（边界） | 20% | scorer 排序与真实排序在 top-k 区域不一致的样本 |

Phase 2、3 必须等 Phase 1 训出一个 v0 scorer 才能启动。

---

## 11. Scorer 模型

### 11.1 输入与张量形状

```
inputs:
  z_x  : (B, 848)
  z_p  : (B, 192) or (1, 192) broadcast
  z_t  : (B, 848) or (1, 848) broadcast
outputs:
  μ    : (B,)
  σ    : (B,)  via softplus, with floor=1e-3
```

### 11.2 网络结构

```
f_x:  z_x → LayerNorm → Linear(848, 256) → GELU → Linear(256, 256) → LayerNorm
f_p:  z_p → LayerNorm → Linear(192, 128) → GELU → Linear(128, 128) → LayerNorm
f_t:  z_t → LayerNorm → Linear(848, 256) → GELU → Linear(256, 256) → LayerNorm

# 条件融合：FiLM 调制 + 双线性
cond = concat(h_p, h_t)                # 384
γ, β = Linear(384, 256) × 2
h    = γ ⊙ h_x + β                     # 256
bilin = Bilinear(h_x, h_p, 64)         # 64
out  = concat(h, bilin) → Linear(320, 256) → GELU → 头

μ_head: Linear(256, 1)
σ_head: Linear(256, 1) → softplus + 1e-3
```

参数量约 1.5M，部署时单条样本前向 ≈ 0.05ms（GPU）。

### 11.3 损失

```
L = λ_rank · L_rank^{hi}
  + λ_reg  · L_reg^{hi}
  + λ_proxy · L_proxy^{lo}
  + λ_unc  · L_unc^{hi}
```

默认权重：`λ_rank=1.0, λ_reg=0.3, λ_proxy=0.5, λ_unc=0.2`。

#### Pairwise ranking

在同 `(p, t)` bucket 内随机抽对 `(i, j)`：

```
L_rank = - log σ( (μ̂_i - μ̂_j) · sign(u^hi_i - u^hi_j) )
```

#### Heteroscedastic NLL

```
L_unc = 0.5 · ((u^hi - μ̂)² / σ̂²) + 0.5 · log σ̂²
```

### 11.4 训练计划

- **阶段 A（低保真预训练）**：仅用 `u^lo` 标签训练 scorer，3 epochs，AdamW lr=3e-4。
- **阶段 B（联合精调）**：加入 `u^hi`，所有损失启用，2 epochs，lr=1e-4。

> 不直接 mixing 是为了避免低保真噪声主导初始训练；先让 scorer 学到 z_x×z_p×z_t 的结构，再用稀疏高保真校正。

---

## 12. 数据选择策略

### 12.1 评分

```
q(x) = μ̂(x, p*, t*) - λ · σ̂(x, p*, t*),  λ=0.2  (default)
```

### 12.2 聚类

- 聚类空间：`e_x.joint`（768 维）；
- 算法：MiniBatch K-Means；
- `k = max(50, ⌊√N⌋)`；
- 对长尾 cluster size，做 `min_cluster_size = max(20, 0.1·B/k)` 保护。

### 12.3 配额分配

```
v_k = mean( top 10% of {q_i : x_i ∈ C_k} )    # 簇级 utility
b_k = round( B · (v_k^+)^α · |C_k|^{1-α} / Z )
α   = 0.6  (default; ablation α ∈ {0, 0.3, 0.6, 0.9, 1.0})
```

若 `b_k > |C_k|` 则上限剪裁，余量按 `v_k^+` 比例重分配给其他簇。

### 12.4 簇内选择

按 q_i 取 top-`b_k`。

---

## 13. OOD 判定与校准

### 13.1 ID 判定

- 训练完成后，对训练用 PEFT 配置集合，记录 `μ_z_p` 与 `Σ_z_p`；
- 新 PEFT `p*` 的 ID 分数：

```
d²(p*) = (z_{p*} - μ_z_p)^T Σ_z_p^{-1} (z_{p*} - μ_z_p)
```

- 阈值 `τ_id = quantile(d²(p_train), 0.95)`。

### 13.2 校准模式

若 `d²(p*) > τ_id`：

1. 抽样 200 / 500 条样本（两档对照，见实验设计 §6.3）；
2. 对 `p*` 计算高保真 `u^hi`（horizon=1，单 anchor，约 1/4 完整成本）；
3. 冻结 scorer 主体，仅训练一个 calibration head：

```
μ_cal = μ̂ + W_cal · [z_x; z_{p*}; z_t] + b_cal
```

4. 用这些样本做线性回归 fit `W_cal, b_cal`；
5. 全量打分时使用 `μ_cal`。

> **实现边界（native 短程更新后端）**：步骤 2 的高保真短程更新由 `hi_fidelity.native_peft` 实现，仅支持 `lora / ia3 / adapter / bitfit`（prefix / ptuning 需 prompt/KV 注入，无法表示为 `nn.Linear` 包装，见 `native_peft.SUPPORTED_FAMILIES`）。因此 **prefix / ptuning 目标无法生成校准标签**，E5 中这类 L2 配置只能走 zero-shot，按失败边界如实报告（实验设计 §6.5）；同属 L2 的 BitFit 可正常校准。校准标签由 `scripts/experiments/build_calib_labels.py` 生成。

---

## 14. 数据存储与缓存

### 14.1 目录布局

```
runs/<exp_id>/
  features/
    sample_features.parquet       # 一行一个样本 (e_x, d_x, a_x)
    sample_grad_signature/        # 分片 npy: site_id_xxx.npy, 形状 (N, d_proj)
  task/
    sketches/<task>_<seed>.json
    z_t_<task_id>.npy             # 任务条件向量 z_t（每任务一份，runner 读取）
    task_grad_<task_id>.npy       # 任务梯度签名（每任务一份）
  peft/
    configs.jsonl                 # 已编码的 z_p 与原始 yaml
    fingerprints.npy              # 若启用
  labels/
    lo_fidelity.parquet           # (sample_id, peft_id, task_id, u_lo)
    hi_fidelity.parquet           # (..., u_hi, σ_est)
  scorer/
    ckpt_a.pt, ckpt_b.pt
  selection/
    target_<peft>_<task>/
      scored.parquet
      selected.txt
  cost/
    accounting.jsonl
```

### 14.2 缓存键

```
feature_cache_key  = sha256(selector_model_id + sample_id + feat_version)
grad_sig_cache_key = sha256(selector_model_id + sample_id + site_id + global_seed)
hi_fidelity_key    = sha256(anchor_id + peft_id + sample_id + task_id + horizon + seed)
```

---

## 15. 训练 / 应用流水线接口

### 15.1 Offline pipeline 入口

```python
def run_offline(
    *,
    meta_pool: DatasetLike,
    peft_space: list[PEFTConfig],
    tasks: list[TaskConfig],
    selector_model: str,
    cfg: OfflineConfig,
    workdir: Path,
) -> Path:
    """ 返回 scorer ckpt 路径。 """
```

### 15.2 Apply pipeline 入口

```python
def run_apply(
    *,
    candidate_pool: DatasetLike,
    peft_target: PEFTConfig,
    task_target: TaskConfig,
    budget: int | float,
    scorer_ckpt: Path,
    cfg: ApplyConfig,
    workdir: Path,
) -> list[SampleID]:
    """ 返回选中样本的 id 列表。 """
```

### 15.3 调用示例

```python
from pcu_select.data import JsonlPool, load_sketch
from pcu_select.peft_space.schema import load_peft_config
from pcu_select.types import ApplyConfig, TaskConfig

sketch = load_sketch("runs/exp1/task/sketches/gsm8k_0.json")
selected = run_apply(
    candidate_pool=JsonlPool.from_jsonl("data/alpaca_100k.jsonl"),
    peft_target=load_peft_config("configs/peft/lora_r16_qkvo.yaml"),
    task_target=TaskConfig(name="gsm8k", task_id=sketch.task_id, sketch=sketch),
    budget=0.1,
    scorer_ckpt=Path("runs/exp1/scorer/ckpt_b.pt"),
    cfg=ApplyConfig(lambda_unc=0.2, cluster_alpha=0.6),
    workdir=Path("runs/exp1"),
)
```

---

## 16. 成本核算与 break-even

### 16.1 必须记录的字段

每次离线/应用阶段写一条 `cost_event` 到 `cost/accounting.jsonl`：

```
{ "stage": "feat" | "lo" | "hi" | "scorer_train" | "apply_score" | "apply_select" | "target_train",
  "wall_time_sec": ..., "gpu_id": ..., "gpu_hours": ...,
  "peak_mem_mb": ..., "disk_written_mb": ...,
  "peft_id": "...", "task_id": "...", "n_samples": ... }
```

### 16.2 Break-even 计算脚本

```python
def break_even_T(C_offline, C_apply, C_specific):
    return C_offline / max(1e-6, C_specific - C_apply)
```

论文必须把上式渲染为曲线（X 轴：目标 PEFT 数量 T；Y 轴：总成本）。

---

## 17. 实现里程碑（最小闭环优先）

```
M1: feature extraction + activation signature  → 1 周
M2: site mask + α_p^ω + low-fidelity proxy     → 1 周
M3: scorer model + low-fidelity-only 训练      → 1 周
M4: hi-fidelity short-update + labeler         → 1.5 周
M5: scorer 联合训练 + 选择 + 端到端微调对比     → 1.5 周
```

M3 末尾即可跑原方案 §20 的三个最小验证实验。

---

## 18. 未覆盖项与显式假设

下列内容本设计文档**有意不细化**，由实验结果驱动决定：

1. λ_rank / λ_reg / λ_proxy / λ_unc 的最终权重——超参搜索；
2. `α`（cluster quota 倾斜）的最终值——消融决定；
3. functional fingerprint 是否进入主结果——消融决定；
4. anchor 数从 2 升到 3 的成本/收益——后续实验；
5. 选择空间是否换成 DPP / submodular——若 adaptive cluster 失败的备选；
6. 跨 backbone family 的扩展——明确放到 future work。

---

## 19. 文件 → 模块 → 责任清单

| 模块路径 | 主要责任 |
|---|---|
| [src/pcu_select/types.py](../src/pcu_select/types.py) | 全局 dataclass / 协议定义 |
| [src/pcu_select/data/](../src/pcu_select/data/) | dataset、sample iter |
| [src/pcu_select/peft_space/schema.py](../src/pcu_select/peft_space/schema.py) | PEFTConfig 解析 |
| [src/pcu_select/peft_space/site_mask.py](../src/pcu_select/peft_space/site_mask.py) | `site_mask_of`, `alpha_pω` |
| [src/pcu_select/peft_space/encoder.py](../src/pcu_select/peft_space/encoder.py) | z_p 构造 |
| [src/pcu_select/features/](../src/pcu_select/features/) | e_x / d_x / a_x |
| [src/pcu_select/task_cond/encoder.py](../src/pcu_select/task_cond/encoder.py) | z_t 池化 |
| [src/pcu_select/proxy/hooks.py](../src/pcu_select/proxy/hooks.py) | 站点 forward / grad hook |
| [src/pcu_select/proxy/projection.py](../src/pcu_select/proxy/projection.py) | 随机投影矩阵 |
| [src/pcu_select/proxy/lo_fidelity.py](../src/pcu_select/proxy/lo_fidelity.py) | u_lo 计算 |
| [src/pcu_select/hi_fidelity/anchors.py](../src/pcu_select/hi_fidelity/anchors.py) | anchor checkpoint 管理 |
| [src/pcu_select/hi_fidelity/short_update.py](../src/pcu_select/hi_fidelity/short_update.py) | h-step PEFT-only 更新 |
| [src/pcu_select/hi_fidelity/sampler.py](../src/pcu_select/hi_fidelity/sampler.py) | 三阶段三元组采样 |
| [src/pcu_select/hi_fidelity/labeler.py](../src/pcu_select/hi_fidelity/labeler.py) | u_hi 计算与 RankNorm |
| [src/pcu_select/scorer/model.py](../src/pcu_select/scorer/model.py) | scorer nn.Module |
| [src/pcu_select/scorer/losses.py](../src/pcu_select/scorer/losses.py) | rank / reg / proxy / NLL |
| [src/pcu_select/scorer/trainer.py](../src/pcu_select/scorer/trainer.py) | 两阶段训练循环 |
| [src/pcu_select/scorer/inference.py](../src/pcu_select/scorer/inference.py) | 批前向打分 |
| [src/pcu_select/selection/cluster.py](../src/pcu_select/selection/cluster.py) | MiniBatch KMeans |
| [src/pcu_select/selection/adaptive_quota.py](../src/pcu_select/selection/adaptive_quota.py) | 簇配额 + 簇内 top-k |
| [src/pcu_select/selection/selector.py](../src/pcu_select/selection/selector.py) | 端到端 selector |
| [src/pcu_select/ood/calibration.py](../src/pcu_select/ood/calibration.py) | Mahalanobis + calibration head |
| [src/pcu_select/pipeline/offline.py](../src/pcu_select/pipeline/offline.py) | offline 全流程编排 |
| [src/pcu_select/pipeline/apply.py](../src/pcu_select/pipeline/apply.py) | apply 全流程编排 |
| [src/pcu_select/cost/accounting.py](../src/pcu_select/cost/accounting.py) | gpu-hours 与 break-even |

代码骨架见对应文件。

---

## 20. 验证设计正确性的三个最小实验（M3 之后即可跑）

1. **相关性实验**：抽 200 个 `(x, p, t)` 三元组，计算 `u^lo`，再算高保真 `u^hi`，比对 Spearman ρ。若 ρ < 0.3，则站点定义或 α 公式需修。
2. **PEFT condition 必要性**：训练两个 scorer——有/无 z_p 输入。在 hold-out 三元组上比较 NDCG@K。差距应 ≥ 5%。
3. **同等成本基线对比**：把本方法和 RDS+ / PPL / LESS 限制在相同 wall-time 内运行，看选出的子集在目标 PEFT 上的训练后 val loss。本方法应至少持平或胜出，特别是在多 PEFT 共享 scorer 时。

只要这三个实验通过，整体设计闭环成立，后续是工程优化与消融实验。
