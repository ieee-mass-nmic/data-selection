# README — 合成实验数据（SYNTHETIC DATA）

> ⚠️ **重要声明：本目录下的数字不是真实实验结果。**
>
> 这些数据由 [`gen_synthetic_data.py`](gen_synthetic_data.py) **人工合成**，用于在真实
> 离线/GPU 流水线尚未运行时，让既有的绘图脚本（`scripts/plots/plot_*.py`）有可渲染的
> 输入，便于检查图表样式、论文排版与代码连通性。
>
> **没有**加载任何 7B backbone，**没有**做任何 PEFT 微调，**没有**做任何短程更新
> （short-update）标注。所有 `metric` / `eval_loss` / `u_hi` 等数值都是按"符合常识的量级 +
> 合理噪声 + 故意的不完美"由固定随机种子生成的**占位数**。**严禁**把它们写进论文当作实验
> 证据，或据此下任何科学结论。

唯一的例外见下文 **Table 1**：它是由代码库自身的解析式参数计数器**真实计算**得到的
（零 GPU、与 `z_p` 编码器同源），不是合成数。

---

## 0. 如何复现

```bash
pip install -e ".[viz]"            # 需要 matplotlib / pyarrow

# 1) 重新生成全部合成数据（固定 master seed = 20260629，完全可复现）
python result/gen_synthetic_data.py

# 2) 重新生成 Table 1（真实计算，零 GPU）
python scripts/experiments/build_table1.py --model llama2-7b --out-dir result/tables

# 3) 重新绘制全部图表
cd scripts/plots
R=../../result/data; F=../../result/figs
python plot_e1.py            --results $R/E1.jsonl --out-dir $F
python plot_e2.py            --results $R/E2.jsonl --cost-model $R/E2_cost_model.json --out-dir $F
python plot_e3.py            --results $R/E3.jsonl --out-dir $F
python plot_e4.py            --results $R/E4.jsonl --overlap $R/E4_overlap.json --out-dir $F
python plot_e5.py            --results $R/E5.jsonl --out-dir $F
python plot_motivation_f1.py --values  $R/motivation/values.parquet --signal u_hi   --out-dir $F
python plot_motivation_f1.py --values  $R/motivation/values.parquet --signal u_grad --out-dir $F
python plot_motivation_f2.py --results $R/MOT_F2.jsonl --out-dir $F
```

数据 schema 与真实 runner（`scripts/experiments/run_e*.py`）写出的
`ResultRow`（[src/pcu_select/experiments/results.py](../src/pcu_select/experiments/results.py)）
**完全一致**，因此把真实 runner 的输出替换到 `result/data/` 即可用同一套绘图脚本。

---

## 1. 目录结构

```
result/
  README_SYNTHETIC_DATA.md     ← 本文件
  gen_synthetic_data.py        ← 合成数据生成器（固定种子）
  data/                        ← 合成的原始结果（等价于真实 runs/<exp>/results/）
    E1.jsonl  E2.jsonl  E2_cost_model.json
    E3.jsonl  E4.jsonl  E4_overlap.json  E5.jsonl
    MOT_F2.jsonl
    motivation/values.parquet
  figs/                        ← 由绘图脚本渲染的图（.png）+ 脚本副产 .csv
  tables/                      ← Table 1（真实计算）+ 各图脚本导出的汇总表 .csv
```

---

## 2. 每个产物 ↔ 对应实验（设计文档 §9 图表清单）

文档依据：[实验设计](../docs/pcu_select_experiment_design.md) · [动机实验](../docs/pcu_select_motivation_design.md)。

### 动机实验（Motivation — intro 章，先导验证「数据价值随 PEFT 改变」）

| 产物 | 文件 | 对应 | 数据源（合成函数） |
|---|---|---|---|
| **Table 1** | `tables/table1.{md,csv,tex}` | 动机 §2：PEFT 结构差异（改的位置/算子/参数量横跨两个数量级） | **真实计算**（`build_table1.py`，非合成） |
| **Figure 1** | `figs/F1_disagreement_u_hi.png` · `figs/F1_disagreement_u_grad.png` · `tables/F1_structural_*.csv` | 动机 §3：逐样本价值排序的不一致 + 噪声地板 | `gen_motivation_f1()` → `data/motivation/values.parquet` |
| **Figure 2** | `figs/F2_transfer.png` · `tables/F2_transfer_raw.csv` | 动机 §4：跨 PEFT 迁移矩阵（对角占优） | `gen_motivation_f2()` → `data/MOT_F2.jsonl` |

### 主实验 E1–E5

| 产物 | 文件 | 对应 | 数据源 |
|---|---|---|---|
| **T1** | `figs/T1_method_x_peft.png` · `tables/T1_method_x_peft.csv` | **E1** 方法 × PEFT × 任务大表（budget=10%, 7B） | `data/E1.jsonl` |
| **F1** | `figs/F1_budget_sensitivity.png` | **E1** budget(5/10/30%) 敏感性 | `data/E1.jsonl` |
| **F2** | `figs/F2_break_even.png` | **E2** 总 GPU-h vs T，break-even T\* | `data/E2.jsonl` + `data/E2_cost_model.json` |
| **F3** | `figs/F3_pareto.png` | **E2** 性能 vs 总成本 Pareto | 同上 |
| **T2** | `figs/T2_ablation.png` · `tables/T2_ablation.csv` | **E3** 消融逐项掉点（性能 + NDCG@K） | `data/E3.jsonl` |
| **F4 / F4b** | `figs/F4_alpha_sweep.png` · `figs/F4b_strategy.png` | **E3** α 拐点 / 选择策略（轴 G） | `data/E3.jsonl` |
| **F5** | `figs/F5_mismatch.png` | **E4-c** 错配矩阵（对角占优） | `data/E4.jsonl`（`method=pcu_mismatch`） |
| **F6** | `figs/F6_config_vs_selection.png` | **E4-a** 配置差异 → 选择差异 | `data/E4_overlap.json` |
| **F7** | `figs/F7_levels_modes.png` | **E5** L0/L1/L2 × (zero-shot/cal-200/cal-500) | `data/E5.jsonl` |
| **F8** | `figs/F8_d2_vs_degradation.png` | **E5** Mahalanobis d² vs 性能衰减 | `data/E5.jsonl` |

> 注：`figs/` 下还有几个由绘图脚本顺手导出的 `.csv`（T1/T2/结构汇总/原始迁移矩阵），
> 已同步复制一份到 `tables/` 便于查阅；二者内容相同。

---

## 3. 合成数据是怎么"刻意做得不完美"的（满足生成要求）

为了避免"过度规律/过度完美"，生成器在固定种子下注入了多层结构化扰动与不利证据：

1. **保留随机波动 + 异方差 + 离群跑**：每个 (方法 × PEFT × 任务 × budget) 跑 3 个目标训练
   seed；噪声是**异方差的**（每个 cell 的 σ 在 0.8–1.35×base 间按确定性键变化），并以 ~5%
   概率叠加一次"特别好/特别差的训练跑"离群扰动——而非处处同一个干净高斯。

2. **方法排序不是全局一致（列序逐任务抖动）**：引入 `方法×任务` 与 `方法×PEFT` 的确定性
   交互项，使得**每个任务的方法名次会重排**，中段基线（RDS+/Diversity/IFD/S2L）经常彼此
   反超，而不是每列都是同一条单调梯度。

3. **方法不总是占优（少数掉队）**：
   - **E1/T1**：PCU 在 **5 个 SEEN PEFT 中赢 4 个**，在 `L-r8-mlp` 上**输给 LESS**
     （对应设计 §2.4 弱回退主张 B）。**逐任务**看：PCU 拿下 GSM8K/MMLU，**LESS 拿下
     HumanEval/TyDiQA**；二者**总均值几乎打平**（PCU 仅高 ≈0.1 个点），20 个 (PEFT×任务)
     最优格 **PCU 与 LESS 各占 10**——没有任何"横扫"。
   - **E2**：PCU 相对**便宜的 RDS+/Random 在原始成本上永远不占优**（无 break-even），
     只对每 PEFT 重算的影响力方法（LESS/Influence）才在 **T\*≈5.2** 后摊薄获胜——
     这是一个**诚实的负面结论**。

4. **不让所有实验支持同一结论**：
   - **E5-L2**：`PRE-l16`/`PT-l32`（prefix/ptuning）**无法生成校准标签**，zero-shot
     **直接失败、甚至低于 Random**（设计 §6.5 失败边界）；同属 L2 的 `BF` 可正常校准并恢复。
     且**同一 level 内各配置不再是同一个数**（按 d² 大小有不同程度衰减/恢复）。
   - **E3**：`pcu_no_fingerprint` 几乎不掉点（≈ 0），即"fingerprint 模块无明显增益、应降级为
     可选"——对方法**不利**的消融结论。各消融的掉点幅度**随 (PEFT,任务) 变化**，且排序指标
     NDCG 与下游 metric **不完全同向**（如 α=0 的 NDCG 偏高但下游偏低）。

5. **非线性 + 非对称的几何**（而非干净直线/对称梯度）：
   - **E4-c / F5 错配矩阵**：错配掉点对 `(src,tgt)` **非对称**、对配置距离**非线性**，并带较大
     噪声——少数近对角格甚至**略高于对角**（1.03–1.05），但**平均仍对角占优**（7 列中 6 列对角
     为该列最大）。
   - **E4-a / F6**、**动机 F2 迁移矩阵**：重叠/迁移随结构距离的衰减是**带曲率 + 逐对散点**的，
     不是一条直线。

6. **不夸张**：方法间差距是个位数百分点（PCU 比 LESS 总均值仅高 ≈0.1）、α 有**内部最优**
   （0.6，两端更差）、calibration 有收益但**补不满**（cal-500 接近但不超过 LESS 上界）。

7. **多 seed 合理方差 + 严格可复现**：master seed = `20260629`，各实验用 `seed+常数` 派生独立
   RNG；所有"确定性不均匀扰动"用 **md5(repr) 哈希**实现（修复了早期用 Python 内建 `hash()`
   导致跨进程不可复现的问题）。重跑生成器**逐字节一致**（已校验 md5 相同）。

8. **动机实验的噪声地板成立**：F1 的同一 PEFT 自一致性 ρ_intra≈0.79 **高于**跨 PEFT 的
   ρ_inter，且分歧**随结构距离单调增大**（同容量 0.66 > 同 family 换 placement 0.44 >
   跨 family 0.28），全部低于噪声地板（动机 §3.3 判据）。每个 PEFT 的共享分量载荷带**确定性
   抖动**，使非对角相关值**散开**、个别对**打破严格单调**，而非落在整齐的 bucket 常数上。

---

## 4. 量纲约定

- 主指标 `metric`：**越大越好**，取任务原生单位（GSM8K=Exact-Match %、HumanEval=pass@1 %、
  MMLU=accuracy %、TyDiQA=F1）。`metric_name` 字段标注具体指标。
- `eval_loss`：始终可得的 held-out response-LM 损失（与 metric 弱反相关，仅作占位）。
- 成本字段 `select_gpu_h` / `target_train_gpu_h` / `E2_cost_model.json`：单位 GPU-hours，
  离线一次性成本（feat+lo+hi+scorer）约 31.4 GPU-h，影响力基线每 PEFT 重算 6.0 GPU-h。
- 排序指标 `spearman/kendall_tau/ndcg_at_k/topk_hit_rate/pairwise_acc`：对照高保真真值的
  机制性指标，仅对有 dense 打分的方法填值，Random/Length 等留空（NaN）。

---

## 5. 用真实结果替换

当真实离线流水线 + E1–E5 runner 跑通后，把 `runs/<exp>/results/` 下的真实文件复制/软链到
`result/data/` 同名位置（schema 一致），即可用 §0 的同一组绘图命令得到真实图表，并**删除本
README 的合成声明**。在此之前，请始终把本目录视为"样式占位"，不得作为实验证据引用。
