# PCU-Select

PEFT-Conditional Utility Selection — 跨 PEFT 复用的数据高效微调框架。

详见：
- [研究方案（原版）](docs/pcu_select_research_plan.md)
- [完整设计文档（实现版）](docs/pcu_select_design.md)

## 快速结构

```
src/pcu_select/
  types.py              # 数据契约
  data/                 # 候选池、sketch loader
  peft_space/           # PEFT schema、site mask、z_p 编码
  features/             # e_x / d_x / a_x
  task_cond/            # z_t set-pooling
  proxy/                # 站点 hook、随机投影、u_lo
  hi_fidelity/          # anchor、短程更新、采样、u_hi
  scorer/               # 多塔 + FiLM + 双塔损失 + 训练
  selection/            # 聚类 + 自适应配额
  ood/                  # Mahalanobis + calibration
  pipeline/             # offline / apply 编排
  cost/                 # GPU-hours + break-even
```

