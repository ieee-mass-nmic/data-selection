# PCU-Select

PEFT-Conditional Utility Selection — 跨 PEFT 复用的数据高效微调框架。

详见：
- [研究方案（原版）](docs/pcu_select_research_plan.md)
- [完整设计文档（实现版）](docs/pcu_select_design.md)

## 本地验证

项目要求 Python >= 3.10；本地优先使用仓库虚拟环境或其他 3.10+ 环境。

```bash
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m mypy
.venv/bin/python -m pytest -q
```

轻量流程 smoke test 不会下载 7B 模型或启动 GPU 训练，可用来确认脚本入口和
PEFT registry 物化链路：

```bash
.venv/bin/python scripts/experiments/dump_peft_registry.py \
  --model llama2-7b \
  --out-dir /private/tmp/pcu_registry_smoke
```

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
