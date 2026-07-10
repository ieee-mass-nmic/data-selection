附录表 17 是非常明显的不一致性。

论文称：

Reuse-one-LESS 是在源配置 L-r8-qv 上运行一次 LESS；
在目标同样为 L-r8-qv 时，它与 per-PEFT LESS “by construction” 相等；
使用的是三个 matched target-training seeds。

但表中给出的结果是：

Reuse-one-LESS：34.80 ± 0.67
Per-PEFT LESS：34.80 ± 0.40

如果二者在该列确实是同一选中子集、同一目标 PEFT、同一训练配置、同一组种子，并且“按构造相等”，那么逐种子结果应当相同，因而平均值和标准差都应相同。现在只是平均值恰好相同，而标准差不同。