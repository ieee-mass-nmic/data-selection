最严重的问题是 GPU-hour 口径。论文明确说 GPU-hours 按 “wall seconds × GPU count / 3600” 计算，并且使用 8×A100-80GB。它又声称目标训练是 Llama-2-7B，global batch 128，1000 steps，max length 1024，10% 预算下约 128K sample-passes、约 4.3 个 epoch。

但同一篇论文又说 shared target-training cost 只有 0.83 GPU-hours per configuration。按它自己的定义，8 张 GPU 下 0.83 GPU-hours 只等于约 6.2 分钟 wall time，也就是 1000 step 约 0.37 秒/step。对 Llama-2-7B、batch 128、最长 1024 tokens 的 PEFT 训练来说，这个数字非常可疑；即使冻结 backbone，只训练 adapter，也仍需完整前向和大量反向传播。论文没有给出吞吐、平均序列长度、packing、flash-attn、梯度检查点等能支撑这个速度的证据。

第二个成本矛盾更直接。论文称 high-fidelity label 平均每个 triple 需要 1.89 wall-clock seconds，8 GPU 计费下，500 个 label 的成本应约为：

500×1.89×8/3600≈2.1 GPU-hours

但论文在 OOD 校准部分说 500 个校准 label 只增加 0.60 GPU-hours。除非校准 label 的生成流程显著不同、复用了已有标签、或没有按同样 GPU-hour 口径计费，否则这个数字与前文不一致。论文没有解释这种差异。

这两个问题都直接影响本文最核心的“amortization / cost saving”主张。性能结果只是 near-tie，真正卖点是 72.9 vs 158.0 GPU-hours；如果成本口径错了，论文核心结论会被削弱。