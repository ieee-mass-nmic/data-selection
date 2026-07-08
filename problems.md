
## 一、总体审稿结论

**建议：大修后再审 / 若为顶会短周期审稿，倾向 Weak Reject。**

论文选题有价值：它指出数据选择不应只依赖“任务—样本”关系，还应显式依赖 PEFT 配置，因为 LoRA、Adapter、IA³ 等方法暴露的可训练子空间不同，样本梯度在不同子空间中的效用可能发生变化。这一问题定义较清晰，PCU-Select 也给出了比较完整的工程化方案，包括 intervention-site gradient signature、结构化 PEFT 编码、多保真 utility label、uncertainty penalty 与 coverage-aware selection。

但目前稿件仍存在若干会影响审稿判断的问题：一是主实验和附录之间有不少维度、编号、表述和交叉引用不一致；二是核心实验主要集中在一个 backbone、四个任务和五个 seen PEFT 配置上，泛化性证据不足；三是主文过度依赖异质指标的平均值，虽然作者在附录中有所补充，但主结论仍容易被理解为比实际更强；四是 PCU-Select 相比 RDS+、Influence 的收益并不大，相比 per-PEFT LESS 只是统计等价而非显著优越，因此需要更强的成本—性能论证和更严谨的消融实验；五是 PEFT 编码、OOD family embedding、24-site 近似、高保真标签生成等关键技术细节还没有完全讲清楚。

整体而言，论文有明确创新点和潜在发表价值，但需要在实验严谨性、方法可复现性、表述一致性和结论边界方面做较大幅度修订。

---

## 二、论文主要优点

1. **问题切入点较新。** 现有数据选择方法通常把样本价值看作任务相关但 PEFT 无关的量，而本文强调 PEFT 配置会改变可训练子空间，因此样本 utility 应该是 (u(x,p,t)) 而不是 (u(x,t))。这个问题定义是论文最有价值的部分。

2. **方法设计具有结构化意识。** PCU-Select 没有简单使用 PEFT family one-hot，而是尝试编码 site mask、capacity、operator type、training recipe，并用 24 个 intervention sites 作为样本、任务与 PEFT 之间的公共坐标系。这比单纯的 family-conditioned selector 更有说服力。

3. **实验包含较强基线。** 论文不仅比较 Random、RDS+、Influence，还加入 per-PEFT LESS 作为强基线，并明确承认 PCU-Select 与 LESS 是统计等价而非显著胜出。这一点比很多只与弱基线比较的稿件更严谨。

4. **成本分析是论文的重要贡献。** 作者没有只报告性能，还讨论了 PCU-Select 的 offline cost 如何在多个 target PEFT 配置上摊销，并给出与 LESS per-target gradient recomputation 的 break-even 分析。

5. **附录信息量较大。** 附录补充了 per-task results、baseline specifications、candidate pool provenance、decontamination、PEFT registry、算法伪代码和 OOD transfer 结果，这些内容对复现和审稿都有帮助。

---

## 三、主要问题与审稿关注点

### 1. 结论强度略高于当前证据

主文和摘要强调 PCU-Select 比 PEFT-agnostic selectors 更好，并与 per-PEFT LESS 统计等价。但从 Table 2 看，PCU-Select 相比 RDS+ 的平均收益为 **0.74 points**，相比 Influence 的平均收益约 **0.50 points**，相比 LESS 只有 **0.04 points**。这些收益并不大，尤其是在四个任务指标尺度不同的情况下，平均值的解释空间有限。

作者在附录 Table 5 中承认 PCU-Select 对 LESS 的表现是任务依赖的：MMLU 上优于 LESS，TyDiQA 上显著落后，GSM8K 与 HumanEval 基本在噪声范围内。这个信息非常关键，不应主要放在附录中。建议将 per-task 对 LESS 的比较移入主文，否则主结论容易显得过度概括。

### 2. 主实验的泛化性有限

论文的主要结论来自一个 backbone family，即 Llama-2-7B；任务只有 GSM8K、HumanEval、MMLU、TyDiQA；seen PEFT 配置只有 5 个。对于一个“PEFT-conditioned”方法而言，PEFT 配置空间很大，而 5 个 seen configurations 对一个 192 维 PEFT code 来说支撑不足。

尤其需要澄清：主实验中的四个任务是否都参与了 scorer 训练？seen PEFT 配置是否也参与了 high-fidelity utility label 构建？如果 PCU-Select 在相同 task 和 PEFT configuration 上获取过 high-fidelity labels，再用于这些 target 的 downstream selection，那么这更像是“已见配置上的 amortized selector”，而不是对新 task 或新 PEFT 的泛化。论文目前对此边界说得不够直接。

### 3. 24-site 表示是核心假设，但缺少充分验证

PCU-Select 的核心近似是用 8 个层 × 3 个 module-output sites，即 24 个 sites，作为所有 PEFT 方法的公共坐标系。这个设计直观合理，但也是方法成败的关键假设。论文自己也说完整 site-space ablation 留给未来工作，这对主贡献来说偏弱。

例如，LoRA q/v 与 LoRA q/k/v/o 在 site-level abstraction 中可能都映射到 attention-output sites，仅通过 capacity 或 target-module set 区分；但 q、k、v、o 投影的梯度几何差异可能很大。类似地，IA³ 的 multiplicative scaling、adapter 的 residual bottleneck、prefix tuning 的 KV prefix 并不一定能被同一类 module-output gradient 充分近似。若不做 site-space granularity 的消融，方法的解释力会受到质疑。

### 4. 方法细节存在若干不一致或不清楚之处

主文第 4 页附近写 PEFT code (z_p \in \mathbb{R}^{192})，由 96 维 site/operator mask、16 维 capacity、16 维 recipe、64 维 family/operator embedding 组成；但 Figure 2 中 PEFT repr 标注为 “128 = mask96 + cap16 + recipe16”，缺少 64 维 embedding。这是明显不一致。

此外，文中说 site/operator mask 是 (24 \times 4)，但 operator prior 又列出 additive low-rank、multiplicative gates、bottlenecks、prefix、bias shifts 五类。这里到底是 4 类 operator channel，还是 5 类 operator family？BitFit 如何编码？Prefix/P-tuning 的 unseen family embedding 如何初始化？这些都需要明确说明。

### 5. 高保真 utility label 与真实 full-training utility 的关系仍不充分

论文使用 short-horizon sketch-loss reduction 作为 high-fidelity utility label，并承认它不是 full-training marginal contribution。这个设计有现实意义，但必须证明它与最终 downstream selection performance 有稳定相关性。当前论文主要报告 ranking metrics 和 downstream results，但缺少直接分析：short-horizon utility 与 full fine-tuning 后的样本贡献或子集表现之间的相关性有多强？在 TyDiQA 上 PCU-Select 落后 LESS，说明该 proxy 对某些任务可能偏差明显。

### 6. 统计分析仍需增强

论文对 PCU-Select vs LESS 做了 paired difference、bootstrap CI、Wilcoxon 和 TOST，这比一般稿件更完整。但仍有几个问题：

第一，20 个 PEFT × task cells 并非完全独立，任务和 PEFT 配置之间存在嵌套结构。作者提到 task-stratified bootstrap，但需要更明确地描述 bootstrap unit 和 resampling procedure。

第二，TOST 的 ±1.0-point equivalence margin 用在四个异质任务指标的平均值上并不自然。一个 MMLU accuracy point、一个 TyDiQA F1 point 和一个 HumanEval Pass@1 point 的实际意义不同。建议使用 task-normalized score 或 per-task equivalence margin。

第三，当前 std 主要来自 target-training seeds，但 selection scorer、task sketch、high-fidelity label sampling、candidate ordering 等选择过程是否固定为 seed 0？如果固定，则误差条低估了实际 pipeline variance。

---

## 四、短期内可以即时调整的修改意见

以下修改不需要新增大规模实验，主要涉及内容一致性、表述严谨性、补充说明、排版和结果呈现。

### A. 摘要与贡献表述

1. **降低摘要中的平均性能表述强度。**
   摘要中报告 “improves 10% budget average over random by 2.89 and over RDS+ by 0.74” 是可以的，但应紧接说明该 average 是 heterogeneous task-native metrics 的 arithmetic average，不应被解释为统一尺度上的绝对提升。建议改成：
   “On the arithmetic average of task-native percentage metrics, used only as a compact summary and accompanied by per-task results, …”

2. **把 “statistically equivalent to LESS” 的边界说得更清楚。**
   建议在摘要中写明：这是在 **seen PEFT configurations**、当前四个任务和 Llama-2-7B backbone 上成立；对于 unseen families 需要 calibration。否则读者容易误解为 PCU-Select 普遍替代 LESS。

3. **避免 “consistently improves” 这类可能过强的表述。**
   PCU-Select 相比 PEFT-agnostic selectors 在主表平均上提升，但相比 LESS 在 TyDiQA 上明显落后。建议把 “consistently improves over selectors” 限定为 “PEFT-agnostic selectors” 或 “on average across the evaluated cells”。

4. **明确论文主要卖点是 amortization，而不是性能显著超过 LESS。**
   目前摘要中已经提到成本摊销，但建议更突出：PCU-Select 的价值在于接近 LESS 的质量，同时避免每个 PEFT 配置重新构建 gradient datastore。

### B. 引言和问题定义

5. **修复缺失交叉引用。**
   文中多处出现 “Sec. .”、“Appendix ”、“Appendix Table 11” 等不完整引用。需要全文检查 LaTeX labels，确保所有章节、表格、图和附录编号正确。

6. **把 cross-PEFT 与 cross-backbone 明确区分。**
   论文标题和引言强调 cross-PEFT，但当前证据不是 cross-backbone。建议在引言贡献或 Limitations 中提前说明：本文研究固定 backbone family 下的 cross-PEFT data selection。

7. **解释 “same-family capacity changes” 与 “placement changes” 的定义。**
   Figure 1 和 Figure 4 都依赖 PEFT configuration distance，但主文对 rank、module set、layer placement、family distance 的定义较晚且不够直观。建议在引言或方法中提前给一个简短定义。

8. **更清楚地区分 motivation study 与 main evaluation。**
   Figure 1 使用 20K candidate examples、two task sketches、independent anchor/seed replicates；但读者不清楚这些 sketches 来自哪些任务，是否对所有任务平均，是否使用了同一 pool。建议在 Figure 1 caption 或正文补充实验设置。

### C. 方法部分细节

9. **修正 PEFT code 维度不一致。**
   主文说 (z_p \in \mathbb{R}^{192})，Figure 2 标注为 128。应统一。如果最终采用 192 维，Figure 2 应改为：
   “PEFT repr (z_p): 192 = mask96 + cap16 + recipe16 + family/operator64”。

10. **说明 site/operator mask 的 operator channel 到底有几类。**
    若是 (24 \times 4)，请列出 4 个 channel；若包含 low-rank、multiplicative、bottleneck、prefix、bias-shift 五类，则应改为 (24 \times 5) 或解释 bias-shift 如何并入某个 channel。

11. **补充 PEFT family/operator embedding 的 OOD 初始化方式。**
    对 Prefix、P-Tuning、BitFit 等 unseen families，如果 family/operator embedding 是 learned embedding，那么未在训练支持中出现的 family 如何获得 embedding？是随机初始化、operator-level 共享、手工编码，还是 calibration 后学习？这会直接影响 Figure 5 的可信度。

12. **明确 target-module set 如何进入 PEFT code。**
    LoRA q/v 与 q/k/v/o 在 24-site 层面可能都激活 attention-output sites。需要说明 q、k、v、o 的差异是通过 recipe vector、capacity vector、operator mask，还是额外 module-set encoding 表达。

13. **补充 16 维 difficulty/format statistics 的具体列表。**
    目前只说 sample representation 包含 16 difficulty and format statistics，但没有列出。建议在主文或附录列出全部统计量，例如 token length、response length、instruction length、code flag、math flag、language flag、perplexity、loss 等。

14. **说明 768 维 semantic embedding 的来源。**
    RDS+ 和 PCU-Select 都使用该 embedding，因此必须说明 frozen external sentence encoder 的具体模型、版本、是否多语言、是否在候选池或 benchmark 上训练过、是否会引入额外数据优势。

15. **定义 module-output gradient signature 的损失函数。**
    文中说 response-token loss gradient，但 MMLU 是 multiple-choice log-likelihood scoring，TyDiQA 是 span answer generation，HumanEval 是 code generation。需要说明对不同任务 sketch 计算 (g_t^\omega) 时使用的 loss 形式是否一致。

16. **解释 high-fidelity label 中 “sample + 7 in-bucket fillers” 的影响。**
    附录说每个 high-fidelity update 使用 sampled example 加 7 个 in-bucket fillers。这样 label 不再是单个样本的纯效用，而是含有 filler 影响的 mini-batch utility。建议说明是否对 filler 做了随机重复、是否减去了 filler-only baseline、label 方差如何控制。

17. **补充 low-fidelity utility 与 high-fidelity utility 的归一化关系。**
    Eq. (4) 的 cosine 加权和可能为负，而问题定义中 (u(x,p,t)\in[0,1]) 是 rank-normalized short-horizon utility。需要说明 low-fidelity proxy 是否也 rank-normalized，何时归一化，ranking loss 是在同一 (p,t) bucket 内还是跨 bucket。

18. **明确 boundary sampling 的流程。**
    高保真标签的 10K triples 分为 coverage、uncertainty、boundary sampling，但 boundary sampling 所谓 “scorer-vs-true rank disagreement” 在 true label 未知时如何获得？是否是迭代式 active learning？需要写清楚。

19. **说明 uncertainty (\hat{\sigma}) 的校准方式。**
    附录提到 ECE=0.043，但主文只把 (\hat{\sigma}) 作为 penalty。建议主文简述 uncertainty 不是概率置信度，而是 heteroscedastic regression error proxy，并说明其验证方式。

20. **解释 cluster quota 中 (v_k^+) 的定义。**
    Eq. (7) 中 (v_k^+) 在主文没有足够直观说明。建议明确：(v_k) 是 cluster 内 top 10% examples 的 mean conservative utility，(v_k^+=\max(v_k,0))。

21. **说明 (k=\max(50,\lfloor\sqrt{N}\rfloor)) 对 300K pool 的实际值。**
    对 300K，(\sqrt{N}\approx548)，cluster 数较大。建议报告实际 cluster 数、平均 cluster size，以及 clustering 的 CPU/GPU 时间。

### D. 实验设置与结果呈现

22. **将 per-task 结果提前到主文。**
    Table 4 和 Table 5 目前在附录，但它们对解释主结果至关重要。建议主文至少加入一个 compact per-task table，显示 PCU-Select vs RDS+、Influence、LESS 的 per-task 平均差异。

23. **在 Table 2 caption 中明确平均方式。**
    目前 Table 2 的列是 PEFT 配置，值是四个任务 native metrics 的平均。建议 caption 明确：每个 PEFT 列是 GSM8K EM、HumanEval Pass@1、MMLU Acc、TyDiQA F1 的 arithmetic mean after percentage scaling。

24. **为 PCU-Select vs RDS+、Influence 也报告统计检验。**
    论文对 LESS 的统计检验很详细，但对 RDS+ 和 Influence 主要只给平均差。建议补充 paired bootstrap CI 和 Wilcoxon test，至少给出 PCU-Select over RDS+ 的 95% CI。

25. **解释 target-training seeds 与 selection seeds 的区别。**
    附录说 downstream std 来自 3 个 target-training seeds，而 selection-scorer training、task sketch sampling 和 candidate-pool ordering 固定为 seed 0。建议在主文说明，否则读者会误以为误差条包含整个选择流程的不确定性。

26. **补充 selection stability。**
    即使不新增大规模实验，也可以用已有 Figure 1 或 Table 12 结果说明 selected subset 的 Jaccard stability。建议报告 PCU-Select 在不同 sketch seeds 或 scorer seeds 下的 overlap 和 downstream variance。

27. **修正 Full-pool reference 的表述。**
    “PCU-Select at 10% recovers 96% of full-data average” 使用异质指标平均值，解释性有限。建议改为 per-task recovery ratio，或把该句放到附录，避免主文过度强调。

28. **说明 30% budget 超过 full-pool 的原因时更谨慎。**
    论文说 30% PCU-Select 稍高于 full pool，因为能去除 noisy examples。这个解释合理但未被直接验证。建议改成：“possibly because selected subsets discard low-utility/noisy examples, although we do not claim this as a measured oracle effect.”

29. **补充 token budget 统计。**
    当前预算按 example count 计算，但不同 selector 可能选择长度分布不同的样本，导致 token budget 和训练 compute 不一致。建议至少报告每个 selector 选择子集的平均 token length、总 token 数，说明差异是否显著。

30. **澄清 fixed 1000 steps 与不同 budget 的关系。**
    在 5%、10%、30% budget 下固定 1000 steps 会导致不同 epoch 数。建议说明这是 compute-controlled 还是 step-controlled 设置，并补充 equal-epoch 或 equal-token 结果到长期计划中。

31. **补充 MMLU、HumanEval、TyDiQA 评测细节。**
    MMLU 需说明是 zero-shot、few-shot 还是只做 option likelihood；HumanEval 需说明 Pass@1 的 sampling seed、unit-test harness；TyDiQA 需说明语言集合、answer normalization 和 prompt 模板。

32. **把 Balanced-Random 的结果补进附录。**
    主文说 Balanced-Random “fold into appendix”，但当前附录没有看到对应完整表格。建议补充，或删去“in appendix”这类承诺。

33. **补充所有提到的 heuristic baselines 的结果。**
    文中提到 length、loss、perplexity、IFD、embedding-nearest-neighbor、diversity clustering 等 baseline，但主表只呈现 Random、RDS+、Influence、LESS。建议至少在附录提供完整 baseline table，否则会被认为选择性报告。

34. **LESS 的 “faithful reimplementation” 需要更谨慎。**
    如果没有与原 LESS 公开结果做 sanity check，建议改成 “our per-PEFT LESS-style implementation”。若保留 “faithful”，需说明与原 LESS 的关键实现一致性，包括 warmup checkpoints、Adam preconditioning、projection dimension、validation sketch influence 等。

35. **Table 10 的 NDCG@K 定义需要澄清。**
    K 被说成 10% budget size，但 high-fidelity held-out labels 只有有限 triples。NDCG@K 是在 300K pool 上计算，还是在 held-out labeled subset 上计算？如果是在 labeled subset 上，K 不能直接等于 30K。需要修正定义。

### E. OOD 与 calibration 部分

36. **把 Figure 5 与 OOD 讨论放在同一位置。**
    Figure 5 出现在 references 前，但主文对它的引用和解释较短。建议在 OOD subsection 中直接放 Figure 5 或 Table 13 的 compact 版本。

37. **说明 Mahalanobis distance 在仅有 5 个 seen configurations 时如何稳定估计。**
    文中说使用 shrinkage-regularized covariance，并只做粗粒度 level assignment。建议给出 shrinkage 参数、阈值设置、是否使用 leave-one-config validation 校准。

38. **明确 calibration labels 的成本组成。**
    论文说 500 labels cost 0.23 GPU-hours。建议拆分为 label generation、residual head fitting、selection scoring，并与 LESS 的 per-target cost 放在同一表中。

39. **说明 calibration 是否改变 break-even 点。**
    主文说 OOD calibration shifts break-even to at most 1.7。但如果多个 OOD target 都需要 calibration，成本应是 (11.2 + 0.23T) 而不是一次性 0.23。建议明确公式。

40. **OOD family 的 zero-shot 结果要谨慎解释。**
    如果 Prefix/P-Tuning/BitFit 的 embedding 未在 scorer training 中出现，zero-shot PCU-Select 的机制需要解释。否则建议弱化 OOD-family zero-shot claim，只强调 calibration 后可恢复。

### F. 图表、排版与文字

41. **Figure 1 的 diagonal 说明容易混淆。**
    热图 diagonal 是 self-correlation 1.00，但 caption 又提到 independent same-PEFT replicates ρ=0.79。建议在图中单独用旁注或小表展示 replicate agreement，避免读者误以为 diagonal 代表重复实验一致性。

42. **Figure 2 的字体和维度标注需要修正。**
    图中信息量大，字体偏小。建议简化为主流程图，把维度和公式放到图注或表格中。尤其要修正 PEFT repr 128/192 的矛盾。

43. **Figure 3 caption 应明确是 selection-only GPU-hours。**
    当前正文说明 shared target-training cost 被排除，但图标题和 y-axis 可能让读者误解为总训练成本。建议 y-axis 改为 “selection-only GPU-hours”。

44. **Figure 4 的 metric 单位需要说明。**
    Caption 中说 mean matched metric 21.25 vs mismatched 19.79，但没有清楚说明这是什么任务或哪些任务的平均。建议写明是 LoRA configuration scan on which tasks/metrics。

45. **Table 6 与 decontamination 数量需要统一。**
    Table 6 的 source counts 加总为 300K，但附录又说 benchmark decontamination removes 1,742 candidates。需要说明 decontamination 是在 300K finalization 前还是后；如果后又补样保持 300K，需要说明 replacement procedure。

46. **补全或删除 “Competition Disclosure”。**
    附录第 16 页出现 “Competition Disclosure” 标题但没有内容。这会给人 unfinished draft 的印象。若目标会议要求 disclosure，请补齐；否则删除。

47. **统一 IA³、PEFT、LoRA scaling 等符号。**
    PDF 中部分 IA³ 排版断裂，LoRA scaling α 与 Eq. (3) 中 site weight (\alpha_p^\omega) 容易混淆。建议把 LoRA scaling 写成 (\alpha_{\text{LoRA}})，把 site weight 写成 (w_p^\omega) 或 (\tilde{w}_p^\omega)。

48. **减少过度防御式文字。**
    论文多处强调“不把平均值解释为任务可比”“不是 oracle upper bound”“cheap selectors remain preferable”等，这些诚实但过多会影响行文流畅度。建议把关键 caveats 保留在摘要、结果和 limitations，其他地方简化。

49. **将 “released with result bundle” 改成匿名审稿兼容措辞。**
    如果当前 submission 不能公开代码或 bundle，建议写 “will be released upon acceptance” 或 “included in the supplementary anonymized artifact”，避免与匿名审稿政策冲突。

50. **检查参考文献年份和匿名性。**
    Related Work 中有 2026 年 arXiv paper。需要确认其是否真实、是否公开、是否与作者存在匿名冲突。如果是 concurrent or unpublished work，应按会议规范处理。

---

## 五、中长期可以优化和补充的修改意见

以下建议通常需要新增实验、重新训练或更大规模分析，适合大修、期刊扩展版或后续工作。

### 1. 做真正的 held-out PEFT 与 held-out task 泛化实验

当前 main results 主要是 seen configurations。建议设计三类更严格的泛化设置：

1. **Leave-one-PEFT-config-out：** 每次拿掉一个 PEFT 配置训练 scorer，在该配置上测试 selection performance。
2. **Leave-one-PEFT-family-out：** 例如训练时不看 Adapter，只在 LoRA 和 IA³ 上训练，然后测试 Adapter；或不看 IA³，测试 IA³。
3. **Leave-one-task-out：** 训练 scorer 时不使用某个任务的 high-fidelity labels，只用该任务 sketch 做 inference，测试是否能泛化到新任务。

这三类实验能直接证明 PCU-Select 的 structured PEFT code 和 task sketch 是否真的具有泛化能力，而不仅是对固定 registry 的摊销。

### 2. 扩展 backbone family

目前只有 Llama-2-7B。建议至少增加一个不同结构或不同 tokenizer 的 backbone，例如 Mistral、Llama-3 系列、Qwen 系列或更小/更大的 Llama 变体。重点不是追求 SOTA，而是验证 intervention-site representation 是否能跨 backbone family 迁移，或者至少说明必须每个 backbone family 重新构建 cache。

理想实验包括：

* 同一 candidate pool、同一任务、同一 PEFT registry，在两个 backbone 上分别训练 PCU-Select；
* 用 Llama-2-7B 上训练的 scorer 或 site prior 迁移到另一个 backbone，观察是否完全失效；
* 分析 24-site 选择在不同深度模型中的映射是否稳定。

### 3. 系统消融 intervention-site 设计

24-site abstraction 是论文核心，应补充系统消融：

* 4 层、8 层、16 层采样；
* 仅 attention sites、仅 MLP sites、attention+MLP、不含 block residual、含 block residual；
* module-output site vs projection-level site，例如 q/k/v/o/up/down/gate 分开；
* projection dimension 128/256/512；
* 是否使用 post-residual block site；
* learned site weights vs hand-designed operator priors。

如果 projection-level 表示明显提升 LoRA q/v 与 q/k/v/o 的区分能力，论文可以进一步解释当前 24-site 设计的效率—精度折中。

### 4. 验证 short-horizon utility 与 full-training utility 的相关性

建议构建小规模但更接近 oracle 的验证：

* 随机抽取若干 candidate examples 或 small subsets；
* 对每个样本/子集进行更长 horizon 的 PEFT update；
* 测量 sketch-loss reduction 与最终 downstream metric 的 Spearman/Kendall 相关；
* 分任务报告，尤其是 TyDiQA；
* 比较 low-fidelity proxy、high-fidelity short-horizon label、LESS influence 与 full-training outcome 的相关性。

这能解释为什么 PCU-Select 在 MMLU 上较好，而在 TyDiQA 上落后。

### 5. 增加更强和更全面的数据选择基线

当前主表中的 PEFT-agnostic baselines 较少。建议补充：

* 质量/复杂度/多样性类 instruction selection 方法；
* DPP/log-det/submodular diversity selection；
* IFD、perplexity、loss、length 的完整结果；
* embedding nearest neighbor、diversity clustering、RDS+ 的不同组合；
* DataInf 或 LoRA-specific influence 方法；
* “PCU without high-fidelity labels but with PEFT-weighted proxy” 作为 cheap PEFT-conditioned baseline。

尤其需要回答：PCU-Select 的收益来自 PEFT conditioning，还是来自更强的 feature/scorer/coverage machinery？

### 6. 做 token-budget 与 compute-budget 控制实验

按 example count 选择 10% 可能不公平。建议新增两种设置：

1. **固定 token budget：** 每个 selector 选取总 token 数相同的子集。
2. **固定 training FLOPs 或 wall-clock：** 控制总训练 token、steps 和 sequence length。

这能排除某些 selector 因选择短样本或长样本而获得隐性训练预算优势。

### 7. 增加 selection pipeline 的随机性评估

目前报告的是 target-training seed 方差，建议新增：

* task sketch seed variance；
* high-fidelity label sampling seed variance；
* scorer initialization seed variance；
* clustering seed variance；
* selected subset Jaccard stability；
* downstream performance stability。

如果 PCU-Select 的 subset 对 seed 很敏感，那么实际使用成本和可靠性会受影响。

### 8. 扩展 OOD calibration 曲线

目前只报告 200 和 500 labels。建议补充更细的 active calibration curve：

* 0、50、100、200、500、1000 labels；
* random calibration vs uncertainty sampling vs boundary sampling；
* calibration label cost vs downstream gain；
* calibration 后与 LESS 的 cost-performance Pareto curve。

这能更清楚地说明 PCU-Select 在 OOD PEFT family 上是否真的比 per-target LESS 更划算。

### 9. 更广泛的任务类型

四个任务覆盖数学、代码、知识选择、多语言 QA，但仍不足以支撑通用 instruction-data selection。建议加入：

* summarization；
* translation；
* dialogue/instruction following；
* safety/alignment；
* retrieval-style QA；
* long-context reasoning；
* domain-specific tasks，例如 biomedical/legal/finance；
* 更多 multilingual generation tasks。

特别是 TyDiQA 的失败表明 multilingual span extraction 与当前 site-gradient proxy 的适配不足，应专门增加多语言和抽取式任务分析。

### 10. 增加定性分析

建议展示不同 PEFT 配置下 PCU-Select 选出的样本差异，例如：

* LoRA q/v vs LoRA MLP 选出的样本在 domain、length、reasoning type 上有何差异；
* Adapter vs IA³ 选出的样本是否更偏向不同 cluster；
* PCU-Select 与 RDS+、LESS 的 top examples 有哪些重叠和差异；
* TyDiQA 中 PCU-Select 选错了哪些类型的样本。

这种 qualitative evidence 能增强论文的可解释性，也有助于说明“PEFT-conditioned utility”不是纯粹数值现象。

### 11. 进一步理论化 intervention-site approximation

目前论文主要是经验方法。若要增强理论贡献，可以补充推导：

* module-output gradient 与 PEFT parameter gradient 的关系；
* 对 LoRA、Adapter、IA³ 分别说明为何 output-site gradient 能近似其可训练子空间中的更新方向；
* 分析何时该近似失效，例如 multiplicative gates、prefix tuning、bias-only tuning；
* 给出 approximation error 与 site granularity 的关系。

即使不做严格定理，一个清晰的线性化推导也会提升方法可信度。

### 12. 扩展成本模型

当前成本模型主要比较 PCU-Select 与 LESS 的 selection-only GPU-hours。建议进一步报告：

* offline cache 的磁盘占用；
* CPU preprocessing 和 I/O 时间；
* memory footprint；
* 不同 candidate pool size 下的 scaling curve；
* 不同 backbone size 下的 scaling curve；
* 多任务、多 PEFT、多 pool 复用时的摊销公式；
* 与 cheap selectors 的 cost-performance Pareto frontier。

因为 PCU-Select 的实践价值主要来自 amortization，成本模型越完整，论文说服力越强。

---

## 六、可执行的修订优先级

**最高优先级：**

1. 修复所有交叉引用、维度不一致和图表标注问题。
2. 将 per-task 对 LESS 的结果移入主文，降低异质指标平均值的中心地位。
3. 澄清 scorer 训练与 downstream evaluation 之间的 task/PEFT split，避免被质疑数据泄漏或 target leakage。
4. 明确 PEFT code、operator channels、OOD family embedding、high-fidelity label 生成流程。
5. 为 PCU-Select vs RDS+/Influence 补充 paired CI 或统计检验。
6. 补全所有提到但未展示的 baselines 和 Balanced-Random 结果。

**中等优先级：**

1. 增加 site-space ablation，至少比较 24-site 与 projection-level 或 attention/MLP-only 版本。
2. 增加 selection seed/sketch seed 稳定性分析。
3. 补充 token budget 统计，排除长度偏差。
4. 完善 OOD calibration cost 公式和 embedding 初始化说明。
5. 改写摘要和结论，突出“接近 LESS 但可摊销”，而不是暗示性能普遍更优。

**长期优先级：**

1. 增加 cross-backbone、leave-one-task-out、leave-one-family-out 实验。
2. 验证 short-horizon utility 与 full-training utility 的相关性。
3. 扩展任务和 PEFT family，特别是多语言、抽取式 QA、prefix/prompt/BitFit 类方法。
4. 做完整 cost-performance Pareto 分析。

---

## 七、最终评价

这篇论文的核心想法是有价值的：**数据选择应当条件化于 PEFT 配置**。PCU-Select 的结构化设计也比简单的 PEFT family one-hot 更有潜力。当前稿件已经有较完整的实验框架和附录，但主张要成立，还需要更清楚地区分“seen registry 上的摊销效果”和“对新 PEFT/新任务的泛化能力”，并修复多个会影响可信度的技术细节与表述不一致。

如果作者能在修订中补充关键消融、澄清实验拆分、修正图表和维度问题，并把 per-task 结论前置，论文有机会达到可接收水平。
