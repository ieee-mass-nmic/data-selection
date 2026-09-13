from pathlib import Path

ROOT=Path('/Users/bytedance/code/data-selection')
source=(ROOT/'tmp/midterm_assessment_paper/build_paper_form.py').read_text()
source=source.replace("TMP=ROOT/'tmp/midterm_assessment_paper'", "TMP=ROOT/'tmp/midterm_assessment_revision'")
source=source.replace("BASE=ROOT/'output/midterm_assessment/林肯达_研究生学位论文中期考核表.docx'", "BASE=ROOT/'output/midterm_assessment/林肯达_中期考核表_论文式图文版.docx'")
source=source.replace("OUT=ROOT/'output/midterm_assessment/林肯达_中期考核表_论文式图文版.docx'", "OUT=ROOT/'output/midterm_assessment/林肯达_中期考核表_公式与正文实验修订版.docx'")
source=source.replace("(TMP/'method_content.json')", "(ROOT/'tmp/midterm_assessment_paper/method_content.json')")
source=source.replace("(TMP/'figure_review.json')", "(ROOT/'tmp/midterm_assessment_paper/figure_review.json')")
start=source.index('# Native Office Math constructors;')
end=source.index('# The official form labels remain;')
source=source[:start]+(ROOT/'tmp/midterm_assessment_revision/math_layout.py').read_text()+'\n'+source[end:]

motivation='''
heading2(ic,'1.1 背景动机实验')
addp(ic,'动机实验考察同一候选样本在不同 PEFT 配置下是否具有相同效用排序。实验设计固定候选池和目标任务，分别构造各配置的短更新效用排序，以 Spearman 相关系数比较整体排序，以前 5% 样本集合的 Jaccard 重叠衡量高价值样本的一致性；同配置的独立重复用于提供噪声参照。进一步将源配置选出的子集用于其他目标配置，检查排序差异是否影响下游训练效果。')
fig(TMP/'assets/motivation.png','图 1  不同 PEFT 配置的效用排序与前 5% 样本重叠',14.2,cell=ic)
addp(ic,'注：图源为论文引言中的动机分析图；生成脚本对非对角元素统一加 0.10。',size=9,indent=False,line=14,after=4)
addp(ic,'图 1 显示的趋势是：配置结构越接近，排序与高价值样本越一致；跨家族变化则可能产生更大差异。该结果支持的研究假设是：数据价值应同时依赖目标任务和可训练子空间，复用公共计算时仍应保留配置条件。')
'''
source=source.replace("major(ic,'2 创新点')",motivation+"\nmajor(ic,'2 创新点')")
source=source.replace("'图 1  PCU-Select 总体架构：共享信号提取、多保真效用学习与目标条件化选择'","'图 2  PCU-Select 总体架构：共享信号提取、多保真效用学习与目标条件化选择'")

experiments='''
major(wc,'3 实验')
heading2(wc,'3.1 实验设置')
for text in [
'实验按论文正文的三个问题展开：已见配置上的下游质量是否接近逐配置 LESS，公共计算能否降低新增配置的选择成本，以及评分器在未见配置上如何通过少量标签校准。采用 Llama-2-7B 骨干，从去污染后的 30 万条混合指令数据中选择 10%，即 3 万条样本。主实验覆盖 GSM8K、HumanEval、MMLU 和 TyDiQA，分别使用精确匹配率、Pass@1、准确率和 F1，并对四项百分制指标等权平均。',
'每个任务摘要含 32 个训练或开发侧示例，且与评测数据分离；HumanEval 使用 MBPP 和 CodeAlpaca 构造摘要。主实验采用 AD-b64、IA3-attnmlp、L-r16-qkvo、L-r8-mlp、L-r8-qv 五种已见配置。各选择方法固定骨干、PEFT 配方及评测协议，下游训练使用 1000 步 AdamW、全局批量 128、最大序列长度 1024 和余弦学习率调度。计时平台为 8 张 NVIDIA H20-96GB，GPU 小时等于墙钟小时乘以 GPU 数。',
'以下按照论文正文整理实验结果。表中转录均值；描述性统计不用于独立样本显著性结论。']:
 addp(wc,text)

heading2(wc,'3.2 实验一 已见配置上的下游质量对比')
addp(wc,'本实验比较条件选数与逐配置选择的下游质量。Random 不使用任务与 PEFT 条件；RDS+ 按冻结表征与任务摘要的相似度选数；Influence 使用不区分 PEFT 的共享梯度存储；LESS 为每种配置单独建库。PCU-Select 复用表征和评分器，为各目标生成子集。')
addp(wc,'表 1  五种已见 PEFT 配置的主实验结果',bold=True,indent=False,align=WD_ALIGN_PARAGRAPH.CENTER,before=4,after=4,keep=True)
edata=json.loads((ROOT/'tmp/midterm_assessment_paper/experiment_revision_review.json').read_text())
qrows=[[row[0]]+[f'{x:.2f}' for x in row[1:]] for row in edata['experiment_1_quality']['means']]
t=smalltable(wc,['方法','AD-b64','IA3-\\nattnmlp','L-r16-\\nqkvo','L-r8-\\nmlp','L-r8-qv','平均'],qrows,[2.6,1.9,2.2,2.2,1.9,1.9,2.1])
keep_caption_with_table(t)
for c in t.rows[-1].cells:
 for p in c.paragraphs:
  for r in p.runs:r.bold=True
addp(wc,'注：改编自论文正文主实验表，仅保留均值。每个配置列为四个任务原生百分制指标的等权平均，最后一列对五种配置取平均。',size=9,indent=False,line=14,after=4)
addp(wc,'表 1 中 PCU-Select 的总体平均为 35.18，LESS 为 35.14，两者相差 0.04 个指标点；相对 RDS+ 和 Influence 分别增加 0.74 和 0.50 个点。该结果支持在当前注册集合上“平均质量接近”的阶段性判断。配置间仍存在差异：例如在 L-r8-mlp 上，PCU-Select 为 34.56，低于 LESS 的 35.13，因此不能将宏平均接近解释为所有配置和任务均占优。')

heading2(wc,'3.3 实验二 跨配置复用与摊销成本')
addp(wc,'本实验区分只支付一次的固定或共享成本，以及每新增一种 PEFT 配置的边际成本；每种配置均服务四个任务。除前述方法外，加入 Reuse-one-LESS：把一个来源配置选出的任务子集复用于所有目标配置，用于检验直接复用子集的质量代价。表 2 对照质量和选择成本，均排除共同的下游微调开销。')
addp(wc,'表 2  五种已见配置的质量与选择成本',bold=True,indent=False,align=WD_ALIGN_PARAGRAPH.CENTER,before=4,after=4,keep=True)
crows=[]
for row in edata['experiment_2_cost']['rows']:
 name,fixed,marg,total,quality,gap=row
 crows.append([name,f'{fixed:.1f}',f'{marg:.2f}',f'{total:.1f}',f'{quality:.2f}',f'{gap:+.2f}' if gap else '0.00'])
t=smalltable(wc,['方法','固定/共享\\n成本','新增配置\\n成本','五配置\\n总成本','质量','相对 LESS\\n差值'],crows,[3.8,2.2,2.3,2.3,2.0,2.2])
for c in t.rows[-1].cells:
 for p in c.paragraphs:
  for r in p.runs:r.bold=True
addp(wc,'注：改编自论文正文质量与成本表，成本单位为 GPU 小时；质量为四任务与五配置的宏平均，差值单位为指标点。该表按既定五配置、四任务集合计账。',size=9,indent=False,line=14,after=4)
addp(wc,'PCU-Select 将主要成本集中到共享离线阶段：一次性开销为 144.0 GPU 小时，新增四任务配置的成本为 1.51 GPU 小时；LESS 每个配置为 63.2 GPU 小时，边际选择成本比约为 42。Reuse-one-LESS 的质量为 34.24，低于 RDS+ 的 34.44，说明直接复用既有子集可能损失目标匹配性。PCU-Select 的复用对象是公共表征和评分器，目标配置仍保留各自子集。')
equation('cost_general')
addp(wc,'式（6）表示累计选择成本，P 为配置数，Q 为任务数；offline 表示共享离线开销，pair 为一次配置—任务组合的评分与选数成本，data 和 rank 分别表示 LESS 的建库及任务排名成本。固定 Q=4 且目标无须额外校准时，累计成本为 144+1.51P 与 63.2P，其交点为：')
equation('break_even')
addp(wc,'因此，按上述成本模型，从第三种配置开始体现摊销收益；服务五种配置时，累计选择成本分别为 151.6 与 316.0 GPU 小时，成本比约为 2.09。若仅向已有配置增加一个任务，PCU-Select 与 LESS 分别为约 0.38 和 0.30 GPU 小时，收益主要来自跨配置复用。若每个新配置的四个任务均需 500 条标签校准，则每配置另加 4.2 GPU 小时，交点移至约 2.50。上述倍数均不代表端到端训练加速。')

heading2(wc,'3.4 实验三 未见配置上的少量标签校准')
addp(wc,'本实验检查评分器超出训练注册范围后的适用边界。评测前按结构支持程度分组：L0 接近已见配置，L1 为同一家族内的较大结构变化，L2 为未见家族。图 3 汇总各组目标在 GSM8K、HumanEval 和 MMLU 三个任务上的平均差距，纵轴为 LESS 减去 PCU-Select 的分数，数值越小越好。')
addp(wc,'Cal-200 和 Cal-500 分别使用 200 条与 500 条目标配置—任务标签，冻结共享评分器，仅拟合零初始化的线性残差头。每条标签只使用一个已适应检查点和一步更新，对应成本分别为每个配置—任务 0.42 与 1.05 GPU 小时。该设计检验少量局部监督能否修正超出注册范围后的排序偏差。')
fig(TMP/'assets/calibration.png','图 3  注册范围外配置的校准结果（差距越小越好）',10.3)
addp(wc,'注：图源为论文正文，汇总三个任务。L0 未报告校准结果，图中空缺不表示零差距。',size=9,indent=False,line=14,after=4)
addp(wc,'当前稿件中，L0 的直接评分差距为 0.34 个点；L1 由 2.08 经 Cal-200 降至 0.91，再经 Cal-500 降至 0.30。Prefix/P-Tuning 由 6.97 降至 2.17 和 1.99，增加 300 条标签仅再改善 0.18 个点。该趋势表明，同家族偏移可通过少量校准缩小差距，跨家族目标仍可能保留较大误差，需结合结构支持、校准成本和目标质量选择使用范围。')
'''
start=source.index("major(wc,'3 实验')")
end=source.index("major(wc,'4 总结')")
source=source[:start]+experiments+'\n'+source[end:]
source=source.replace("heading2(wc,'4.1 研究总结与不足');blocks('summary')", "heading2(wc,'4.1 研究总结与不足');blocks('summary')\naddp(wc,'下一阶段将整理动机分析、校准与正文实验的运行记录，使背景假设、方法设计与实验结论逐项对应。')")
source=source.replace("major(wc,'4 总结')\n", "major(wc,'4 总结').paragraph_format.space_before=Pt(30)\n")
source=source.replace("inter=TMP/'authored.docx';doc.save(inter)","normalize_body_math_and_properties()\ninter=TMP/'authored.docx';doc.save(inter)")
source=source.replace("'docProps/core.xml'}", "'docProps/core.xml','word/settings.xml'}")
source=source.replace("len(tree.xpath('//m:oMath',namespaces=W))==eqcount", "len(tree.xpath('//m:oMathPara',namespaces=W))==eqcount")
source=source.replace("len(tree.xpath('//w:drawing',namespaces=W))==3", "len(tree.xpath('//w:drawing',namespaces=W))==4")
source=source.replace("'figures':2", "'figures':3").replace('Original figures: 2','Original figures: 3')
(ROOT/'tmp/midterm_assessment_revision/generated_builder.py').write_text(source)
exec(compile(source,str(ROOT/'tmp/midterm_assessment_revision/generated_builder.py'),'exec'))
