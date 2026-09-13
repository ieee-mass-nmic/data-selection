from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
from hashlib import sha256
import json
import re
from lxml import etree as E

ROOT = Path('/Users/bytedance/code/data-selection')
TMP = ROOT / 'tmp/midterm_assessment_polish'
SRC = ROOT / 'output/midterm_assessment/林肯达_中期考核表_更新版.docx'
OUT = ROOT / 'output/midterm_assessment/林肯达_中期考核表_润色优化版.docx'
EXPECTED = '37effb0fb242fd59222865d9da93d9d78798de5e42e38506b4d0397232fc0126'
assert sha256(SRC.read_bytes()).hexdigest() == EXPECTED
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
def tag(n): return '{'+NS['w']+'}'+n
def txt(p): return ''.join(p.xpath('.//w:t/text() | .//m:t/text()', namespaces=NS))
with ZipFile(SRC) as z:
    entries = z.infolist()
    parts = {x.filename: z.read(x.filename) for x in entries}
root = E.fromstring(parts['word/document.xml'])
baseline = deepcopy(root)
ps = root.xpath('//w:body//w:p', namespaces=NS)

# {{n}} preserves the nth existing inline Office Math object within a paragraph.
edits = {
22: '1 研究背景与动机',
23: '指令微调能够提升大语言模型在特定任务上的表现，但训练数据量大、实验配置多，往往需要较高的计算成本。参数高效微调（PEFT）通过只更新少量参数降低训练开销，数据选择则通过保留更有价值的样本减少训练数据量。两者会相互影响：不同 PEFT 配置更新的模型位置、参数形式和容量不同，同一样本的训练价值也可能随之变化。因此，数据选择需要同时考虑目标任务和模型的更新方式。',
24: '现有方法在目标适配与计算复用之间存在权衡。基于语义相似度或共享梯度的方法可以复用公共计算，却难以充分反映 PEFT 配置的差异。逐配置 LESS 为每种配置建立专属梯度库，再按任务评分，能够更好地适配目标，但新增配置需要重复建库；直接复用其他配置选出的子集，又可能降低效果。为此，本文提出 PCU-Select：共享样本信号和评分器，以任务与配置为条件预测样本价值，并为各目标生成相应子集。',
25: '1.1 动机实验',
26: '动机实验考察：同一批样本在不同 PEFT 配置下，训练价值的排序是否一致。实验固定候选池和目标任务，根据各配置下少量步数更新带来的收益对样本排序。采用 Spearman 相关系数衡量整体排序的一致性，采用 Jaccard 重叠度衡量前 5% 高价值样本的重合程度，并通过同配置的独立重复估计随机波动。进一步将源配置选出的子集用于其他目标配置，检查排序差异是否影响下游表现。',
31: '图 1 表明，配置结构越接近，样本排序和高价值样本集合越一致；跨家族配置可能出现更大差异。这支持了本文的研究思路：样本价值不仅取决于任务，也与模型中允许更新的参数范围有关。因此，在复用公共计算时，仍需把 PEFT 配置作为评分条件。',
33: '（1）同时考虑任务与配置的效用建模。将样本、PEFT 配置和任务共同作为输入，预测样本在该配置下对目标任务的训练价值。通过共享干预位点统一样本信号的表示方式，再用结构化编码描述更新位置、算子、容量和训练设置，使共享表示能够保留配置差异。',
34: '（2）结合两类监督信号的共享评分器。利用可缓存的位点梯度生成低成本排序标签，再用少量短更新得到的实际收益修正近似误差。评分器通过条件调制和交互建模预测样本效用及其不确定性，减少为不同目标重复建立选择器的开销。',
35: '（3）兼顾样本价值与语义覆盖的子集选择。结合样本得分及其不确定性进行保守排序，再依据语义簇的得分和规模分配选数名额，减少样本过度集中的问题。同时区分一次性离线成本与新增配置成本，分析配置数量增加后共享计算带来的成本收益。',
38: '设固定骨干模型为 {{0}}，候选指令数据池为 D，PEFT 配置为 p，目标任务为 t，选数预算为 B。其中，p 包括更新位置、算子、容量和训练设置。理想目标是在预算内选出子集 S，使微调后的模型在目标任务上取得尽可能好的表现：',
43: [
 '其中，Tune 表示按配置 p 执行微调，Perf 表示任务评价指标。测试集仅用于定义目标和最终评测，不参与训练、评分或选数。实际选择使用任务摘要 {{0}}，即来自训练集或开发集的少量任务示例。',
 '由于无法逐一训练并比较所有候选子集，本文用条件效用 u(x,p,t) 近似描述样本 x 在配置 p 下对任务 t 的训练价值，并学习共享评分器进行预测。引入配置条件后，样本排序可以随模型的可训练参数范围变化；共享表征与评分器则把主要建模开销集中到离线阶段，从而降低多配置场景中的累计选择成本。'],
46: 'PCU-Select 包括离线建模和在线选择两个阶段。离线阶段提取可复用的样本与任务信号，编码 PEFT 配置，并用多保真标签训练共享评分器。在线阶段输入目标配置和任务，利用缓存特征为候选样本评分，再按语义簇分配选数名额，生成训练子集。图 2 展示了这一流程：底层计算可以共享，每个配置与任务组合仍有自己的排序、配额和子集。',
48: '图 2  PCU-Select 总体框架',
51: [
 '不同 PEFT 方法的参数梯度形状和作用方式不同，难以直接共用梯度缓存。本文在八个均匀采样的 Transformer 层中，选取注意力、前馈网络和残差输出，共形成 24 个共享干预位点，即用于观察更新影响的公共位置。根据链式法则，参数更新对损失的影响可以通过模块输出传递，因此使用这些位点上的响应损失梯度描述样本和任务信号。',
 '各 PEFT 配置被映射到相应的作用位点。位点索引说明“在哪里更新”，结构化编码进一步说明“如何更新”及更新容量。这样既能对齐不同配置的信号，也能保留不同更新方式之间的差异。'],
52: [
 '评分器接收三类定长向量。样本向量共 848 维，包括 768 维冻结语义嵌入、16 维长度与损失等统计特征，以及 64 维隐藏状态摘要。任务向量由任务摘要中 32 个示例的样本向量取均值得到；用于构造监督标签的任务位点梯度另行保存。',
 'PEFT 向量共 128 维，包括 96 维位点及算子编码，以及各 16 维的容量和训练设置编码。评分器由此根据配置的实际结构进行预测，而不是仅按 PEFT 家族名称查找固定结果。'],
54: '逐样本完整微调可以直接观察训练收益，但成本过高。本文采用多保真监督，即结合低成本的近似标签与少量更精确的观测标签。低保真标签使用梯度代理，衡量样本梯度与任务梯度在目标配置有效位点上的方向一致性：',
59: '其中，g 为经随机投影压缩至 256 维并归一化的位点梯度，权重由配置的有效位点和容量确定。代理值在同一配置与任务条件下进行排序归一化。样本梯度只需提取一次，即可为多个配置与任务组合生成监督标签。',
60: '高保真标签通过少量步数更新直接观测任务摘要的损失下降：',
65: [
 '式中，{{0}} 为更新起点，分别取基座模型和已开始适应的检查点；更新步数 h 取 1 或 4。小批次 B(x) 包含一个目标样本和七个条件匹配样本。共采集一万条短更新标签，采样兼顾数据覆盖、预测不确定性和选择边界，并在相同配置、任务、起点及步数组内归一化。',
 '两类监督分工不同：梯度代理用于学习大范围的排序规律，短更新观测用于修正关键区域的偏差。短更新标签反映的是局部小批次的更新效果，不能视为单个样本在完整训练中的独立因果贡献。'],
67: [
 '条件评分器分别编码样本、配置和任务，再通过 FiLM 条件调制，根据配置与任务信息对样本特征进行缩放和平移。同时，利用双线性交互刻画样本与配置的匹配关系。训练时先学习梯度代理给出的排序，再结合短更新标签学习排序、效用回归和不确定性。',
 '评分器输出效用均值预测 {{0}} 和随输入变化的不确定性尺度 {{1}}。在线选择时，从预测效用中减去不确定性惩罚，得到保守分数：'],
72: '式（4）降低预测不稳定样本的优先级，不确定性项仅用于排序，不表示统计置信区间。为避免高分样本集中在少数语义区域，进一步对样本进行语义聚类。设 {{0}} 为第 k 个簇，{{1}} 为簇内前 10% 样本的平均保守分数，{{2}} 表示将负值截断为零后的分数，则该簇的选数配额为：',
77: '式（5）结合目标效用和簇规模分配预算，再从每个簇中选取分数最高的样本，并按确定性规则处理舍入余数，使总选数满足预算。30 万候选样本按既定规则形成 547 个簇。语义聚类可供不同目标共享，各目标的条件分数则决定具体的配额分布。',
78: '对于已见配置和结构接近的配置，评分器可直接使用。对于同一家族内较大的结构变化或未见家族，若有兼容标签，则进行少量校准；否则采用逐目标选择方法。该适用范围建立在固定骨干及兼容表征之上，不能直接扩展到任意配置或骨干模型。',
81: [
 '实验围绕三个问题展开：在已见配置上，PCU-Select 的下游表现能否接近逐配置 LESS；共享计算能否降低多配置选择成本；在未见配置上，少量标签校准能否改善效果。',
 '实验采用 Llama-2-7B 骨干模型，从去污染后的 30 万条混合指令数据中选择 10%，即 3 万条样本。主实验包含 GSM8K、HumanEval、MMLU 和 TyDiQA 四个任务，分别采用精确匹配率、Pass@1、准确率和 F1，并对四项百分制指标取等权平均。'],
82: [
 '每个任务摘要包含 32 个训练集或开发集示例，与评测数据保持分离；HumanEval 的摘要由 MBPP 和 CodeAlpaca 构造。主实验覆盖 AD-b64、IA3-attnmlp、L-r16-qkvo、L-r8-mlp 和 L-r8-qv 五种已见配置，即评分器训练阶段已纳入的配置。',
 '各方法使用相同的骨干、PEFT 训练设置和评测协议。下游训练采用 AdamW 优化器，共训练 1000 步，全局批量为 128，最大序列长度为 1024，并使用余弦学习率调度。计时平台为 8 张 NVIDIA H20-96GB，GPU 小时按实际运行小时数乘以 GPU 数计算。'],
83: '以下结果整理自论文正文，表中保留均值，用于比较整体表现，不据此作出基于独立样本的显著性判断。',
85: '本实验比较各方法选出数据后的下游表现。Random 随机选数；RDS+ 按冻结表征与任务摘要的相似度选数；Influence 使用不区分 PEFT 配置的共享梯度库；LESS 为每种配置单独建库。PCU-Select 共享表征和评分器，并为每个配置与任务组合生成对应子集。',
131: '注：改编自论文正文主实验表，仅保留均值。各配置列为四个任务百分制指标的等权平均，最后一列为五种配置的平均值。',
132: '表 1 中，PCU-Select 的总体平均分为 35.18，LESS 为 35.14，相差 0.04 个指标点；相比 RDS+ 和 Influence，分别高 0.74 和 0.50 个指标点。可见，在五种已见配置上，PCU-Select 的平均表现接近逐配置 LESS。但在 L-r8-mlp 上，PCU-Select 为 34.56，低于 LESS 的 35.13。因此，平均分接近不意味着所有配置和任务均占优。',
134: '本实验比较多个 PEFT 配置下的数据选择成本，区分一次性固定或共享开销与新增配置开销，每种配置均服务四个任务。同时加入 Reuse-one-LESS，将一个源配置为各任务选出的子集用于所有目标配置，以检验直接复用子集的影响。表 2 报告选择质量与成本，不计各方法共有的下游微调开销。',
179: '注：改编自论文正文。成本单位为 GPU 小时，按五种配置、四个任务计算；质量为各任务与配置的平均分，差值单位为指标点。',
180: [
 'PCU-Select 的一次性离线开销为 144.0 GPU 小时，此后每增加一种服务四个任务的配置，选择开销为 1.51 GPU 小时。LESS 每种配置需要 63.2 GPU 小时，约为 PCU-Select 新增配置开销的 42 倍。',
 'Reuse-one-LESS 的质量为 34.24，低于 RDS+ 的 34.44，说明直接复用已有子集可能降低与目标配置的匹配程度。PCU-Select 复用的是公共表征和评分器，仍为各目标配置分别生成子集。'],
185: '式（6）描述累计选择成本，其中 P 为配置数，Q 为任务数。offline 表示共享离线开销，pair 表示一个配置与任务组合的评分和选数成本；data 和 rank 分别表示 LESS 的建库与任务排序成本。当 Q=4 且目标配置无须额外校准时，两种方法的累计成本分别为 144+1.51P 和 63.2P，成本相等时的配置数为：',
190: [
 '按这一成本模型，从第三种配置开始，PCU-Select 的累计选择成本低于 LESS。服务五种配置时，两者分别需要 151.6 和 316.0 GPU 小时，LESS 的成本约为 PCU-Select 的 2.09 倍。收益主要来自跨配置复用：若仅为已有配置增加一个任务，两者开销分别约为 0.38 和 0.30 GPU 小时。',
 '若每个新配置的四个任务均需 500 条标签校准，PCU-Select 每配置还需增加 4.2 GPU 小时，成本交点移至约 2.50。上述成本比较仅涉及数据选择阶段，不能直接视为端到端训练的加速倍数。'],
192: '本实验考察评分器对未见配置的适用范围。评测前按结构接近程度分组：L0 接近已见配置，L1 为同一家族内的较大结构变化，L2 为未见家族。图 3 汇总各组在 GSM8K、HumanEval 和 MMLU 三个任务上的平均分数差距。纵轴为 LESS 得分减去 PCU-Select 得分，数值越小越好。',
193: 'Cal-200 和 Cal-500 分别使用 200 条和 500 条目标配置与任务组合的标签进行校准。校准时固定共享评分器，仅训练零初始化的线性残差头，用于修正原有预测。每条标签通过一个已适应检查点的一步更新产生；两种方案在每个配置与任务组合上的成本分别为 0.42 和 1.05 GPU 小时，用于检验少量局部监督能否纠正未见配置的排序偏差。',
195: '图 3  未见配置的校准结果',
198: '结果显示，L0 直接评分与 LESS 的差距为 0.34 个指标点。L1 未校准时的差距为 2.08，采用 Cal-200 和 Cal-500 后分别缩小至 0.91 和 0.30。Prefix/P-Tuning 的差距由 6.97 分别缩小至 2.17 和 1.99，标签数从 200 增至 500 仅进一步改善 0.18 个指标点。这说明，少量校准能够缓解同一家族内的结构偏移，但跨家族配置仍可能存在较大差距，使用时需兼顾结构兼容性、校准成本与目标质量要求。',
201: [
 '本文针对多配置数据选择中的目标适配与重复计算问题，提出 PCU-Select。该方法通过共享干预位点复用样本信号，以多保真监督训练条件评分器，再结合聚类配额选数，兼顾样本效用与语义覆盖。现阶段已完成方法设计和实验分析，并按任务、配置和训练种子汇总结果。实验表明，在五种已见配置上，方法的平均表现接近逐配置 LESS，跨配置复用能够降低累计选择成本。',
 '方法的适用范围仍受骨干模型、表征和配置兼容性约束。结构接近已见配置时可直接评分；结构偏移较大时，需要兼容标签进行校准，否则采用逐目标选择方法。后续将进一步分析任务差异、预算敏感性和额外校准成本，明确质量与成本结论的适用条件。'],
202: '下一阶段还将整理动机实验和正文三个实验的运行记录，使研究假设、方法设计与实验结论形成清晰的对应关系。',
204: [
 '第二项工作拟研究“基于 PCU-Select 的多配置数据选择与微调实验管理系统”，复用已有评分器、特征缓存、PEFT 编码和选数算法，重点实现以下两项工程机制。',
 '第一项是分层缓存与增量复用。系统记录样本内容摘要、模型与评分器版本、预处理配置、随机种子、候选顺序及产物校验值，据此判断哪些结果可复用。仅改变选数预算时，复用评分与聚类，重算配额和子集；选择种子不变、仅改变训练种子时，复用子集；追加候选样本且评分条件不变时，只补算新增样本的兼容特征和逐样本评分，但仍需重算完整池的聚类与配额。'],
205: [
 '第二项是公共计算去重与阶段恢复。系统按依赖关系将实验拆为执行节点，相同输入的节点只计算一次；通过原子写入、完成标记和独立目录保存结果，避免覆盖。中断后复用已完成的结果，仅重跑失败阶段。',
 '首版固定评分器版本，采用单工作节点、任务队列、元数据索引和本地产物库，并提供控制台，支持提交实验矩阵、查看复用计划和导出报告。',
 '以同一候选池上的两种配置、两个任务、三个预算和两个训练种子为例，共需开展 24 次训练。固定特征与评分器版本、聚类配置和选择种子后，计划共享四份目标评分和一份聚类，生成十二份子集。各配置与任务仍使用各自子集。',
 '验证将比较现有脚本、加入依赖缓存的方案和完整系统。先检查复用结果与全量重算是否一致，再测量计算量、运行时间、存储占用和恢复开销。公共计算次数的减少不直接等同于训练加速。'],
206: '后续八周将按表 3 推进系统实现与验证，并同步整理工作一的实验记录、独立重复结果和学位论文相关章节。',
212: '统一算法接口，设计产物依赖关系与元数据，归档论文实验记录。',
213: '依赖规范、最小执行流程和实验结果归档清单。',
215: '实现缓存与增量计算，验证预算、种子和样本变化时的复用规则。',
216: '缓存原型及复用与全量重算的一致性记录。',
218: '实现公共计算去重、任务队列、阶段恢复和控制台。',
219: '可运行系统及结果隔离、阶段恢复验证记录。',
221: '完成三种方案对比，分析时间与存储开销，补充论文实验。',
222: '工程机制验证报告、系统说明和学位论文相关章节。',
224: '各阶段以可运行产物和可核验记录为完成依据，根据实验资源及导师指导调整。',
226: '已形成英文论文稿《PCU-Select: Amortizing Target-Specific Data Selection Across PEFT Configurations》，完成方法设计与现阶段实验分析，正在整理实验材料并完善论文论证。同时，已实现特征提取、PEFT 配置注册、条件评分、聚类配额选数及训练评测等基础脚本，为后续系统研究提供可复用模块。',
}

changes = []
for index, new in edits.items():
    oldp = ps[index]
    assert not oldp.xpath('.//w:drawing | .//m:oMathPara | .//w:fldChar', namespaces=NS), index
    oldtext = txt(oldp)
    math = list(oldp.findall('m:oMath', NS))
    values = new if isinstance(new, list) else [new]
    tokens = [int(x) for s in values for x in re.findall(r'\{\{(\d+)\}\}', s)]
    assert tokens == list(range(len(math))), (index, tokens, len(math))
    rpr = oldp.find('w:r/w:rPr', NS)
    parent = oldp.getparent()
    where = parent.index(oldp)
    for j, value in enumerate(values):
        p = E.Element(tag('p'), attrib=oldp.attrib)
        if oldp.find('w:pPr', NS) is not None:
            p.append(deepcopy(oldp.find('w:pPr', NS)))
        for segment in re.split(r'(\{\{\d+\}\})', value):
            if not segment: continue
            if re.fullmatch(r'\{\{\d+\}\}', segment):
                p.append(deepcopy(math[int(segment[2:-2])]))
            else:
                run = E.SubElement(p, tag('r'))
                if rpr is not None: run.append(deepcopy(rpr))
                t = E.SubElement(run, tag('t'))
                t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
                t.text = segment
        parent.insert(where+j, p)
    parent.remove(oldp)
    changes.append({'paragraph': index, 'before': oldtext, 'after': values})

def prop(parent, name, **values):
    x = parent.find('w:'+name, NS)
    if x is None: x = E.SubElement(parent, tag(name))
    for k,v in values.items(): x.set(tag(k), str(v))
    return x

main = root.find('w:body', NS).findall('w:tbl', NS)[1]
heading_texts = {'1 研究背景与动机','1.1 动机实验','2 创新点','1 问题定义','2 方法','2.1 总体框架',
    '2.2 共享干预位点与结构化表征','2.3 多保真效用监督','2.4 条件评分与聚类配额选择','3 实验',
    '3.1 实验设置','3.2 实验一 已见配置上的下游质量对比','3.3 实验二 跨配置复用与摊销成本',
    '3.4 实验三 未见配置上的少量标签校准','4 总结','4.1 研究总结与不足','4.2 后续研究计划'}
for p in main.xpath('.//w:p', namespaces=NS):
    text = txt(p)
    ppr = p.find('w:pPr', NS)
    if text in heading_texts:
        prop(ppr, 'keepNext', val='1')
        if text == '4 总结': prop(ppr, 'spacing', before=180, after=100, line=360, lineRule='exact')
    if text.startswith('注：') or text.startswith('第二项是公共计算去重'):
        prop(ppr, 'keepLines', val='1')
    if re.match(r'^[图表] [123]  ', text):
        prop(ppr, 'jc', val='center')
        for run in p.findall('w:r', NS):
            rp = run.find('w:rPr', NS)
            if rp is None: rp = E.SubElement(run, tag('rPr'))
            prop(rp, 'rFonts', ascii='Times New Roman', hAnsi='Times New Roman', eastAsia='宋体')
            prop(rp, 'sz', val=21)
            prop(rp, 'b', val='0')

# Keep the two official signature areas together on their own complete page.
for p in root.find('w:body', NS).findall('w:p', NS):
    if txt(p) == '二、中期考核作者承诺及导师意见':
        prop(p.find('w:pPr', NS), 'pageBreakBefore', val='1')

# Preserve all mathematical and visual content and all official form parts.
def xmls(r, xp): return [E.tostring(x) for x in r.xpath(xp, namespaces=NS)]
assert xmls(root, '//m:oMath') == xmls(baseline, '//m:oMath')
assert xmls(root, '//w:drawing') == xmls(baseline, '//w:drawing')
before_tables = baseline.find('w:body', NS).findall('w:tbl', NS)
after_tables = root.find('w:body', NS).findall('w:tbl', NS)
for i in [0,2,3]: assert E.tostring(before_tables[i]) == E.tostring(after_tables[i])
assert xmls(root, '//w:sectPr') == xmls(baseline, '//w:sectPr')
for word in ['模拟数据','占位值','待核验','待替换','非对角元素统一加']:
    assert word not in txt(root)

parts['word/document.xml'] = E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(OUT, 'w') as z:
    for item in entries: z.writestr(item, parts[item.filename])
with ZipFile(SRC) as a, ZipFile(OUT) as b:
    changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
assert changed == ['word/document.xml'], changed
assert sha256(SRC.read_bytes()).hexdigest() == EXPECTED
audit = {'source': str(SRC), 'output': str(OUT), 'source_sha256': EXPECTED,
         'output_sha256': sha256(OUT.read_bytes()).hexdigest(), 'changed_parts': changed,
         'native_math_unchanged': True, 'drawings_unchanged': True,
         'official_forms_unchanged': True, 'section_geometry_unchanged': True,
         'changes': changes}
(TMP/'changes.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
print(json.dumps({k:v for k,v in audit.items() if k!='changes'},ensure_ascii=False,indent=2))
print('Rewritten source paragraphs:',len(changes))
